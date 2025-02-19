import argparse
import logging
import math
import os
import random
import time
import yaml
import torch

import numpy as np

import torchvision.transforms as transforms
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as tdata

from torchvision import datasets
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
from models.experimental import attempt_load
#from models.yolo import Model
from copy import deepcopy
from pathlib import Path
#from threading import Thread

from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel

from models.common import CustomClassificationNet

logger = logging.getLogger(__name__)

def train_step(model, dataloader, loss_fn, optimizer, device):
    
    # Select mode
    model.train()

    # Initialize metrices
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Send data to GPU/CPU
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)          # forward
        loss   = loss_fn(y_pred,y) # calculate loss
        train_loss += loss.item()  # accumulate loss

        optimizer.zero_grad()
        loss.backward()            # backward
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model, dataloader, loss_fn, device):

    # Select mode
    model.eval() 
    
    # Initialize metrices
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model, train_dataloader, test_dataloader, optimizer, write_dir, device, loss_fn, epochs):
    
    # Initialize results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)

        test_loss, test_acc   = test_step(model=model,
                                          dataloader=test_dataloader,
                                          loss_fn=loss_fn,
                                          device=device)
        
        # Print training progress
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        final_epoch = epoch + 1 == epochs
        if (not opt.nosave):
            if epoch % 50 == 49:
                torch.save(model.state_dict(), write_dir / 'epoch_{:03d}.pt'.format(epoch))
            if final_epoch: 
                torch.save(model.state_dict(), write_dir / 'last.pt')

    # 6. Return the filled results at the end of the epochs
    return results

def train_classifier(opt, device):
    # Cache opt data
    save_dir, epochs, batch_size, total_batch_size, weights, rank, num_workers = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank, opt.workers

    # Make directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    # Image transforms
    data_transform_train = transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
        transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
    ])

    data_transform_test = transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
    ])

    # Setup datasets
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    
    train_dir = data_dict['train']
    test_dir  = data_dict['test']
    num_cls   = data_dict['nc'] # number of classes

    trainset    = datasets.ImageFolder(root=train_dir, transform=data_transform_train, target_transform=None)
    trainloader = tdata.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    testset     = datasets.ImageFolder(root=test_dir, transform=data_transform_test)                                       
    testloader  = tdata.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True) 

    # Check data
    img, lbl = next(iter(trainloader))
    print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
    print(f"Label shape: {lbl.shape}") 

    img, lbl = next(iter(testloader))
    print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
    print(f"Label shape: {lbl.shape}")                                    

    # Define network
    model = CustomClassificationNet(num_cls).to(device)

    loss_fn = nn.CrossEntropyLoss()
    if opt.adam:
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=[0.9,0.95], eps=1e-16, weight_decay=0.00005)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.975, weight_decay=0.0001)

    # Start timing
    start_time = time.time()

    # Train model
    results = train(model=model, 
                    train_dataloader=trainloader,
                    test_dataloader=testloader,
                    optimizer=optimizer,
                    write_dir=wdir,
                    device=device,
                    loss_fn=loss_fn, 
                    epochs=epochs)

    # End the timer and print out how long it took
    end_time = time.time()
    print(f"Total training time: {(end_time-start_time)/60:.3f} mins")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='pre-trained_wts/yolov7-w6.pt', help='initial weights path')
    parser.add_argument('--data', type=str, default='data/dronedatasetclassify.yaml', help='data.yaml path')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=2, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[128, 128], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', default = False, help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', default = True, help='only test final epoch')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', default=False, help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/trainclassifier', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--save_period', type=int, default=10, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    opt = parser.parse_args()

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)

    if opt.resume:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data = check_file(opt.data)  # check files
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Train
    logger.info(opt)
    
    #tb_writer = None  # init loggers
    if opt.global_rank in [-1, 0]:
        #prefix = colorstr('tensorboard: ')
        #logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
        #tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        logger.info("Start Training")
        train_classifier(opt, device)

    '''
    # NORMAL TRAINING FUNCTION
    # ------------------------------------------------------------------ 
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
                   
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i+1):.3f}')
        final_epoch = epoch + 1 == epochs
        if epoch % 20 == 0:
            torch.save(model.state_dict(), wdir / 'epoch_{:03d}.pt'.format(epoch))

    print('Finished Training')
    '''