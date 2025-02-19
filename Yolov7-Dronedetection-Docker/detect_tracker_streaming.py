# Made changes to the original script - P. Bhattacharya, P. Nowak

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from flask import Flask, Response, render_template

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from trackers.filter_detections import filter_detections
from trackers.xcorr.xcorrtracker import xcorrtrack
from models.common import CustomClassificationNet

import threading

### Flask -------------------------------
outputFrame = None
lock = threading.Lock()
tt = 0

app = Flask(__name__)

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")
# --------------------------------------

# Detect and track ----------------------------------------------------------------------------------------------------------------------
def detect(save_img=False):
    
    global outputFrame, lock, tt
    source, weights, view_img, save_txt, imgsz, trace, istracking, tracker, stream_img = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.tracking, opt.tracker, opt.stream_img
    
    save_img = True # Original - not opt.nosave and not source.endswith('.txt')  # save inference images

    # Live stream input
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) or source.lower().startswith(('udpsrc'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    ### Custom -  For tracking with xcorr
    bboxes_old, bboxes_pred, confs_old, confs_pred, len_old = [], [], [], [], 0
    
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Model tracing - optimizes the model
    if trace:
        model = TracedModel(model, device, opt.img_size)
    
    # Uses lower precision to save the parameters
    if half:
        model.half()  # to FP16
        mmr = round(torch.cuda.get_device_properties(0).total_memory/1000000000)
        if mmr < 16.0:
            t = 4
        elif mmr < 32.0:
            t = 3
        else:
            t = 2

    ### Custom - Second-stage classifier - implemented but no model is trained, hence incomplete and not tested
    classify = False
    if classify:
        # Original
        # modelc = load_classifier(name='resnet101', n=2)  # initialize
        # modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
        modelc = CustomClassificationNet(2)
        modelc.load_state_dict(torch.load('runs/trainclassifier/exp/weights/best.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        ### Custom -  Get input udp stream (Gstreamer)
        if source == 'udpsrc': 
            pipeline = 'udpsrc port=5001 ! application/x-rtp,payload=96,encoding-name=H264 ! rtpjitterbuffer mode=1 latency=50 ! rtph264depay ! h264parse ! decodebin ! videoconvert ! appsink'
            dataset = LoadStreams(pipeline, img_size=imgsz, stride=stride, mem=t)
            fps_s,w_s,h_s  = dataset.fps, dataset.imgs[0].shape[1], dataset.imgs[0].shape[0]
        # Load webcam image
        else:
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, mem=t)
            fps_s,w_s,h_s  = dataset.fps, dataset.imgs[0].shape[1], dataset.imgs[0].shape[0]
    else:
        # Get video
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        fps_s,w_s,h_s  = round(dataset.cap.get(cv2.CAP_PROP_FPS)), int(dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ### Custom -  Prepare stream output (Gstreamer)
    if stream_img:
        out_send = cv2.VideoWriter('appsrc ! queue ! videoconvert ! queue ! video/x-raw,format=I420 ! x264enc speed-preset=veryfast tune=zerolatency bitrate=400 ! rtph264pay ! udpsink host=' + opt.stream_ip + ' port=' + opt.stream_port ,cv2.CAP_GSTREAMER,0, fps_s, (w_s,h_s), True)
        
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    itr = 0
    for path, img, im0s, vid_cap in dataset:
        t00 = time_synchronized()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply Non Maximal Supression (NMS)
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier (Not implemented - no model available)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            ### Custom -  Tracking
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                if istracking:
                    if tracker == 'xcorr':                      
                        # Use tracker
                        if len(bboxes_old):
                            bboxes_pred, confs_pred = xcorrtrack(imold, im0, 12, bboxes_old, det[:, :4], confs_old, det[:,4],opt.conf_thres,opt.min_thres,itr)
                            bboxes_pred = bboxes_pred.to(device)
                            confs_pred  = confs_pred.to(device)

                        if len(bboxes_pred):
                            det[:, :4] = bboxes_pred # comment to not use the xcorr predicted box, if unreliable
                            det[:,4]   = confs_pred
                        
                        confs_old  = det[:,4]   # confs_pred
                        bboxes_old = det[:, :4] # bboxes_pred
                        imold, len_old      = im0, len(det)
                else:
                    # Reset tracking variables
                    confs_old, bboxes_old, bboxes_pred, confs_pred  = [], [], [], []
            else:
                # Reset tracking variables
                confs_old, bboxes_old, bboxes_pred, confs_pred  = [], [], [], []

            if len(det):
                #det[:,4] = 0.0
                det = filter_detections(det,opt.min_thres,im0)            
            
            if len(det):
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {c}{'s' * (n > 1)}, "  # add to string # names[int(c)]

                # Write results
                # det = trim(det) # -> for viewing just use the detections(det) with higher confidence scores
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

            # Print time (inference + NMS)
            itr = itr+1
            tt = tt + ((t1-t00)+(t2-t1) + (t3-t2))
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS, ({(1E3 * (tt/itr)):.1f}ms) Total')
            
            with lock:
                outputFrame = im0.copy()
                
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            if stream_img:
                out_send.write(im0)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

#-----------------------------------------------------------------------------------------------------------------------------------

# Flask ----------------------------------------------------------------------
# encode frames for Flask
def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

# Flask feed - should be implemented in main
@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")

# Main---------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/color/exp_cat/weights/best.pt', help='model.pt path(s)')
    # -- source
    # -----------------------------------------------------------
    # file/folder, 0 - webcam (was default from yolo, not tested)
    # video - 'datasets/infrared/FLIR0000023.mp4'
    # gstreamer stream - 'udpsrc'
    parser.add_argument('--source', type=str, default= 'datasets/color/00009.mp4', help='source')
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.005, help='object confidence threshold')
    parser.add_argument('--min-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--tracking', type=bool, default=True, help='Enable or disable tracking')
    parser.add_argument('--tracker', type=str, default='xcorr', help='select tracker') # only one is implemented now
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', default = True, help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', default=False, help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--stream-img', action='store_true', default = True, help='stream results')
    parser.add_argument('--stream-ip', type=str, default='10.246.57.28', help='ip where the stream with detection is sent to')
    parser.add_argument('--stream-port', type=str, default='5000', help='port where the stream is send to')    
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()