import numpy as np
import torch
from utils.general import im2patch

im = torch.zeros(1,4,4)
im[:,0:2,0:2] = 0
im[:,0:2,2:4] = 64
im[:,2:4,0:2] = 128
im[:,2:4,2:4] = 256
psz = 2
bs = int(np.ceil((im.shape[1]/psz)*(im.shape[2]/psz)))

newim = im2patch(im, psz, bs)
print(newim.shape)
print(im)
for i in range(bs):
    print(newim[i,:,:,:])