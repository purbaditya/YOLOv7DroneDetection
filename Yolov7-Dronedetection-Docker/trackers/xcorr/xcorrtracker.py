### Custom - set of methods including cross-correlation based bbox prediction or registration for tracking or reducing dropped detection
# P. Bhattacharya

import os
import cv2
import torch
import numpy as np
from PIL import Image
from utils.general import bbox_iou, bbox_alpha_iou, box_iou, box_giou, box_diou, box_ciou, xywh2xyxy
# import matplotlib.pyplot as plt

### Predict shift via xcorrelation
def find_bbox(obj, patch, bbox, scale):
    if scale > 1:
        patch = cv2.resize(patch, (int(scale*patch.shape[0]), int(scale*patch.shape[1])), interpolation=cv2.INTER_CUBIC)
        obj = cv2.resize(obj, (int(scale*obj.shape[0]), int(scale*obj.shape[1])), interpolation=cv2.INTER_CUBIC)
    cc = cv2.matchTemplate(patch, obj, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(cc)

    shift = [max_loc[0] - int(cc.shape[1] / 2), max_loc[1] - int(cc.shape[0] / 2)]
    if np.abs(shift[0]) > 10:
        shift[0] = 0
    if np.abs(shift[1]) > 10:
        shift[1] = 0
    
    p = [bbox[0] + shift[0], bbox[1] + shift[1]]
    bbox_pred = [p[0], p[1], p[0] + obj.shape[1], p[1] + obj.shape[0]]

    return bbox_pred, max_val

### Register bounding boxes between consecutive frames, add new box or delete old / false box according to requirements
def match_bbox(bboxes, bboxes_prev, conf_prev, conf):
    lenbp = len(bboxes_prev)
    lenb  = len(bboxes)
    d = np.zeros([lenb,lenbp])
    s = np.zeros([lenb,lenbp])

    # number of boxes in previous frame was more
    if lenbp > lenb:
        for idx in range(lenb):
            bbox = bboxes.cpu().numpy()[idx]
            for id in range(lenbp):
                bbox_prev = bboxes_prev.cpu().numpy()[id]
                d[idx,id] = np.sqrt(np.square(bbox[0] + (bbox[2]-bbox[0])/2 - bbox_prev[0] - (bbox_prev[2]-bbox_prev[0])/2) + np.square(bbox[1] + (bbox[3]-bbox[1])/2 - bbox_prev[1] - (bbox_prev[3]-bbox_prev[1])/2))
                s[idx,id] = np.abs((np.abs(bbox[2]-bbox[0])*np.abs(bbox[3]-bbox[1])) - (np.abs(bbox_prev[2]-bbox_prev[0])*np.abs(bbox_prev[3]-bbox_prev[1])))
            d_id = d[idx,:]
            p_id  = np.where(d_id == d_id.min())
            s_id = s[idx,:] 
            if len(p_id[0]) > 1:
                p_id = np.where(s_id == s_id.min())
            if idx !=  p_id[0][0]:
                tmpb = bboxes_prev[p_id[0][0]]
                tmpc = conf_prev[p_id[0][0]]
                bboxes_prev[p_id[0][0]] = bboxes_prev[idx]
                conf_prev[p_id[0][0]] = conf_prev[idx]
                bboxes_prev[idx] = tmpb
                conf_prev[idx] = tmpc
        bboxes_prev = bboxes_prev[0:lenb]
        conf_prev = conf_prev[0:lenb]

    # number of boxes in previous frame was less
    elif lenbp < lenb:
        for id in range(lenbp):
            bbox_prev = bboxes_prev.cpu().numpy()[id]
            for idx in range(lenb):
                bbox = bboxes.cpu().numpy()[idx]
                d[idx,id] = np.sqrt(np.square(bbox[0] + (bbox[2]-bbox[0])/2 - bbox_prev[0] - (bbox_prev[2]-bbox_prev[0])/2) + np.square(bbox[1] + (bbox[3]-bbox[1])/2 - bbox_prev[1] - (bbox_prev[3]-bbox_prev[1])/2))
                s[idx,id] = np.abs((np.abs(bbox[2]-bbox[0])*np.abs(bbox[3]-bbox[1])) - (np.abs(bbox_prev[2]-bbox_prev[0])*np.abs(bbox_prev[3]-bbox_prev[1])))
            d_id = d[:,id]
            p_id  = np.where(d_id == d_id.min())
            s_id = s[idx,:] 
            if len(p_id[0]) > 1:
                p_id = np.where(s_id == s_id.min())
            if id !=  p_id[0][0]:
                tmpb = bboxes[p_id[0][0]]
                tmpc = conf[p_id[0][0]]
                bboxes[p_id[0][0]] = bboxes[id]
                conf[p_id[0][0]] = conf[id]
                bboxes[id] = tmpb
                conf[id] = tmpc
            bboxes_prev = torch.cat([bboxes_prev, bboxes[lenbp:]])
            conf_prev = torch.cat([conf_prev, conf[lenbp:]])

    # number of boxes in consecutive frames is equal
    else:
        for idx in range(lenb):
            bbox = bboxes.cpu().numpy()[idx]
            for id in range(lenbp):
                bbox_prev = bboxes_prev.cpu().numpy()[id]
                d[idx,id] = np.sqrt(np.square(bbox[0] + (bbox[2]-bbox[0])/2 - bbox_prev[0] - (bbox_prev[2]-bbox_prev[0])/2) + np.square(bbox[1] + (bbox[3]-bbox[1])/2 - bbox_prev[1] - (bbox_prev[3]-bbox_prev[1])/2))
                s[idx,id] = np.abs((np.abs(bbox[2]-bbox[0])*np.abs(bbox[3]-bbox[1])) - (np.abs(bbox_prev[2]-bbox_prev[0])*np.abs(bbox_prev[3]-bbox_prev[1])))
            d_id = d[idx,:]
            p_id  = np.where(d_id == d_id.min())
            s_id = s[idx,:] 
            if len(p_id[0]) > 1:
                p_id = np.where(s_id == s_id.min())
            if idx !=  p_id[0][0]:
                tmpb = bboxes_prev[p_id[0][0]]
                tmpc = conf_prev[p_id[0][0]]
                bboxes_prev[p_id[0][0]] = bboxes_prev[idx]
                conf_prev[p_id[0][0]] = conf_prev[idx]
                bboxes_prev[idx] = tmpb
                conf_prev[idx] = tmpc
    
    return bboxes, bboxes_prev, conf, conf_prev

### Calculate intersection over union
def bb_iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

### Perform bounding box tracking or prevent bounding box drops
def xcorrtrack(image_prev, image, stride, bboxes_prev, bboxes, conf_prev, conf, conf_thrsh, min_conf_thrsh, framenum):
    width, height = image.shape[1], image.shape[0]
    bboxes_pred = []
    confs_pred  = []
    scale = 2
    
    if len(bboxes) > 1 or len(bboxes_prev) > 1:
        bboxes, bboxes_prev, conf, conf_prev = match_bbox(bboxes, bboxes_prev, conf_prev, conf)

    for idx in range(len(bboxes)):
        bbox_prev = bboxes_prev.cpu().numpy().astype('int16')[idx]
        bbox      = bboxes.cpu().numpy().astype('int16')[idx]
        cnf_prev  = conf_prev.cpu().numpy()[idx]
        cnf       = conf.cpu().numpy()[idx]

        obj       = image_prev[int(bbox_prev[1]):int(bbox_prev[3]), int(bbox_prev[0]):int(bbox_prev[2]), 0]
        sz        = np.multiply(obj.shape[0],obj.shape[1])

        patch = image[max(0, int(bbox_prev[1]) - stride):min(height, int(bbox_prev[3]) + stride),
                                 max(0, int(bbox_prev[0]) - stride):min(width, int(bbox_prev[2]) + stride), 0]
        
        # condition to detect a bbox with low prediction score (below threshold) but with required correlation to the prior bounding box
        # having the rquired intersection over union (iou) and high confidence score (above threshold).  
        # The above depends on the success of the first detection. For the very first frame it will simply select the bbox with
        # confidence score higher than the required threshold or else will discard it.

        if sz > 0 and cnf_prev >= min_conf_thrsh and cnf > conf_thrsh and cnf < min_conf_thrsh and abs(sum(bbox-bbox_prev)) > 0:
            bbox_pred, max_val = find_bbox(obj, patch, bbox_prev, scale)
            iou = bb_iou(bbox_pred, bbox)
            if iou > 0.005:
                bbox_pred = bbox
                conf_pred = cnf_prev

            # If iou is lower than a set value, xcorr value threshold alone determines the detection of the missed box as a prediction.
            elif max_val > 0.25:
                bbox_pred = bbox_pred
                conf_pred = cnf_prev
            else:
                bbox_pred = bbox_pred
                conf_pred = 0
        else:
            bbox_pred = bbox
            conf_pred = cnf
        
        # avoid appending empty values to the list if any
        if bbox_pred != []: 
            bboxes_pred.append(bbox_pred)
            confs_pred.append(conf_pred)

    if bboxes_pred !=[]:
        bboxes_pred = torch.tensor(np.array(bboxes_pred).astype('float32'))
        confs_pred  = torch.tensor(np.array(confs_pred).astype('float32'))
    return bboxes_pred, confs_pred