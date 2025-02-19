def filter_detections(det, threshold, im):
    h = (det[:,3]-det[:,1])
    w = (det[:,2]-det[:,0])
    s = det[:,4]
    H = im.shape[1]
    W = im.shape[0]

    # confidence
    det = det[s>threshold]
    if len(det):
        # size
        det = det[((det[:,3]-det[:,1])*(det[:,2]-det[:,0]))/(H*W)<0.7]
        det = det[(det[:,3]-det[:,1])/H<0.8]
        det = det[(det[:,2]-det[:,0])/W<0.8]
        det = det[((det[:,3]-det[:,1])*(det[:,2]-det[:,0]))/(H*W)>0.00015]

    return det