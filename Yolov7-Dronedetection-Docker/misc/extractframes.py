import cv2
import os

video_path = 'runs/detect/00009_notracking_0pt5/00009.mp4'
cap = cv2.VideoCapture(video_path)

if cap.isOpened():
    savepath = '/'.join(video_path.split('/')[:3]) + '/images'
    os.makedirs(savepath, exist_ok=True)
    length = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    
    n = 0
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(savepath + '/' + str(n+1).zfill(length) + '.png', frame)
            n += 1
        else:
            break
