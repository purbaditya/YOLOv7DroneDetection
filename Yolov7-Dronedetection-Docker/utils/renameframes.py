import os
import shutil
import glob

filename = 'VOC_label_00010'
extension = '/*.xml'
foldername = "datasets/dronedataset_color/labels/valid/"+filename+extension
newfolder="datasets/dronedataset_color/labels/valid"

filelist= glob.glob(foldername)
iter = 0
for x in filelist:
    newname = x[0:57]+filename[-5:]+"_"+x[57:]
    os.rename(x,newname)
    shutil.move(newname,newfolder)
    iter=iter+1
    print(iter)
