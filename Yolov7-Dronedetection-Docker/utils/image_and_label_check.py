import os
import re
import shutil

images =     "datasets/dronedataset_color/images/valid"
labels =     "datasets/dronedataset_color/labels/valid"
newfolder=   "datasets/dronedataset_color/extra/extra_images_valid"
difffolder = "datasets/dronedataset_color/extra/difficult/valid"

imageslist= os.listdir(images)
labelslist= os.listdir(labels)
def listnum(list):
    V=[]
    for x in list:
        c=re.search(r'(.+?)\.',x)
        c=c.group()
        V.append(c)
    return V

imageslist=listnum(imageslist)
labelslist=listnum(labelslist)

extraimages=[]
for i in imageslist:
    if i not in labelslist:
        extraimages.append(i)

for s in extraimages:
    shutil.move(images+'/'+s+'jpg',newfolder)

for s in labelslist:
    if os.path.getsize(labels+'/'+s+'txt')==0:
        os.remove(labels+'/'+s+'txt')
        shutil.move(images+'/'+s+'jpg',difffolder)
