# Annotation Fileformats

Overview at [roboflow](https://roboflow.com/formats)

---

## [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) (.xml)
The annotation format originally created for the Visual Object Challenge (VOC) has become a common interchange format for object detection labels.
Many labeling tools can export it including [LabelImg](https://github.com/tzutalin/labelImg), [CVAT](https://github.com/openvinotoolkit/cvat), [VoTT](https://github.com/microsoft/VoTT)
No known model directly consumes Pascal VOC (.xml).

### Annotation from Original Pascal VOC Dataset

Bounding box coordinates are pixel relative integers

```
<annotation>
    <filename>2012_004324.jpg</filename>
    <folder>VOC2012</folder>
    <object>
        <name>person</name>
        <bndbox>
            <xmax>269</xmax>
            <xmin>189</xmin>
            <ymax>211</ymax>
            <ymin>11</ymin>
        </bndbox>
        <difficult>0</difficult>
        <pose>Unspecified</pose>
        <point>
            <x>231</x>
            <y>59</y>
        </point>
    </object>
    <segmented>0</segmented>
    <size>
        <depth>3</depth>
        <height>333</height>
        <width>500</width>
    </size>
    <source>
        <annotation>PASCAL VOC2012</annotation>
        <database>The VOC2012 Database</database>
        <image>flickr</image>
    </source>
</annotation>
```

### Example Annotation from [LabelImg](https://github.com/tzutalin/labelImg)

Extends the standard by adding a `<path>` tag

```
<annotation>
    <folder>demo image</folder>
    <filename>air-monitoring-211124_1920.jpg</filename>
    <path>.../demo image/air-monitoring-211124_1920.jpg</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>1920</width>
        <height>1440</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>drone</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>395</xmin>
            <ymin>103</ymin>
            <xmax>1563</xmax>
            <ymax>781</ymax>
        </bndbox>
    </object>
    <object>
        <name>dog</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>665</xmin>
            <ymin>891</ymin>
            <xmax>1092</xmax>
            <ymax>1210</ymax>
        </bndbox>
    </object>
</annotation>
```

---

## COCO (.json)
The JSON format is used by the MS COCO dataset release by Microsoft in 2015.

* [Offical Data format Documentation](https://cocodataset.org/#format-data)
* [Offical Results format Documentation](https://cocodataset.org/#format-results)
* [Helpful Reference](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch)

### Overview

The JSON file is separated in the following sections:

```
{
    "info": {...},
    "licenses": [...],
    "images": [...],
    "annotations": [...],
    "categories": [...], // Not in Captions annotations
    "segment_info": [...] // Only in Panoptic annotations
}
```

There content is explained in the following chapter

### Sections

#### Info

Contains the general info of the dataset

```
"info": {
    "description": "COCO 2017 Dataset",
    "url": "http://cocodataset.org",
    "version": "1.0",
    "year": 2017,
    "contributor": "COCO Consortium",
    "date_created": "2017/09/01" // datetime
}
```

#### Licenses

```
"licenses": [
    {
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License"
    },
    ...
]
```

#### Images

```
"images": [
    {
        "license": 4,
        "file_name": "000000397133.jpg",
        "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
        "height": 427,
        "width": 640,
        "date_captured": "2013-11-14 17:02:52",
        "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
        "id": 397133
    },
    {
        "license": 1,
        "file_name": "000000037777.jpg",
        "coco_url": "http://images.cocodataset.org/val2017/000000037777.jpg",
        "height": 230,
        "width": 352,
        "date_captured": "2013-11-14 20:55:31",
        "flickr_url": "http://farm9.staticflickr.com/8429/7839199426_f6d48aa585_z.jpg",
        "id": 37777
    },
    ...
]
```

#### Annotations

A detailed description is done in the JSON snipped separated by a "//"
(this does not follow any convention, because JSON format does not allow comments by default)

Bounding box coordinates are pixel relative floats

```
"annotations": [
    {
        "segmentation": [[510.66,423.01,511.72,420.03,...,510.45,423.01]], // polygon based, list of vertices (x, y pixel positions)
        "area": 702.1057499999998, // occluded area of the polygon
        "iscrowd": 0, // used to label group/cluster of objects, only ~1% in dataset
        "image_id": 289343, // unique identifier per image
        "bbox": [
            473.07, // x of top left corner
            395.93, // y
            38.65,  // width
            28.67   // height
        ],
        "category_id": 18, // unique identifier per class (in this case 'dog')
        "id": 1768
    },
    {
        "segmentation": [[ ...]],
        "area": 27718.476299999995,
        "iscrowd": 0,
        "image_id": 61471,
        "bbox": [
            272.1,
            200.23,
            151.97,
            279.77
        ],
        "category_id": 18,
        "id": 1773
    },
    ...
]
```

#### Categories

```
"categories": [
    {
        "supercategory": "person",
        "id": 1,
        "name": "person"
    },
    {
        "supercategory": "vehicle",
        "id": 2,
        "name": "bicycle"
    },
    {
        "supercategory": "vehicle",
        "id": 3,
        "name": "car"
    }
    ...
]
```

---

## Open Images (.csv)
Released by Amazon with their [OpenImages](https://storage.googleapis.com/openimages/web/index.html) dataset.
Instead of an numeric class ID the class name is a a string of the form "/m/xxxxx"

Bounding box coordinates are normalized values (floats) in range [0-1]

```
ImageID,LabelName,XMin,XMax,YMin,YMax,IsGroupOf
0001eeaf4aed83f9,/m/0cmf2,0.022673031,0.9642005,0.07103825,0.80054647,0
000595fe6fee6369,/m/02xwb,0.1413844,0.179676,0.67627496,0.73170733,0
000595fe6fee6369,/m/02xwb,0.21354933,0.2533137,0.29933482,0.35476717,0
000595fe6fee6369,/m/02xwb,0.23269513,0.28865978,0.49002218,0.54545456,0
```

## YOLO (.txt)
There is no large public dataset that uses the YOLO annotation format.
But because many models use it to read the annotations, it is described here.

For each image there is a annotation file. Every line represents one bounding box. The format is:
```
class_id center_x center_y width height
```

001.txt
```
0 0.051562 0.887963 0.029167 0.027778
0 0.089583 0.892593 0.030208 0.029630
2 0.139844 0.923148 0.033854 0.048148
1 0.169271 0.933796 0.021875 0.030556
1 0.339583 0.973148 0.027083 0.029630
1 0.153125 0.595370 0.009375 0.007407
1 0.164844 0.599537 0.008854 0.012037
```

The name of the .txt file corresponds to the name of the image file.

A separate file contains the mapping of class_id to class name.
The file convention varies between the different implementations.
Here is an example for YOLO in Pytorch, where the mapping takes place in a yaml file:

data.yaml
```
nc: 3
names: ['person', 'land', 'sea']
```

## TF-Record (.tfrecord)

The [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) uses a proprietary binary file format called TFRecord.

The dataset contains images and annotation all together in one binary file that can only be read by a TF-Record API.
