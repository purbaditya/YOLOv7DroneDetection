# Dataset Preprocessing
A list of handpicked scripts to analyse, manipulate, rearrange and convert the most common object detection dataset annotation formats

## Annotation Fileformats

Different object detection datasets contain different annotation standards and files formats.
A detailed explanation of the most common formats together with some example annotations can be found in [Annotation Fileformats](Annotation_Fileformats.md).

## Manual Annotation Workflow
1. Use the open source tool [LabelImg](https://github.com/tzutalin/labelImg) to label your images by hand and store the annotations in the __Pascal VOC__ (`.xml`) format.
2. Use [xml_annotation_changer.py](xml_annotation_changer.py) script to analyse, change or delete labels.
3. Convert the annotation from __Pascal VOC__ (`.xml`) into __COCO__ (`.json`), to analyze with COCO-dataset-explorer
4. Convert the annotation from __Pascal VOC__ (`.xml`) into __Open Images__ (`.csv`)
5. Convert the annotation from __Open Images__ (`.csv`) into __TFRecord__ (`.tfrecord`)

### Convert the annotation from __Pascal VOC__ (`.xml`) into __COCO__ (`.json`)

```bash
python3 voc2coco.py ./path_to_xml_files ./output.json
```
