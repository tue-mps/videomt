# Prepare Datasets 

A dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog)
for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc).
This document explains how to setup the builtin datasets so they can be used by the above APIs.
[Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html) gives a deeper dive on how to use `DatasetCatalog` and `MetadataCatalog`,
and how to add new datasets to them.

VidEoMT has builtin support for a few datasets.
The datasets are assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  ytvis_2019/
  ytvis_2021/
  ovis/
  VIPSeg/
  VSPW_480p/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.



## Expected dataset structure for [YouTube-VIS 2019](https://competitions.codalab.org/competitions/20128):

```
ytvis_2019/
  {train,valid,test}.json
  {train,valid,test}/
    Annotations/
    JPEGImages/
```

## Expected dataset structure for [YouTube-VIS 2021](https://competitions.codalab.org/competitions/28988):

```
ytvis_2021/
  {train,valid,test}.json
  {train,valid,test}/
    JPEGImages/
    instances.json
    
```

## Expected dataset structure for [YouTube-VIS 2022](https://codalab.lisn.upsaclay.fr/competitions/3410#participate-get_data):

```
ytvis_2022/
  {train,valid,test}.json
  gt_{short,long}.json
  {train,valid,test}/
    JPEGImages/
    instances.json
  
```

## Expected dataset structure for [OVIS](http://songbai.site/ovis/):

```
ovis/
  annotations/
    annotations_{train,valid,test}.json
  {train,valid,test}/
```

## Expected dataset structure for [VIPSeg](https://github.com/VIPSeg-Dataset/VIPSeg-Dataset):

After downloading the VIPSeg dataset, it still needs to be processed according to the official script (`/datasets/utils/vipseg_change2_720p.py`). To save time, you can directly download the processed VIPSeg dataset from [baiduyun](https://pan.baidu.com/s/1SMausnr6pVDJXTGISeFMuw) (password is `dvis`). 
```
VIPSeg/
  VIPSeg_720P/
    images/
    panomasksRGB/
    panoptic_gt_VIPSeg_{train,val,test}.json
```

## Expected dataset structure for [VSPW](https://github.com/VSPW-dataset/VSPW-dataset-download):
```
VSPW_480p/
  data/
    video_1/
      mask/
      origin/   
  train.txt
  val.txt
  test.txt
  data.txt
  abel_num_dic_final.json
  

```

## Register your own dataset:

- If it is a VIS/VPS/VSS dataset, convert it to YTVIS/VIPSeg/VSPW format. If it is a image instance dataset, convert it to COCO format.
- Register it in `/videomt/data_video/datasets/{builtin,vps,vss}.py`

## Convert COCO to YTVIS / OVIS format

VidEoMT provides a helper script (copied from DVIS++) to convert COCO annotations into subsets compatible with **YTVIS 2019**, **YTVIS 2021**, and **OVIS**, by filtering COCO categories.

The script expects the COCO annotation file to be located at:

```bash
$DETECTRON2_DATASETS/coco/annotations/instances_train2017.json
```

It generates the following files under:

```bash
$DETECTRON2_DATASETS/coco/annotations/

coco2ytvis2019_train.json
coco2ytvis2021_train.json
coco2ovis_train.json
```

To run the conversion script:
```bash
python tools/convert_coco_to_vis.py
```