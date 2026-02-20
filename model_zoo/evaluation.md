## Evaluation
To evaluate a pre-trained VidEoMT model , first prepare the datasets by following the instructions in this [link](../datasets/README.md) and download the trained weights from [here](dinov2.md). Once these are set up, run:


### YouTube-VIS 2019, YouTube-VIS 2021, OVIS<sup>*</sup>

```bash
python train_net_video.py \
  --num-gpus 1 \
  --config-file /path/to/config.yaml \
  --eval-only MODEL.WEIGHTS /path/to/weight.pth
  MODEL.BACKBONE.TEST.WINDOW_SIZE 1 \ 
  OUTPUT_DIR /path/to/output
```

🔧 Replace `/path/to/config.yaml` with the path to the config file.  
🔧 Replace `/path/to/weight.pth` with the path to the checkpoint to evaluate.   
🔧 Replace `/path/to/output` with the path to the output folder.  
🔧 Change the value of `--num-gpus` to the number of GPUs available to you.


### YouTube-VIS 2022

```bash
python train_net_video.py \
  --num-gpus 1 \
  --config-file /path/to/config.yaml \
  --eval-only MODEL.WEIGHTS /path/to/weight.pth
  MODEL.BACKBONE.TEST.WINDOW_SIZE 1 \ 
  OUTPUT_DIR /path/to/output
```
After generating the inference images using the above command, to calculate the AP for long videos, run:

``` bash
python utils/yt2022_evaluate.py /path/to/dataset/ytvis_2022  /path/to/output/inference
```
🔧 Replace `/path/to/dataset` with the path to the dataset folder.  
🔧 Replace `/path/to/output` with the path to the output folder.  

### VIPSeg

```bash
python train_net_video.py \
  --num-gpus 1 \
  --config-file /path/to/config.yaml \
  --eval-only MODEL.WEIGHTS /path/to/weight.pth
  MODEL.BACKBONE.TEST.WINDOW_SIZE 1 \ 
  OUTPUT_DIR /path/to/output
```
After generating the inference images using the above command, to calculate the VPQ and STQ, run:

``` bash
DATAROOT='/path/to/dataset/VIPSeg_720P/panomasksRGB'
IMGSAVEROOT='/path/to/output/inference'
GT_JSONFILE='/path/to/dataset/VIPSeg_720P/panoptic_gt_VIPSeg_val.json'

# ###VPQ
python utils/eval_vpq_vspw.py --submit_dir $IMGSAVEROOT --truth_dir $DATAROOT --pan_gt_json_file $GT_JSONFILE
# # ###STQ
python utils/eval_stq_vspw.py --submit_dir $IMGSAVEROOT --truth_dir $DATAROOT --pan_gt_json_file $GT_JSONFILE
```
🔧 Replace `/path/to/dataset` with the path to the dataset folder.  
🔧 Replace `/path/to/output` with the path to the output folder.  

### VSPW

```bash
python train_net_video.py \
  --num-gpus 1 \
  --config-file /path/to/config.yaml \
  --eval-only MODEL.WEIGHTS /path/to/weight.pth
  MODEL.BACKBONE.TEST.WINDOW_SIZE 1 \ 
  OUTPUT_DIR /path/to/output
```
After generating the inference images using the above command, to calculate the mVC<sub>16</sub> and mIoU, run:

``` bash
DATAROOT='/path/to/dataset/VSPW_480p'
GT_JSONFILE='/path/to/output/inference'

python utils/eval_miou_vspw.py  $DATAROOT $IMGSAVEROOT
python utils/eval_vc_vspw.py $DATAROOT  $IMGSAVEROOT

```
🔧 Replace `/path/to/dataset` with the path to the dataset folder.  
🔧 Replace `/path/to/output` with the path to the output folder.  
