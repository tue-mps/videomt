# -------------------------------------------------------------------
# Copied from DVIS++ code, https://github.com/zhang-tao-whu/DVIS_Plus/tree/main
# Used under MIT License.
# -------------------------------------------------------------------
import os
import json
import sys

sys.path.append('../..')

from videomt.data_video.datasets.ytvis import (
    COCO_TO_YTVIS_2019,
    COCO_TO_YTVIS_2021,
    COCO_TO_OVIS
)

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
_root = os.path.join('../..', _root)

convert_list = [
    (COCO_TO_YTVIS_2019, 
        os.path.join(_root, "coco/annotations/instances_train2017.json"),
        os.path.join(_root, "coco/annotations/coco2ytvis2019_train.json"), "COCO to YTVIS 2019:"),
  
    (COCO_TO_YTVIS_2021, 
        os.path.join(_root, "coco/annotations/instances_train2017.json"),
        os.path.join(_root, "coco/annotations/coco2ytvis2021_train.json"), "COCO to YTVIS 2021:"),
 
    (COCO_TO_OVIS, 
        os.path.join(_root, "coco/annotations/instances_train2017.json"),
        os.path.join(_root, "coco/annotations/coco2ovis_train.json"), "COCO to OVIS:"),
]

for convert_dict, src_path, out_path, msg in convert_list:
    src_f = open(src_path, "r")
    out_f = open(out_path, "w")
    src_json = json.load(src_f)


    out_json = {}
    for k, v in src_json.items():
        if k != 'annotations':
            out_json[k] = v

    converted_item_num = 0
    out_json['annotations'] = []
    for anno in src_json['annotations']:
        if anno["category_id"] not in convert_dict:
            continue

        out_json['annotations'].append(anno)
        converted_item_num += 1

    json.dump(out_json, out_f)
    print(msg, converted_item_num, "items converted.")
