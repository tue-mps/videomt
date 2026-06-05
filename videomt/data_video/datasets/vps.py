import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager


def load_video_vspw_vps_json(json_file, image_dir, gt_dir):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, cate_id_thingstaff):
        isthing = cate_id_thingstaff[segment_info['category_id']]
        segment_info["isthing"] = isthing

        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    videoid_img_dic = {}
    for video_ in json_info["videos"]:
        videoid_img_dic[video_['video_id']] = {}
        for imgimg in video_['images']:
            videoid_img_dic[video_['video_id']][imgimg['id']] = {'width': imgimg['width'], 'height': imgimg['height'],
                                                                 'file_name': imgimg['file_name']}

    cate_id_thingstaff = {}
    for cate in json_info['categories']:
        cate_id_thingstaff[cate['id']] = cate['isthing']

    ret = []
    for ann in json_info["annotations"]:
        video_id = ann["video_id"]
        anns = ann['annotations']
        image_files = []
        label_files = []
        sem_label_files = []
        segments_infos = []
        for image in anns:
            image_id = image['image_id']

            # TODO: currently we assume image and label has the same filename but
            # different extension, and images have extension ".jpg" for COCO. Need
            # to make image extension a user-provided argument if we extend this
            # function to support other COCO-like datasets.
            image_file = os.path.join(image_dir, video_id,
                                      videoid_img_dic[video_id][image_id]['file_name'].split('.')[0] + '.jpg')
            image_files.append(image_file)

            label_file = os.path.join(gt_dir, video_id, image["file_name"])
            label_files.append(label_file)

            #            sem_label_file = os.path.join(semseg_dir, image["file_name"])
            #            sem_label_files.append(sem_label_file)

            segments_info = image["segments_info"]
            segments_info = [_convert_category_id(seg_info, cate_id_thingstaff) for seg_info in segments_info]
            segments_infos.append(segments_info)
        ret.append(
            {
                "file_names": image_files,
                "width": videoid_img_dic[video_id][image_id]['width'],
                "height": videoid_img_dic[video_id][image_id]['height'],
                "video_id": video_id,
                "pan_seg_file_names": label_files,
                #        "sem_seg_file_names": sem_label_files,
                "segments_infos": segments_infos,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    #    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    #    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    #    assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    # print(ret[0])
    # exit()
    return ret


def register_video_vspw_vps_json(
        name, metadata, image_root, panoptic_root, panoptic_json, instances_json=None
):
    """
    Register a "standard" version of ADE20k panoptic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".
    Args:
        name (str): the name that identifies a dataset,
            e.g. "ade20k_panoptic_train"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images in COCO format
        panoptic_json (str): path to the json panoptic annotation file in COCO format
        sem_seg_root (none): not used, to be consistent with
            `register_coco_panoptic_separated`.
        instances_json (str): path to the json instance annotation file
    """
    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_video_vspw_vps_json(
            panoptic_json, image_root, panoptic_root
        ),
    )
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type=None,
        ignore_label=255,
        label_divisor=100,
        **metadata,
    )


def get_metadata(json_file):
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.

    categories = [{"id": 0, "name": "wall", "isthing": 0, "color": [120, 120, 120]},
                  {"id": 1, "name": "ceiling", "isthing": 0, "color": [180, 120, 120]},
                  {"id": 2, "name": "door", "isthing": 1, "color": [6, 230, 230]},
                  {"id": 3, "name": "stair", "isthing": 0, "color": [80, 50, 50]},
                  {"id": 4, "name": "ladder", "isthing": 1, "color": [4, 200, 3]},
                  {"id": 5, "name": "escalator", "isthing": 0, "color": [120, 120, 80]},
                  {"id": 6, "name": "Playground_slide", "isthing": 0, "color": [140, 140, 140]},
                  {"id": 7, "name": "handrail_or_fence", "isthing": 0, "color": [204, 5, 255]},
                  {"id": 8, "name": "window", "isthing": 1, "color": [230, 230, 230]},
                  {"id": 9, "name": "rail", "isthing": 0, "color": [4, 250, 7]},
                  {"id": 10, "name": "goal", "isthing": 1, "color": [224, 5, 255]},
                  {"id": 11, "name": "pillar", "isthing": 0, "color": [235, 255, 7]},
                  {"id": 12, "name": "pole", "isthing": 0, "color": [150, 5, 61]},
                  {"id": 13, "name": "floor", "isthing": 0, "color": [120, 120, 70]},
                  {"id": 14, "name": "ground", "isthing": 0, "color": [8, 255, 51]},
                  {"id": 15, "name": "grass", "isthing": 0, "color": [255, 6, 82]},
                  {"id": 16, "name": "sand", "isthing": 0, "color": [143, 255, 140]},
                  {"id": 17, "name": "athletic_field", "isthing": 0, "color": [204, 255, 4]},
                  {"id": 18, "name": "road", "isthing": 0, "color": [255, 51, 7]},
                  {"id": 19, "name": "path", "isthing": 0, "color": [204, 70, 3]},
                  {"id": 20, "name": "crosswalk", "isthing": 0, "color": [0, 102, 200]},
                  {"id": 21, "name": "building", "isthing": 0, "color": [61, 230, 250]},
                  {"id": 22, "name": "house", "isthing": 0, "color": [255, 6, 51]},
                  {"id": 23, "name": "bridge", "isthing": 0, "color": [11, 102, 255]},
                  {"id": 24, "name": "tower", "isthing": 0, "color": [255, 7, 71]},
                  {"id": 25, "name": "windmill", "isthing": 0, "color": [255, 9, 224]},
                  {"id": 26, "name": "well_or_well_lid", "isthing": 0, "color": [9, 7, 230]},
                  {"id": 27, "name": "other_construction", "isthing": 0, "color": [220, 220, 220]},
                  {"id": 28, "name": "sky", "isthing": 0, "color": [255, 9, 92]},
                  {"id": 29, "name": "mountain", "isthing": 0, "color": [112, 9, 255]},
                  {"id": 30, "name": "stone", "isthing": 0, "color": [8, 255, 214]},
                  {"id": 31, "name": "wood", "isthing": 0, "color": [7, 255, 224]},
                  {"id": 32, "name": "ice", "isthing": 0, "color": [255, 184, 6]},
                  {"id": 33, "name": "snowfield", "isthing": 0, "color": [10, 255, 71]},
                  {"id": 34, "name": "grandstand", "isthing": 0, "color": [255, 41, 10]},
                  {"id": 35, "name": "sea", "isthing": 0, "color": [7, 255, 255]},
                  {"id": 36, "name": "river", "isthing": 0, "color": [224, 255, 8]},
                  {"id": 37, "name": "lake", "isthing": 0, "color": [102, 8, 255]},
                  {"id": 38, "name": "waterfall", "isthing": 0, "color": [255, 61, 6]},
                  {"id": 39, "name": "water", "isthing": 0, "color": [255, 194, 7]},
                  {"id": 40, "name": "billboard_or_Bulletin_Board", "isthing": 0, "color": [255, 122, 8]},
                  {"id": 41, "name": "sculpture", "isthing": 1, "color": [0, 255, 20]},
                  {"id": 42, "name": "pipeline", "isthing": 0, "color": [255, 8, 41]},
                  {"id": 43, "name": "flag", "isthing": 1, "color": [255, 5, 153]},
                  {"id": 44, "name": "parasol_or_umbrella", "isthing": 1, "color": [6, 51, 255]},
                  {"id": 45, "name": "cushion_or_carpet", "isthing": 0, "color": [235, 12, 255]},
                  {"id": 46, "name": "tent", "isthing": 1, "color": [160, 150, 20]},
                  {"id": 47, "name": "roadblock", "isthing": 1, "color": [0, 163, 255]},
                  {"id": 48, "name": "car", "isthing": 1, "color": [140, 140, 140]},
                  {"id": 49, "name": "bus", "isthing": 1, "color": [250, 10, 15]},
                  {"id": 50, "name": "truck", "isthing": 1, "color": [20, 255, 0]},
                  {"id": 51, "name": "bicycle", "isthing": 1, "color": [31, 255, 0]},
                  {"id": 52, "name": "motorcycle", "isthing": 1, "color": [255, 31, 0]},
                  {"id": 53, "name": "wheeled_machine", "isthing": 0, "color": [255, 224, 0]},
                  {"id": 54, "name": "ship_or_boat", "isthing": 1, "color": [153, 255, 0]},
                  {"id": 55, "name": "raft", "isthing": 1, "color": [0, 0, 255]},
                  {"id": 56, "name": "airplane", "isthing": 1, "color": [255, 71, 0]},
                  {"id": 57, "name": "tyre", "isthing": 0, "color": [0, 235, 255]},
                  {"id": 58, "name": "traffic_light", "isthing": 0, "color": [0, 173, 255]},
                  {"id": 59, "name": "lamp", "isthing": 0, "color": [31, 0, 255]},
                  {"id": 60, "name": "person", "isthing": 1, "color": [11, 200, 200]},
                  {"id": 61, "name": "cat", "isthing": 1, "color": [255, 82, 0]},
                  {"id": 62, "name": "dog", "isthing": 1, "color": [0, 255, 245]},
                  {"id": 63, "name": "horse", "isthing": 1, "color": [0, 61, 255]},
                  {"id": 64, "name": "cattle", "isthing": 1, "color": [0, 255, 112]},
                  {"id": 65, "name": "other_animal", "isthing": 1, "color": [0, 255, 133]},
                  {"id": 66, "name": "tree", "isthing": 0, "color": [255, 0, 0]},
                  {"id": 67, "name": "flower", "isthing": 0, "color": [255, 163, 0]},
                  {"id": 68, "name": "other_plant", "isthing": 0, "color": [255, 102, 0]},
                  {"id": 69, "name": "toy", "isthing": 0, "color": [194, 255, 0]},
                  {"id": 70, "name": "ball_net", "isthing": 0, "color": [0, 143, 255]},
                  {"id": 71, "name": "backboard", "isthing": 0, "color": [51, 255, 0]},
                  {"id": 72, "name": "skateboard", "isthing": 1, "color": [0, 82, 255]},
                  {"id": 73, "name": "bat", "isthing": 0, "color": [0, 255, 41]},
                  {"id": 74, "name": "ball", "isthing": 1, "color": [0, 255, 173]},
                  {"id": 75, "name": "cupboard_or_showcase_or_storage_rack", "isthing": 0, "color": [10, 0, 255]},
                  {"id": 76, "name": "box", "isthing": 1, "color": [173, 255, 0]},
                  {"id": 77, "name": "traveling_case_or_trolley_case", "isthing": 1, "color": [0, 255, 153]},
                  {"id": 78, "name": "basket", "isthing": 1, "color": [255, 92, 0]},
                  {"id": 79, "name": "bag_or_package", "isthing": 1, "color": [255, 0, 255]},
                  {"id": 80, "name": "trash_can", "isthing": 0, "color": [255, 0, 245]},
                  {"id": 81, "name": "cage", "isthing": 0, "color": [255, 0, 102]},
                  {"id": 82, "name": "plate", "isthing": 1, "color": [255, 173, 0]},
                  {"id": 83, "name": "tub_or_bowl_or_pot", "isthing": 1, "color": [255, 0, 20]},
                  {"id": 84, "name": "bottle_or_cup", "isthing": 1, "color": [255, 184, 184]},
                  {"id": 85, "name": "barrel", "isthing": 1, "color": [0, 31, 255]},
                  {"id": 86, "name": "fishbowl", "isthing": 1, "color": [0, 255, 61]},
                  {"id": 87, "name": "bed", "isthing": 1, "color": [0, 71, 255]},
                  {"id": 88, "name": "pillow", "isthing": 1, "color": [255, 0, 204]},
                  {"id": 89, "name": "table_or_desk", "isthing": 1, "color": [0, 255, 194]},
                  {"id": 90, "name": "chair_or_seat", "isthing": 1, "color": [0, 255, 82]},
                  {"id": 91, "name": "bench", "isthing": 1, "color": [0, 10, 255]},
                  {"id": 92, "name": "sofa", "isthing": 1, "color": [0, 112, 255]},
                  {"id": 93, "name": "shelf", "isthing": 0, "color": [51, 0, 255]},
                  {"id": 94, "name": "bathtub", "isthing": 0, "color": [0, 194, 255]},
                  {"id": 95, "name": "gun", "isthing": 1, "color": [0, 122, 255]},
                  {"id": 96, "name": "commode", "isthing": 1, "color": [0, 255, 163]},
                  {"id": 97, "name": "roaster", "isthing": 1, "color": [255, 153, 0]},
                  {"id": 98, "name": "other_machine", "isthing": 0, "color": [0, 255, 10]},
                  {"id": 99, "name": "refrigerator", "isthing": 1, "color": [255, 112, 0]},
                  {"id": 100, "name": "washing_machine", "isthing": 1, "color": [143, 255, 0]},
                  {"id": 101, "name": "Microwave_oven", "isthing": 1, "color": [82, 0, 255]},
                  {"id": 102, "name": "fan", "isthing": 1, "color": [163, 255, 0]},
                  {"id": 103, "name": "curtain", "isthing": 0, "color": [255, 235, 0]},
                  {"id": 104, "name": "textiles", "isthing": 0, "color": [8, 184, 170]},
                  {"id": 105, "name": "clothes", "isthing": 0, "color": [133, 0, 255]},
                  {"id": 106, "name": "painting_or_poster", "isthing": 1, "color": [0, 255, 92]},
                  {"id": 107, "name": "mirror", "isthing": 1, "color": [184, 0, 255]},
                  {"id": 108, "name": "flower_pot_or_vase", "isthing": 1, "color": [255, 0, 31]},
                  {"id": 109, "name": "clock", "isthing": 1, "color": [0, 184, 255]},
                  {"id": 110, "name": "book", "isthing": 0, "color": [0, 214, 255]},
                  {"id": 111, "name": "tool", "isthing": 0, "color": [255, 0, 112]},
                  {"id": 112, "name": "blackboard", "isthing": 0, "color": [92, 255, 0]},
                  {"id": 113, "name": "tissue", "isthing": 0, "color": [0, 224, 255]},
                  {"id": 114, "name": "screen_or_television", "isthing": 1, "color": [112, 224, 255]},
                  {"id": 115, "name": "computer", "isthing": 1, "color": [70, 184, 160]},
                  {"id": 116, "name": "printer", "isthing": 1, "color": [163, 0, 255]},
                  {"id": 117, "name": "Mobile_phone", "isthing": 1, "color": [153, 0, 255]},
                  {"id": 118, "name": "keyboard", "isthing": 1, "color": [71, 255, 0]},
                  {"id": 119, "name": "other_electronic_product", "isthing": 0, "color": [255, 0, 163]},
                  {"id": 120, "name": "fruit", "isthing": 0, "color": [255, 204, 0]},
                  {"id": 121, "name": "food", "isthing": 0, "color": [255, 0, 143]},
                  {"id": 122, "name": "instrument", "isthing": 1, "color": [0, 255, 235]},
                  {"id": 123, "name": "train", "isthing": 1, "color": [133, 255, 0]}]

    thing_classes = [k["name"] for k in categories if k['isthing']]
    thing_colors = [k["color"] for k in categories if k['isthing']]
    stuff_classes = [k["name"] for k in categories if not k['isthing']]
    stuff_colors = [k["color"] for k in categories if not k['isthing']]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    thing_classes_id = [k['id'] for k in categories if k['isthing']]
    meta['thing_classes_id'] = thing_classes_id

    categories_ = {}
    for cat in categories:
        categories_.update({cat['id']: cat})
    meta['categories'] = categories_

    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    thing_classes_id = [k['id'] for k in categories if k['isthing']]
    stuff_classes_id = [k['id'] for k in categories if not k['isthing']]
    for i, id_ in enumerate(thing_classes_id):
        thing_dataset_id_to_contiguous_id[id_] = id_
    for i, id_ in enumerate(stuff_classes_id):
        stuff_dataset_id_to_contiguous_id[id_] = id_
    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.

    # in order to use sem_seg evaluator

    return meta


_PREDEFINED_SPLITS_PANOVSPW = {
    "panoVSPW_vps_video_train": (
        "VIPSeg/VIPSeg_720P/images",
        "VIPSeg/VIPSeg_720P/panomasksRGB",
        "VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_train.json",
    ),
    "panoVSPW_vps_video_val": (
        "VIPSeg/VIPSeg_720P/images",
        "VIPSeg/VIPSeg_720P/panomasksRGB",
        "VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_val.json",
    ),
    "panoVSPW_vps_video_test": (
        "VIPSeg/VIPSeg_720P/images",
        "VIPSeg/VIPSeg_720P/panomasksRGB",
        "VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_test.json",
    ),
}


def register_all_video_panoVSPW(root):
    for (
            prefix,
            (image_root, panoptic_root, panoptic_json),
    ) in _PREDEFINED_SPLITS_PANOVSPW.items():
        metadata = get_metadata(os.path.join(root, panoptic_json))
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_video_vspw_vps_json(
            prefix,
            metadata,
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            #            os.path.join(root, semantic_root),
            os.path.join(root, panoptic_json),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_video_panoVSPW(_root)

