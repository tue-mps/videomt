import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances, Boxes
from detectron2.projects.point_rend import ColorAugSSDTransform
from panopticapi.utils import rgb2id

from .utils import Video_BitMasks, Video_Boxes
import random

__all__ = ["PanopticDatasetVideoMapper"]


class PanopticDatasetVideoMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
            self,
            is_train=True,
            is_tgt=True,  # not used, vps not support category mapper
            *,
            augmentations,
            image_format,
            ignore_label,
            thing_ids_to_continue_dic,
            stuff_ids_to_continue_dic,
            #sample
            sampling_frame_num: int = 2,
            sampling_frame_range: int = 5,
            reverse_agu: bool = False,
            src_dataset_name: str = "",  # not used
            tgt_dataset_name: str = "",  # not used
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.thing_ids_to_continue_dic = thing_ids_to_continue_dic
        self.stuff_ids_to_continue_dic = stuff_ids_to_continue_dic

        self.sampling_frame_num = sampling_frame_num
        self.sampling_frame_range = sampling_frame_range
        self.sampling_frame_ratio = 1.0
        self.reverse_agu = reverse_agu

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        if is_train:
            augs = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                )
            ]
            if cfg.INPUT.CROP.ENABLED:
                augs.append(
                    T.RandomCrop_CategoryAreaConstraint(
                        cfg.INPUT.CROP.TYPE,
                        cfg.INPUT.CROP.SIZE,
                        1.0,
                    )
                )
            augs.append(T.RandomFlip())
        else:
            # Resize
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            sample_style = "choice"
            augs = [T.ResizeShortestEdge(min_size, max_size, sample_style)]

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[-1])
        ignore_label = meta.ignore_label

        #######
        thing_ids = list(meta.thing_dataset_id_to_contiguous_id.values())
        thing_ids_to_continue_dic = {}
        for ii, id_ in enumerate(sorted(thing_ids)):
            thing_ids_to_continue_dic[id_] = ii

        stuff_ids = list(meta.stuff_dataset_id_to_contiguous_id.values())
        stuff_ids_to_continue_dic = {}
        for ii, id_ in enumerate(sorted(stuff_ids)):
            stuff_ids_to_continue_dic[id_] = ii

        #######
        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        reverse_agu = cfg.INPUT.REVERSE_AGU

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            # "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "thing_ids_to_continue_dic": thing_ids_to_continue_dic,
            "stuff_ids_to_continue_dic": stuff_ids_to_continue_dic,
            # sample
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "reverse_agu": reverse_agu,
        }
        return ret

    def select_frames(self, video_length):
        """
        Args:
            video_length (int): length of the video

        Returns:
            selected_idx (list[int]): a list of selected frame indices
        """
        if self.sampling_frame_range * 2 + 1 == self.sampling_frame_num:
            # sample continuous frames
            if self.sampling_frame_num > video_length:
                selected_idx = np.arange(0, video_length)
                selected_idx_ = np.random.choice(selected_idx, self.sampling_frame_num - len(selected_idx))
                selected_idx = selected_idx.tolist() + selected_idx_.tolist()
                sorted(selected_idx)
            else:
                if video_length == self.sampling_frame_num:
                    start_idx = 0
                else:
                    start_idx = random.randrange(video_length - self.sampling_frame_num)
                end_idx = start_idx + self.sampling_frame_num
                selected_idx = np.arange(start_idx, end_idx).tolist()
            if self.reverse_agu and random.random() < 0.5:
                selected_idx = selected_idx[::-1]
            return selected_idx

        ref_frame = random.randrange(video_length)
        start_idx = max(0, ref_frame-self.sampling_frame_range)
        end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)

        selected_idx = np.random.choice(
            np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame+1, end_idx))),
            self.sampling_frame_num - 1,
        )
        selected_idx = selected_idx.tolist() + [ref_frame]
        selected_idx = sorted(selected_idx)
        return selected_idx

    def convert2ytvis(self, dataset_dict):
        # Convert the labels to the same format as vis
        ret = {}
        ret["image"] = [item[0] for item in torch.split(dataset_dict["video_images"], 1, dim=0)]
        if not self.is_train:
            dataset_dict.update(ret)
            return
        ret["instances"] = []
        ori_instances = dataset_dict['instances']

        if not ori_instances.has("gt_masks"):
            image_shape = (ret["image"][0].shape[-2], ret["image"][0].shape[-1])
            for i in range(len(dataset_dict["frame_idx"])):
                instances = Instances(image_shape)
                instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
                instances.gt_classes = torch.tensor([])
                instances.gt_ids = torch.tensor([])
                ret["instances"].append(instances)
            dataset_dict.update(ret)
            return

        masks = ori_instances.gt_masks.tensor
        classes = ori_instances.gt_classes
        assert masks.size(1) == len(dataset_dict["frame_idx"])

        image_shape = (ret["image"][0].shape[-2], ret["image"][0].shape[-1])

        for i in range(len(dataset_dict["frame_idx"])):
            instances = Instances(image_shape)
            instances.gt_masks = masks[:, i]
            instances.gt_classes = copy.deepcopy(classes)
            instances.gt_ids = torch.arange(0, masks.size(0))
            ret["instances"].append(instances)

        dataset_dict.update(ret)
        return

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        video_length = len(dataset_dict['file_names'])
        if self.is_train:
            index_list = self.select_frames(video_length)
        else:
            index_list = range(video_length)
        dataset_dict["video_len"] = video_length
        dataset_dict["frame_idx"] = index_list

        # if video length less than sample length, pad with last frame
        # if video length large than sample length, sample continuous frames
        select_filenames = []
        select_pan_seg_file_names = []
        select_segments_infos = []

        for idx in index_list:
            select_filenames.append(dataset_dict['file_names'][idx])
            if "pan_seg_file_names" in dataset_dict:
                select_pan_seg_file_names.append(dataset_dict["pan_seg_file_names"][idx])
                select_segments_infos.append(dataset_dict['segments_infos'][idx])

            else:
                select_pan_seg_file_names.append(None)
                select_segments_infos.append(None)

        insid_catid_dic = {}  # ins-cat dict
        input_images = []
        input_panoptic_seg = []
        for ii_, (file_name, pan_seg_file_name, segments_infos) in enumerate(
                zip(select_filenames, select_pan_seg_file_names, select_segments_infos)):

            if segments_infos is not None and self.is_train:
                for segments_info in segments_infos:
                    class_id = segments_info["category_id"]
                    ins_id = segments_info['id']
                    if not segments_info['iscrowd']:
                        if ins_id not in insid_catid_dic:
                            insid_catid_dic[ins_id] = class_id

            if ii_ == 0:
                image = utils.read_image(file_name, format=self.img_format)
                utils.check_image_size(dataset_dict, image)
                if pan_seg_file_name is not None and self.is_train:
                    pan_seg_gt = utils.read_image(pan_seg_file_name, "RGB")
                else:
                    pan_seg_gt = None

                aug_input = T.AugInput(image, sem_seg=pan_seg_gt)
                aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
                image = aug_input.image
                pan_seg_gt = aug_input.sem_seg

            else:
                image = utils.read_image(file_name, format=self.img_format)
                utils.check_image_size(dataset_dict, image)
                image = transforms.apply_image(image)
                if pan_seg_file_name is not None and self.is_train:
                    pan_seg_gt = utils.read_image(pan_seg_file_name, "RGB")
                else:
                    pan_seg_gt = None

                # apply the same transformation to panoptic segmentation
                if pan_seg_gt is not None:
                    pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

            if pan_seg_gt is not None:
                pan_seg_gt = rgb2id(pan_seg_gt)

            # Pad image and segmentation label here!
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

            input_images.append(image.unsqueeze(0))
            input_panoptic_seg.append(pan_seg_gt)

        input_images = torch.cat(input_images, 0)
        dataset_dict["video_images"] = input_images
        if not self.is_train:
            self.convert2ytvis(dataset_dict)
            return dataset_dict

        image_shape = (input_images.shape[-2], input_images.shape[-1])
        input_panoptic_seg = np.stack(input_panoptic_seg)
        unique_ids = np.unique(input_panoptic_seg)

        instances = Instances(image_shape)
        classes = []
        masks = []
        for insid_, class_id_ in insid_catid_dic.items():
            if class_id_ not in self.thing_ids_to_continue_dic:
                if insid_ in unique_ids:
                    classes.append(self.stuff_ids_to_continue_dic[class_id_] + len(self.thing_ids_to_continue_dic))
                    masks.append(input_panoptic_seg == insid_)
                continue

            if insid_ in unique_ids:
                classes.append(self.thing_ids_to_continue_dic[class_id_])
                masks.append(input_panoptic_seg == insid_)

        classes = np.array(classes)
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            pass
        else:
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            masks = Video_BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks

        dataset_dict["instances"] = instances

        # align to ytvis target format
        self.convert2ytvis(dataset_dict)

        return dataset_dict