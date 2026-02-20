# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from:
# - the MinVIS repository https://github.com/zhang-tao-whu/DVIS/blob/main/dvis/meta_architecture.py
# Used Nvidia Source Code License-NC.
# - the CAVIS repository https://github.com/Seung-Hun-Lee/CAVIS/blob/main/cavis/cavis.py
# Used under the MIT License.
# ---------------------------------------------------------------

import logging
import re
from typing import Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import einops

from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from torchvision.transforms.v2.functional import resize, pad

from .criterion_videomt import VideoSetCriterion, loss_reid
from .modeling.matcher import VideoHungarianMatcher, VideoHungarianMatcher_Consistent
from .utils.memory import retry_if_cuda_oom

@META_ARCH_REGISTRY.register()
class videomt(nn.Module):
    """
    Copied from "https://github.com/NVlabs/MinVIS".
    """

    @configurable
    def __init__(
        self,
        *,
        img_size: int,
        backbone: Backbone,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # video
        num_frames,
        window_inference,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.img_size = img_size
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_frames = num_frames
        self.window_inference = window_inference

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)

        # Loss parameters:
        deep_supervision = cfg.MODEL.BACKBONE.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.BACKBONE.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.BACKBONE.CLASS_WEIGHT
        dice_weight = cfg.MODEL.BACKBONE.DICE_WEIGHT
        mask_weight = cfg.MODEL.BACKBONE.MASK_WEIGHT

        # building criterion
        matcher = VideoHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.BACKBONE.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.BACKBONE.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = VideoSetCriterion(
            backbone.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.BACKBONE.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.BACKBONE.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.BACKBONE.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "img_size": cfg.MODEL.BACKBONE.IMG_SIZE,
            "backbone": backbone,
            "criterion": criterion,
            "num_queries": cfg.MODEL.BACKBONE.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.BACKBONE.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.BACKBONE.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.BACKBONE.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "window_inference": cfg.MODEL.BACKBONE.TEST.WINDOW_INFERENCE
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        if not self.training and self.window_inference:
            outputs = self.run_window_inference(images.tensor, window_size=3)
        else:
            outputs = self.backbone(images.tensor)

        if self.training:
            # mask classification target
            targets = self.prepare_targets(batched_inputs, images)
            outputs, targets = self.frame_decoder_loss_reshape(outputs, targets)

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            outputs = self.post_processing(outputs)

            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            mask_cls_result = mask_cls_results[0]
            mask_pred_result = mask_pred_results[0]
            first_resize_size = (images.tensor.shape[-2], images.tensor.shape[-1])

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            return retry_if_cuda_oom(self.inference_video)(
                mask_cls_result,
                mask_pred_result,
                image_size,
                height,
                width,
                first_resize_size)

    def frame_decoder_loss_reshape(self, outputs, targets):
        outputs['pred_masks'] = einops.rearrange(outputs['pred_masks'], 'b q t h w -> (b t) q () h w')
        outputs['pred_logits'] = einops.rearrange(outputs['pred_logits'], 'b t q c -> (b t) q c')
        if 'aux_outputs' in outputs:
            for i in range(len(outputs['aux_outputs'])):
                outputs['aux_outputs'][i]['pred_masks'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_masks'], 'b q t h w -> (b t) q () h w'
                )
                outputs['aux_outputs'][i]['pred_logits'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_logits'], 'b t q c -> (b t) q c'
                )
        gt_instances = []
        for targets_per_video in targets:
            num_labeled_frames = targets_per_video['ids'].shape[1]
            for f in range(num_labeled_frames):
                labels = targets_per_video['labels']
                ids = targets_per_video['ids'][:, [f]]
                masks = targets_per_video['masks'][:, [f], :, :]
                gt_instances.append({"labels": labels, "ids": ids, "masks": masks})
        return outputs, gt_instances

    def match_from_embds(self, tgt_embds, cur_embds):
        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0, 1))
        cost_embd = 1 - cos_sim
        C = 1.0 * cost_embd
        C = C.cpu()
        indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
        indices = indices[1]  # permutation that makes current aligns to target
        return indices

    def post_processing(self, outputs):
        pred_logits, pred_masks, pred_embds = outputs['pred_logits'], outputs['pred_masks'], outputs['pred_embds']

        pred_logits = pred_logits[0]
        pred_masks = einops.rearrange(pred_masks[0], 'q t h w -> t q h w')
        pred_embds = einops.rearrange(pred_embds[0], 'c t q -> t q c')

        pred_logits = list(torch.unbind(pred_logits))
        pred_masks = list(torch.unbind(pred_masks))
        pred_embds = list(torch.unbind(pred_embds))

        out_logits = []
        out_masks = []
        out_embds = []
        out_logits.append(pred_logits[0])
        out_masks.append(pred_masks[0])
        out_embds.append(pred_embds[0])

        # match the instances frame by frame
        for i in range(1, len(pred_logits)):
            indices = self.match_from_embds(out_embds[-1], pred_embds[i])

            out_logits.append(pred_logits[i][indices, :])
            out_masks.append(pred_masks[i][indices, :, :])
            out_embds.append(pred_embds[i][indices, :])

        out_logits = sum(out_logits)/len(out_logits)
        out_masks = torch.stack(out_masks, dim=1)  # q h w -> q t h w

        out_logits = out_logits.unsqueeze(0)
        out_masks = out_masks.unsqueeze(0)

        outputs['pred_logits'] = out_logits
        outputs['pred_masks'] = out_masks

        return outputs

    def run_window_inference(self, images_tensor, window_size=30):
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1
        out_list = []
        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i+1) * window_size

            out = self.backbone(images_tensor[start_idx:end_idx])
           
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']
            out['pred_masks'] = out['pred_masks'].detach().cpu().to(torch.float32)
            out_list.append(out)

        # merge outputs
        outputs = {}
        outputs['pred_logits'] = torch.cat([x['pred_logits'] for x in out_list], dim=1).detach()
        outputs['pred_masks'] = torch.cat([x['pred_masks'] for x in out_list], dim=2).detach()
        outputs['pred_embds'] = torch.cat([x['pred_embds'] for x in out_list], dim=2).detach()

        return outputs

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                if isinstance(targets_per_frame.gt_masks, BitMasks):
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor
                else:  # polygon
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)

            gt_classes_per_video = targets_per_frame.gt_classes[valid_idx]          # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]                          # N, num_frames

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()          # N, num_frames, H, W
            gt_instances[-1].update({"masks": gt_masks_per_video})

        return gt_instances

    def inference_video(self, pred_cls, pred_masks, img_size, output_height, output_width, first_resize_size):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            labels = torch.arange(
                self.backbone.num_classes,
                device=self.device
            ).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            # keep top-10 predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(10, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.backbone.num_classes
            topk_indices = topk_indices.to(pred_masks.device)
            pred_masks = pred_masks[topk_indices]

            pred_masks = F.interpolate(
                pred_masks, size=first_resize_size, mode="bilinear", align_corners=False
            )

            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )

            masks = pred_masks > 0.

            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output


@META_ARCH_REGISTRY.register()
class videomt_segmenter(videomt):
    

    @configurable
    def __init__(
        self,
        *,
        img_size: int,
        backbone: Backbone,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # video
        num_frames,
        window_inference,
        max_num,
        max_iter_num,
        window_size,
        task,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
         
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            # video
            num_frames: number of frames sampled during training
            window_inference: if the GPU memory is insufficient to predict the entire video at
                once, inference needs to be performed clip by clip
            num_class: the categories number of the dataset
            max_num: the maximum number of instances retained for a video, only used in VIS
            max_iter_num: the iter nums
            window_size: the number of images processed by the segmenter at a time
            task: VIS, VSS or VPS
        """
        super().__init__(
            backbone=backbone,
            img_size=img_size,
            criterion=criterion,
            num_queries=num_queries,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            # video
            num_frames=num_frames,
            window_inference=window_inference,
        )
        self.max_num = max_num
        self.iter = 0
        self.max_iter_num = max_iter_num

        self.window_size = window_size
        self.task = task
        assert self.task in ['vis', 'vss', 'vps'], "Only support vis, vss and vps !"
        inference_dict = {
            'vis': self.inference_video_vis,
            'vss': self.inference_video_vss,
            'vps': self.inference_video_vps,
        }
        self.inference_video_task = inference_dict[self.task]

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)

        # Loss parameters:
        deep_supervision = cfg.MODEL.BACKBONE.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.BACKBONE.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.BACKBONE.CLASS_WEIGHT
        dice_weight = cfg.MODEL.BACKBONE.DICE_WEIGHT
        mask_weight = cfg.MODEL.BACKBONE.MASK_WEIGHT

        # building criterion
        matcher = VideoHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.BACKBONE.TRAIN_NUM_POINTS,
        )
        
        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }

        if deep_supervision:
            dec_layers = cfg.MODEL.BACKBONE.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        weight_dict.update(
            {
                "loss_ctx": 2.0,
                "loss_ctx_aux": 2.0,
                "loss_proto": 2.0,
                "loss_proto_aux": 2.0,
            }
        )

        losses = ["labels", "masks"]

        criterion = VideoSetCriterion(
            backbone.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.BACKBONE.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.BACKBONE.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.BACKBONE.IMPORTANCE_SAMPLE_RATIO,
        )

        max_iter_num = cfg.SOLVER.MAX_ITER

        return {
            "img_size": cfg.MODEL.BACKBONE.IMG_SIZE,
            "backbone": backbone,
            "criterion": criterion,
            "num_queries": cfg.MODEL.BACKBONE.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.BACKBONE.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.BACKBONE.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.BACKBONE.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "window_inference": cfg.MODEL.BACKBONE.TEST.WINDOW_INFERENCE,
            "max_num": cfg.MODEL.BACKBONE.TEST.MAX_NUM,
            "max_iter_num": max_iter_num,
            "window_size": cfg.MODEL.BACKBONE.TEST.WINDOW_SIZE,
            "task": cfg.MODEL.BACKBONE.TEST.TASK,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def validation_data_preprocessing(self,img_size,images):

        img_shape = images.tensor.shape[-2:]
        factor = min(
                img_size / img_shape[0],
                img_size / img_shape[1],
            )

        scaled_size = [round(s * factor) for s in img_shape]
        pil_img = Image.fromarray(images.tensor[0].permute(1, 2, 0).cpu().numpy())
        pil_img = pil_img.resize((scaled_size[1], scaled_size[0]), Image.BILINEAR)
        images.tensor = (
            torch.from_numpy(np.array(pil_img))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(images.tensor.device)
        )

        pad_h = max(0, img_size - images.tensor.shape[-2])
        pad_w = max(0, img_size - images.tensor.shape[-1])
        padding = [0, 0, pad_w, pad_h]

        images.tensor = pad(images.tensor, padding)
        return images, scaled_size

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            dict:
                For specific task, the dict contains the following keys:
                * For VIS:
                    "image_size": (output_height, output_width).
                    "pred_scores": score for per instance.
                    "pred_labels": class for per instance.
                    "pred_masks": list[Tensor], bit-masks for per instance, Tensor shape is (t, h, w).
                    "pred_ids": list, query ids for per instance, list length is N.
                    "task": "vis",
                * For VSS:
                    "image_size": (output_height, output_width).
                    "pred_masks": A Tensor that represents the
                        per-pixel segmentation prediced by the head.
                        The prediction has shape (t, h, w) that represents
                        the category ID for each pixel.
                    "task": "vss".
                * For VPS:
                    "image_size": (output_height, output_width).
                    "pred_masks": Tensor, shape is (t, h, w),
                        that represents the unique ID for the object which each pixel belong to.
                    "segments_infos": list[dict], info dicts for per object.
                        Info dict including unique ID, category ID and isthing.
                    "pred_ids": list, query ids for per thing and stuff, list length is N.
                    "task": "vps".
        """

        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        

        if not self.training and self.window_inference:
            outputs = self.run_window_inference(images.tensor, window_size=self.window_size)
        else:
            outputs = self.backbone(images.tensor)
       

        if self.training:
            targets = self.prepare_targets(batched_inputs, images)
            outputs, targets = self.frame_decoder_loss_reshape(outputs, targets)
            
            losses = self.criterion(outputs, targets)
            
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
                                
            return losses

        else:
            outputs = self.post_processing(outputs)
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            pred_ids = outputs["ids"]

            mask_cls_result = mask_cls_results[0]
            mask_pred_result = mask_pred_results[0]
            pred_id = pred_ids[0]
            first_resize_size = (images.tensor.shape[-2], images.tensor.shape[-1])

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            return retry_if_cuda_oom(self.inference_video_task)(
                mask_cls_result, mask_pred_result, image_size, height, width, first_resize_size, pred_id
            )   

    def post_processing(self, outputs):

        pred_logits, pred_masks, pred_embds = outputs['pred_logits'], outputs['pred_masks'], outputs['pred_embds']
        pred_logits = pred_logits[0]
        pred_masks = einops.rearrange(pred_masks[0], 'q t h w -> t q h w')
        pred_embds = einops.rearrange(pred_embds[0], 'c t q -> t q c')

        pred_logits = list(torch.unbind(pred_logits))
        pred_masks = list(torch.unbind(pred_masks))
        pred_embds = list(torch.unbind(pred_embds))

        out_logits = []
        out_masks = []
        out_embds = []
        out_logits.append(pred_logits[0])
        out_masks.append(pred_masks[0])
        out_embds.append(pred_embds[0])

        # match the instances frame by frame
        for i in range(1, len(pred_logits)):
            indices = self.match_from_embds(out_embds[-1], pred_embds[i])

            out_logits.append(pred_logits[i][indices, :])
            out_masks.append(pred_masks[i][indices, :, :])
            out_embds.append(pred_embds[i][indices, :])

        out_logits = sum(out_logits)/len(out_logits)
        out_masks = torch.stack(out_masks, dim=1)  # q h w -> q t h w

        out_logits = out_logits.unsqueeze(0)
        out_masks = out_masks.unsqueeze(0)

        outputs['pred_logits'] = out_logits
        outputs['pred_masks'] = out_masks
        outputs['ids'] = [torch.arange(0, outputs['pred_masks'].size(1))]

        return outputs

    def run_window_inference(self, images_tensor, window_size=30):
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1
        out_list = []
        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i+1) * window_size

            out = self.backbone(images_tensor[start_idx:end_idx])
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']
            del out['pred_reid_embed'], out['mask_features'], out['pred_embds_without_norm']
            out['pred_masks'] = out['pred_masks'].detach().cpu().to(torch.float32)
            out_list.append(out)

        # merge outputs
        outputs = {}
        outputs['pred_logits'] = torch.cat([x['pred_logits'] for x in out_list], dim=1).detach()
        outputs['pred_masks'] = torch.cat([x['pred_masks'] for x in out_list], dim=2).detach()
        outputs['pred_embds'] = torch.cat([x['pred_embds'] for x in out_list], dim=2).detach()

        return outputs

    def inference_video_vis(
        self, pred_cls, pred_masks, img_size, output_height, output_width,
        first_resize_size, pred_id, aux_pred_cls=None,
    ):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            if aux_pred_cls is not None:
                aux_pred_cls = F.softmax(aux_pred_cls, dim=-1)[:, :-1]
                scores = torch.maximum(scores, aux_pred_cls.to(scores))
            labels = torch.arange(
                self.backbone.num_classes, device=self.device
            ).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            # keep top-K predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.max_num, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.backbone.num_classes
            topk_indices = topk_indices.to(pred_masks.device)
            pred_masks = pred_masks[topk_indices]
            pred_ids = pred_id[topk_indices]

            # interpolation to original image size
            pred_masks = F.interpolate(
                pred_masks, size=first_resize_size, mode="bilinear", align_corners=False
            )
            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )
            masks = pred_masks > 0.
            del pred_masks

            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_ids = pred_ids.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []
            out_ids = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
            "pred_ids": out_ids,
            "task": "vis",
        }

        return video_output

    def inference_video_vps(
        self, pred_cls, pred_masks, img_size, output_height, output_width,
        first_resize_size, pred_id, aux_pred_cls=None,
    ):
        pred_cls = F.softmax(pred_cls, dim=-1)
        if aux_pred_cls is not None:
            aux_pred_cls = F.softmax(aux_pred_cls, dim=-1)[:, :-1]
            pred_cls[:, :-1] = torch.maximum(pred_cls[:, :-1], aux_pred_cls.to(pred_cls))
        mask_pred = pred_masks
        scores, labels = pred_cls.max(-1)

        # filter out the background prediction
        keep = labels.ne(self.backbone.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_ids = pred_id[keep]
        cur_masks = mask_pred[keep]

        # interpolation to original image size
        cur_masks = F.interpolate(
            cur_masks, size=first_resize_size, mode="bilinear", align_corners=False
        )
        cur_masks = cur_masks[:, :, :img_size[0], :img_size[1]].sigmoid()
        cur_masks = F.interpolate(
            cur_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
        )
        cur_prob_masks = cur_scores.view(-1, 1, 1, 1).to(cur_masks.device) * cur_masks

        # initial panoptic_seg and segments infos
        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((cur_masks.size(1), h, w), dtype=torch.int32, device=cur_masks.device)
        segments_infos = []
        out_ids = []
        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask
            return {
                "image_size": (output_height, output_width),
                "pred_masks": panoptic_seg.cpu(),
                "segments_infos": segments_infos,
                "pred_ids": out_ids,
                "task": "vps",
            }
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)  # (t, h, w)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class < len(self.metadata.thing_dataset_id_to_contiguous_id)
                # filter out the unstable segmentation results
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)
                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue
                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1
                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_infos.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )
                    out_ids.append(cur_ids[k])

            return {
                "image_size": (output_height, output_width),
                "pred_masks": panoptic_seg.cpu(),
                "segments_infos": segments_infos,
                "pred_ids": out_ids,
                "task": "vps",
            }

    def inference_video_vss(
        self, pred_cls, pred_masks, img_size, output_height, output_width,
        first_resize_size, pred_id, aux_pred_cls=None,
    ):
        mask_cls = F.softmax(pred_cls, dim=-1)[..., :-1]
        if aux_pred_cls is not None:
            aux_pred_cls = F.softmax(aux_pred_cls, dim=-1)[:, :-1]
            mask_cls = torch.maximum(mask_cls, aux_pred_cls.to(mask_cls))
        mask_pred = pred_masks
        # interpolation to original image size
        cur_masks = F.interpolate(
            mask_pred, size=first_resize_size, mode="bilinear", align_corners=False
        )
        cur_masks = cur_masks[:, :, :img_size[0], :img_size[1]].sigmoid()
        cur_masks = F.interpolate(
            cur_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
        )
        
        cur_masks = cur_masks.to(mask_cls.device)
        semseg = torch.einsum("qc,qthw->cthw", mask_cls, cur_masks)
        sem_score, sem_mask = semseg.max(0)
        sem_mask = sem_mask
        return {
                "image_size": (output_height, output_width),
                "pred_masks": sem_mask.cpu(),
                "task": "vss",
            }


@META_ARCH_REGISTRY.register()
class videomt_online(videomt):
    
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        img_size: int,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # video
        num_frames,
        window_inference,
        max_num,
        max_iter_num,
        window_size,
        task,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            # video
           
            num_frames: number of frames sampled during training
            window_inference: if the GPU memory is insufficient to predict the entire video at
                once, inference needs to be performed clip by clip
            num_class: the categories number of the dataset
            max_num: the maximum number of instances retained for a video, only used in VIS
            max_iter_num: the iter nums
            window_size: the number of images processed by the segmenter at a time
            task: VIS, VSS or VPS
        """
        super().__init__(
            backbone=backbone,
            img_size=img_size,
            criterion=criterion,
            num_queries=num_queries,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            # video
            num_frames=num_frames,
            window_inference=window_inference,
        )
 

        self.max_num = max_num
        self.iter = 0
        self.max_iter_num = max_iter_num

        self.window_size = window_size
        self.task = task
        assert self.task in ['vis', 'vss', 'vps'], "Only support vis, vss and vps !"
        inference_dict = {
            'vis': self.inference_video_vis,
            'vss': self.inference_video_vss,
            'vps': self.inference_video_vps,
        }
        self.inference_video_task = inference_dict[self.task]


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)


        # Loss parameters:
        deep_supervision = cfg.MODEL.BACKBONE.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.BACKBONE.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.BACKBONE.CLASS_WEIGHT
        dice_weight = cfg.MODEL.BACKBONE.DICE_WEIGHT
        mask_weight = cfg.MODEL.BACKBONE.MASK_WEIGHT

        # building criterion
        matcher = VideoHungarianMatcher_Consistent(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.BACKBONE.TRAIN_NUM_POINTS,
            frames=cfg.INPUT.SAMPLING_FRAME_NUM
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }


        if deep_supervision:
            dec_layers = cfg.MODEL.BACKBONE.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
          

        weight_dict.update(
            {
                "loss_reid": 2.0,
                "loss_reid_aux": 2.0,
            }
        )

        losses = ["labels", "masks"]

        criterion = VideoSetCriterion(
            backbone.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.BACKBONE.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.BACKBONE.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.BACKBONE.IMPORTANCE_SAMPLE_RATIO,
        )

    

        max_iter_num = cfg.SOLVER.MAX_ITER

        return {
            "backbone": backbone,
            "img_size": cfg.MODEL.BACKBONE.IMG_SIZE,
            "criterion": criterion,
            "num_queries": cfg.MODEL.BACKBONE.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.BACKBONE.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.BACKBONE.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.BACKBONE.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "window_inference": cfg.MODEL.BACKBONE.TEST.WINDOW_INFERENCE,
            "max_num": cfg.MODEL.BACKBONE.TEST.MAX_NUM,
            "max_iter_num": max_iter_num,
            "window_size": cfg.MODEL.BACKBONE.TEST.WINDOW_SIZE,
            "task": cfg.MODEL.BACKBONE.TEST.TASK,
        }

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            dict:
                For specific task, the dict contains the following keys:
                * For VIS:
                    "image_size": (output_height, output_width).
                    "pred_scores": score for per instance.
                    "pred_labels": class for per instance.
                    "pred_masks": list[Tensor], bit-masks for per instance, Tensor shape is (t, h, w).
                    "pred_ids": list, query ids for per instance, list length is N.
                    "task": "vis",
                * For VSS:
                    "image_size": (output_height, output_width).
                    "pred_masks": A Tensor that represents the
                        per-pixel segmentation prediced by the head.
                        The prediction has shape (t, h, w) that represents
                        the category ID for each pixel.
                    "task": "vss".
                * For VPS:
                    "image_size": (output_height, output_width).
                    "pred_masks": Tensor, shape is (t, h, w),
                        that represents the unique ID for the object which each pixel belong to.
                    "segments_infos": list[dict], info dicts for per object.
                        Info dict including unique ID, category ID and isthing.
                    "pred_ids": list, query ids for per thing and stuff, list length is N.
                    "task": "vps".
        """
        # for running demo on very long videos
        if 'keep' in batched_inputs[0].keys():
            self.keep = batched_inputs[0]['keep']
        else:
            self.keep = False

        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        if not self.training and self.window_inference:
            outputs = self.run_window_inference(images.tensor, window_size=self.window_size)
        else:
            
            outputs = self.backbone(images.tensor, resume=self.keep)

        if self.training:

            targets = self.prepare_targets(batched_inputs, images)
            outputs, targets = self.frame_decoder_loss_reshape(outputs, targets)
            losses = self.criterion(outputs, targets)

            # Apply weights and filter
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    losses.pop(k)
            return losses

        else:
            outputs = self.post_processing(outputs)
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            pred_ids = outputs["ids"]

            mask_cls_result = mask_cls_results[0]
            mask_pred_result = mask_pred_results[0]
            pred_id = pred_ids[0]
            first_resize_size = (images.tensor.shape[-2], images.tensor.shape[-1])

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            return retry_if_cuda_oom(self.inference_video_task)(
                mask_cls_result, mask_pred_result, image_size, height, width, first_resize_size, pred_id
            )

    def frame_decoder_loss_reshape(self, outputs, targets):
        outputs['pred_masks'] = einops.rearrange(outputs['pred_masks'], 'b q t h w -> (b t) q () h w')
        outputs['pred_logits'] = einops.rearrange(outputs['pred_logits'], 'b t q c -> (b t) q c')
      
        if 'aux_outputs' in outputs:
            for i in range(len(outputs['aux_outputs'])):
                outputs['aux_outputs'][i]['pred_masks'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_masks'], 'b q t h w -> (b t) q () h w'
                )
                outputs['aux_outputs'][i]['pred_logits'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_logits'], 'b t q c -> (b t) q c'
                )
        gt_instances = []
        for targets_per_video in targets:
            num_labeled_frames = targets_per_video['ids'].shape[1]
            for f in range(num_labeled_frames):
                labels = targets_per_video['labels']
                ids = targets_per_video['ids'][:, [f]]
                masks = targets_per_video['masks'][:, [f], :, :]
                gt_instances.append({"labels": labels, "ids": ids, "masks": masks})
        return  outputs, gt_instances
   

    def post_processing(self, outputs, aux_logits=None):
        """
        average the class logits and append query ids
        """
        pred_logits = outputs['pred_logits']
        pred_logits = pred_logits[0]  # (t, q, c)
        out_logits = torch.mean(pred_logits, dim=0).unsqueeze(0)
        if aux_logits is not None:
            aux_logits = aux_logits[0]
            aux_logits = torch.mean(aux_logits, dim=0)  # (q, c)
        outputs['pred_logits'] = out_logits
        outputs['ids'] = [torch.arange(0, outputs['pred_masks'].size(1))]
        if aux_logits is not None:
            return outputs, aux_logits
        return outputs

    def run_window_inference(self, images_tensor, window_size=30):
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1
        out_list = []
        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i+1) * window_size
            # segmeter inference
            if i != 0 or self.keep:
                out = self.backbone(images_tensor[start_idx:end_idx],resume=True)
            else:
                out = self.backbone(images_tensor[start_idx:end_idx])
           
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']
          
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']
            out['pred_logits'] = out['pred_logits'].to(torch.float32).detach().cpu()
            out['pred_masks'] = out['pred_masks'].to(torch.float32).detach().cpu()
          
            out_list.append(out)

        # merge outputs
        outputs = {}
        outputs['pred_logits'] = torch.cat([x['pred_logits'] for x in out_list], dim=1)
        outputs['pred_masks'] = torch.cat([x['pred_masks'] for x in out_list], dim=2)

        return outputs

    def inference_video_vis(
        self, pred_cls, pred_masks, img_size, output_height, output_width,
        first_resize_size, pred_id, aux_pred_cls=None,
    ):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            if aux_pred_cls is not None:
                aux_pred_cls = F.softmax(aux_pred_cls, dim=-1)[:, :-1]
                scores = torch.maximum(scores, aux_pred_cls.to(scores))
            labels = torch.arange(
                self.backbone.num_classes, device=self.device
            ).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            # keep top-K predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.max_num, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.backbone.num_classes
            pred_masks = pred_masks[topk_indices]
            pred_ids = pred_id[topk_indices]

            # interpolation to original image size
            pred_masks = F.interpolate(
                pred_masks, size=first_resize_size, mode="bilinear", align_corners=False
            )
            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )
            masks = pred_masks > 0.
            del pred_masks

            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_ids = pred_ids.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []
            out_ids = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
            "pred_ids": out_ids,
            "task": "vis",
        }

        return video_output

    def inference_video_vps(
        self, pred_cls, pred_masks, img_size, output_height, output_width,
        first_resize_size, pred_id, aux_pred_cls=None,
    ):
        pred_cls = F.softmax(pred_cls, dim=-1)
        if aux_pred_cls is not None:
            aux_pred_cls = F.softmax(aux_pred_cls, dim=-1)[:, :-1]
            pred_cls[:, :-1] = torch.maximum(pred_cls[:, :-1], aux_pred_cls.to(pred_cls))
        mask_pred = pred_masks
        scores, labels = pred_cls.max(-1)

        # filter out the background prediction
        keep = labels.ne(self.backbone.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_ids = pred_id[keep]
        cur_masks = mask_pred[keep]

        # interpolation to original image size
        cur_masks = F.interpolate(
            cur_masks, size=first_resize_size, mode="bilinear", align_corners=False
        )
        cur_masks = cur_masks[:, :, :img_size[0], :img_size[1]].sigmoid()
        cur_masks = F.interpolate(
            cur_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
        )
        cur_prob_masks = cur_scores.view(-1, 1, 1, 1).to(cur_masks.device) * cur_masks

        # initial panoptic_seg and segments infos
        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((cur_masks.size(1), h, w), dtype=torch.int32, device=cur_masks.device)
        segments_infos = []
        out_ids = []
        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask
            return {
                "image_size": (output_height, output_width),
                "pred_masks": panoptic_seg.cpu(),
                "segments_infos": segments_infos,
                "pred_ids": out_ids,
                "task": "vps",
            }
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)  # (t, h, w)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class < len(self.metadata.thing_dataset_id_to_contiguous_id)
                # filter out the unstable segmentation results
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)
                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue
                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1
                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_infos.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )
                    out_ids.append(cur_ids[k])

            return {
                "image_size": (output_height, output_width),
                "pred_masks": panoptic_seg.cpu(),
                "segments_infos": segments_infos,
                "pred_ids": out_ids,
                "task": "vps",
            }

    def inference_video_vss(
        self, pred_cls, pred_masks, img_size, output_height, output_width,
        first_resize_size, pred_id, aux_pred_cls=None,
    ):
        mask_cls = F.softmax(pred_cls, dim=-1)[..., :-1]
        if aux_pred_cls is not None:
            aux_pred_cls = F.softmax(aux_pred_cls, dim=-1)[:, :-1]
            mask_cls = torch.maximum(mask_cls, aux_pred_cls.to(mask_cls))
        mask_pred = pred_masks
        # interpolation to original image size
        cur_masks = F.interpolate(
            mask_pred, size=first_resize_size, mode="bilinear", align_corners=False
        )
        cur_masks = cur_masks[:, :, :img_size[0], :img_size[1]].sigmoid()
        cur_masks = F.interpolate(
            cur_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
        )

        # cur_masks = cur_masks.to(mask_cls.device)
        semseg = torch.einsum("qc,qthw->cthw", mask_cls, cur_masks)
        sem_score, sem_mask = semseg.max(0)
        sem_mask = sem_mask
        return {
                "image_size": (output_height, output_width),
                "pred_masks": sem_mask.cpu(),
                "task": "vss",
            }
        
    def get_cl_loss_ref(self, outputs, referecne_match_result):
        references = outputs['pred_references']  # t q c

        # per frame
        contrastive_items = []
        for i in range(references.size(0)):
            if i == 0:
                continue
            frame_reference = references[i]  # (q, c)
            frame_reference_ = references[i - 1]  # (q, c)

            if i != references.size(0) - 1:
                frame_reference_next = references[i + 1]
            else:
                frame_reference_next = None

            frame_ref_gt_indices = referecne_match_result[i]

            gt2ref = {}
            for i_ref, i_gt in zip(frame_ref_gt_indices[0], frame_ref_gt_indices[1]):
                gt2ref[i_gt.item()] = i_ref.item()
            # per instance
            for i_gt in gt2ref.keys():
                i_ref = gt2ref[i_gt]

                anchor_embeds = frame_reference[[i_ref]]
                pos_embeds = frame_reference_[[i_ref]]
                neg_range = list(range(0, i_ref)) + list(range(i_ref + 1, frame_reference.size(0)))
                neg_embeds = frame_reference_[neg_range]

                num_positive = pos_embeds.shape[0]
                # concate pos and neg to get whole constractive samples
                pos_neg_embedding = torch.cat(
                    [pos_embeds, neg_embeds], dim=0)
                # generate label, pos is 1, neg is 0
                pos_neg_label = pos_neg_embedding.new_zeros((pos_neg_embedding.shape[0],),
                                                            dtype=torch.int64)  # noqa
                pos_neg_label[:num_positive] = 1.

                # dot product
                dot_product = torch.einsum(
                    'ac,kc->ak', [pos_neg_embedding, anchor_embeds])
                aux_normalize_pos_neg_embedding = nn.functional.normalize(
                    pos_neg_embedding, dim=1)
                aux_normalize_anchor_embedding = nn.functional.normalize(
                    anchor_embeds, dim=1)

                aux_cosine_similarity = torch.einsum('ac,kc->ak', [aux_normalize_pos_neg_embedding,
                                                                   aux_normalize_anchor_embedding])
                contrastive_items.append({
                    'dot_product': dot_product,
                    'cosine_similarity': aux_cosine_similarity,
                    'label': pos_neg_label})

                if frame_reference_next is not None:
                    pos_embeds = frame_reference_next[[i_ref]]
                    neg_range = list(range(0, i_ref)) + list(range(i_ref + 1, frame_reference.size(0)))
                    neg_embeds = frame_reference_next[neg_range]

                    num_positive = pos_embeds.shape[0]
                    # concate pos and neg to get whole constractive samples
                    pos_neg_embedding = torch.cat(
                        [pos_embeds, neg_embeds], dim=0)
                    # generate label, pos is 1, neg is 0
                    pos_neg_label = pos_neg_embedding.new_zeros((pos_neg_embedding.shape[0],),
                                                                dtype=torch.int64)  # noqa
                    pos_neg_label[:num_positive] = 1.

                    # dot product
                    dot_product = torch.einsum(
                        'ac,kc->ak', [pos_neg_embedding, anchor_embeds])
                    aux_normalize_pos_neg_embedding = nn.functional.normalize(
                        pos_neg_embedding, dim=1)
                    aux_normalize_anchor_embedding = nn.functional.normalize(
                        anchor_embeds, dim=1)

                    aux_cosine_similarity = torch.einsum('ac,kc->ak', [aux_normalize_pos_neg_embedding,
                                                                       aux_normalize_anchor_embedding])
                    contrastive_items.append({
                        'dot_product': dot_product,
                        'cosine_similarity': aux_cosine_similarity,
                        'label': pos_neg_label})

        losses = loss_reid(contrastive_items, outputs['pred_references'], 'loss_reid')
        return losses

