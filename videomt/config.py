# -*- coding: utf-8 -*-
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.config import CfgNode as CN

def add_videomt_config(cfg):
    """
    Add config for videomt.
    """
    # NOTE: configs from original maskformer
    # data config
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1
    cfg.INPUT.SAMPLING_FRAME_NUM = 2
    cfg.INPUT.SAMPLING_FRAME_RANGE = 20
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.BACKBONE = CN()

    # loss
    cfg.MODEL.BACKBONE.DEEP_SUPERVISION = True
    cfg.MODEL.BACKBONE.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.BACKBONE.CLASS_WEIGHT = 1.0
    cfg.MODEL.BACKBONE.DICE_WEIGHT = 1.0
    cfg.MODEL.BACKBONE.MASK_WEIGHT = 20.0
    cfg.MODEL.BACKBONE.NAME = "VidEoMT"

    # transformer config
    cfg.MODEL.BACKBONE.NHEADS = 8
    cfg.MODEL.BACKBONE.DROPOUT = 0.1
    cfg.MODEL.BACKBONE.DIM_FEEDFORWARD = 2048
    cfg.MODEL.BACKBONE.ENC_LAYERS = 0
    cfg.MODEL.BACKBONE.DEC_LAYERS = 6
    cfg.MODEL.BACKBONE.PRE_NORM = False
    cfg.MODEL.BACKBONE.WINDOW_INFERENCE = True

    cfg.MODEL.BACKBONE.HIDDEN_DIM = 256
    cfg.MODEL.BACKBONE.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.BACKBONE.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.BACKBONE.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.BACKBONE.TEST = CN()
    cfg.MODEL.BACKBONE.TEST.SEMANTIC_ON = True
    cfg.MODEL.BACKBONE.TEST.INSTANCE_ON = False
    cfg.MODEL.BACKBONE.TEST.PANOPTIC_ON = False
    cfg.MODEL.BACKBONE.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.BACKBONE.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.BACKBONE.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    cfg.MODEL.BACKBONE.TEST.WINDOW_INFERENCE = True

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.BACKBONE.SIZE_DIVISIBILITY = 32


    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

   

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.BACKBONE.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.BACKBONE.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.BACKBONE.IMPORTANCE_SAMPLE_RATIO = 0.75

    """Add video instance segmentation configuration options."""
    # Input configuration
    cfg.INPUT.REVERSE_AGU = False

    # Mask former test configuration
    cfg.MODEL.BACKBONE.TEST.WINDOW_SIZE = 3
    cfg.MODEL.BACKBONE.TEST.TASK = 'vis'
    cfg.MODEL.BACKBONE.TEST.MAX_NUM = 20

    # Dataset configuration
    cfg.DATASETS.DATASET_RATIO = [1.0]
    cfg.DATASETS.DATASET_NEED_MAP = [False]
    cfg.DATASETS.DATASET_TYPE = ['video_instance']
    cfg.DATASETS.DATASET_TYPE_TEST = ['video_instance']

    # Pseudo data augmentation
    cfg.INPUT.PSEUDO = CN()
    cfg.INPUT.PSEUDO.AUGMENTATIONS = ['rotation']
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768)
    cfg.INPUT.PSEUDO.MAX_SIZE_TRAIN = 768
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN_SAMPLING = "choice_by_clip"
    cfg.INPUT.PSEUDO.CROP = CN()
    cfg.INPUT.PSEUDO.CROP.ENABLED = False
    cfg.INPUT.PSEUDO.CROP.TYPE = "absolute_range"
    cfg.INPUT.PSEUDO.CROP.SIZE = (384, 600)

    # Large-scale jittering augmentation
    cfg.INPUT.LSJ_AUG = CN()
    cfg.INPUT.LSJ_AUG.ENABLED = False
    cfg.INPUT.LSJ_AUG.IMAGE_SIZE = 1024
    cfg.INPUT.LSJ_AUG.MIN_SCALE = 0.1
    cfg.INPUT.LSJ_AUG.MAX_SCALE = 2.0

    # Training configuration
    cfg.SEED = 42
    cfg.DATALOADER.NUM_WORKERS = 4


    cfg.MODEL.BACKBONE.IMG_SIZE = 640
    cfg.MODEL.BACKBONE.PATCH_SIZE = 16
    cfg.MODEL.BACKBONE.NUM_CLASSES = 133
    cfg.MODEL.BACKBONE.NUM_Q = 200
    cfg.MODEL.BACKBONE.MASKED = False
    cfg.MODEL.BACKBONE.MODEL_NAME = "vit_large_patch14_reg4_dinov2"
    cfg.MODEL.BACKBONE.TRACKER_BLOCKS= [22,23]
    cfg.MODEL.BACKBONE.SEGMENTER_BLOCKS = [20,21,22,23]
    cfg.MODEL.BACKBONE.ATTN_MASK_ANNEALING_ENABLED = True
    cfg.MODEL.BACKBONE.START_STEPS = [14782, 29564, 44346, 59128]
    cfg.MODEL.BACKBONE.END_STEPS = [29564, 44346, 59128, 73910]
    cfg.MODEL.BACKBONE.DYNAMIC_EVAL = False

    cfg.MODEL.BACKBONE.REID_BRANCH= True
    cfg.MODEL.BACKBONE.CONTEXT_FILTER_SIZE= 9
    cfg.MODEL.BACKBONE.HIDDEN_DIM= 1024


    cfg.SOLVER.LLRD = 0.8
    cfg.SOLVER.POLY_POWER = 0.9
    cfg.SOLVER.WARMUP_STEPS =[500,1000]
  