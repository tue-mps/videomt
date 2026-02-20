# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# This file are adapted from:
# - the EoMT repository https://github.com/tue-mps/eomt/tree/master/models/vit.py
# Used under the MIT License.
# ---------------------------------------------------------------
import torch.nn as nn
import timm


class ViT(nn.Module):
    def __init__(self, img_size: int, name, patch_size=16):
        super().__init__()

        self.backbone = timm.create_model(
            name,
            pretrained=True,
            img_size=img_size,
            patch_size=patch_size,
            num_classes=0,
            dynamic_img_size=True,
        )
