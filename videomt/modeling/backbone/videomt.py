# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from:
# - the EoMT repository https://github.com/tue-mps/eomt/tree/master/models
# Used under the MIT License.
# ---------------------------------------------------------------
import logging
import math
import einops
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import Backbone, BACKBONE_REGISTRY

from .vit import ViT
from .scale_block import ScaleBlock


class VidEoMT_CLASS(nn.Module):
    def __init__(
        self,
        img_size,
        num_classes,
        name,
        num_frames,
        num_q: int = 200,
        segmenter_blocks: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.encoder = ViT(img_size, name)
        self.encoder.backbone.patch_embed.strict_img_size = False
        self.encoder.backbone.patch_embed.dynamic_img_pad = False
        self.encoder.backbone.dynamic_img_size = True
        self.num_q = num_q
        self.segmenter_blocks = segmenter_blocks
        self.num_blocks = len(self.segmenter_blocks)
        self.mask_classification = True
        self.num_frames = num_frames
        self.embed_dim = self.encoder.backbone.embed_dim
        self.last_query_embed = None

        self.register_buffer("attn_mask_probs", torch.ones(self.num_blocks))
        self.q = nn.Embedding(num_q, self.encoder.backbone.embed_dim)
        self.query_updater = nn.Linear(self.embed_dim, self.embed_dim)
        self.class_head = nn.Linear(self.encoder.backbone.embed_dim, num_classes + 1)

        self.mask_head = nn.Sequential(
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
        )

        patch_size = self.encoder.backbone.patch_embed.patch_size
        max_patch_size = max(patch_size[0], patch_size[1])
        num_upscale = max(1, int(math.log2(max_patch_size)) - 2)

        self.upscale = nn.Sequential(
            *[ScaleBlock(self.encoder.backbone.embed_dim) for _ in range(num_upscale)],
        )

  

    def _predict(self, x: torch.Tensor,H_g,W_g):
        q = x[:, : self.num_q, :]

        class_logits = self.class_head(q)

        x = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, H_g,W_g
        )

        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x)
        )

        return mask_logits, class_logits


    def _attn(self, module: nn.Module, x: torch.Tensor, mask: Optional[torch.Tensor]):
        B, N, C = x.shape

        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)

        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)

        dropout_p = module.attn_drop.p if self.training else 0.0

        if module.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, mask, dropout_p)
        else:
            attn = (q @ k.transpose(-2, -1)) * module.scale
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = module.attn_drop(attn)
            x = attn @ v

        x = module.proj_drop(module.proj(x.transpose(1, 2).reshape(B, N, C)))
        return x

    def _clear_memory(self):
        del self.last_query_embed
        self.last_query_embed = None
        return

    def backbone_patch_token(self, x, blocks):
        for block in blocks:
            x = x + block.drop_path1(
                block.ls1(self._attn(block.attn, block.norm1(x), None))
            )
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        return x
    
    def segmenter(self, x, blocks, H_g, W_g, resume: bool = False):

        attn_mask = None
        mask_logits_per_frame, class_logits_per_frame = [], []
        mask_logits_per_frame_list , class_logits_per_frame_list = [], []
        bt = x.shape[0]
        bs = bt // self.num_frames if self.training else 1
        t = bt // bs
        x = einops.rearrange(x, '(b t) n c -> b t n c', t=t)
        attn_mask = None
        x_outputs = []

        for i in range(t):
            mask_logits_per_layer, class_logits_per_layer = [], []
            if i == 0 and resume is False:
                self._clear_memory()
                for j, block in enumerate(blocks):
                    if j == 0:
                        x_frame = torch.cat(
                            (self.q.weight[None, :, :], x[:,i,:,:]), dim=1
                                )
                    if self.training:
                        mask_logits, class_logits = self._predict(x_frame, H_g, W_g)
                        mask_logits_per_layer.append(mask_logits)
                        class_logits_per_layer.append(class_logits)
                
                        
                    x_frame = x_frame + block.drop_path1(
                        block.ls1(self._attn(block.attn, block.norm1(x_frame), attn_mask))
                    )
                    x_frame = x_frame + block.drop_path2(block.ls2(block.mlp(block.norm2(x_frame))))

                x_outputs.append(x_frame)
                propag_query = self.query_updater(x_frame[:, : self.num_q, :])
                # Weighted combination
                self.last_query_embed = propag_query + self.q.weight[None, :, :]
            else:
                for j, block in enumerate(blocks):
                    if j == 0:
                        x_frame = torch.cat(
                            (self.last_query_embed, x[:,i,:,:]), dim=1
                                )
                    if self.training:
                        mask_logits, class_logits = self._predict(x_frame, H_g, W_g)
                        mask_logits_per_layer.append(mask_logits)
                        class_logits_per_layer.append(class_logits)
                    
                    x_frame = x_frame + block.drop_path1(
                        block.ls1(self._attn(block.attn, block.norm1(x_frame), attn_mask))
                    )
                    x_frame = x_frame + block.drop_path2(block.ls2(block.mlp(block.norm2(x_frame))))
                    
                x_outputs.append(x_frame)
                propag_query = self.query_updater(x_frame[:, : self.num_q, :])
                # Weighted combination
                self.last_query_embed = propag_query + self.q.weight[None, :, :]


            if self.training:
                mask_logits_per_layer = torch.stack(mask_logits_per_layer, dim=0)  # (layers, b, q, h, w)
                class_logits_per_layer = torch.stack(class_logits_per_layer, dim=0)  # (layers, b, q, class) # (layers, b, t, q, c)
                
                mask_logits_per_frame.append(mask_logits_per_layer) 
                class_logits_per_frame.append(class_logits_per_layer)
                
        
        if self.training:
            mask_logits_per_frame = torch.stack(mask_logits_per_frame, dim=1)  # (L, t, b, q, h, w)
            class_logits_per_frame = torch.stack(class_logits_per_frame, dim=1)  # (L, t, b, q, class)
            

            
            for i in range(len(mask_logits_per_frame)):
                mask_logits_per_frame_list.append(mask_logits_per_frame[i].permute(1, 2, 0, 3, 4)) # from (L, t, b, q, h, w) to (b q t h w)
            

            for i in range(len(class_logits_per_frame)):
                class_logits_per_frame_list.append(class_logits_per_frame[i].permute(1, 0, 2, 3)) # from (L, t, b, q, class) to (b t q c)
        

        x_outputs = torch.stack(x_outputs, dim=0)  # (t, b, N, c) to (t*b, N, c)
        x_outputs = x_outputs.reshape(t * bs, x_outputs.shape[2], x_outputs.shape[3])
        mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x_outputs), H_g, W_g)
        mask_logits = einops.rearrange(mask_logits, '(b t) q h w -> b q t h w', t=t)
        class_logits = einops.rearrange(class_logits, '(b t) q c -> b t q c', t=t)

        out = {
            'pred_logits': class_logits,
            'pred_masks': mask_logits,
            'aux_outputs': self._set_aux_loss(
                class_logits_per_frame_list if self.mask_classification else None,
                mask_logits_per_frame_list,
            ),
        }

        return out
    

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class, outputs_seg_masks)
            ]
        else:
            
            return [{"pred_masks": b} for b in outputs_seg_masks]

  
    def forward(self, x: torch.Tensor,resume=False):
        x = self.encoder.backbone.patch_embed(x)
        _,H_g,W_g,_= x.shape
        x = self.encoder.backbone._pos_embed(x)
        x = self.encoder.backbone.patch_drop(x)
        x = self.encoder.backbone.norm_pre(x)

        # backbone blocks for patch tokens
        x = self.backbone_patch_token(x, self.encoder.backbone.blocks[0:self.segmenter_blocks[0]])
        
        # segmenter blocks for genarating quries and feature masks
        start_idx = self.segmenter_blocks[0]
        end_idx = self.segmenter_blocks[-1] + 1 
        out = self.segmenter(x, self.encoder.backbone.blocks[start_idx:end_idx], H_g, W_g,
                           resume=resume)
        return out


@BACKBONE_REGISTRY.register()
class VidEoMT(VidEoMT_CLASS, Backbone):
    def __init__(self, cfg, input_shape):
        self.img_size=cfg.MODEL.BACKBONE.IMG_SIZE
        self.num_q = cfg.MODEL.BACKBONE.NUM_OBJECT_QUERIES
        self.num_classes=cfg.MODEL.BACKBONE.NUM_CLASSES
        self.name=cfg.MODEL.BACKBONE.MODEL_NAME
        self.segmenter_blocks=cfg.MODEL.BACKBONE.SEGMENTER_BLOCKS 
        self.num_frames = cfg.INPUT.SAMPLING_FRAME_NUM  
        
        
        super().__init__(
            img_size=self.img_size,
            num_classes=self.num_classes,
            name=self.name,
            segmenter_blocks=self.segmenter_blocks,
            num_q = self.num_q,
            num_frames= self.num_frames,
            
        )

       