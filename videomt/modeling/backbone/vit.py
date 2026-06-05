# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# This file are adapted from:
# - the EoMT repository https://github.com/tue-mps/eomt/tree/master/models/vit.py
# Used under the MIT License.
# ---------------------------------------------------------------
import logging
import torch.nn as nn
import timm
import os 
import torch
from transformers import AutoModel

class ViT(nn.Module):
    def __init__(self, img_size: int, name, patch_size=16):
        super().__init__()

        if self._offline_env_enabled():
            self._force_timm_hf_local_only()

        # DINOv3 (loaded via huggingface transformers); detected by "v3" in the
        # model name. DINOv2 / other timm models use the original timm path.
        if "v3" in name:
            self.backbone = AutoModel.from_pretrained(
                name,
                **self._get_transformers_offline_kwargs(),
            )
            img_size_tuple = (img_size, img_size) if isinstance(img_size, int) else img_size
            self._transformers_to_timm(img_size_tuple)
            self.backbone.is_v3 = True
        else:
            self.backbone = timm.create_model(
                name,
                pretrained=True,
                img_size=img_size,
                patch_size=patch_size,
                num_classes=0,
                dynamic_img_size=True,
            )
            self.backbone.is_v3 = False

    def _transformers_to_timm(self, img_size):
        """Patch a HuggingFace DINOv3 backbone so it exposes the same attribute
        names as a timm ViT (`patch_embed`, `blocks`, `embed_dim`, etc.)."""
        patch_size = self.backbone.embeddings.config.patch_size
        grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)

        self.backbone.embed_dim = self.backbone.embeddings.config.hidden_size
        self.backbone.patch_embed = self.backbone.embeddings
        self.backbone.patch_embed.patch_size = (patch_size, patch_size)
        self.backbone.num_prefix_tokens = (
            self.backbone.patch_embed.config.num_register_tokens + 1
        )
        self.backbone.patch_embed.grid_size = grid_size

        self.backbone.blocks = self.backbone.layer

        del self.backbone.embeddings, self.backbone.patch_embed.mask_token, self.backbone.layer

    @staticmethod
    def _offline_env_enabled() -> bool:
        return bool(os.getenv("HF_HUB_OFFLINE") or os.getenv("TRANSFORMERS_OFFLINE"))

    @staticmethod
    def _force_timm_hf_local_only() -> None:
        if getattr(timm, "_hf_hub_local_only_patched", False):
            return
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return

        try:
            import timm.models._hub as timm_hub
        except Exception:
            return

        def _hf_hub_download_local_only(*args, **kwargs):
            kwargs.setdefault("local_files_only", True)
            # Always set cache_dir from HF_HOME when offline mode is enabled
            # This ensures consistency with where weights were downloaded
            hf_home = os.getenv("HF_HOME")
            if hf_home:
                # Always override cache_dir with HF_HOME to ensure consistency
                original_cache_dir = kwargs.get("cache_dir")
                kwargs["cache_dir"] = hf_home
                # Verify cache directory exists
                if not os.path.exists(hf_home):
                    error_msg = f"HF_HOME cache directory does not exist: {hf_home}. Please ensure weights are downloaded first."
                    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                        logging.error(error_msg)
                    raise FileNotFoundError(error_msg)
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    model_id = args[0] if args else kwargs.get("repo_id", "unknown")
                    filename = args[1] if len(args) > 1 else kwargs.get("filename", "unknown")
                    logging.info(f"Loading from HF_HOME cache_dir: {hf_home} (overriding {original_cache_dir}) for {model_id}/{filename}")
            else:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    logging.warning("HF_HOME not set, using default cache location")
            try:
                return hf_hub_download(*args, **kwargs)
            except Exception as e:
                # Provide helpful error message if files not found
                if "LocalEntryNotFoundError" in str(type(e).__name__) or "Cannot find" in str(e):
                    model_id = args[0] if args else kwargs.get("repo_id", "unknown")
                    filename = args[1] if len(args) > 1 else kwargs.get("filename", "unknown")
                    cache_dir_used = kwargs.get("cache_dir", "default")
                    error_msg = (
                        f"Model weights not found in cache. "
                        f"Model: {model_id}, File: {filename}, Cache: {cache_dir_used}. "
                        f"Please run scripts/download_weights.py with HF_HOME={cache_dir_used} set."
                    )
                    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                        logging.error(error_msg)
                    raise FileNotFoundError(error_msg) from e
                raise

        timm_hub.hf_hub_download = _hf_hub_download_local_only
        timm._hf_hub_local_only_patched = True

    @staticmethod
    def _get_transformers_offline_kwargs() -> dict:
        """Get kwargs for transformers AutoModel.from_pretrained() when offline mode is enabled."""
        kwargs = {}
        if ViT._offline_env_enabled():
            kwargs["local_files_only"] = True
            # Set cache_dir from HF_HOME if available
            hf_home = os.getenv("HF_HOME")
            if hf_home:
                kwargs["cache_dir"] = hf_home
        return kwargs