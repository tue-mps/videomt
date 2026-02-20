import argparse
import logging
import time
import numpy as np
from collections import Counter
import tqdm
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from fvcore.nn import FlopCountAnalysis, flop_count_table


from videomt import (
    YTVISDatasetMapper,
    CocoClipDatasetMapper,
    PanopticDatasetVideoMapper,
    SemanticDatasetVideoMapper,
    YTVISEvaluator,
    VPSEvaluator,
    VSSEvaluator,
    add_videomt_config,
    build_combined_loader,
    build_detection_train_loader,
    build_detection_test_loader,
)

from detectron2.structures import ImageList

logger = logging.getLogger("vps-analysis")


# -------------------- Setup --------------------
def setup_cfg(args):
    cfg = get_cfg()
    add_videomt_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(["MODEL.WEIGHTS", args.model_weights] + args.opts)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    setup_logger()
    return cfg

def build_test_loader(cfg, dataset_name, dataset_type):
        mapper_dict = {
            'video_instance': YTVISDatasetMapper,
            'video_panoptic': PanopticDatasetVideoMapper,
            'video_semantic': SemanticDatasetVideoMapper,
        }
        if dataset_type not in mapper_dict.keys():
            raise NotImplementedError
        mapper = mapper_dict[dataset_type](cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

# -------------------- Custom Forward --------------------
def forward_custom(model, batched_inputs):
    if 'keep' in batched_inputs[0].keys():
        model.keep = batched_inputs[0]['keep']
    else:
        model.keep = False

    images = []
    for video in batched_inputs:
        for frame in video["image"]:
            images.append(frame.to(model.device))
    images = [(x - model.pixel_mean) / model.pixel_std for x in images]
    images = ImageList.from_tensors(images, model.size_divisibility)
    

    return model , images.tensor

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        if isinstance(out, dict):
            if "pred_logits" in out:
                return out["pred_logits"]  
            elif "pred_masks" in out:
                return out["pred_masks"]   
            else:
                for v in out.values():
                    if torch.is_tensor(v):
                        return v

        if isinstance(out, (list, tuple)):
            return out[0]
        return out

def _sanitize_for_jit(x):
    if isinstance(x, range):
        return list(x)
    if isinstance(x, dict):
        return {k: _sanitize_for_jit(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_sanitize_for_jit(v) for v in x)
    return x


def measure_flops(model, data_loader):
    
    total_flops_list = []
    wrapped_model = ModelWrapper(model)

    for idx, batch in zip(tqdm.trange(100), data_loader):
        batch_clean = _sanitize_for_jit(batch)

        with torch.no_grad():
                flops_encoder = FlopCountAnalysis(wrapped_model, batch_clean)
                flops_encoder.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
                total_flops_list.append(flops_encoder.total()/len(batch[0]['file_names']))
                
    print("\n" + "-" * 50)
    print(
        "Total GFlops: {:.2f} ± {:.2f}".format(
            np.mean(total_flops_list) / 1e9, np.std(total_flops_list) / 1e9
        )
    )
    print("=" * 50)
    # Parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")



# -------------------- FPS Online --------------------
def measure_fps_online(model, data_loader, warmup_iters):
  
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_time = 0
    repetitions = 0

    print(f"Running FPS test (warmup={warmup_iters} iters)...")
    from torch.amp.autocast_mode import autocast
    
    with autocast(dtype=torch.float16, device_type="cuda"):
        for idx, batch in enumerate(data_loader):
            with torch.no_grad():
                model, images=forward_custom(model,batch)
                if idx < warmup_iters:
                    i = 0 
                    for image in images:
                        image = image.unsqueeze(0)
                        if i != 0 or model.keep:
                            out = model.backbone(image,resume=True)
                        else:
                            out= model.backbone(image)
                        i += 1
                    print(f"[Warmup {idx+1}/{warmup_iters}]")
                    continue
                i = 0
                for image in images:
                    image = image.unsqueeze(0)
                    torch.cuda.synchronize()
                    start.record()
                    if i != 0 or model.keep:
                            out = model.backbone(image,resume=True)
                    else:
                            out = model.backbone(image)
                    end.record()
                    torch.cuda.synchronize()
                    elapsed = start.elapsed_time(end)
                    i += 1
                    

                    total_time += elapsed
                    repetitions += 1

                print(f"[Batch {idx}] Proccesed Frames online: {repetitions}, Time: {total_time:.2f}s, "
                    f"Average FPS online: {1000/(total_time/repetitions):.2f}")

# -------------------- Main --------------------
def main(args):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.suppress_errors = True
    torch.set_float32_matmul_precision("medium")
    cfg = setup_cfg(args)
    model = build_model(cfg)
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.window_inference = True
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    data_loader = build_test_loader(cfg,cfg.DATASETS.TEST[0],cfg.DATASETS.DATASET_TYPE_TEST[0])

    if args.task == "fps":
        model.backbone = torch.compile(model.backbone, fullgraph=False, dynamic=True)
        measure_fps_online(model, data_loader, warmup_iters=args.warmup_iters)
        
    elif args.task == "flops":
        measure_flops(model, data_loader)

    else:
        raise ValueError("Unknown task. Use --task flop or --task fps")


# -------------------- Args --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VPS Model FLOPs or FPS")
    parser.add_argument("--task", choices=["flops", "fps"], required=True, help="Task to run")
    parser.add_argument("--config-file", required=True, help="Path to config file")
    parser.add_argument("--model-weights", required=True, help="Path to model checkpoint")
    parser.add_argument("--warmup-iters", type=int, default=200, help="Warmup iterations for FPS")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[], help="Additional config options")
    args = parser.parse_args()
    main(args)