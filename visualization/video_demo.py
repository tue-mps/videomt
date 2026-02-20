# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py

import torch
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import time
import tqdm

# from torch.cuda.amp import autocast

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from videomt import  add_videomt_config
from predictor import VisualizationDemo, VisualizationDemo_windows


def setup_cfg(args):
	# load config from file and command-line arguments
	cfg = get_cfg()
	add_deeplab_config(cfg)
	add_videomt_config(cfg)
	cfg.merge_from_file(args.config_file)
	cfg.merge_from_list(args.opts)
	cfg.freeze()
	return cfg

def get_parser():
	parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
	parser.add_argument(
		"--config-file",
		default="configs/youtubevis_2019/video_maskformer2_R50_bs32_8ep_frame.yaml",
		metavar="FILE",
		help="path to config file",
	)
	parser.add_argument(
		"--input",
		help="directory of input video frames",
		required=True,
	)
	parser.add_argument(
		"--output",
		help="directory to save output frames",
		required=True,
	)
	parser.add_argument(
		"--confidence_threshold",
		type=float,
		default=0.5,
		help="Minimum score for instance predictions to be shown",
	)
	parser.add_argument(
		"--windows_size",
		type=int,
		default=20,
		help="Windows size for semi-offline mode",
	)
	parser.add_argument(
		"--opts",
		help="Modify config options using the command-line 'KEY VALUE' pairs",
		default=[],
		nargs=argparse.REMAINDER,
	)
	return parser

if __name__ == "__main__":
	mp.set_start_method("spawn", force=True)
	args = get_parser().parse_args()
	setup_logger(name="fvcore")
	logger = setup_logger()
	logger.info("Arguments: " + str(args))

	cfg = setup_cfg(args)

	demo = VisualizationDemo_windows(cfg)

	assert args.input and args.output

	video_root = args.input
	output_root = args.output
	score_threshold = args.confidence_threshold
	windows_size = args.windows_size


	os.makedirs(output_root, exist_ok=True)
	
	frames_path = video_root
	frames_path = glob.glob(os.path.expanduser(os.path.join(frames_path, '*.???')))
	frames_path.sort()
	if windows_size == -1:
		windows_size = len(frames_path)
	start_time = time.time()
	vid_frames = []
	_frames_path = []
	instances = set()
	for i, path in enumerate(tqdm.tqdm(frames_path)):
		img = read_image(path, format="BGR")
		_frames_path.append(path)
		vid_frames.append(img)
		if len(vid_frames) == windows_size or i == len(frames_path) - 1:
			# do inference
			with torch.amp.autocast(device_type="cuda"):
				if i < windows_size:
					predictions, visualized_output = demo.run_on_video(vid_frames, keep=False)
				else:
					predictions, visualized_output = demo.run_on_video(vid_frames, keep=True)
			# do save
			for path, _vis_output in zip(_frames_path, visualized_output):
				out_filename = os.path.join(output_root, os.path.basename(path))
				_vis_output.save(out_filename)
			if 'pred_ids' in predictions.keys():
				for id in predictions['pred_ids']:
					instances.add(id)
			del visualized_output, vid_frames, _frames_path, predictions

			vid_frames = []
			_frames_path = []

	logger.info(
		"detected {} instances per frame in {:.2f}s".format(
			len(set(instances)), time.time() - start_time
		)
	)

