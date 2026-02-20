# -------------------------------------------------------------------
# Copied from DVIS++ code, https://github.com/zhang-tao-whu/DVIS_Plus/tree/main
# Used under MIT License.
# -------------------------------------------------------------------
import logging
import numpy as np
import os
import torch

from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager

from PIL import Image


class VSSEvaluator(DatasetEvaluator):
    """
    Only for save the prediction results in VSPW format
    """

    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        use_fast_impl=True,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm", "keypoints".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instances_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl

        if tasks is not None and isinstance(tasks, CfgNode):
            self._logger.warning(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it from predictions should be better
        else:
            self._tasks = tasks

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)
        self.ignore_val = self._metadata.ignore_label
        dataset_id_to_contiguous_id = self._metadata.stuff_dataset_id_to_contiguous_id
        self.contiguous_id_to_dataset_id = {}
        for i, key in enumerate(dataset_id_to_contiguous_id.keys()):
            self.contiguous_id_to_dataset_id.update({i: key})

        self._do_evaluation = False

    def reset(self):
        self._predictions = []
        PathManager.mkdirs(self._output_dir)

    def process(self, inputs, outputs):
        """
         save semantic segmentation result as an image
        """
        assert len(inputs) == 1, "More than one inputs are loaded for inference!"

        video_id = inputs[0]["video_id"]
        image_names = [inputs[0]['file_names'][idx] for idx in inputs[0]["frame_idx"]]
        img_shape = outputs['image_size']
        sem_seg_result = outputs['pred_masks'].numpy().astype(np.uint8)  # (t, h, w, 3)
        sem_seg_result_ = np.zeros_like(sem_seg_result, dtype=np.uint8) + 255
        unique_cls = np.unique(sem_seg_result)
        for cls in unique_cls:
            if cls == self.ignore_val:
                continue
            cls_ = self.contiguous_id_to_dataset_id[cls]
            sem_seg_result_[sem_seg_result == cls] = cls_
        sem_seg_result = sem_seg_result_
        for i, image_name in enumerate(image_names):
            image_ = Image.fromarray(sem_seg_result[i])
            if not os.path.exists(os.path.join(self._output_dir, video_id)):
                os.makedirs(os.path.join(self._output_dir, video_id))
            image_.save(os.path.join(self._output_dir, video_id, image_name.split('/')[-1].split('.')[0] + '.png'))
        return

    def evaluate(self):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        return {}