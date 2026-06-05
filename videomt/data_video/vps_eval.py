import itertools
import json
import logging
import numpy as np
import os
import torch


import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager


from PIL import Image
from panopticapi.utils import rgb2id
from panopticapi.utils import IdGenerator


class VPSEvaluator(DatasetEvaluator):
    """
    Only for save the prediction results in VIPSeg format
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
        thing_dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
        stuff_dataset_id_to_contiguous_id = self._metadata.stuff_dataset_id_to_contiguous_id
        self.contiguous_id_to_thing_dataset_id = {}
        self.contiguous_id_to_stuff_dataset_id = {}
        for i, key in enumerate(thing_dataset_id_to_contiguous_id.values()):
            self.contiguous_id_to_thing_dataset_id.update({i: key})
        for i, key in enumerate(stuff_dataset_id_to_contiguous_id.values()):
            self.contiguous_id_to_stuff_dataset_id.update({i: key})
        json_file = PathManager.get_local_path(self._metadata.panoptic_json)

        self._do_evaluation = False

    def reset(self):
        self._predictions = []
        PathManager.mkdirs(self._output_dir)
        if not os.path.exists(os.path.join(self._output_dir, 'pan_pred')):
            os.makedirs(os.path.join(self._output_dir, 'pan_pred'), exist_ok=True)

    def process(self, inputs, outputs):
        """
        save panoptic segmentation result as an image
        """
        assert len(inputs) == 1, "More than one inputs are loaded for inference!"
        color_generator = IdGenerator(self._metadata.categories)

        video_id = inputs[0]["video_id"]
        image_names = [inputs[0]['file_names'][idx] for idx in inputs[0]["frame_idx"]]
        img_shape = outputs['image_size']
        pan_seg_result = outputs['pred_masks']
        segments_infos = outputs['segments_infos']
        segments_infos_ = []

        pan_format = np.zeros((pan_seg_result.shape[0], img_shape[0], img_shape[1], 3), dtype=np.uint8)
        for segments_info in segments_infos:
            id = segments_info['id']
            is_thing = segments_info['isthing']
            sem = segments_info['category_id']
            if is_thing:
                sem = self.contiguous_id_to_thing_dataset_id[sem]
            else:
                sem = self.contiguous_id_to_stuff_dataset_id[sem - len(self.contiguous_id_to_thing_dataset_id)]

            mask = pan_seg_result == id
            color = color_generator.get_color(sem)
            pan_format[mask] = color

            dts = []
            dt_ = {"category_id": int(sem), "iscrowd": 0, "id": int(rgb2id(color))}
            for i in range(pan_format.shape[0]):
                area = mask[i].sum()
                index = np.where(mask[i].numpy())
                if len(index[0]) == 0:
                    dts.append(None)
                else:
                    if area == 0:
                        dts.append(None)
                    else:
                        x = index[1].min()
                        y = index[0].min()
                        width = index[1].max() - x
                        height = index[0].max() - y
                        dt = {"bbox": [x.item(), y.item(), width.item(), height.item()], "area": int(area)}
                        dt.update(dt_)
                        dts.append(dt)
            segments_infos_.append(dts)
        #### save image
        annotations = []
        for i, image_name in enumerate(image_names):
            image_ = Image.fromarray(pan_format[i])
            if not os.path.exists(os.path.join(self._output_dir, 'pan_pred', video_id)):
                os.makedirs(os.path.join(self._output_dir, 'pan_pred', video_id))
            image_.save(os.path.join(self._output_dir, 'pan_pred', video_id, image_name.split('/')[-1].split('.')[0] + '.png'))
            annotations.append({"segments_info": [item[i] for item in segments_infos_ if item[i] is not None], "file_name": image_name.split('/')[-1]})
        self._predictions.append({'annotations': annotations, 'video_id': video_id})

    def evaluate(self):
        """
        save jsons
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            file_path = os.path.join(self._output_dir, 'pred.json')
            with open(file_path, 'w') as f:
                json.dump({'annotations': predictions}, f)
        return {}