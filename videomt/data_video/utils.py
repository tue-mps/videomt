from detectron2.structures.masks import BitMasks
from detectron2.structures import Instances,Boxes
import torch
import numpy as np
from typing import Any, Iterator, List, Union,Tuple

from detectron2.utils.env import TORCH_VERSION

if TORCH_VERSION < (1, 8):
    _maybe_jit_unused = torch.jit.unused
else:

    def _maybe_jit_unused(x):
        return x

class Video_BitMasks(BitMasks):
    """
    This class stores the segmentation masks for all objects in one video, in
    the form of bitmaps.

    Attributes:
        tensor: bool Tensor of N,T,H,W, representing N instances in the video.
    """
    def __init__(self, tensor: Union[torch.Tensor, np.ndarray]):
     """
     Args:
         tensor: bool Tensor of N,T,H,W, representing N instances in the video.
     """
     device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
     tensor = torch.as_tensor(tensor, dtype=torch.bool, device=device)
     assert tensor.dim() == 4, tensor.size()
     self.image_size = tensor.shape[2:]
     self.tensor = tensor

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "BitMasks":
        """
        Returns:
            BitMasks: Create a new :class:`BitMasks` by indexing.

        The following usage are allowed:

        1. `new_masks = masks[3]`: return a `BitMasks` which contains only one mask.
        2. `new_masks = masks[2:10]`: return a slice of masks.
        3. `new_masks = masks[vector]`, where vector is a torch.BoolTensor
           with `length = len(masks)`. Nonzero elements in the vector will be selected.

        Note that the returned object might share storage with this object,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return BitMasks(self.tensor[item].view(1, -1))
        m = self.tensor[item]
        assert m.dim() == 4, "Indexing on BitMasks with {} returns a tensor with shape {}!".format(
            item, m.shape
        )
        return BitMasks(m)

    def nonempty(self) -> torch.Tensor:
        """
        Find masks that are non-empty.

        Returns:
            Tensor: a BoolTensor which represents
                whether each mask is empty (False) or non-empty (True).
        """
        N,T,H,W = self.tensor.size()
        return self.tensor.reshape(N*T,H*W).any(dim=1).reshape(N,T)


class Video_Boxes(Boxes):
    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a NxTx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
#        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
#            tensor = tensor.reshape((-1, 5)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 3 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor

    def clone(self) -> "Video_Boxes":
        """
        Clone the Boxes.
        Returns:
            Boxes
        """
        return Video_Boxes(self.tensor.clone())


    @_maybe_jit_unused
    def to(self, device: torch.device):
        # Boxes are assumed float32 and does not support to(dtype)
        return Video_Boxes(self.tensor.to(device=device))


    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.
        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        area = (box[:,:, 2] - box[:, :,0]) * (box[:, :,3] - box[:,:, 1])
        return area

    def clip(self, box_size: Tuple[int, int]) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].
        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        h, w = box_size
        x1 = self.tensor[:,:, 0].clamp(min=0, max=w)
        y1 = self.tensor[:,:, 1].clamp(min=0, max=h)
        x2 = self.tensor[:, :,2].clamp(min=0, max=w)
        y2 = self.tensor[:, :,3].clamp(min=0, max=h)
        self.tensor = torch.stack((x1, y1, x2, y2), dim=-1)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.
        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:,:, 2] - box[:, :,0]
        heights = box[:,:, 3] - box[:,:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep



    def __getitem__(self, item) -> "Video_Boxes":
        """
        Args:
            item: int, slice, or a BoolTensor
        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.
        The following usage are allowed:
        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.
        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Video_Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 3, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Video_Boxes(b)


    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        return (self.tensor[:,:, :2] + self.tensor[:, :,2:]) / 2
    def frame_num(self):
        return self.tensor.size(1)

    @classmethod
    @_maybe_jit_unused
    def cat(cls, boxes_list: List["Video_Boxes"]) -> "Video_Boxes":
        """
        Concatenates a list of Boxes into a single Boxes
        Arguments:
            boxes_list (list[Boxes])
        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, Video_Boxes) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes



