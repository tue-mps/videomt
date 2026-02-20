import numpy as np
import os
from PIL import Image
import sys
from tqdm import tqdm


class Evaluator(object):
    """Evaluator for semantic segmentation metrics."""

    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def beforeval(self):
        isval = np.sum(self.confusion_matrix, axis=1) > 0
        self.confusion_matrix = self.confusion_matrix * isval

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix)
        )
        isval = np.sum(self.confusion_matrix, axis=1) > 0
        MIoU = np.nansum(MIoU * isval) / isval.sum()
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix)
        )
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        # VSPW label preprocessing
        gt_image[gt_image == 0] = 255
        gt_image = gt_image - 1

        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


if __name__ == "__main__":
    eval_ = Evaluator(124)
    eval_.reset()

    DIR = sys.argv[1]
    split = 'val.txt'

    with open(os.path.join(DIR, split), 'r') as f:
        lines = f.readlines()
        videolist = [line[:-1] for line in lines]

    PRED = sys.argv[2]
    for video in tqdm(videolist):
        for tar in os.listdir(os.path.join(DIR, 'data', video, 'mask')):
            pred = os.path.join(PRED, video, tar)
            tar_ = Image.open(os.path.join(DIR, 'data', video, 'mask', tar))
            tar_ = np.array(tar_)
            tar_ = tar_[np.newaxis, :]
            pred_ = Image.open(pred)
            pred_ = np.array(pred_)
            pred_ = pred_[np.newaxis, :]
            eval_.add_batch(tar_, pred_)

    Acc = eval_.Pixel_Accuracy()
    Acc_class = eval_.Pixel_Accuracy_Class()
    mIoU = eval_.Mean_Intersection_over_Union()
    FWIoU = eval_.Frequency_Weighted_Intersection_over_Union()
    print(f"Acc: {Acc}, Acc_class: {Acc_class}, mIoU: {mIoU}, FWIoU: {FWIoU}")