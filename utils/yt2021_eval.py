# -------------------------------------------------------------------
# Copied from DVIS++ code, https://github.com/zhang-tao-whu/DVIS_Plus/tree/main
# Used under MIT License
# -------------------------------------------------------------------
import subprocess
import sys
import os
import numpy as np

from ytvis_api.ytvos import YTVOS
from ytvis_api.ytvoseval import YTVOSeval


def _summarize(self, ap=1, iouThr=None, areaRng='all', maxDets=100):
    """Summarize evaluation results."""
    p = self.params
    iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
    titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
    typeStr = '(AP)' if ap == 1 else '(AR)'
    iouStr = ('{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1])
              if iouThr is None else '{:0.2f}'.format(iouThr))

    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

    if ap == 1:
        s = self.eval['precision']
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, :, aind, mind]
    else:
        s = self.eval['recall']
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, aind, mind]

    mean_s = -1 if len(s[s > -1]) == 0 else np.mean(s[s > -1])
    print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
    return mean_s


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_folder_with_results.json>")
        sys.exit(1)

    submit_dir = sys.argv[1]
    output_dir = submit_dir

    # Ground truth file path (update with actual path)
    truth_file = "/p/scratch/objectsegvideo/narges/video_datasets/ytvis_2021/validation_gt.json"

    submit_file = os.path.join(submit_dir, 'results.json')
    if not os.path.isfile(submit_file):
        print(f"ERROR: {submit_file} doesn't exist")
        sys.exit(1)

    if not os.path.isfile(truth_file):
        print(f"ERROR: {truth_file} doesn't exist")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Evaluation
    gts = YTVOS(truth_file)
    res = gts.loadRes(submit_file)
    ytvosEval = YTVOSeval(gts, res, 'segm')
    ytvosEval.evaluate()
ytvosEval.accumulate()
ytvosEval.summarize()

# ---- Save results ----
output_filename = os.path.join(output_dir, 'scores.txt')
with open(output_filename, 'w') as output_file:
    output_file.write('mAP: {}\n'.format(_summarize(ytvosEval, 1)))
    output_file.write('AP50: {}\n'.format(_summarize(ytvosEval, 1, iouThr=.5, maxDets=ytvosEval.params.maxDets[2])))
    output_file.write('AP75: {}\n'.format(_summarize(ytvosEval, 1, iouThr=.75, maxDets=ytvosEval.params.maxDets[2])))
    output_file.write('AR1: {}\n'.format(_summarize(ytvosEval, 0, maxDets=ytvosEval.params.maxDets[0])))
    output_file.write('AR10: {}\n'.format(_summarize(ytvosEval, 0, maxDets=ytvosEval.params.maxDets[1])))

print(f"Saved evaluation results to {output_filename}")
