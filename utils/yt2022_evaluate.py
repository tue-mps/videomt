# -------------------------------------------------------------------
# Copied from DVIS++ code, https://github.com/zhang-tao-whu/DVIS_Plus/tree/main
# Used under MIT License
# -------------------------------------------------------------------
import subprocess
import sys

def install(package):
    subprocess.check_call("pip install " + package, shell=True)



import sys
import os
import numpy as np
from ytvis_api.ytvos import YTVOS
from ytvis_api.ytvoseval import YTVOSeval


def _summarize(self, ap=1, iouThr=None, areaRng='all', maxDets=100):
  p = self.params
  iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
  titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
  typeStr = '(AP)' if ap == 1 else '(AR)'
  iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
      if iouThr is None else '{:0.2f}'.format(iouThr)

  aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
  mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
  if ap == 1:
    # dimension of precision: [TxRxKxAxM]
    s = self.eval['precision']
    # IoU
    if iouThr is not None:
      t = np.where(iouThr == p.iouThrs)[0]
      s = s[t]
    s = s[:, :, :, aind, mind]
  else:
    # dimension of recall: [TxKxAxM]
    s = self.eval['recall']
    if iouThr is not None:
      t = np.where(iouThr == p.iouThrs)[0]
      s = s[t]
    s = s[:, :, aind, mind]
  if len(s[s > -1]) == 0:
    mean_s = -1
  else:
    mean_s = np.mean(s[s > -1])
  print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
  return mean_s


input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = output_dir 
truth_dir = input_dir

if not os.path.isdir(submit_dir):
  print("%s doesn't exist" % submit_dir)

submit_file = os.path.join(submit_dir, 'results.json')
truth_file = os.path.join(truth_dir, 'gt.json')
truth_short_file = os.path.join(truth_dir, 'gt_short.json')
truth_long_file = os.path.join(truth_dir, 'gt_long.json')

if not os.path.isfile(submit_file):
  print("%s doesn't exist" % submit_file)

gts = YTVOS(truth_file)
res = gts.loadRes(submit_file)

gts_short = YTVOS(truth_short_file)
ytvosEval_short = YTVOSeval(gts_short, res, 'segm')
ytvosEval_short.evaluate()
ytvosEval_short.accumulate()
ytvosEval_short.summarize()

gts_long = YTVOS(truth_long_file)
ytvosEval_long = YTVOSeval(gts_long, res, 'segm')
ytvosEval_long.evaluate()
ytvosEval_long.accumulate()
ytvosEval_long.summarize()

output_filename = os.path.join(output_dir, 'scores.txt')
output_file = open(output_filename, 'w')
output_file.write('mAP: {}\n'.format((_summarize(ytvosEval_short, 1)+_summarize(ytvosEval_long, 1))/2))
output_file.write('mAP_S: {}\n'.format(_summarize(ytvosEval_short, 1)))
output_file.write('AP50_S: {}\n'.format(
    _summarize(ytvosEval_short, 1, iouThr=.5, maxDets=ytvosEval_short.params.maxDets[2])))
output_file.write('AP75_S: {}\n'.format(
    _summarize(ytvosEval_short, 1, iouThr=.75, maxDets=ytvosEval_short.params.maxDets[2])))
output_file.write('AR1_S: {}\n'.format(
    _summarize(ytvosEval_short, 0, maxDets=ytvosEval_short.params.maxDets[0])))
output_file.write('AR10_S: {}\n'.format(
    _summarize(ytvosEval_short, 0, maxDets=ytvosEval_short.params.maxDets[1])))
output_file.write('mAP_L: {}\n'.format(_summarize(ytvosEval_long, 1)))
output_file.write('AP50_L: {}\n'.format(
    _summarize(ytvosEval_long, 1, iouThr=.5, maxDets=ytvosEval_long.params.maxDets[2])))
output_file.write('AP75_L: {}\n'.format(
    _summarize(ytvosEval_long, 1, iouThr=.75, maxDets=ytvosEval_long.params.maxDets[2])))
output_file.write('AR1_L: {}\n'.format(
    _summarize(ytvosEval_long, 0, maxDets=ytvosEval_long.params.maxDets[0])))
output_file.write('AR10_L: {}\n'.format(
    _summarize(ytvosEval_long, 0, maxDets=ytvosEval_long.params.maxDets[1])))

output_file.close()
