# VidEoMT Model Zoo - DINOv2

> FPS measured on NVIDIA H100 with default torch.compile.

## Video Instance Segmentation

### YouTube-VIS 2019

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">AP</th>
<th valign="bottom">AR<sub>10</sub></th>
<th valign="bottom">FPS</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: VidEoMT-S  -->
<tr><td align="left"><a href="../configs/ytvis19/videomt/vit-small/videomt_online_ViTS.yaml">VidEoMT-S</a></td>
<td align="center">52.8</td>
<td align="center">62.2</td>
<td align="center">294</td>
<td align="center"><a href="https://huggingface.co/tue-mps/VidEoMT/resolve/main/yt_2019_vit_small_52.8.pth?download=true">Model Weights</a></td>
</tr>
<!-- ROW: VidEoMT-B -->
<tr><td align="left"><a href="../configs/ytvis19/videomt/vit-base/videomt_online_ViTB.yaml">VidEoMT-B</a></td>
<td align="center">58.2</td>
<td align="center">66.5</td>
<td align="center">251</td>
<td align="center"><a href="https://huggingface.co/tue-mps/VidEoMT/resolve/main/yt_2019_vit_base_58.2.pth?download=true">Model Weights</a></td>
</tr>
<!-- ROW: VidEoMT-L 640x640 -->
<tr><td align="left"><a href="../configs/ytvis19/videomt/vit-large/videomt_online_ViTL.yaml">VidEoMT-L</a></td>
<td align="center">68.6</td>
<td align="center">73.9</td>
<td align="center">160</td>
<td align="center"><a href="https://huggingface.co/tue-mps/VidEoMT/resolve/main/yt_2019_vit_large_68.6.pth?download=true">Model Weights</a></td>
</tr>
</tbody></table>  




### YouTube-VIS 2021

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">AP</th>
<th valign="bottom">AR<sub>10</sub></th>
<th valign="bottom">FPS</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<tr><td align="left"><a href="../configs/ytvis21/videomt/vit-large/videomt_online_ViTL.yaml">VidEoMT-L</a></td>
<td align="center">63.1</td>
<td align="center">68.1</td>
<td align="center">160</td>
<td align="center"><a href="https://huggingface.co/tue-mps/VidEoMT/resolve/main/yt_2021_vit_large_63.1.pth?download=true">Model Weights</a></td>

</tbody></table>

### YouTube-VIS 2022

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">AP<sup>L</sup></th>
<th valign="bottom">AR<sup>L</sup><sub>10</sub></th>
<th valign="bottom">FPS</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<tr><td align="left"><a href="../configs/ytvis22/videomt/vit-large/videomt_online_ViTL.yaml">VidEoMT-L</a></td>
<td align="center">42.6</td>
<td align="center">48.1</td>
<td align="center">161</td>
<td align="center"><a href="https://huggingface.co/tue-mps/VidEoMT/resolve/main/yt_2022_vit_large_42.6.pth?download=true">Model Weights</a></td>

</tbody></table>

### OVIS

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">AP</th>
<th valign="bottom">AR<sub>10</sub></th>
<th valign="bottom">FPS</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<tr><td align="left"><a href="../configs/ovis/videomt/vit-large/videomt_online_ViTL.yaml">VidEoMT-L</a></td>
<td align="center">52.5</td>
<td align="center">57.5</td>
<td align="center">115</td>
<td align="center"><a href="https://huggingface.co/tue-mps/VidEoMT/resolve/main/ovis_vit_large_52.5.pth?download=true">Model Weights</a></td>

</tbody></table>

## Video Panoptic Segmentation

### VIPSeg

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">VPQ</th>
<th valign="bottom">STQ</th>
<th valign="bottom">FPS</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: VidEoMT-L 1024x1024 -->
<tr><td align="left"><a href="../configs/VIPSeg/videomt/vit-large/videomt_Online_ViTL.yaml">VidEoMT-L</a></td>
<td align="center">55.2</td>
<td align="center">48.9</td>
<td align="center">75</td>
<td align="center"><a href="https://huggingface.co/tue-mps/VidEoMT/resolve/main/vipseg_vit_large_55.2.pth?download=true">Model Weights</a></td>
</tr>
</tbody></table>

## Video Semantic Segmentation
### VSPW

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">mVC<sub>16</sub></th>
<th valign="bottom">mIoU</th>
<th valign="bottom">FPS</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: VidEoMT-L 512x512 -->
<tr><td align="left"><a href="../configs/VSPW/videomt/vit-large/videomt_online_ViTL.yaml">VidEoMT-L</a></td>
<td align="center">95.0</td>
<td align="center">64.9</td>
<td align="center">73</td>
<td align="center"><a href="https://huggingface.co/tue-mps/VidEoMT/resolve/main/vspw_vit_large_95.0_64.9.pth?download=true">Model Weights</a></td>
</tr>
</tbody></table>

