# VidEoMT Model Zoo - DINOv3

> For DINOv3 benchmarking, enable fused QKV to get FPS closer to the DINOv2 setup.

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
<tr><td align="left"><a href="../configs/ytvis19/videomt/dinov3/vit-large/videomt_online_ViTL.yaml">VidEoMT-L</a></td>
<td align="center">68.9</td>
<td align="center">74.8</td>
<td align="center">133</td>
<td align="center"><a href="https://huggingface.co/tue-mps/VidEoMT/resolve/main/dinov3_online/yt_2019_dinov3_68.9.pth?download=true">Model Weights</a></td>
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
<tr><td align="left"><a href="../configs/ytvis21/videomt/dinov3/vit-large/videomt_online_ViTL.yaml">VidEoMT-L</a></td>
<td align="center">63.2</td>
<td align="center">69.2</td>
<td align="center">133</td>
<td align="center"><a href="https://huggingface.co/tue-mps/VidEoMT/resolve/main/dinov3_online/yt_2021_dinov3_63.2.pth?download=true">Model Weights</a></td>
</tr>
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
<tr><td align="left"><a href="../configs/ytvis22/videomt/dinov3/vit-large/videomt_online_ViTL.yaml">VidEoMT-L</a></td>
<td align="center">42.8</td>
<td align="center">50.3</td>
<td align="center">133</td>
<td align="center"><a href="https://huggingface.co/tue-mps/VidEoMT/resolve/main/dinov3_online/yt_2022_dinov3_42.8.pth?download=true">Model Weights</a></td>
</tr>
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
<tr><td align="left"><a href="../configs/VIPSeg/videomt/dinov3/vit-large/videomt_Online_ViTL.yaml">VidEoMT-L</a></td>
<td align="center">55.1</td>
<td align="center">47.1</td>
<td align="center">133</td>
<td align="center"><a href="https://huggingface.co/tue-mps/VidEoMT/resolve/main/dinov3_online/vipsseg_dinov3_55.1_47.1.pth?download=true">Model Weights</a></td>
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
<tr><td align="left"><a href="../configs/VSPW/videomt/dinov3/vit-large/videomt_online_ViTL.yaml">VidEoMT-L</a></td>
<td align="center">94.4</td>
<td align="center">64.0</td>
<td align="center">133</td>
<td align="center"><a href="https://huggingface.co/tue-mps/VidEoMT/resolve/main/dinov3_online/vspw_dinov3_94.4_64.0.pth?download=true">Model Weights</a></td>
</tr>
</tbody></table>