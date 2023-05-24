# AB3DMOT

<b>3D Multi-Object Tracking: A Baseline and New Evaluation Metrics (IROS 2020, ECCVW 2020)</b>

This repository contains the official python implementation for our full paper at IROS 2020 "[3D Multi-Object Tracking: A Baseline and New Evaluation Metrics](http://www.xinshuoweng.com/papers/AB3DMOT/proceeding.pdf)" and short paper "[AB3DMOT: A Baseline for 3D Multi-Object Tracking and New Evaluation Metrics](http://www.xinshuoweng.com/papers/AB3DMOT_eccvw/camera_ready.pdf)" at ECCVW 2020. Our project website and video demos are [here](http://www.xinshuoweng.com/projects/AB3DMOT/). If you find our paper or code useful, please cite our papers:

```
@article{Weng2020_AB3DMOT, 
author = {Weng, Xinshuo and Wang, Jianren and Held, David and Kitani, Kris}, 
journal = {IROS}, 
title = {{3D Multi-Object Tracking: A Baseline and New Evaluation Metrics}}, 
year = {2020} 
}
```
```
@article{Weng2020_AB3DMOT_eccvw, 
author = {Weng, Xinshuo and Wang, Jianren and Held, David and Kitani, Kris}, 
journal = {ECCVW}, 
title = {{AB3DMOT: A Baseline for 3D Multi-Object Tracking and New Evaluation Metrics}}, 
year = {2020} 
}
```

<img align="center" width="100%" src="https://github.com/xinshuoweng/AB3DMOT/blob/master/main1.gif">
<img align="center" width="100%" src="https://github.com/xinshuoweng/AB3DMOT/blob/master/main2.gif">

## Overview

- [News](#news)
- [Introduction](#introduction)
- [Installation](#installation)
- [Quick Demo on KITTI](#quick-demo-on-kitti)
- [Benchmarking](#benchmarking)
- [Acknowledgement](#acknowledgement)

## News

- Feb. 27, 2022: Added support to the nuScenes dataset and updated README
- Feb. 26, 2022: Refactored code libraries and signficantly improved performance on KITTI 3D MOT
- Aug. 06, 2020: Extend abstract (one oral) accepted at two ECCV workshops: [WiCV](https://sites.google.com/view/wicvworkshop-eccv2020/), [PAD](https://sites.google.com/view/pad2020/accepted-papers?authuser=0)
- Jul. 05, 2020: 2D MOT results on KITTI for all three categories released
- Jul. 04, 2020: Code modularized and a minor bug in KITTI evaluation for DontCare objects fixed
- Jun. 30, 2020: Paper accepted at IROS 2020
- Jan. 10, 2020: New metrics sAMOTA added and results updated
- Aug. 21, 2019: Python 3 supported
- Aug. 21, 2019: 3D MOT results on KITTI "Pedestrian" and "Cyclist" categories released
- Aug. 19, 2019: A minor bug in orientation correction fixed
- Jul. 9, 2019: Code and 3D MOT results on KITTI "Car" category released, support Python 2 only

## Introduction

3D multi-object tracking (MOT) is an essential component technology for many real-time applications such as autonomous driving or assistive robotics. However, recent works for 3D MOT tend to focus more on developing accurate systems giving less regard to computational cost and system complexity. In contrast, this work proposes a simple yet accurate real-time baseline 3D MOT system. We use an off-the-shelf 3D object detector to obtain oriented 3D bounding boxes from the LiDAR point cloud. Then, a combination of 3D Kalman filter and Hungarian algorithm is used for state estimation and data association. Although our baseline system is a straightforward combination of standard methods, we obtain the state-of-the-art results. To evaluate our baseline system, we propose a new 3D MOT extension to the official KITTI 2D MOT evaluation along with two new metrics. Our proposed baseline method for 3D MOT establishes new state-of-the-art performance on 3D MOT for KITTI, improving the 3D MOTA from 72.23 of prior art to 76.47. Surprisingly, by projecting our 3D tracking results to the 2D image plane and compare against published 2D MOT methods, our system places 2nd on the official KITTI leaderboard. Also, our proposed 3D MOT method runs at a rate of 214.7 FPS, 65 times faster than the state-of-the-art 2D MOT system. 

## Installation

Please follow carefully our provided [installation instructions](docs/INSTALL.md), to avoid errors when running the code.

## Quick Demo on KITTI

To quickly get a sense of our method's performance on the KITTI dataset, one can run the following command after installation of the code. This step does not require you to download any dataset (a small set of data is already included in this code repository).

```
python3 main.py --dataset KITTI --split val --det_name pointrcnn
python3 scripts/post_processing/trk_conf_threshold.py --dataset KITTI --result_sha pointrcnn_val_H1
python3 scripts/post_processing/visualization.py --result_sha pointrcnn_val_H1_thres --split val
```

## Benchmarking

We provide instructions (inference, evaluation and visualization) for reproducing our method's performance on various supported datasets ([KITTI](docs/KITTI.md), [nuScenes](docs/nuScenes.md)) for benchmarking purposes. 

## Real-Time Tracking in ROS

Special thanks to Pardis for the development of the real-time version running in ROS. Code can be found [here](https://github.com/PardisTaghavi/real_time_tracking_AB3DMOT).

## Acknowledgement

The idea of this method is inspired by "[SORT](https://github.com/abewley/sort)"

