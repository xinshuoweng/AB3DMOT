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
- [Dependencies](#dependencies)
- [3D Object Detection](#3d-object-detection)
- [3D Multi-Object Tracking](#3d-multi-object-tracking)
- [Acknowledgement](#acknowledgement)

## News
- Feb. 26, 2022: Refactor the code and libraries and signficantly improve performance on KITTI 3D MOT evaluation
- Aug. 06, 2020: Extended abstract (one oral) accepted at two ECCV workshops: [WiCV](https://sites.google.com/view/wicvworkshop-eccv2020/), [PAD](https://sites.google.com/view/pad2020/accepted-papers?authuser=0)
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

## Dependencies:
This code requires the following packages:
1. scikit-learn==0.19.2
2. filterpy==1.4.5
3. numba==0.43.1
4. matplotlib==2.2.3
5. pillow==6.2.2
6. opencv-python==4.2.0.32
7. glob2==0.6
8. llvmlite==0.32.1

One can either use the system python or create a virtual enviroment (venv for python3) specifically for this project (https://www.pythonforbeginners.com/basics/how-to-use-python-virtualenv). To install required dependencies on the system python, please run the following command at the root of this code:
```
$ cd path/to/AB3DMOT
$ pip3 install -r requirements.txt
```
To install required dependencies on the virtual environment of the python, please run the following command at the root of this code:
```
$ pip3 install venv
$ python3 -m venv env
$ source env/bin/activate
$ pip3 install -r requirements.txt
```

Additionally, this code depends on my personal toolbox: https://github.com/xinshuoweng/Xinshuo_PyToolbox. Please install the toolbox by:

*1. Clone the github repository.*
~~~shell
git clone https://github.com/xinshuoweng/Xinshuo_PyToolbox
~~~

*2. Install dependency for the toolbox.*
~~~shell
cd Xinshuo_PyToolbox
pip3 install -r Xinshuo_PyToolbox/requirements.txt
~~~

Please add the path to the code to your PYTHONPATH in order to load the library appropriately. For example, if the code is located at /home/user/workspace/code/AB3DMOT, please add the following to your ~/.profile:
```
$ export PYTHONPATH=${PYTHONPATH}:/home/user/workspace/code/AB3DMOT
$ export PYTHONPATH=${PYTHONPATH}:/home/user/workspace/code/AB3DMOT/Xinshuo_PyToolbox
```

## 3D Object Detection
For convenience, we provide the 3D detection of PointRCNN on the KITTI MOT dataset at "./data/KITTI/" for car, pedestrian and cyclist splits. Our detection results follow the format of the KITTI 3D Object Detection Challenge (format definition can be found in the object development toolkit here: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) except that the order is switched. We show an example of detection as follows:

Frame | Type   | 2D BBOX (x1, y1, x2, y2)       | Score | 3D BBOX (h, w, l, x, y, z, rot_y) | Alpha  | 
------|:------:|:------------------------------:|:----------:|:---------------------------------:|:-------------:
 0    | 2 (car) | 726.4, 173.69, 917.5, 315.1 |  13.85     | 1.56, 1.58, 3.48, 2.57, 1.57, 9.72, -1.56 | -1.82 | 
 
## 3D Multi-Object Tracking

### Inference
To run our tracker on the KITTI/nuScenes MOT validation set with the provided detection:

```
$ python3 main.py --cfg KITTI
$ python3 main.py --cfg nuScenes
```
To run our tracker on the test set with the provided detection, one can change the "split" entry in the config file from "val" to "test" and then run the same above command, or simply run
```
$ python3 main.py --cfg KITTI --split test
$ python3 main.py --cfg nuScenes --split test
```
Then, the results will be saved to the "./results/KITTI" or "./results/nuScenes" folder. 

In detail, running above command will generate a folder named "detname_split_H1" that includes results combined from all categories, and also folders named "detname_category_split_H1" representing results for each category. Under each result folder, the "./data_0" subfolders are used for MOT evaluation, which follow the format of the KITTI Multi-Object Tracking Challenge (format definition can be found in the tracking development toolkit here: http://www.cvlibs.net/datasets/kitti/eval_tracking.php). Also, the "./trk_withid_0" subfolders are used for visualization only, which follow the format of KITTI 3D Object Detection challenge except that we add an ID in the last column.

### 3D MOT Evaluation on KITTI MOT Validation Set

To reproduce the quantitative **3D MOT** results of our 3D MOT system on the KITTI MOT **validation** set with a threshold of 0.25 3D IoU during evaluation, please run:
```
$ python3 evaluation/evaluate_kitti3dmot.py pointrcnn_val_H1 1 3D 0.25
```
Or you can use the threshold of 0.5 3D IoU during evaluation:
```
$ python3 evaluation/evaluate_kitti3dmot.py pointrcnn_val_H1 1 3D 0.5
```

Then, the results should be exactly same as below, except for the FPS which might vary across individual machines. The overall performance is the performance averaged over three categoeries for sAMOTA, AMOTA, AMOTP, MOTA, MOTP and the summed over three categories for IDS, FRAG, FP, FN. Note that, please run the code when the CPU is not occupied by other programs otherwise you might not achieve similar speed as reported in our paper.

#### Results with PointRCNN

Results evaluated with the 0.25 3D IoU threshold:

 Category       | sAMOTA | AMOTA | AMOTP | MOTA | MOTP | IDS | FRAG | FP | FN | FPS 
--------------- |:----------:|:---------:|:--------:|:-------:|:------:|:---:|:--:|:---:|:--:|:---:
 *Car*          |  93.34     | 45.51     | 78.49     | 86.47    |  79.40  |  0  | 15   | 368  | 766  | 108.7
 *Pedestrian*   |  82.73     | 34.72     | 62.54     | 73.86    |  67.58  |  4  | 62   | 589  | 1965 | 119.2
 *Cyclist*      |  93.78     | 47.88     | 81.97     | 84.79    |  77.23  |  1  | 3    | 114  | 90   | 980.7
 *Overall*      |  89.62     | 42.70     | 74.33     | 81.71    |  74.74  |  5  | 80   | 1071 | 2821 | -
 
Results evaluated with the 0.5 3D IoU threshold:

 Category       | sAMOTA | AMOTA | AMOTP | MOTA | MOTP | IDS | FRAG | FP | FN | FPS 
--------------- |:----------:|:---------:|:--------:|:-------:|:------:|:---:|:--:|:---:|:--:|:---:
 *Car*          |  92.57     | 44.85     | 78.69     | 84.81    |  79.82  |  0  | 49   | 456  | 817  | 108.7
 *Pedestrian*   |  77.68     | 31.50     | 59.58     | 68.19    |  68.55  |  2  | 132  | 888  | 2223 | 119.2
 *Cyclist*      |  92.05     | 46.17     | 82.21     | 83.38    |  77.52  |  1  | 5    | 124  | 99   | 980.7
 *Overall*      |  87.43     | 40.84     | 73.49     | 78.79    |  75.30  |  3  | 186  | 1468 | 3139 | -

Results evaluated with the 0.7 3D IoU threshold:

 Category       | sAMOTA | AMOTA | AMOTP | MOTA | MOTP | IDS | FRAG | FP | FN | FPS 
--------------- |:----------:|:---------:|:--------:|:-------:|:------:|:---:|:--:|:---:|:--:|:---:
 *Car*          |  74.96     | 30.60     | 69.58     | 62.48    |  82.64  |  0  | 173   | 1065 | 2079  | 108.7

Note that the results are slightly higher than our original IROS 2020 paper due to some improvements we made in the code. We will describe those in a follow up report very soon.

### 2D MOT Evaluation on KITTI MOT Validation Set

To reproduce the quantitative **2D MOT** results of our 3D MOT system on KITTI MOT **validation** set, please run:
```
$ python3 evaluation/evaluate_kitti3dmot.py pointrcnn_val_H1 1 2D 0.5
```

Then, the results should be exactly same as below, except for the FPS which might vary across individual machines. 

 Category       | sAMOTA | AMOTA | AMOTP | MOTA | MOTP | IDS | FRAG | FP | FN | FPS 
--------------- |:----------:|:---------:|:--------:|:-------:|:------:|:---:|:--:|:---:|:--:|:---:
 *Car*          |  93.08     | 45.30     | 84.58     | 85.98    |  86.95  |   2 | 25   | 394  | 779  | 108.7
 *Pedestrian*   |  69.70     | 24.38     | 55.77     | 60.41    |  67.18  | 119 | 477  | 1075 | 2681 | 119.2
 *Cyclist*      |  91.62     | 45.92     | 87.51     | 83.01    |  85.55  |   0 | 7    | 130  | 99   | 980.7
 *Overall*      |  84.80     | 38.53     | 75.95     | 76.47    |  79.89  | 121 | 509  | 1599 | 3559 | -
  
### 2D MOT Evaluation on KITTI MOT Test Set

To reproduce the quantitative **2D MOT** results of our 3D MOT system on KITTI MOT **test set**, please run the following: 
```
$ python3 KITTI_trk_conf_threshold.py --result_sha pointrcnn_Car_test_H1
$ python3 KITTI_trk_conf_threshold.py --result_sha pointrcnn_Pedestrian_test_H1
$ python3 KITTI_trk_conf_threshold.py --result_sha pointrcnn_Cyclist_test_H1
$ python3 combine_trk_cat.py --dataset KITTI --split test --suffix H1_thres
```
Then, compress the folder below and upload to http://www.cvlibs.net/datasets/kitti/user_submit.php for KITTI 2D MOT evaluation. Note that KITTI does not release the ground truth labels to users, so we have to use the official KITTI 2D MOT evaluation server for evaluation, which does not include our new metrics.
```
$ ./results/KITTI/pointrcnn_test_H1_thres/data_0
```
The results should be similar to our entry on the KITTI 2D MOT leaderboard (http://www.cvlibs.net/datasets/kitti/eval_tracking.php). 

### Visualization

To visualize the qualitative results of our 3D MOT system on images shown in the paper (Note that the opencv3 is required by this step, please check the opencv version if there is an error):
  ```
  $ python3 visualization.py --result_sha pointrcnn_test_H1_thres --split test
  ```
Visualization results are then saved to "./results/KITTI/pointrcnn_test_H1_thres/trk_image_vis" and "./results/KITTI/pointrcnn_test_H1_thres/trk_video_vis". If one wants to visualize the results on the entire sequences, please download the KITTI MOT dataset http://www.cvlibs.net/datasets/kitti/eval_tracking.php and move the image_02 (we have already prepared the calib data for you) data to the "./data/KITTI/resources" folder.
 
### Acknowledgement
The idea of this method is inspired by "[SORT](https://github.com/abewley/sort)"
