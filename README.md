# AB3DMOT

<b>3D Multi-Object Tracking: A Baseline and New Evaluation Metrics (IROS 2020, ECCVW 2020)</b>

This repository contains the official python implementation for our full paper at IROS 2020 "[3D Multi-Object Tracking: A Baseline and New Evaluation Metrics](http://www.xinshuoweng.com/papers/AB3DMOT/camera_ready.pdf)" and short paper "[AB3DMOT: A Baseline for 3D Multi-Object Tracking and New Evaluation Metrics](http://www.xinshuoweng.com/papers/AB3DMOT_eccvw/camera_ready.pdf)" at ECCVW 2020. Our project website and video demos are [here](http://www.xinshuoweng.com/projects/AB3DMOT/). If you find our paper or code useful, please cite our papers:

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

<img align="center" src="https://github.com/xinshuoweng/AB3DMOT/blob/master/main.gif">

## Overview
- [News](#news)
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [3D Object Detection](#3d-object-detection)
- [3D Multi-Object Tracking](#3d-multi-object-tracking)
- [Acknowledgement](#acknowledgement)

## News
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
This code depends on my personal toolbox: https://github.com/xinshuoweng/Xinshuo_PyToolbox. Please install the toolbox by

*1. Clone the github repository.*
~~~shell
git clone https://github.com/xinshuoweng/Xinshuo_PyToolbox
~~~

*2. Install dependency for the toolbox.*
~~~shell
cd Xinshuo_PyToolbox
pip install -r requirements.txt
~~~

Additionaly, it also requires the following packages:
1. scikit-learn==0.19.2
2. filterpy==1.4.5
3. numba==0.43.1
4. matplotlib==2.2.3
5. pillow==6.2.2
6. opencv-python==3.4.3.18
7. glob2==0.6
8. llvmlite==0.32.1 (for python 3.6) or llvmlite==0.31.0 (for python 2.7)

One can either use the system python or create a virtual enviroment (virtualenv for python2, venv for python3) specifically for this project (https://www.pythonforbeginners.com/basics/how-to-use-python-virtualenv). To install required dependencies on the system python, please run the following command at the root of this code:
```
$ cd path/to/AB3DMOT
$ pip install -r requirements.txt
```
To install required dependencies on the virtual environment of the python (e.g., virtualenv for python2), please run the following command at the root of this code:
```
$ pip install virtualenv
$ virtualenv .
$ source bin/activate
$ pip install -r requirements.txt
```
Please add the path to the code to your PYTHONPATH in order to load the library appropriately. For example, if the code is located at /home/user/workspace/code/AB3DMOT, please add the following to your ~/.profile:
```
$ export PYTHONPATH=${PYTHONPATH}:/home/user/workspace/code/AB3DMOT
$ export PYTHONPATH=${PYTHONPATH}:/home/user/workspace/code/Xinshuo_PyToolbox
```

## 3D Object Detection
For convenience, we provide the 3D detection of PointRCNN on the KITTI MOT dataset at "./data/KITTI/" for car, pedestrian and cyclist splits. Our detection results follow the format of the KITTI 3D Object Detection Challenge (format definition can be found in the object development toolkit here: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) except that the order is switched. We show an example of detection as follows:

Frame | Type   | 2D BBOX (x1, y1, x2, y2)       | Score | 3D BBOX (h, w, l, x, y, z, rot_y) | Alpha  | 
------|:------:|:------------------------------:|:----------:|:---------------------------------:|:-------------:
 0    | 2 (car) | 726.4, 173.69, 917.5, 315.1 |  13.85     | 1.56, 1.58, 3.48, 2.57, 1.57, 9.72, -1.56 | -1.82 | 
 
## 3D Multi-Object Tracking

### Inference
To run our tracker on the KITTI MOT validation set with the provided detection:

```
$ python main.py pointrcnn_Car_val
$ python main.py pointrcnn_Pedestrian_val
$ python main.py pointrcnn_Cyclist_val
```
To run our tracker on the KITTI MOT test set with the provided detection:

```
$ python main.py pointrcnn_Car_test
$ python main.py pointrcnn_Pedestrian_test
```
Then, the results will be saved to "./results" folder. In detail, results in "./results/data" folder are used for MOT evaluation, which follow the format of the KITTI Multi-Object Tracking Challenge (format definition can be found in the tracking development toolkit here: http://www.cvlibs.net/datasets/kitti/eval_tracking.php). On the other hand, results in "./results/trk_withid" folder are used for visualization only, which follow the format of KITTI 3D Object Detection challenge except that we add an ID in the last column.

Note that, please run the code when the CPU is not occupied by other programs otherwise you might not achieve similar speed as reported in our paper.

### 3D MOT Evaluation on KITTI MOT Validation Set

To reproduce the quantitative **3D MOT** results of our 3D MOT system on KITTI MOT **validation** set, please run:
  ```
  $ python evaluation/evaluate_kitti3dmot.py pointrcnn_Car_val
  $ python evaluation/evaluate_kitti3dmot.py pointrcnn_Pedestrian_val
  $ python evaluation/evaluate_kitti3dmot.py pointrcnn_Cyclist_val
  ```
Then, the results should be exactly same as below, except for the FPS which might vary across individual machines. The overall performance is the performance averaged over three categoeries for sAMOTA, AMOTA, AMOTP, MOTA, MOTP and the summed over three categories for IDS, FRAG, FP, FN.

 Category       | sAMOTA | AMOTA | AMOTP | MOTA | MOTP | IDS | FRAG | FP | FN | FPS 
--------------- |:----------:|:---------:|:--------:|:-------:|:------:|:---:|:--:|:---:|:--:|:---:
 *Car*          |  93.28     | 45.43     | 77.41     | 86.24    |  78.43  |  0  | 15   | 365 | 708  | 207.4
 *Pedestrian*   |  74.39     | 29.77     | 53.90     | 69.50    |  67.77  |  1  | 74   | 276 | 2708 | 470.1
 *Cyclist*      |  72.94     | 37.95     | 63.03     | 79.82    |  76.55  |  0  | 4    | 55  | 217  | 1241.6
 *Overall*      |  80.20     | 37.72     | 64.78     | 78.52    |  74.25  |  1  | 93   | 696 | 3713 | -
 
### 2D MOT Evaluation on KITTI MOT Validation Set

To reproduce the quantitative **2D MOT** results of our 3D MOT system on KITTI MOT **validation** set, please run:
  ```
  $ python evaluation/evaluate_kitti3dmot.py pointrcnn_Car_val 2D
  $ python evaluation/evaluate_kitti3dmot.py pointrcnn_Pedestrian_val 2D
  $ python evaluation/evaluate_kitti3dmot.py pointrcnn_Cyclist_val 2D
  ```
Then, the results should be exactly same as below, except for the FPS which might vary across individual machines. 

 Category       | sAMOTA | AMOTA | AMOTP | MOTA | MOTP | IDS | FRAG | FP | FN | FPS 
--------------- |:----------:|:---------:|:--------:|:-------:|:------:|:---:|:--:|:---:|:--:|:---:
 *Car*          |  93.01     | 45.21     | 84.61     | 85.70    |  86.99  |  2  | 24   | 391  | 805  | 207.4
 *Pedestrian*   |  65.89     | 24.29     | 49.15     | 59.76    |  67.27  |  52 | 371  | 683  | 3203 | 470.1
 *Cyclist*      |  72.09     | 37.65     | 67.47     | 78.78    |  85.40  |  0  | 8    | 64   | 222  | 1241.6
 *Overall*      |  77.00     | 35.72     | 67.08     | 74.75    |  79.89  |  54 | 403  | 1138 | 4230 | -
  
### 2D MOT Evaluation on KITTI MOT Test Set

To reproduce the quantitative **2D MOT** results of our 3D MOT system on KITTI MOT **test set**, please run the following: 
```
  $ python trk_conf_threshold.py pointrcnn_Car_test
  $ python trk_conf_threshold.py pointrcnn_Pedestrian_test
  $ python combine_trk_cat.py
  ```
Then, compress the folder below and upload to http://www.cvlibs.net/datasets/kitti/user_submit.php for KITTI 2D MOT evaluation. Note that KITTI does not release the ground truth labels to users, so we have to use the official KITTI 2D MOT evaluation server for evaluation, which does not include our new metrics.
  ```
  $ ./results/pointrcnn_test_thres/data
  ```
The results should be similar to our entry shown below on the KITTI 2D MOT leaderboard. Note that we only have results for Car and Pedestrian because KITTI 2D MOT benchmark only supports to evaluate these two categories, not including the Cyclist. 

 Category       | MOTA (%) | MOTP (%)| MT (%) | ML (%) | IDS | FRAG |  FP  |   FN  | FPS 
--------------- |:--------:|:-------:|:------:|:------:|:---:|:----:|:----:|:-----:|:---:
 *Car*          | 83.84    |  85.24  | 66.92  | 11.38  |  9  | 224  | 1059 | 4491  | 214.7
 *Pedestrian*   | 39.63    |  64.87  | 16.84  | 41.58  | 170 | 940  | 2098 | 11707 | 351.8

### Visualization

To visualize the qualitative results of our 3D MOT system on images shown in the paper (Note that the opencv3 is required by this step, please check the opencv version if there is an error):
  ```
  $ python visualization.py pointrcnn_Car_test_thres
  ```
Visualization results are then saved to "./results/pointrcnn_test_thres/trk_image_vis". If one wants to visualize the results on the entire sequences, please download the KITTI MOT dataset http://www.cvlibs.net/datasets/kitti/eval_tracking.php and move the image_02 (we have already prepared the calib data for you) data to the "./data/KITTI/resources" folder.
 
### Acknowledgement
Part of the code is borrowed from "[SORT](https://github.com/abewley/sort)"
