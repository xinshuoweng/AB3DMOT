# KITTI Inference

## Dataset Preparation

* Please download the official [KITTI multi object tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) dataset (mainly left color images, velodyne point cloud, GPS/IMU data, training labels and camera calibration data are needed). Then uncompress, and put the data under "./data/KITTI" folder (either hard copy or soft symbolic link) in the following structure:
```
AB3DMOT
├── data
│   ├── KITTI
│   │   │── tracking
│   │   |   │── training
│   │   │   │   ├──calib & velodyne & label_02 & image_02 & oxts
│   │   │   │── testing
│   │   │   │   ├──calib & velodyne & image_02 & oxts
├── AB3DMOT_libs
├── configs
```

## 3D Object Detection

For convenience, we provide the 3D detection of PointRCNN on the KITTI MOT dataset at "./data/KITTI/detection". Our detection results follow the format of the KITTI 3D Object Detection Challenge (format definition can be found in the object development toolkit here: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) except that the order is switched. We show an example of detection as follows:

Frame |   Type  |   2D BBOX (x1, y1, x2, y2)  | Score |    3D BBOX (h, w, l, x, y, z, rot_y)      | Alpha | 
------|:-------:|:---------------------------:|:-----:|:-----------------------------------------:|:-----:|
 0    | 2 (car) | 726.4, 173.69, 917.5, 315.1 | 13.85 | 1.56, 1.58, 3.48, 2.57, 1.57, 9.72, -1.56 | -1.82 | 
 
## 3D Multi-Object Tracking

To run our tracker on the KITTI MOT validation set with the provided PointRCNN detections:

```
python3 main.py --dataset KITTI --det_name pointrcnn
```

In detail, running above command will generate a folder named "pointrcnn_val_H1" that includes results combined from all categories, and also folders named "pointrcnn_category_val_H1" representing results for each category. Under each result folder, "./data_0" subfolders are used for MOT evaluation, which follow the format of the KITTI Multi-Object Tracking Challenge (format definition can be found in the tracking development toolkit here: http://www.cvlibs.net/datasets/kitti/eval_tracking.php). Also, "./trk_withid_0" subfolders are used for visualization only, which follow the format of KITTI 3D Object Detection challenge except that we add an ID in the last column.

To run our tracker on the test set with the provided PointRCNN detections, one can simply run:
```
python3 main.py --dataset KITTI --det_name pointrcnn --split test
```
Then, the results will be saved to the "./results/KITTI/pointrcnn_test_H1" folder. 

### 3D MOT Evaluation on KITTI MOT Validation Set

To reproduce the quantitative **3D MOT** results of our 3D MOT system on the KITTI MOT **validation** set with a threshold of 0.25 3D IoU during evaluation, please run:
```
python3 scripts/KITTI/evaluate.py pointrcnn_val_H1 1 3D 0.25
```
Or you can use the threshold of 0.5 and 0.7 3D IoU during evaluation:
```
python3 scripts/KITTI/evaluate.py pointrcnn_val_H1 1 3D 0.5
python3 scripts/KITTI/evaluate.py pointrcnn_Car_val_H1 1 3D 0.7
```

Then, the results should be exactly same as below, except for the FPS which might vary across individual machines. The overall performance is the performance averaged over three categoeries for sAMOTA, MOTA, MOTP and the summed over three categories for IDS, FRAG, FP, FN. Note that, please run the code when CPUs are not occupied by other programs otherwise you might not achieve similar speed as reported in our paper.

#### PointRCNN + AB3DMOT (KITTI val set)

Results evaluated with the 0.25 3D IoU threshold:

 Category       | sAMOTA |  MOTA  |  MOTP  | IDS | FRAG |  FP  |  FN  |  FPS 
--------------- |:------:|:------:|:------:|:---:|:----:|:----:|:----:|:----:|
 *Car*          | 93.34  | 86.47  |  79.40 |  0  | 15   | 368  | 766  | 108.7
 *Pedestrian*   | 82.73  | 73.86  |  67.58 |  4  | 62   | 589  | 1965 | 119.2
 *Cyclist*      | 93.78  | 84.79  |  77.23 |  1  | 3    | 114  | 90   | 980.7
 *Overall*      | 89.62  | 81.71  |  74.74 |  5  | 80   | 1071 | 2821 | -
 
Results evaluated with the 0.5 3D IoU threshold:

 Category       | sAMOTA |  MOTA  |  MOTP  | IDS | FRAG |  FP  |  FN  |  FPS 
--------------- |:------:|:------:|:------:|:---:|:----:|:----:|:----:|:-----:
 *Car*          | 92.57  | 84.81  | 79.82  |  0  | 49   | 456  | 817  | 108.7
 *Pedestrian*   | 77.68  | 68.19  | 68.55  |  2  | 132  | 888  | 2223 | 119.2
 *Cyclist*      | 92.05  | 83.38  | 77.52  |  1  | 5    | 124  | 99   | 980.7
 *Overall*      | 87.43  | 78.79  | 75.30  |  3  | 186  | 1468 | 3139 | -

Results evaluated with the 0.7 3D IoU threshold:

 Category       | sAMOTA |  MOTA  |  MOTP  | IDS | FRAG |  FP  |  FN  |  FPS 
--------------- |:------:|:------:|:------:|:---:|:----:|:----:|:----:|:-----:
 *Car*          | 74.96  | 62.48  |  82.64 |  0  | 173  | 1065 | 2079 | 108.7

Note that the performance is higher than our original IROS 2020 paper due to some improvements we made in the code. We will describe those in a follow up report very soon.

### 2D MOT Evaluation on KITTI MOT Validation Set

To reproduce the quantitative **2D MOT** results of our 3D MOT system on KITTI MOT **validation** set, please run:
```
python3 scripts/KITTI/evaluate.py pointrcnn_val_H1 1 2D 0.5
```

Then, the results should be exactly same as below, except for the FPS which might vary across individual machines. 

 Category       | sAMOTA |  MOTA  |  MOTP  | IDS | FRAG |  FP  |  FN  |  FPS 
--------------- |:------:|:------:|:------:|:---:|:----:|:----:|:----:|:-----:
 *Car*          | 93.08  | 85.98  | 86.95  |   2 | 25   | 394  | 779  | 108.7
 *Pedestrian*   | 69.70  | 60.41  | 67.18  | 119 | 477  | 1075 | 2681 | 119.2
 *Cyclist*      | 91.62  | 83.01  | 85.55  |   0 | 7    | 130  | 99   | 980.7
 *Overall*      | 84.80  | 76.47  | 79.89  | 121 | 509  | 1599 | 3559 | -
  
### 2D MOT Evaluation on KITTI MOT Test Set

To reproduce the quantitative **2D MOT** results of our 3D MOT system on KITTI MOT **test set**, please run the following: 
```
python3 scripts/post_processing/trk_conf_threshold.py --dataset KITTI --result_sha pointrcnn_test_H1
```

Then, compress the folder below and upload to http://www.cvlibs.net/datasets/kitti/user_submit.php for KITTI 2D MOT evaluation. Note that KITTI does not release the ground truth labels to users, so we have to use the official KITTI 2D MOT evaluation server for evaluation, which does not include our new metrics.
```
./results/KITTI/pointrcnn_test_H1_thres/data_0
```

The results should be similar to our entry on the KITTI 2D MOT leaderboard (http://www.cvlibs.net/datasets/kitti/eval_tracking.php). 

## Visualization

To visualize the qualitative results of our 3D MOT system on images shown in the paper (Note that the opencv3 is required by this step, please check the opencv version if there is an error):
```
python3 scripts/post_processing/visualization.py --dataset KITTI --result_sha pointrcnn_test_H1_thres --split test
```
  
Visualization results are then saved to "./results/KITTI/pointrcnn_test_H1_thres/trk_image_vis" and "./results/KITTI/pointrcnn_test_H1_thres/trk_video_vis"