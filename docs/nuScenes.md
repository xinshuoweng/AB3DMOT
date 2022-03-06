# nuScenes Inference

## Additional Dependencies:

To run experiments on nuScenes, the following packages need to be installed:
1. nuscenes-devkit==1.1.9
2. motmetrics<=1.1.3
3. pandas>=0.24

To install required dependencies on the virtual environment of the python, please run the following commands:

```
cd path/to/AB3DMOT
source env/bin/activate
cd scripts/nuScenes
pip3 install -r requirements.txt
cd ../../
```

## Dataset Preparation

* Please download the official [nuScenes full dataset (v1.0)](https://www.nuscenes.org/download), then uncompress, and put the data under "./data/nuScenes/data" folder (either hard copy or soft symbolic link) in the following structure:
```
AB3DMOT
├── data
│   ├── nuScenes
│   │   │── data
│   │   |   │── samples
│   │   |   │── sweeps
│   │   |   │── v1.0-mini
│   │   |   │── v1.0-test
│   │   |   │── v1.0-trainval
├── AB3DMOT_libs
├── configs
```

Because our code processes data in the KITTI format, one must run the following code to convert the nuScenes raw data into the KITTI format:
```
python3 scripts/nuScenes/export_kitti.py nuscenes_gt2kitti_trk --split val
python3 scripts/nuScenes/export_kitti.py nuscenes_gt2kitti_trk --split test
```

The above code will generate nuScenes GT data at "./data/nuScenes/nuKITTI/tracking" following the KITTI format. Please check if the data has the following structure:
```
AB3DMOT/data
├── nuScenes
│   ├── nuKITTI
│   │   │── tracking
│   │   │   │── produced
│   │   │   │   ├──correspondence & split
│   │   │   │── test
│   │   │   │   ├──calib & image_02 & oxts & velodyne 
│   │   |   │── val
│   │   │   │   ├──calib & image_02 & label_02 & oxts & velodyne 
├── AB3DMOT_libs
├── configs
```

## 3D Object Detection

For convenience, we provide the pre-processed 3D detection of Megvii/CenterPoint on the nuScenes dataset at "./data/nuScenes/detection". These detections share the same format as the KITTI detections introduced [here](docs/KITTI.md) for easy processing by our code. We show an example of detection as follows:

Frame |   Type  |   2D BBOX (x1, y1, x2, y2)  | Score  |    3D BBOX (h, w, l, x, y, z, rot_y)      | Alpha | 
------|:-------:|:---------------------------:|:------:|:-----------------------------------------:|:-----:|
 0    | 2 (car) | 726.4, 173.69, 917.5, 315.1 | 0.9357 | 1.56, 1.58, 3.48, 2.57, 1.57, 9.72, -1.56 | -10.0 | 
 
Note that these detection results are converted from the offcial format of the nuScenes 3D Object Detection Challenge (format definition can be found [here](https://www.nuscenes.org/object-detection/?externalData=all&mapData=all&modalities=Any)). To convert your own nuScenes detections (following nuScenes detection format) into our format, we provide the conversion code at "./scripts/nuScenes/export_kitti.py". For example, given your raw nuScenes detection file at "./data/nuScenes/data/produced/results/detection/detname/results_val.json", simply run the following:
```
python3 scripts/nuScenes/export_kitti.py nuscenes_obj_result2kitti --result_name detname --split val
```

The above code will generate detection results at "./data/nuScenes/nuKITTI/object/produced/results/val/detname/data", which strictly follow the format of the KITTI object detection challenge and can be evaluated using the standard KITTI detection eval code [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). Then, we need to pre-process those files into "./data/nuScenes/detection" with the following code:
```
python3 scripts/pre_processing/convert_det2input.py --dataset nuScenes --split val --det_name detname
```

You will see the input files under "./data/nuScenes/detection/detname_cat_val", where cat means the category name. We split the detections into every category and also a single folder including all categories. In other words, you will see folders such as "detname_Car_val", "detname_Pedestrian_val", "detname_Truck_val", "detname_all_val", etc. These folders are ready to be used by our main.py for AB3DMOT.

## 3D Multi-Object Tracking

Once the data is properly prepared, everything is as easy as running for KITTI inference. For example, one can run AB3DMOT using the megvii detections on the validation set with the following command:
```
python3 main.py --dataset nuScenes --det_name megvii --split val
python3 main.py --dataset nuScenes --det_name centerpoint --split val
```
Then, the results from all categories will be saved to the "./results/nuScenes/megvii_val_H1" or "./results/nuScenes/centerpoint_val_H1" folders, similar to the KITTI tracking result format as introduced [here](docs/KITTI.md). 

### 3D MOT Evaluation on the nuScenes Tracking Validation Set

To reproduce the quantitative **3D MOT** results of our 3D MOT system on the nuScenes tracking **validation** set, we need to first convert the result format into the [nuScenes tracking result format](https://www.nuscenes.org/tracking/?externalData=all&mapData=all&modalities=Any) and then run the [official nuScenes MOT evaluation code](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/tracking/evaluate.py). For simplicity, we have made a local copy of the evaluation code in this repository, please run:
```
python3 scripts/nuScenes/export_kitti.py kitti_trk_result2nuscenes --result_name megvii_val_H1 --split val
python3 scripts/nuScenes/evaluate.py --result_path ./results/nuScenes/megvii_val_H1/results_val.json
```

Then, the results should be exactly same as below:

#### Megvii + AB3DMOT (nuScenes val set)

 Category     | sAMOTA |  MOTA  |  MOTP  | IDS | FRAG |  FP   |  FN   
--------------|:------:|:------:|:------:|:---:|:----:|:-----:|:-----:
 *Car*        | 0.786  | 0.667  | 0.314  | 353 | 275  | 6556  | 12493 
 *Pedestrian* | 0.775  | 0.648  | 0.267  | 299 | 162  | 3740  | 4916 
 *Bicycle*    | 0.284  | 0.259  | 0.167  |  1  |  2   | 211   | 1264
 *Motorcycle* | 0.545  | 0.473  | 0.297  | 14  | 20   | 252   | 775   
 *Bus*        | 0.768  | 0.661  | 0.506  |  0  | 13   |  92   | 625   
 *Trailer*    | 0.362  | 0.324  | 0.728  |  3  | 20   | 201   | 1436   
 *Truck*      | 0.583  | 0.454  | 0.419  | 30  | 27   | 1303  | 3934   
 *Overall*    | 0.586  | 0.498  | 0.385  | 769 | 532  | 12361 | 25375
 
Note that the results are different from (actually higher than) numbers reported in our original IROS 2020 paper because results in our IROS 2020 paper are obtained using a different evaluation code with a different matching threshold/metric (the official nuScenes tracking evaluation code was still under development at that time).

### 3D MOT Evaluation on the nuScenes Tracking Test Set

To reproduce the quantitative **3D MOT** results of our 3D MOT system on the nuScenes tracking **test set**, please run the following: 
```
python3 main.py --dataset nuScenes --det_name megvii --split test
python3 scripts/nuScenes/export_kitti.py kitti_trk_result2nuscenes --result_name megvii_test_H1 --split test
```

Then, compress the result file at "./results/nuScenes/megvii_test_H1/results_test.json" into a zip file and upload [here](https://eval.ai/web/challenges/challenge-page/476/submission) for official nuScenes 3D MOT evaluation. Note that nuScenes does not release the ground truth labels to users, so we have to use the official nuScenes 3D MOT evaluation server for evaluation. The results should be similar to our entry on the nuScenes 3D MOT leaderboard below:

#### Megvii + AB3DMOT (nuScenes test set)

 Category     | sAMOTA |  MOTA  |  MOTP  | IDS | FRAG |  FP   |  FN   
--------------|:------:|:------:|:------:|:---:|:----:|:-----:|:-----:
 *Car*        | 0.771  | 0.636  | 0.310  | 515 | 365  | 9836  | 14595 
 *Pedestrian* | 0.764  | 0.624  | 0.242  | 376 | 237  | 4984  | 7422
 *Bicycle*    | 0.266  | 0.233  | 0.190  |  3  |  4   | 164   | 1509
 *Motorcycle* | 0.494  | 0.426  | 0.312  | 11  | 14   | 198   | 907   
 *Bus*        | 0.645  | 0.525  | 0.509  | 10  | 10   | 364   | 434   
 *Trailer*    | 0.513  | 0.437  | 0.595  | 10  | 19   | 220   | 1214   
 *Truck*      | 0.531  | 0.392  | 0.426  | 23  | 32   | 2120  | 3109   
 *Overall*    | 0.569  | 0.468  | 0.369  | 948 | 681  | 17886 | 29190

### Quick 3D MOT Evaluation on the nuScenes Tracking Validation Set (Unofficial)

The above evaluation process follows the same procesdure as the official nuScenes tracking evaluation, but the scripts take a long time to run. So we also provide a quick evaluation script that is adapted from KITTI evaluation. To proceed, simply run the following commands:
```
python3 scripts/nuScenes/evaluate_quick.py megvii_val_H1 1 val
```

The results will not be the same as above because there are differences in the implementation of the evaluation code but we do observe similar trend when using two scripts (i.e., when our method is clearly improved and get higher numbers in one evaluation code, we also obtain higher numbers when using the other evaluation code).

## Visualization

To visualize the qualitative results of our 3D MOT system on nuScenes, please run:
```
python3 scripts/post_processing/trk_conf_threshold.py --dataset nuScenes-- result_sha megvii_val_H1
python3 scripts/post_processing/visualization.py --dataset nuScenes --result_sha megvii_val_H1_thres --split val
```

The first script filters out some low-score tracklets which are likely false positives. These low-score tracklets are included in the raw results because nuScenes uses sAMOTA metric for evaluation that requires all tracklets regardless of the score. After running the second script, visualization results are then saved to "./results/nuScenes/megvii_val_H1_thres/trk_image_vis" and "./results/nuScenes/megvii_val_H1_thres/trk_video_vis". Note that the opencv3 is required by this step, please check the opencv version if there is an error.