# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.
# Licensed under the Creative Commons [see licence.txt]

"""
This script converts nuScenes data to KITTI format and KITTI results to nuScenes.
It is used for compatibility with software that uses KITTI-style annotations.

We do not encourage this, as:
- KITTI has only front-facing cameras, whereas nuScenes has a 360 degree horizontal fov.
- KITTI has no radar data.
- The nuScenes database format is more modular.
- KITTI fields like occluded and truncated cannot be exactly reproduced from nuScenes data.
- KITTI has different categories.

Limitations:
- We don't specify the KITTI imu_to_velo_kitti projection in this code base.
- We map nuScenes categories to nuScenes detection categories, rather than KITTI categories.
- Attributes are not part of KITTI and therefore set to '' in the nuScenes result format.
- Velocities are not part of KITTI and therefore set to 0 in the nuScenes result format.
- This script uses the `train` and `val` splits of nuScenes, whereas standard KITTI has `training` and `testing` splits.

This script includes three main functions:
- nuscenes_gt_to_kitti(): Converts nuScenes GT annotations to KITTI format.
- render_kitti(): Render the annotations of the (generated or real) KITTI dataset.
- kitti_res_to_nuscenes(): Converts a KITTI detection result to the nuScenes detection results format.

To launch these scripts run:
- python export_kitti.py nuscenes_gt_to_kitti_obj --nusc_kitti_root ~/nusc_kitti
- python export_kitti.py render_kitti --nusc_kitti_root ~/nusc_kitti --render_2d False
- python export_kitti.py kitti_res_to_nuscenes --nusc_kitti_root ~/nusc_kitti
Note: The parameter --render_2d specifies whether to draw 2d or 3d boxes.

To work with the original KITTI dataset, use these parameters:
 --nusc_kitti_root /data/sets/kitti --split training

See https://www.nuscenes.org/object-detection for more information on the nuScenes result format.
"""

import os, sys, json, numpy as np, fire
from typing import List, Dict, Any
from shutil import copyfile
from pyquaternion import Quaternion
from xinshuo_io import mkdir_if_missing, load_list_from_folder, fileparts

# load nuScenes libraries
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import Box
from nuscenes.utils.splits import create_splits_logs, create_splits_scenes
from AB3DMOT_libs.nuScenes2KITTI_helper import load_correspondence, load_correspondence_inverse
from AB3DMOT_libs.nuScenes2KITTI_helper import kitti_cam2nuScenes_lidar, nuScenes_transform2KITTI
from AB3DMOT_libs.nuScenes2KITTI_helper import create_KITTI_transform, convert_anno_to_KITTI, save_image, save_lidar
from AB3DMOT_libs.nuScenes_utils import nuScenes_lidar2world, nuScenes_world2lidar, get_sensor_param, split_to_samples, scene_to_samples
from AB3DMOT_libs.nuScenes_utils import box_to_trk_sample_result, create_nuScenes_box, box_to_det_sample_result

# load KITTI libraries
from AB3DMOT_libs.kitti_calib import Calibration, save_calib_file
from AB3DMOT_libs.kitti_trk import Tracklet_3D

class KittiConverter:
    def __init__(self,
                 nusc_kitti_root: str = './data/nuScenes/nuKITTI',   
                 data_root: str = './data/nuScenes/data',
                 result_root: str = './results/nuScenes/',
                 result_name: str = 'megvii_val_H1',         
                 cam_name: str = 'CAM_FRONT',
                 lidar_name: str = 'LIDAR_TOP',            
                 split: str = 'val'):
        """
        :param nusc_kitti_root: Where to write the KITTI-style annotations.
        :param cam_name: Name of the camera to export. Note that only one camera is allowed in KITTI.
        :param lidar_name: Name of the lidar sensor.
        :param image_count: Number of images to convert.
        :param nusc_version: nuScenes version to use.
        :param split: Dataset split to use.
        """
        self.nusc_kitti_root = nusc_kitti_root; mkdir_if_missing(self.nusc_kitti_root)
        self.cam_name = cam_name
        self.lidar_name = lidar_name
        self.split = split
        if split in ['train', 'val', 'trainval']: self.nusc_version = 'v1.0-trainval'
        elif split == 'test':                     self.nusc_version = 'v1.0-test'
        self.result_name = result_name
        self.data_root = data_root
        self.result_root = result_root

        # Select subset of the data to look at.
        self.nusc = NuScenes(version=self.nusc_version, dataroot=data_root, verbose=True)

    def nuscenes_gt2kitti_obj(self):
        """
        Converts nuScenes GT annotations to KITTI object detection format.
        """
        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse
        imsize = (1600, 900)

        token_idx = 0  # Start tokens from 0.

        # Get assignment of scenes to splits.
        split_logs = create_splits_logs(self.split, self.nusc)

        # Create output folders.
        split_file = os.path.join(self.nusc_kitti_root, 'object', 'split', '%s.txt' % self.split); mkdir_if_missing(split_file)
        split_file = open(split_file, 'w')
        label_folder = os.path.join(self.nusc_kitti_root, 'object', self.split, 'label_2')
        calib_folder = os.path.join(self.nusc_kitti_root, 'object', self.split, 'calib')
        image_folder = os.path.join(self.nusc_kitti_root, 'object', self.split, 'image_2')
        lidar_folder = os.path.join(self.nusc_kitti_root, 'object', self.split, 'velodyne')
        for folder in [label_folder, calib_folder, image_folder, lidar_folder]: mkdir_if_missing(folder)
        corres_file = os.path.join(self.nusc_kitti_root, 'object', self.split, 'correspondence.txt'); corres_file = open(corres_file, 'w')

        # Use only the samples from the current split.
        sample_tokens = split_to_samples(self.nusc, split_logs)
        num_samples = len(sample_tokens)

        tokens = []
        count = 0
        for sample_token in sample_tokens:

            # write id to the split file
            split_file.write('%06d\n' % count)
            corres_file.write('%06d %s\n' % (count, sample_token))

            # Get sample data.
            sample = self.nusc.get('sample', sample_token)
            sample_annotation_tokens = sample['anns']

            cam_front_token = sample['data'][self.cam_name]
            lidar_token = sample['data'][self.lidar_name]

            # Retrieve sensor records.
            sd_record_cam = self.nusc.get('sample_data', cam_front_token)
            sd_record_lid = self.nusc.get('sample_data', lidar_token)
            cs_record_cam = self.nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
            cs_record_lid = self.nusc.get('calibrated_sensor', sd_record_lid['calibrated_sensor_token'])

            # Combine transformations and convert to KITTI format.
            # Note: cam uses same conventions in KITTI and nuScenes.
            lid_to_ego = transform_matrix(cs_record_lid['translation'], Quaternion(cs_record_lid['rotation']), inverse=False)
            ego_to_cam = transform_matrix(cs_record_cam['translation'], Quaternion(cs_record_cam['rotation']), inverse=True)
            velo_to_cam = np.dot(ego_to_cam, lid_to_ego)

            # Convert from KITTI to nuScenes LIDAR coordinates, where we apply velo_to_cam.
            velo_to_cam_kitti = np.dot(velo_to_cam, kitti_to_nu_lidar.transformation_matrix)

            # Currently not used.
            imu_to_velo_kitti = np.zeros((3, 4))            # Dummy values.
            imu_to_velo_kitti[0, 0] = imu_to_velo_kitti[1, 1] = imu_to_velo_kitti[2, 2] = 1
            r0_rect = Quaternion(axis=[1, 0, 0], angle=0)   # Dummy values.

            # Projection matrix.
            p_left_kitti = np.zeros((3, 4))
            p_left_kitti[:3, :3] = cs_record_cam['camera_intrinsic']  # Cameras are always rectified.

            # Create KITTI style transforms.
            velo_to_cam_rot   = velo_to_cam_kitti[:3, :3]
            velo_to_cam_trans = velo_to_cam_kitti[:3, 3]

            # Check that the rotation has the same format as in KITTI.
            assert (velo_to_cam_rot.round(0) == np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])).all()
            assert (velo_to_cam_trans[1:3] < 0).all()

            # Retrieve the token from the lidar.
            # Note that this may be confusing as the filename of the camera will include the timestamp of the lidar,
            # not the camera.
            filename_cam_full = sd_record_cam['filename']
            filename_lid_full = sd_record_lid['filename']
            token_idx += 1

            # Convert image (jpg to png).
            src_im_path = os.path.join(self.nusc.dataroot, filename_cam_full)
            dst_im_path = os.path.join(image_folder, '%06d.png' % count)
            if not os.path.exists(dst_im_path):
                im = Image.open(src_im_path)
                im.save(dst_im_path, "PNG")

            # Convert lidar.
            # Note that we are only using a single sweep, instead of the commonly used n sweeps.
            # TODO, upgrade with n sweeps
            src_lid_path = os.path.join(self.nusc.dataroot, filename_lid_full)
            dst_lid_path = os.path.join(lidar_folder, '%06d.bin' % count)
            assert not dst_lid_path.endswith('.pcd.bin')
            pcl = LidarPointCloud.from_file(src_lid_path)
            pcl.rotate(kitti_to_nu_lidar_inv.rotation_matrix)  # In KITTI lidar frame.
            with open(dst_lid_path, "w") as lid_file:
                pcl.points.T.tofile(lid_file)

            # Add to tokens.
            tokens.append(sample_token)

            # Create calibration file.
            kitti_transforms = OrderedDict()
            kitti_transforms['P0'] = np.zeros((3, 4))   # Dummy values.
            kitti_transforms['P1'] = np.zeros((3, 4))   # Dummy values.
            kitti_transforms['P2'] = p_left_kitti       # Left camera transform.
            kitti_transforms['P3'] = np.zeros((3, 4))   # Dummy values.
            kitti_transforms['R0_rect'] = r0_rect.rotation_matrix  # Cameras are already rectified.
            kitti_transforms['Tr_velo_to_cam'] = np.hstack((velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1)))
            kitti_transforms['Tr_imu_to_velo'] = imu_to_velo_kitti
            calib_path = os.path.join(calib_folder, '%06d.txt' % count)
            with open(calib_path, "w") as calib_file:
                for (key, val) in kitti_transforms.items():
                    val = val.flatten()
                    val_str = '%.12e' % val[0]
                    for v in val[1:]:
                        val_str += ' %.12e' % v
                    calib_file.write('%s: %s\n' % (key, val_str))

            # Write label file.
            label_path = os.path.join(label_folder, '%06d.txt' % count)
            print('processing %d/%d, Writing file: %s' % (count+1, num_samples, label_path))
            with open(label_path, "w") as label_file:
                for sample_annotation_token in sample_annotation_tokens:
                    sample_annotation = self.nusc.get('sample_annotation', sample_annotation_token)

                    # Get box in LIDAR frame.
                    _, box_lidar_nusc, _ = self.nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE,
                                                                     selected_anntokens=[sample_annotation_token])
                    box_lidar_nusc = box_lidar_nusc[0]
                    
                    # Truncated: Set all objects to 0 which means untruncated.
                    truncated = 0.0

                    # Occluded: Set all objects to full visibility as this information is not available in nuScenes.
                    occluded = 0

                    # Convert nuScenes category to nuScenes detection challenge category.
                    detection_name = category_to_detection_name(sample_annotation['category_name'])

                    # Skip categories that are not part of the nuScenes detection challenge.
                    if detection_name is None: continue
                    detection_name = detection_name.capitalize()

                    # Convert from nuScenes to KITTI box format.
                    box_cam_kitti = KittiDB.box_nuscenes_to_kitti(
                        box_lidar_nusc, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect)

                    # Project 3d box to 2d box in image, ignore box if it does not fall inside.
                    bbox_2d = KittiDB.project_kitti_box_to_image(box_cam_kitti, p_left_kitti, imsize=imsize)
                    if bbox_2d is None:
                        bbox_2d = (-1, -1, -1, -1)          # add all anno to the converted version

                    # Set dummy score so we can use this file as result.
                    box_cam_kitti.score = 1.0

                    output = KittiDB.box_to_string(name=detection_name, box=box_cam_kitti, bbox_2d=bbox_2d,
                                                   truncation=truncated, occlusion=occluded)

                    # Write to disk.
                    label_file.write(output + '\n')

            count += 1

        split_file.close()

    def nuscenes_gt2kitti_trk(self):
        # Converts nuScenes GT annotations to KITTI tracking format.
        
        # Create output folders.
        split_file = os.path.join(self.nusc_kitti_root, 'tracking/produced/split', '%s.txt' % self.split)
        mkdir_if_missing(split_file); split_file = open(split_file, 'w')
        corres_folder = os.path.join(self.nusc_kitti_root, 'tracking/produced/correspondence', self.split); mkdir_if_missing(corres_folder)
        calib_folder  = os.path.join(self.nusc_kitti_root, 'tracking', self.split, 'calib'); mkdir_if_missing(calib_folder)
        label_folder  = os.path.join(self.nusc_kitti_root, 'tracking', self.split, 'label_02'); mkdir_if_missing(label_folder)
        oxts_folder   = os.path.join(self.nusc_kitti_root, 'tracking', self.split, 'oxts'); mkdir_if_missing(oxts_folder)
        evaluate_file = os.path.join(self.nusc_kitti_root, 'tracking', 'evaluate_tracking.seqmap.%s' % self.split)
        evaluate_file = open(evaluate_file, 'w')

        # go through every scene
        scene_splits = create_splits_scenes(verbose=False)
        scene_names = scene_splits[self.split]
        count_scene = 0
        for scene_name in scene_names:

            # create output folders that have subfolders at every scene
            label_obj_folder = os.path.join(self.nusc_kitti_root, 'tracking', self.split, 'label_2_object', scene_name)
            image_dir = os.path.join(self.nusc_kitti_root, 'tracking', self.split, 'image_02', scene_name)
            lidar_dir = os.path.join(self.nusc_kitti_root, 'tracking', self.split, 'velodyne', scene_name)
            for folder in [label_obj_folder, image_dir, lidar_dir]: mkdir_if_missing(folder)

            # create output files
            corres_file = os.path.join(corres_folder, scene_name+'.txt'); corres_file = open(corres_file, 'w')
            label_file = os.path.join(label_folder, scene_name+'.txt'); label_file = open(label_file, 'w')
            oxts_file = os.path.join(oxts_folder, scene_name+'.json')
            split_file.write('%s\n' % scene_name)

            # Use only the samples from the current sequence.
            sample_tokens = scene_to_samples(self.nusc, scene_name)
            tokens, instance_token_list, ego_pose_list = [], [], []
            frame_id = 0       # frame count in KITTI format
            sys.stdout.write('processing %s, %d/%d\r' % (scene_name, count_scene+1, len(scene_names)))
            sys.stdout.flush()

            # go through each frame
            for sample_token in sample_tokens:
                tokens.append(sample_token)

                # write id to the split file
                corres_file.write('%06d %s\n' % (frame_id, sample_token))

                # get sensor and transformation between KITTI
                pose_record, cs_record_lid, cs_record_cam, filename_lid_full, filename_cam_full = \
                    get_sensor_param(self.nusc, sample_token, cam_name=self.cam_name, output_file=True)
                velo_to_cam_trans, velo_to_cam_rot, r0_rect, p_left_kitti = \
                    nuScenes_transform2KITTI(cs_record_lid, cs_record_cam)

                # save calib
                calib_path = os.path.join(calib_folder, '%s.txt' % scene_name)
                kitti_transforms = create_KITTI_transform(velo_to_cam_trans, velo_to_cam_rot, r0_rect, p_left_kitti)
                save_calib_file(kitti_transforms, calib_path)

                # save sensor data
                save_image(self.nusc, filename_cam_full, image_dir, frame_id)
                save_lidar(self.nusc, filename_lid_full, lidar_dir, frame_id)

                # compute oxts 
                ego_pose = transform_matrix(pose_record['translation'], Quaternion(pose_record['rotation']), inverse=False)
                ego_pose_list.append(ego_pose)

                # Write label file at this frame
                sample = self.nusc.get('sample', sample_token)
                sample_annotation_tokens = sample['anns']
                label_obj_path = os.path.join(label_obj_folder, '%06d.txt' % frame_id)
                with open(label_obj_path, "w") as label_obj_file:
                    
                    # loop through each object annotation at this frame
                    for anno_token in sample_annotation_tokens:
                        output, ID = convert_anno_to_KITTI(self.nusc, anno_token, sample['data'][self.lidar_name], \
                            instance_token_list, velo_to_cam_trans, velo_to_cam_rot, r0_rect, p_left_kitti)
                        if output is None: continue 
                        label_obj_file.write(output + '\n')
                        label_file.write('%d %d %s\n' % (frame_id, ID, output))

                frame_id += 1

            count_scene += 1
            evaluate_file.write('%s empty 000000 %06d\n' % (scene_name, len(sample_tokens)))
            corres_file.close(); label_file.close()

            # save ego pose
            ego_pose_list = np.stack(ego_pose_list, axis=0).tolist()
            with open(oxts_file, 'w') as f:
                json.dump(ego_pose_list, f)

        split_file.close(); evaluate_file.close()

    def nuscenes_obj_result2kitti(self):
        # convert the detection results in NuScenes format to KITTI object format
        # for example, we will need this when using nuScenes detection results for tracking in the KITTI format

        # load correspondences
        corr_file = os.path.join(self.nusc_kitti_root, 'object', 'produced', 'correspondence', self.split+'.txt')
        corr_dict = load_correspondence(corr_file)

        # path
        save_dir = os.path.join(self.nusc_kitti_root, 'object', 'produced', 'results', self.split, self.result_name, 'data'); mkdir_if_missing(save_dir)

        # load results
        result_file = os.path.join(self.data_root, 'produced', 'results', 'detection', self.result_name, 'results_%s.json' % self.split)
        print('opening results file at %s' % (result_file))
        with open(result_file) as json_file:
            data = json.load(json_file)
            num_frames = len(data['results'])
            count = 0
            for sample_token, dets in data['results'].items():
                
                # get sensor and transformation
                pose_record, cs_record_lid, cs_record_cam = get_sensor_param(self.nusc, sample_token, cam_name=self.cam_name)
                velo_to_cam_trans, velo_to_cam_rot, r0_rect, p_left_kitti = \
                    nuScenes_transform2KITTI(cs_record_lid, cs_record_cam)

                # loop through every detection
                frame_index = corr_dict[sample_token]
                save_file = os.path.join(save_dir, frame_index+'.txt'); save_file = open(save_file, 'w')
                sys.stdout.write('processing results for %s.txt: %d/%d\r' % (frame_index, count, num_frames))
                sys.stdout.flush()
                for result_tmp in dets:
                    token_tmp = result_tmp['sample_token']
                    assert token_tmp == sample_token, 'token is different'
                    
                    # create nuScenes box in world coordinate
                    xyz = result_tmp['translation']                 # center_x, center_y, center_z
                    wlh = result_tmp['size']                        # width, length, height
                    rotation = result_tmp['rotation']               # quaternion in the global frame: w, x, y, z
                    name = result_tmp['detection_name'].capitalize()
                    box = Box(xyz, wlh, Quaternion(rotation), name=name, token=sample_token)        # box in global frame

                    # convert to nuScenes lidar coordinate
                    box = nuScenes_world2lidar(box, cs_record_lid, pose_record)  

                    # Convert from nuScenes lidar to KITTI camera format.
                    box_cam_kitti = KittiDB.box_nuscenes_to_kitti(
                        box, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect)

                    # Project 3d box to 2d box in image, ignore box if it does not fall inside.
                    bbox_2d = KittiDB.project_kitti_box_to_image(box_cam_kitti, p_left_kitti, imsize=(1600, 900))
                    if bbox_2d is None: bbox_2d = (-1, -1, -1, -1)          # add all anno to the converted version

                    truncated = 0.0
                    occluded = 0
                    box_cam_kitti.score = result_tmp['detection_score']

                    # KITTI format: type, trunc, occ, alpha, 2D bbox (x1, y1, x2, y2), 
                    # 3D bbox (h, w, l, x, y, z, ry), score
                    result_str = KittiDB.box_to_string(name=name, box=box_cam_kitti, bbox_2d=bbox_2d, truncation=truncated, occlusion=occluded)
                    save_file.write(result_str + '\n')

                save_file.close()
                count += 1

        print('Results saved to: %s' % save_dir)

    def nuscenes_trk_result2kitti(self):
        # convert the nuScenes tracking results in NuScenes format to KITTI format

        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse
        imsize = (1600, 900)

        # to avoid ID the same across categories
        offset = {'Car': 0, 'Bus': 200000, 'Truck': 300000, 'Trailer': 400000, 'Construction_vehicle': 500000, \
                  'Pedestrian': 700000, 'Bicycle': 900000, 'Motorcycle': 1000000}

        # load result file in nuScenes format
        result_file = os.path.join(self.data_root, 'produced/results/tracking', self.result_name, 'results_%s.json' % self.split)
        print('opening results file at %s' % (result_file))
        with open(result_file) as json_file:
            data = json.load(json_file)
        data = data['results']

        # load correspondence files
        save_dir = os.path.join(self.nusc_kitti_root, 'tracking/produced/results', self.split, self.result_name, 'data'); mkdir_if_missing(save_dir)
        correspondence_dir = os.path.join(self.nusc_kitti_root, 'tracking', self.split, 'correspondence')
        file_list, num_seq = load_list_from_folder(correspondence_dir)
        print('number of sequences to process is %d' % num_seq)

        # loop through each sequence
        count = 0
        for corr_file in file_list:
            _, seq_name, _ = fileparts(corr_file)
            corr_dict = self.load_correspondence(corr_file)     # NEED to run in python3 to make sure that the dictionary is ordered
            
            # open the single file for each sequence
            save_file = os.path.join(save_dir, seq_name+'.txt'); save_file = open(save_file, 'w')
            for sample_token, frame_index in corr_dict.items():

                # get token
                sample = self.nusc.get('sample', sample_token)
                cam_front_token = sample['data'][self.cam_name]
                lidar_token = sample['data'][self.lidar_name]

                # get LiDAR sensor
                sd_record_lid = self.nusc.get('sample_data', lidar_token)
                cs_record_lid = self.nusc.get('calibrated_sensor', sd_record_lid['calibrated_sensor_token'])
                sensor_record_lid = self.nusc.get('sensor', cs_record_lid['sensor_token'])
                pose_record_lid = self.nusc.get('ego_pose', sd_record_lid['ego_pose_token'])

                # get camera sensor
                sd_record_cam = self.nusc.get('sample_data', cam_front_token)
                cs_record_cam = self.nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])

                # Combine transformations and convert to KITTI format.
                # Note: cam uses same conventions in KITTI and nuScenes.
                lid_to_ego = transform_matrix(cs_record_lid['translation'], Quaternion(cs_record_lid['rotation']), inverse=False)
                ego_to_cam = transform_matrix(cs_record_cam['translation'], Quaternion(cs_record_cam['rotation']), inverse=True)
                velo_to_cam = np.dot(ego_to_cam, lid_to_ego)

                # Convert from KITTI to nuScenes LIDAR coordinates, where we apply velo_to_cam.
                velo_to_cam_kitti = np.dot(velo_to_cam, kitti_to_nu_lidar.transformation_matrix)

                # Currently not used.
                imu_to_velo_kitti = np.zeros((3, 4))  # Dummy values.
                r0_rect = Quaternion(axis=[1, 0, 0], angle=0)  # Dummy values.

                # Projection matrix.
                p_left_kitti = np.zeros((3, 4))
                p_left_kitti[:3, :3] = cs_record_cam['camera_intrinsic']  # Cameras are always rectified.

                # Create KITTI style transforms.
                velo_to_cam_rot = velo_to_cam_kitti[:3, :3]
                velo_to_cam_trans = velo_to_cam_kitti[:3, 3]

                # Check that the rotation has the same format as in KITTI.
                assert (velo_to_cam_rot.round(0) == np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])).all()
                assert (velo_to_cam_trans[1:3] < 0).all()

                # loop through result in a single frame
                data_frame = data[sample_token]
                sys.stdout.write('processing results for seq %s (%d/%d), frame %s\r' % (seq_name, count, num_seq, frame_index))
                sys.stdout.flush()
                conflict_dict = dict()
                for result_tmp in data_frame:
                    token_tmp = result_tmp['sample_token']
                    assert token_tmp == sample_token, 'token is different'

                    # print('%s, %s, %s' % (seq_name, frame_index, result_tmp['tracking_id']))

                    center = result_tmp['translation']              # center_x, center_y, center_z
                    size = result_tmp['size']                       # width, length, height
                    rotation = result_tmp['rotation']               # quaternion in the global frame: w, x, y, z
                    name = result_tmp['tracking_name'].capitalize()

                    # get object ID
                    if self.result_name in ['SimpleTrack']:
                        max_ID_per_seq = 1000
                        ID = max_ID_per_seq * int(float(result_tmp['tracking_id'].split('_')[1])) + int(float(result_tmp['tracking_id'].split('_')[2]))
                        ID += offset[name]
                    else:
                        ID = int(float(result_tmp['tracking_id']))      # work for EagerMOT & CBMOT
                    if ID in conflict_dict: continue
                    conflict_dict[ID] = 1

                    box = Box(center, size, Quaternion(rotation), name=name, token=sample_token)        # box in global frame

                    # Move box to ego vehicle coord system
                    box.translate(-np.array(pose_record_lid['translation']))
                    box.rotate(Quaternion(pose_record_lid['rotation']).inverse)

                    # Move box to sensor coord system, LiDAR frame
                    box.translate(-np.array(cs_record_lid['translation']))
                    box.rotate(Quaternion(cs_record_lid['rotation']).inverse)

                    # Convert from nuScenes to KITTI box format.
                    box_cam_kitti = KittiDB.box_nuscenes_to_kitti(
                        box, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect)

                    # Project 3d box to 2d box in image, ignore box if it does not fall inside.
                    bbox_2d = KittiDB.project_kitti_box_to_image(box_cam_kitti, p_left_kitti, imsize=imsize)
                    if bbox_2d is None:
                        bbox_2d = (-1, -1, -1, -1)          # add all anno to the converted version

                    truncated = 0.0
                    occluded = 0
                    box_cam_kitti.score = result_tmp['tracking_score']

                    # KITTI format: type, truncation, occlusion, alpha, 2D bbox (x1, y1, x2, y2), 3D bbox (h, w, l, x, y, z, ry), score
                    # result_str = '%s -1 -1 -10 %f %f %f %f %f %f %f %f %f %f %f %f\n' % (obj_type, 0, 0, 0, 0, height, width, length, x, y, z, ry, score)
                    result_str = KittiDB.box_to_string(name=name, box=box_cam_kitti, bbox_2d=bbox_2d, truncation=truncated, occlusion=occluded)

                    save_file.write('%d %d %s\n' % (int(frame_index), ID, result_str))

            save_file.close()
            count += 1

    def kitti_obj_result2nuscenes(self, meta: Dict[str, bool] = None) -> None:
        """
        Converts a KITTI detection result to the nuScenes detection results format.
        :param meta: Meta data describing the method used to generate the result. See nuscenes.org/object-detection.
        """
        
        results_dir = os.path.join(self.nusc_kitti_root, self.split, 'results', 'detection', self.result_name, 'data')
        min_score, max_score = get_min_max_score(results_dir)

        # Dummy meta data, please adjust accordingly.
        if meta is None:
            meta = {
                'use_camera': False,
                'use_lidar': True,
                'use_radar': False,
                'use_map': False,
                'use_external': False,
            }

        # Init.
        results = {}

        # Load the KITTI dataset.
        kitti = KittiDB(root=self.nusc_kitti_root, splits=(self.split, ), result_name=self.result_name)

        # Get assignment of scenes to splits.
        split_logs = create_splits_logs(self.split, self.nusc)

        # Use only the samples from the current split.
        sample_tokens = split_to_samples(self.nusc, split_logs)
        num_samples = len(sample_tokens)

        # get the correspondence between KITTI and NuScenes
        corr_file = os.path.join(self.nusc_kitti_root, self.split, 'correspondence.txt')
        corr_dict = self.load_correspondence(corr_file)

        # convert results
        count = 0
        for sample_token in sample_tokens:
            sys.stdout.write('converting %d/%d\r' % (count, num_samples))
            sys.stdout.flush()

            pose_record, cs_record = get_sensor_param(self.nusc, sample_token)

            # Get the KITTI boxes we just generated in LIDAR frame.
            sample_token_kitti = corr_dict[sample_token]
            kitti_token = '%s_%s' % (self.split, sample_token_kitti)
            boxes = kitti.get_boxes(token=kitti_token)
            
            # Convert KITTI boxes to nuScenes detection challenge result format.
            sample_results = list()
            for box in boxes:

                box = nuScenes_lidar2world(box, cs_record, pose_record)     # convert to nuScenes world

                # normalize score
                box.score = (box.score - min_score) / (max_score - min_score)

                sample_tmp = self.box_to_det_sample_result(sample_token, box)
                sample_results.append(sample_tmp)

            # Store all results for this image.
            results[sample_token] = sample_results
            count += 1

        # Store submission file to disk.
        submission = {
            'meta': meta,
            'results': results
        }
        submission_path = os.path.join(self.nusc_kitti_root, '../results/detection/%s/results_%s.json' % (self.result_name, self.split))
        print('Writing submission to: %s' % submission_path)
        with open(submission_path, 'w') as f:
            json.dump(submission, f, indent=2)

    def kitti_trk_result2nuscenes(self):
        # conver thr KITTI tracks to nuscenes tracks

        # Dummy meta data, please adjust accordingly.
        meta = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
            }

        # Init.
        results = {}

        # path
        tmp_root_dir = os.path.join(self.nusc_kitti_root, 'tracking', self.split)
        results_dir = os.path.join(self.result_root, self.result_name, 'data_0')
        corres_dir = os.path.join(tmp_root_dir, '../produced/correspondence', self.split)
        calib_dir = os.path.join(tmp_root_dir, 'calib')

        # loop over all sequences
        corres_list, num_list = load_list_from_folder(corres_dir)
        for corres_file in corres_list:
            seq_name = fileparts(corres_file)[1]
            sys.stdout.write('converting %s\r' % (seq_name))
            sys.stdout.flush()

            # initialize empty list for all frames
            corr_dict = load_correspondence_inverse(corres_file)    
            for sample_token in corr_dict.values():
                results[sample_token] = list()

            # load calibration
            calib = os.path.join(calib_dir, seq_name+'.txt')
            calib = Calibration(calib)

            # load results in the KITTI format
            result_file = os.path.join(results_dir, seq_name+'.txt')
            KITTI_results = Tracklet_3D(result_file)
            tracklet_data = KITTI_results.data

            # loop through every frame
            for frame, frame_data in tracklet_data.items():

                # retrieve nuScenes data at this frame
                frame_index = '%06d' % frame
                sample_token = corr_dict[frame_index]
                pose_record, cs_record = get_sensor_param(self.nusc, sample_token)

                # loop over every tracks
                for id_tmp, obj in frame_data.items():

                    # obtain nuScenes box
                    box = create_nuScenes_box(obj)      # initialize from KITTI format
                    box = kitti_cam2nuScenes_lidar(box, calib)      # convert to nuScenes lidar
                    box = nuScenes_lidar2world(box, cs_record, pose_record)     # convert to nuScenes world

                    # convert to nuscenes format
                    sample_result = box_to_trk_sample_result(sample_token, box, trk_id=id_tmp)
                    results[sample_token].append(sample_result)

        # Store submission file to disk.
        submission = {
            'meta': meta,
            'results': results
        }
        submission_path = os.path.join(self.result_root, self.result_name, 'results_%s.json' % (self.split))
        mkdir_if_missing(submission_path)
        print('Writing submission to: %s' % submission_path)
        with open(submission_path, 'w') as f:
            json.dump(submission, f, indent=2)

if __name__ == '__main__':
    fire.Fire(KittiConverter)