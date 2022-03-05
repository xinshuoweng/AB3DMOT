import numpy as np, copy
from typing import List, Dict, Any
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from AB3DMOT_libs.kitti_obj import Object_3D

# general helper function used to load nuScenes data

def get_min_max_score(results_dir):
    min_score = 10000
    max_score = -10000
    
    results_list, num_files = load_list_from_folder(results_dir)
    for result_file_tmp in results_list:
        data_all, num_line = load_txt_file(result_file_tmp)
        for data_line in data_all:
            score = float(data_line.split(' ')[-1])
            if score > max_score: max_score = score
            if score < min_score: min_score = score

    return min_score, max_score

def split_to_samples(nusc, split_logs: List[str]) -> List[str]:
    """
    Convenience function to get the samples in a particular split.
    :param split_logs: A list of the log names in this split.
    :return: The list of samples.
    """
    samples = []
    count = 0
    count_filtered = 0
    for sample in nusc.sample:
        scene = nusc.get('scene', sample['scene_token'])
        log = nusc.get('log', scene['log_token'])
        logfile = log['logfile']
        if logfile in split_logs:
            samples.append(sample['token'])

    return samples

def scene_to_samples(nusc, scene_name: str) -> List[str]:
    # obtain the samples in the order from a given scene

    scene_token = nusc.field2token('scene', 'name', scene_name)[0]
    scene = nusc.get('scene', scene_token)
    samples = []

    # get the first sample in the sequence
    first_sample_token = scene['first_sample_token']
    sample = nusc.get('sample', first_sample_token)
    samples.append(first_sample_token)

    # loop over the sequence
    while 1:
        next_sample_token = sample['next']
        sample = nusc.get('sample', next_sample_token)
        samples.append(sample['token'])
        if sample['next'] is '': break

    return samples

###################### nuScenes box class

def create_nuScenes_box(obj: Object_3D):
    # for loading box in KITTI format      

    quat = Quaternion(axis=(0, 1, 0), angle=obj.ry) * Quaternion(axis=(1, 0, 0), angle=np.pi/2)
    box = Box(obj.xyz, obj.wlh, quat, name=obj.type.lower())

    # assign other info
    box.score = obj.s
    if obj.velo_3d is not None:
        box.velocity = np.array(obj.velo_3d)
    else:
        box.velocity = np.array([0.0, 0.0, 0.0])

    return box

###################### nuScenes coordinate transform

def get_sensor_param(nusc, sample_token, lidar_name='LIDAR_TOP', cam_name=None, output_file=False):
    # get the sensor parameters for box transformation later
    # cs_record:    transformation between lidar sensor and ego coordinate
    # pose_record:  transformation between ego and world coordinate

    sample = nusc.get('sample', sample_token)

    # get lidar sensor parameter
    lidar_token = sample['data'][lidar_name]
    sd_record_lid = nusc.get('sample_data', lidar_token)
    cs_record_lid = nusc.get('calibrated_sensor', sd_record_lid['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record_lid['ego_pose_token'])
    # pose_record of lidar and camera is very close, although not the same

    if cam_name is None:
        return pose_record, cs_record_lid
    else:
        # get camera sensor
        cam_token = sample['data'][cam_name]
        sd_record_cam = nusc.get('sample_data', cam_token)
        cs_record_cam = nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])

        # output the original file name/path
        if output_file:
            filename_cam_full = sd_record_cam['filename']
            filename_lid_full = sd_record_lid['filename']
            return pose_record, cs_record_lid, cs_record_cam, filename_lid_full, filename_cam_full
        else:
            return pose_record, cs_record_lid, cs_record_cam

def nuScenes_lidar2world(box, cs_record=None, pose_record=None, \
    sample_token=None, nusc=None, lidar_name='LIDAR_TOP'):
    """
    Transform from nuScenes lidar frame to nuScenes world coordinate.
    """

    box = copy.deepcopy(box)

    if cs_record is None or pose_record is None:
        assert nusc is not None and sample_token is not None, 'error, all data is none for transformation'
        cs_record, pose_record = get_sensor_param(nusc, sample_token)

    # lidar to ego
    box.rotate(Quaternion(cs_record['rotation']))
    box.translate(np.array(cs_record['translation']))

    # ego to global
    box.rotate(Quaternion(pose_record['rotation']))
    box.translate(np.array(pose_record['translation']))

    return box

def nuScenes_world2lidar(box, cs_record=None, pose_record=None, \
    sample_token=None, nusc=None, lidar_name='LIDAR_TOP'):
    """
    Transform from nuScenes lidar frame to nuScenes world coordinate.
    """

    box = copy.deepcopy(box)

    if cs_record is None or pose_record is None:
        assert nusc is not None and sample_token is not None, 'error, all data is none for transformation'
        cs_record, pose_record = get_sensor_param(nusc, sample_token)

    # global to ego
    box.translate(-np.array(pose_record['translation']))
    box.rotate(Quaternion(pose_record['rotation']).inverse)
    
    # ego to lidar
    box.translate(-np.array(cs_record['translation']))
    box.rotate(Quaternion(cs_record['rotation']).inverse)
         
    return box

#################### output formatting

def box_to_sample_result(sample_token: str, box: Box, attribute_name: str = '') -> Dict[str, Any]:
    
    # Prepare box data
    translation = box.center
    size = box.wlh
    rotation = box.orientation.q
    velocity = box.velocity
    detection_name = box.name
    detection_score = box.score

    # Create result dict
    sample_result = dict()
    sample_result['sample_token'] = sample_token
    sample_result['translation'] = translation.tolist()
    sample_result['size'] = size.tolist()
    sample_result['rotation'] = [item * -1 for item in rotation.tolist()]
    sample_result['velocity'] = velocity.tolist()[:2]  # Only need vx, vy.
    sample_result['attribute_name'] = attribute_name

    return sample_result

def box_to_det_sample_result(sample_token: str, box: Box, attribute_name: str = '') -> Dict[str, Any]:
    
    sample_result = box_to_sample_result(sample_token, box, attribute_name)
    sample_result['detection_name'] = box.name
    sample_result['detection_score'] = box.score

    return sample_result

def box_to_trk_sample_result(sample_token: str, box: Box, attribute_name: str = '', \
    trk_id: int = -1) -> Dict[str, Any]:
    
    sample_result = box_to_sample_result(sample_token, box, attribute_name)
    sample_result['tracking_score'] = box.score
    sample_result['tracking_name'] = box.name
    sample_result['tracking_id'] = trk_id    

    return sample_result