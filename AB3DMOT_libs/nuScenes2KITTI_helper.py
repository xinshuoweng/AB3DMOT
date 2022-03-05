import numpy as np, copy, os
from PIL import Image
from collections import OrderedDict
from pyquaternion import Quaternion
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.tracking.utils import category_to_tracking_name
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.kitti import KittiDB
from nuscenes.utils.geometry_utils import transform_matrix, BoxVisibility
from xinshuo_io import load_txt_file

def load_correspondence(corr_file):
    data, num_line = load_txt_file(corr_file)
    data_dict = dict()
    for line_data in data:
        index, token = line_data.split(' ')
        data_dict[token] = index

    return data_dict

def load_correspondence_inverse(corr_file):
    data, num_line = load_txt_file(corr_file)
    data_dict = dict()
    for line_data in data:
        index, token = line_data.split(' ')
        data_dict[index] = token

    return data_dict

###################### nuScenes/KITTI data conversion

kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)

def kitti_cam2nuScenes_lidar(box, calib):
    # convert box in KITTI camera coordinate to nuScenes lidar coordinate

    # Copy box to avoid side-effects.
    box = copy.deepcopy(box)

    # Translate: KITTI defines the box center as the bottom center of the object. We use true center,
    # so we need to add half height in negative y direction, (since y points downwards), to adjust. The
    # center is already given in camera coord system.
    box.translate(np.array([0, -box.wlh[2] / 2, 0]))

    # Transform from KITTI cam to KITTI LIDAR coord system. First transform 
    # from rectified camera to camera, then camera to KITTI lidar.
    box.rotate(Quaternion(matrix=calib.R0).inverse)
    box.translate(-calib.V2C_T)
    box.rotate(Quaternion(matrix=calib.V2C_R).inverse)

    # Rotate to nuscenes lidar.
    box.rotate(kitti_to_nu_lidar)

    return box

def nuScenes_transform2KITTI(cs_record_lid, cs_record_cam):
    # Combine transformations and convert to KITTI format.
    # Note: cam uses same conventions in KITTI and nuScenes.

    # obtain nuScenes transformation
    lid_to_ego = transform_matrix(cs_record_lid['translation'], Quaternion(cs_record_lid['rotation']), inverse=False)
    ego_to_cam = transform_matrix(cs_record_cam['translation'], Quaternion(cs_record_cam['rotation']), inverse=True)
    lid_to_cam_nuScenes = np.dot(ego_to_cam, lid_to_ego)

    # nuScenes lidar coordinate to KITTI camera
    velo_to_cam_kitti = np.dot(lid_to_cam_nuScenes, kitti_to_nu_lidar.transformation_matrix)
    velo_to_cam_rot   = velo_to_cam_kitti[:3, :3]
    velo_to_cam_trans = velo_to_cam_kitti[:3, 3]

    # Check that the rotation has the same format as in KITTI.
    assert (velo_to_cam_rot.round(0) == np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])).all()
    assert (velo_to_cam_trans[1:3] < 0).all()

    # Dummy values.    
    r0_rect = Quaternion(axis=[1, 0, 0], angle=0) 

    # Projection matrix.
    p_left_kitti = np.zeros((3, 4))
    p_left_kitti[:3, :3] = cs_record_cam['camera_intrinsic']  # Cameras are always rectified.

    return velo_to_cam_trans, velo_to_cam_rot, r0_rect, p_left_kitti

def convert_anno_to_KITTI(nusc, anno_token, lidar_token, instance_token_list, \
    velo_to_cam_trans, velo_to_cam_rot, r0_rect, p_left_kitti):

    sample_annotation = nusc.get('sample_annotation', anno_token)

    # add instance token to the list if not yet, create ID for each object
    instance_token = sample_annotation['instance_token']
    if instance_token in instance_token_list:
        ID = instance_token_list.index(instance_token)        # 0-indexed ID
    else:
        ID = len(instance_token_list)
        instance_token_list.append(instance_token)

    # Get box in LIDAR frame.
    _, box_lidar_nusc, _ = nusc.get_sample_data(lidar_token, \
        box_vis_level=BoxVisibility.NONE, selected_anntokens=[anno_token])
    box_lidar_nusc = box_lidar_nusc[0]

    # Convert nuScenes category to nuScenes detection challenge category.
    # Skip categories that are not part of the nuScenes detection challenge.
    obj_name = category_to_tracking_name(sample_annotation['category_name'])
    if obj_name is None: return None, -1
    obj_name = obj_name.capitalize()

    # Convert from nuScenes to KITTI box format.
    box_cam_kitti = KittiDB.box_nuscenes_to_kitti(
        box_lidar_nusc, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect)
    box_cam_kitti.score = 1.0

    # Project 3d box to 2d box in image, ignore box if it does not fall inside.
    bbox_2d = KittiDB.project_kitti_box_to_image(box_cam_kitti, p_left_kitti, imsize=(1600, 900))
    if bbox_2d is None: bbox_2d = (-1, -1, -1, -1)

    # Convert box to output string format.
    output = KittiDB.box_to_string(name=obj_name, box=box_cam_kitti, \
        bbox_2d=bbox_2d, truncation=0.0, occlusion=0)           # no occlusion/truncation data is available

    return output, ID

###################### nuScenes/KITTI saving

def create_KITTI_transform(velo_to_cam_trans, velo_to_cam_rot, r0_rect, p_left_kitti):
    # create KITTI transform from nuScenes data

    # dummy value
    imu_to_velo_kitti = np.zeros((3, 4))  # Dummy values.
    imu_to_velo_kitti[0, 0] = imu_to_velo_kitti[1, 1] = imu_to_velo_kitti[2, 2] = 1

    # save to KITTI format
    kitti_transforms = OrderedDict()
    kitti_transforms['P0'] = np.zeros((3, 4))   # Dummy values.
    kitti_transforms['P1'] = np.zeros((3, 4))   # Dummy values.
    kitti_transforms['P2'] = p_left_kitti       # Left camera transform.
    kitti_transforms['P3'] = np.zeros((3, 4))   # Dummy values.
    kitti_transforms['R0_rect'] = r0_rect.rotation_matrix  # Cameras are already rectified.
    kitti_transforms['Tr_velo_to_cam'] = np.hstack((velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1)))
    kitti_transforms['Tr_imu_to_velo'] = imu_to_velo_kitti

    return kitti_transforms

def save_lidar(nusc, filename_lid_full, lidar_dir, count):
    # save lidar, note that we are only using a single sweep, instead of the commonly used n sweeps.

    src_lid_path = os.path.join(nusc.dataroot, filename_lid_full)
    dst_lid_path = os.path.join(lidar_dir, '%06d.bin' % count)
    assert not dst_lid_path.endswith('.pcd.bin')
    pcl = LidarPointCloud.from_file(src_lid_path)
    pcl.rotate(kitti_to_nu_lidar.inverse.rotation_matrix)  # In KITTI lidar frame.
    with open(dst_lid_path, "w") as lid_file:
        pcl.points.T.tofile(lid_file)

def save_image(nusc, filename_cam_full, image_dir, count):
    # save image (jpg to png).

    src_im_path = os.path.join(nusc.dataroot, filename_cam_full)
    dst_im_path = os.path.join(image_dir, '%06d.png' % count)
    if not os.path.exists(dst_im_path):
        im = Image.open(src_im_path)
        im.save(dst_im_path, "PNG")