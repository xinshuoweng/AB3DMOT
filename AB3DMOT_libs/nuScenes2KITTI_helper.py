import numpy as np, copy
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
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

###################### nuScenes/KITTI coordinate conversion

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

def box_nuscenes_to_kitti_inv(box: Box, velo_to_cam_rot: Quaternion,
    velo_to_cam_trans: np.ndarray, r0_rect: Quaternion,
    kitti_to_nu_lidar_inv: Quaternion = Quaternion(axis=(0, 0, 1), angle=np.pi / 2).inverse) \
        -> Box:
    """
    Transform from nuScenes lidar frame to KITTI reference frame.
    :param box: Instance in nuScenes lidar frame.
    :param velo_to_cam_rot: Quaternion to rotate from lidar to camera frame.
    :param velo_to_cam_trans: <np.float: 3>. Translate from lidar to camera frame.
    :param r0_rect: Quaternion to rectify camera frame.
    :param kitti_to_nu_lidar_inv: Quaternion to rotate nuScenes to KITTI LIDAR.
    :return: Box instance in KITTI reference frame.
    """

    # Copy box to avoid side-effects.
    box = box.copy()

    # KITTI defines the box center as the bottom center of the object.
    # We use the true center, so we need to adjust half height in y direction.
    box.translate(np.array([0, -box.wlh[2] / 2, 0]))

    # Rotate to KITTI rectified camera.
    box.rotate(r0_rect.inverse)
    
    # Transform to KITTI lidar.
    box.translate(velo_to_cam_trans * -1)
    box.rotate(velo_to_cam_rot.inverse)

    # Rotate to nuscenes lidar.
    box.rotate(kitti_to_nu_lidar_inv.inverse)

    return box