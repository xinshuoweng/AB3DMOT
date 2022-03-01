import numpy as np, json
from numba import jit
from xinshuo_io import fileparts

@jit
def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])

@jit
def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

@jit
def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

@jit
def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

@jit
def _poses_from_oxts(oxts_packets):

    """Helper method to compute SE(3) pose matrices from OXTS packets."""
    # https://github.com/pratikac/kitti/blob/master/pykitti/raw.py
        
    er = 6378137.  # earth radius (approx.) in meters

    # compute scale from first lat value
    scale = np.cos(oxts_packets[0].lat * np.pi / 180.)

    t_0 = []    # initial position
    poses = []  # list of poses computed from oxts
    for packet in oxts_packets:
        # Use a Mercator projection to get the translation vector
        tx = scale * packet.lon * np.pi * er / 180.
        ty = scale * er * \
            np.log(np.tan((90. + packet.lat) * np.pi / 360.))
        tz = packet.alt
        t = np.array([tx, ty, tz])

        # We want the initial position to be the origin, but keep the ENU
        # coordinate system
        if len(t_0) == 0:
            t_0 = t

        # Use the Euler angles to get the rotation matrix
        Rx = rotx(packet.roll)
        Ry = roty(packet.pitch)
        Rz = rotz(packet.yaw)
        R = Rz.dot(Ry.dot(Rx))

        # Combine the translation and rotation into a homogeneous transform
        poses.append(transform_from_rot_trans(R, t - t_0))      # store transformation matrix

    return np.stack(poses)

def load_oxts(oxts_file):
    """Load OXTS data from file."""
    # https://github.com/pratikac/kitti/blob/master/pykitti/raw.py

    ext = fileparts(oxts_file)[-1]
    if ext == '.json':        # loading for nuScenes-to-KITTI data
        with open(oxts_file, 'r') as file: 
            imu_poses = json.load(file)
            imu_poses = np.array(imu_poses)

        return imu_poses

    # Extract the data from each OXTS packe per dataformat.txt
    from collections import namedtuple
    OxtsPacket = namedtuple('OxtsPacket',
                            'lat, lon, alt, ' +
                            'roll, pitch, yaw, ' +
                            'vn, ve, vf, vl, vu, ' +
                            'ax, ay, az, af, al, au, ' +
                            'wx, wy, wz, wf, wl, wu, ' +
                            'pos_accuracy, vel_accuracy, ' +
                            'navstat, numsats, ' +
                            'posmode, velmode, orimode')

    oxts_packets = []
    with open(oxts_file, 'r') as f:
        for line in f.readlines():
            line = line.split()
            # Last five entries are flags and counts
            line[:-5] = [float(x) for x in line[:-5]]
            line[-5:] = [int(float(x)) for x in line[-5:]]

            data = OxtsPacket(*line)
            oxts_packets.append(data)

    # Precompute the IMU poses in the world frame
    imu_poses = _poses_from_oxts(oxts_packets)      # seq_frames x 4 x 4

    return imu_poses
    
def get_ego_traj(imu_poses, frame, pref, futf, inverse=False, only_fut=False):
    # compute the motion of the ego vehicle for ego-motion compensation
    # using the current frame as the coordinate
    # current frame means one frame prior to future, and also the last frame of the past
    
    # compute the start and end frame to retrieve the imu poses
    num_frames = imu_poses.shape[0]
    assert frame >= 0 and frame <= num_frames - 1, 'error'
    if inverse:             # pre and fut are inverse, i.e., inverse ego motion compensation
        start = min(frame+pref-1, num_frames-1)
        end   = max(frame-futf-1, -1)
        index = [*range(start, end, -1)]
    else:
        start = max(frame-pref+1, 0)
        end   = min(frame+futf+1, num_frames)
        index = [*range(start, end)]
    
    # compute frame offset due to sequence boundary
    left  = start - (frame-pref+1)
    right = (frame+futf+1) - end

    # compute relative transition compared to the current frame of the ego
    all_world_xyz = imu_poses[index, :3, 3]    # N x 3, only translation, frame = 10-19 for fut only (0-19 for all)
    cur_world_xyz = imu_poses[frame]                        # 4 x 4, frame = 9
    T_world2imu   = np.linalg.inv(cur_world_xyz)            
    all_world_hom = np.concatenate((all_world_xyz, np.ones((all_world_xyz.shape[0], 1))), axis=1)       # N x 4
    all_xyz = all_world_hom.dot(T_world2imu.T)[:, :3]       # N x 3

    # compute relative rotation compared to the current frame of the ego
    all_world_rot = imu_poses[index, :3, :3]   # N x 3 x 3, only rotation
    cur_world_rot = imu_poses[frame, :3, :3]                # 3 x 3, frame = 9
    T_world2imu_rot = np.linalg.inv(cur_world_rot)        
    all_rot_list = list()
    for frame in range(all_world_rot.shape[0]):
        all_rot_tmp = all_world_rot[frame].dot(T_world2imu_rot)   # 3 x 3
        all_rot_list.append(all_rot_tmp)
    
    if only_fut:
        fut_xyz, fut_rot_list = all_xyz[pref-left:], all_rot_list[pref-left:]
        return fut_xyz, fut_rot_list, left, right
    else:
        return all_xyz, all_rot_list, left, right

def egomotion_compensation_ID(traj_id, calib, ego_rot_imu, ego_xyz_imu, left, right, mask=None):
    # traj_id           # N x 3
    # ego_imu can have frames less than pre+fut due to sequence boundary

    # convert trajectory data from rect to IMU for ego-motion compensation
    traj_id_imu = calib.rect_to_imu(traj_id)        # less_pre x 3

    if mask is not None:
        good_index = np.where(mask == 1)[0]
        good_index = (good_index - left).tolist()
        ego_rot_imu = np.array(ego_rot_imu)
        ego_rot_imu = ego_rot_imu[good_index, :].tolist()

    # correct rotation
    for frame in range(traj_id_imu.shape[0]):
        traj_id_imu[frame, :] = np.matmul(ego_rot_imu[frame], traj_id_imu[frame, :].reshape((3, 1))).reshape((3, ))

    # correct transition
    if mask is not None:
        traj_id_imu += ego_xyz_imu[good_index, :]   # les_frames x 3, TODO, need to test which is correct
    else:
        traj_id_imu += ego_xyz_imu[:traj_id_imu.shape[0], :]   # les_frames x 3

    # convert trajectory data back to rect coordinate for visualization
    traj_id_rect = calib.imu_to_rect(traj_id_imu)

    return traj_id_rect