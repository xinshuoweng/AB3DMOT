# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import yaml, numpy as np, warnings, os
from easydict import EasyDict as edict
# from AB3DMOT_libs.model_multi import AB3DMOT_multi
from AB3DMOT_libs.model import AB3DMOT
from AB3DMOT_libs.kitti_oxts import load_oxts
from AB3DMOT_libs.kitti_calib import Calibration
from xinshuo_io import mkdir_if_missing, is_path_exists

def Config(filename):
    listfile1 = open(filename, 'r')
    listfile2 = open(filename, 'r')
    cfg = edict(yaml.safe_load(listfile1))
    settings_show = listfile2.read().splitlines()

    listfile1.close()
    listfile2.close()

    return cfg, settings_show

def load_detection(file):

	# load from raw file
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		dets = np.loadtxt(file, delimiter=',') 	# load detections, N x 15

	if len(dets.shape) == 1: dets = np.expand_dims(dets, axis=0) 	
	if dets.shape[1] == 0:		# if no detection in a sequence
		return [], False
	else:
		return dets, True

def get_subfolder_seq(dataset, split):

	# dataset setting
	if dataset == 'KITTI':				# KITTI
		det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
		
		if split == 'val': subfolder = 'training' 
		elif split == 'test': subfolder = 'testing' 
		else: assert False, 'error'
		
		hw = (720, 1920)
		
		if split == 'train': seq_eval = ['0000', '0002', '0003', '0004', '0005', '0007', '0009', '0011', '0017', '0020']         # train
		if split == 'val':   seq_eval = ['0001', '0006', '0008', '0010', '0012', '0013', '0014', '0015', '0016', '0018', '0019']    # val
		if split == 'test':  seq_eval  = ['%04d' % i for i in range(29)]
	
	elif dataset == 'nuScenes':			# nuScenes
		det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Bicycle', 4: 'Motorcycle', 5: 'Bus', 6: 'Trailer', 7: 'Truck'}
		
		subfolder = split
		hw = (900, 1600)
		
		if split == 'train': seq_eval = get_split()[0]		# 700 scenes
		if split == 'val':   seq_eval = get_split()[1]		# 150 scenes
		if split == 'test':  seq_eval = get_split()[2]      # 150 scenes
	else: assert False, 'error'
		
	return subfolder, det_id2str, hw, seq_eval

def get_saving_dir(eval_dir_dict, seq_name, save_dir, num_hypo):

	# create dir and file for saving
	eval_file_dict, save_trk_dir = dict(), dict()
	for index in range(num_hypo):
		eval_file_dict[index] = os.path.join(eval_dir_dict[index], seq_name + '.txt')
		eval_file_dict[index] = open(eval_file_dict[index], 'w')
		save_trk_dir[index] = os.path.join(save_dir, 'trk_withid_%d' % index, seq_name); mkdir_if_missing(save_trk_dir[index])

	return eval_file_dict, save_trk_dir

def initialize(cfg, data_root, save_dir, subfolder, seq_name, cat, ID_start, hw, log_file):
	# initialize the tracker and provide all path of data needed

	oxts_dir  = os.path.join(data_root, subfolder, 'oxts')
	calib_dir = os.path.join(data_root, subfolder, 'calib')
	image_dir = os.path.join(data_root, subfolder, 'image_02')

	# load ego poses
	oxts = os.path.join(data_root, subfolder, 'oxts', seq_name+'.json')
	if not is_path_exists(oxts): oxts = os.path.join(data_root, subfolder, 'oxts', seq_name+'.txt')
	imu_poses = load_oxts(oxts)                 # seq_frames x 4 x 4

	# load calibration
	calib = os.path.join(data_root, subfolder, 'calib', seq_name+'.txt')
	calib = Calibration(calib)

	# load image for visualization
	img_seq = os.path.join(data_root, subfolder, 'image_02', seq_name)
	vis_dir = os.path.join(save_dir, 'vis_debug', seq_name); mkdir_if_missing(vis_dir)

	# initiate the tracker
	if cfg.hypothesis > 1:
		tracker = AB3DMOT_multi(cfg, cat, calib=calib, oxts=imu_poses, img_dir=img_seq, vis_dir=vis_dir, hw=hw, log=log_file, ID_init=ID_start) 
	elif cfg.hypothesis == 1:
		tracker = AB3DMOT(cfg, cat, calib=calib, oxts=imu_poses, img_dir=img_seq, vis_dir=vis_dir, hw=hw, log=log_file, ID_init=ID_start) 
	else: assert False, 'error'
	
	return tracker

def get_frame_det(dets_all, frame):
	
	# get irrelevant information associated with an object, not used for associationg
	ori_array = dets_all[dets_all[:, 0] == frame, -1].reshape((-1, 1))		# orientation
	other_array = dets_all[dets_all[:, 0] == frame, 1:7] 					# other information, e.g, 2D box, ...
	additional_info = np.concatenate((ori_array, other_array), axis=1)		

	# get 3D box
	dets = dets_all[dets_all[:, 0] == frame, 7:14]		

	dets_frame = {'dets': dets, 'info': additional_info}
	return dets_frame

def save_results(res, save_trk_file, eval_file, det_id2str, frame, score_threshold):

	# box3d in the format of h, w, l, x, y, z, theta in camera coordinate
	bbox3d_tmp, id_tmp, ori_tmp, type_tmp, bbox2d_tmp_trk, conf_tmp = \
		res[0:7], res[7], res[8], det_id2str[res[9]], res[10:14], res[14] 		
	 
	# save in detection format with track ID, can be used for dection evaluation and tracking visualization
	str_to_srite = '%s -1 -1 %f %f %f %f %f %f %f %f %f %f %f %f %f %d\n' % (type_tmp, ori_tmp,
		bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
		bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[6], conf_tmp, id_tmp)
	save_trk_file.write(str_to_srite)

	# save in tracking format, for 3D MOT evaluation
	if conf_tmp >= score_threshold:
		str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (frame, id_tmp, 
			type_tmp, ori_tmp, bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
			bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[6], conf_tmp)
		eval_file.write(str_to_srite)