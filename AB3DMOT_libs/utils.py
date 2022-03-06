# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import yaml, numpy as np, os
from easydict import EasyDict as edict
# from AB3DMOT_libs.model_multi import AB3DMOT_multi
from AB3DMOT_libs.model import AB3DMOT
from AB3DMOT_libs.kitti_oxts import load_oxts
from AB3DMOT_libs.kitti_calib import Calibration
from AB3DMOT_libs.nuScenes_split import get_split
from xinshuo_io import mkdir_if_missing, is_path_exists, fileparts, load_list_from_folder
from xinshuo_miscellaneous import merge_listoflist

def Config(filename):
    listfile1 = open(filename, 'r')
    listfile2 = open(filename, 'r')
    cfg = edict(yaml.safe_load(listfile1))
    settings_show = listfile2.read().splitlines()

    listfile1.close()
    listfile2.close()

    return cfg, settings_show

def get_subfolder_seq(dataset, split):

	# dataset setting
	file_path = os.path.dirname(os.path.realpath(__file__))
	if dataset == 'KITTI':				# KITTI
		det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
		
		if split == 'val': subfolder = 'training' 
		elif split == 'test': subfolder = 'testing' 
		else: assert False, 'error'

		hw = {'image': (375, 1242), 'lidar': (720, 1920)}
		
		if split == 'train': seq_eval = ['0000', '0002', '0003', '0004', '0005', '0007', '0009', '0011', '0017', '0020']         # train
		if split == 'val':   seq_eval = ['0001', '0006', '0008', '0010', '0012', '0013', '0014', '0015', '0016', '0018', '0019']    # val
		if split == 'test':  seq_eval  = ['%04d' % i for i in range(29)]
	
		data_root = os.path.join(file_path, '../data/KITTI') 		# path containing the KITTI root 

	elif dataset == 'nuScenes':			# nuScenes
		det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Bicycle', 4: 'Motorcycle', 5: 'Bus', \
			6: 'Trailer', 7: 'Truck', 8: 'Construction_vehicle', 9: 'Barrier', 10: 'Traffic_cone'}

		subfolder = split
		hw = {'image': (900, 1600), 'lidar': (720, 1920)}

		if split == 'train': seq_eval = get_split()[0]		# 700 scenes
		if split == 'val':   seq_eval = get_split()[1]		# 150 scenes
		if split == 'test':  seq_eval = get_split()[2]      # 150 scenes

		data_root = os.path.join(file_path, '../data/nuScenes/nuKITTI') 	# path containing the nuScenes-converted KITTI root

	else: assert False, 'error, %s dataset is not supported' % dataset
		
	return subfolder, det_id2str, hw, seq_eval, data_root

def get_threshold(dataset, det_name):
	# used for visualization only as we want to remove some false positives, also can be 
	# used for KITTI 2D MOT evaluation which uses a single operating point 
	# obtained by observing the threshold achieving the highest MOTA on the validation set

	if dataset == 'KITTI':
		if det_name == 'pointrcnn': return {'Car': 3.240738, 'Pedestrian': 2.683133, 'Cyclist': 3.645319}
		else: assert False, 'error, detection method not supported for getting threshold' % det_name
	elif dataset == 'nuScenes':
		if det_name == 'megvii': 
			return {'Car': 0.262545, 'Pedestrian': 0.217600, 'Truck': 0.294967, 'Trailer': 0.292775, 
					'Bus': 0.440060, 'Motorcycle': 0.314693, 'Bicycle': 0.284720}
		if det_name == 'centerpoint': 
			return {'Car': 0.269231, 'Pedestrian': 0.410000, 'Truck': 0.300000, 'Trailer': 0.372632, 
					'Bus': 0.430000, 'Motorcycle': 0.368667, 'Bicycle': 0.394146}
		else: assert False, 'error, detection method not supported for getting threshold' % det_name
	else: assert False, 'error, dataset %s not supported for getting threshold' % dataset

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
	if cfg.num_hypo > 1:
		tracker = AB3DMOT_multi(cfg, cat, calib=calib, oxts=imu_poses, img_dir=img_seq, vis_dir=vis_dir, hw=hw, log=log_file, ID_init=ID_start) 
	elif cfg.num_hypo == 1:
		tracker = AB3DMOT(cfg, cat, calib=calib, oxts=imu_poses, img_dir=img_seq, vis_dir=vis_dir, hw=hw, log=log_file, ID_init=ID_start) 
	else: assert False, 'error'
	
	# compute the min/max frame
	frame_list, _ = load_list_from_folder(img_seq)
	frame_list = [fileparts(frame_file)[1] for frame_file in frame_list]

	return tracker, frame_list

def find_all_frames(root_dir, subset, data_suffix, seq_list):
	# warm up to find union of all frames from results of all categories in all sequences
	# finding the union is important because there might be some sequences with only cars while
	# some other sequences only have pedestrians, so we may miss some results if mainly looking
	# at one single category
	# return a dictionary with each key correspondes to the list of frame ID

	# loop through every sequence
	frame_dict = dict()
	for seq_tmp in seq_list:
		frame_all = list()

		# find all frame indexes for each category
		for subset_tmp in subset:
			data_dir = os.path.join(root_dir, subset_tmp, 'trk_withid'+data_suffix, seq_tmp)			# pointrcnn_ped
			if not is_path_exists(data_dir):
				print('%s dir not exist' % data_dir)
				assert False, 'error'

			# extract frame string from this category
			frame_list, _ = load_list_from_folder(data_dir)
			frame_list = [fileparts(frame_tmp)[1] for frame_tmp in frame_list]
			frame_all.append(frame_list)
		
		# merge frame indexes from all categories
		frame_all = merge_listoflist(frame_all, unique=True)
		frame_dict[seq_tmp] = frame_all

	return frame_dict