# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import yaml, numpy as np, warnings
from easydict import EasyDict as edict

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