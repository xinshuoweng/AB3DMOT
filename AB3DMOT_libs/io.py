# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import warnings, numpy as np, os
from xinshuo_io import mkdir_if_missing

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

def get_frame_det(dets_all, frame):
	
	# get irrelevant information associated with an object, not used for associationg
	ori_array = dets_all[dets_all[:, 0] == frame, -1].reshape((-1, 1))		# orientation
	other_array = dets_all[dets_all[:, 0] == frame, 1:7] 					# other information, e.g, 2D box, ...
	additional_info = np.concatenate((ori_array, other_array), axis=1)		

	# get 3D box
	dets = dets_all[dets_all[:, 0] == frame, 7:14]		

	dets_frame = {'dets': dets, 'info': additional_info}
	return dets_frame

def get_saving_dir(eval_dir_dict, seq_name, save_dir, num_hypo):

	# create dir and file for saving
	eval_file_dict, save_trk_dir = dict(), dict()
	for index in range(num_hypo):
		eval_file_dict[index] = os.path.join(eval_dir_dict[index], seq_name + '.txt')
		eval_file_dict[index] = open(eval_file_dict[index], 'w')
		save_trk_dir[index] = os.path.join(save_dir, 'trk_withid_%d' % index, seq_name); mkdir_if_missing(save_trk_dir[index])
	affinity_dir = os.path.join(save_dir, 'affi', seq_name); mkdir_if_missing(affinity_dir)
	affinity_vis = os.path.join(save_dir, 'affi_vis', seq_name); mkdir_if_missing(affinity_vis)

	return eval_file_dict, save_trk_dir, affinity_dir, affinity_vis

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

def save_affinity(affi_data, save_path):
	######### save txt files for faster check, with aligned formatting

	# compute the number of digit for the largest values for better alignment of saving
	min_val, max_val = np.min(affi_data), np.max(affi_data)
	biggest = max(abs(min_val), abs(max_val))
	num_digit = 0
	while True:
		if biggest < 1: break
		num_digit += 1
		biggest = biggest / 10.0
	
	# see if there is a negative sign, so need to a one more digit 
	negative = False
	if min_val < 0: negative = True
	if negative: num_digit += 1

	# add digits depending on the decimals we want to preserve
	decimals = 2
	num_digit += decimals + 1 		# meaning that we want to preserve the dot plus the decimals

	# save
	fmt = '%%%d.%df' % (num_digit, decimals)
	np.savetxt(save_path, affi_data, fmt=fmt, delimiter=', ')