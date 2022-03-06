# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os, numpy as np, sys, argparse
from AB3DMOT_libs.kitti_calib import Calibration
from AB3DMOT_libs.kitti_obj import read_label
from AB3DMOT_libs.utils import get_subfolder_seq, get_threshold
from AB3DMOT_libs.io import load_highlight
from AB3DMOT_libs.vis import vis_image_with_obj
from xinshuo_io import is_path_exists, mkdir_if_missing, load_list_from_folder, fileparts
from xinshuo_miscellaneous import print_log
from xinshuo_video import generate_video_from_folder

def parse_args():
    parser = argparse.ArgumentParser(description='AB3DMOT')
    parser.add_argument('--result_sha', type=str, default='pointrcnn_Car_test_thres', help='name of the result folder')
    parser.add_argument('--dataset', type=str, default='KITTI', help='KITTI, nuScenes')
    parser.add_argument('--split', type=str, default='test', help='train, val, test')
    parser.add_argument('--hypo_index_vis', type=int, default=0, help='the index of hypothesis results for visualization')
    parser.add_argument('--highlight_file', type=str, default=None, help='error to be highlighted')
    args = parser.parse_args()
    return args

def vis(args):
	
	# get config
	file_path = os.path.dirname(os.path.realpath(__file__))
	result_sha = args.result_sha
	det_name = result_sha.split('_')[0]
	result_root = os.path.join(file_path, '../../results', args.dataset, result_sha)
	score_threshold = get_threshold(args.dataset, det_name)
	log = os.path.join(result_root, 'vis_log.txt')
	mkdir_if_missing(log); log = open(log, 'w')

	# load highlighting data if there is
	if args.highlight_file is not None: hl_data_dict = load_highlight(args.highlight_file)
	else: hl_data_dict = None

	# get directory
	subfolder, det_id2str, hw, seq_eval, data_root = get_subfolder_seq(args.dataset, args.split)
	trk_root = os.path.join(data_root, 'tracking', subfolder)

	# change to the local mini dataset for a quick demo
	if not is_path_exists(trk_root):
		print_log('full %s dataset does not exist at %s, fall back to mini dataset for a quick demo' % \
			(args.dataset, trk_root), log=log)
		trk_root = os.path.join(file_path, '../../data/%s/mini' % args.dataset, subfolder)
		assert is_path_exists(trk_root), 'error, unfortunately mini data is missing at %s as well' % trk_root

		# assign sequence in mini data for evaluation on different dataset/split
		if args.dataset == 'KITTI':
			if args.split == 'val' : seq_eval = ['0001', '0016']
			if args.split == 'test': seq_eval = ['0000', '0003']
		# elif args.dataset == 'nuScenes':
		else: assert False, 'mini data does not support for %s-%s' % (args.dataset, args.split)

	# loop through every sequence
	seq_count = 0
	for seq in seq_eval:
		image_dir = os.path.join(trk_root, 'image_02/%s' % seq)
		calib_file = os.path.join(trk_root, 'calib/%s.txt' % seq)
		result_dir = os.path.join(result_root, 'trk_withid_%d/%s' % (args.hypo_index_vis, seq))
		save_3d_bbox_dir = os.path.join(result_dir, '../../trk_image_vis/%s' % seq); mkdir_if_missing(save_3d_bbox_dir)

		# load highlight data this sequence
		if hl_data_dict is not None and seq_count in hl_data_dict: 
			data_hl = hl_data_dict[seq_count]
		else: 					      
			data_hl = None

		# load the list
		images_list, num_images = load_list_from_folder(image_dir)
		print_log('seq %s, number of images to visualize is %d' % (seq, num_images), log=log)
		start_count = 0
		for count in range(start_count, num_images):
			image_tmp = images_list[count]
			if not is_path_exists(image_tmp): 
				count += 1
				continue
			image_index = int(fileparts(image_tmp)[1])

			# load results
			result_tmp = os.path.join(result_dir, '%06d.txt'%image_index)		# load the result
			if not is_path_exists(result_tmp): object_res = []
			else: object_res = read_label(result_tmp)
			print_log('processing index: %d, %d/%d, results from %s' % (image_index, count+1, num_images, result_tmp), log=log, display=False)
			
			# load calibration
			calib_tmp = Calibration(calib_file)			# load the calibration

			# load highlight data this frame
			if data_hl is not None and image_index in data_hl.keys(): id_hl = data_hl[image_index]
			else: id_hl = None

			# filter objects based on object categories and score
			filtered = []
			for object_tmp in object_res:
				obj_type = object_tmp.type
				if obj_type not in det_id2str.values(): continue
				if hasattr(object_tmp, 'score'):
					if object_tmp.score < score_threshold[obj_type]: continue
				filtered.append(object_tmp)

			# visualization and save
			save_path = os.path.join(save_3d_bbox_dir, '%06d.jpg' % (image_index))
			vis_image_with_obj(image_tmp, filtered, [], calib_tmp, hw, save_path=save_path, id_hl=id_hl)
			print_log('number of objects to plot is %d' % (len(filtered)), log=log, display=False)
			count += 1

		# generate video for the image results
		print_log('generating video for seq %s' % (seq), log=log)
		video_file = os.path.join(result_root, 'trk_video_vis', seq+'.mp4'); mkdir_if_missing(video_file)
		if args.dataset == 'KITTI': framerate = 30
		elif args.dataset == 'nuScenes': framerate = 2
		generate_video_from_folder(save_3d_bbox_dir, video_file, framerate=framerate)
		seq_count += 1

if __name__ == "__main__":

	args = parse_args()
	vis(args)