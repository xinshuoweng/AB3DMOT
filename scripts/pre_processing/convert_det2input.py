# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# combine the detection txt from each frame to a single txt per sequence including detection results at all frames
# used to create input files for AB3DMOT

import numpy as np, os, argparse
from AB3DMOT_libs.utils import Config, get_subfolder_seq
from AB3DMOT_libs.nuScenes2KITTI_helper import load_correspondence, load_correspondence_inverse
from AB3DMOT_libs.kitti_obj import read_label
from xinshuo_io import mkdir_if_missing, load_list_from_folder, load_txt_file, fileparts, is_path_exists

def parse_args():
    parser = argparse.ArgumentParser(description='AB3DMOT')
    parser.add_argument('--dataset', type=str, default='nuScenes', help='KITTI, nuScenes')
    parser.add_argument('--split', type=str, default='val', help='train, val, test')
    parser.add_argument('--det_name', type=str, default='centerpoint', help='name of the detection method')
    args = parser.parse_args()
    return args

def combine_dets(dataset, split, det_name):
	
	# source dir
	subfolder, det_id2str, _, seq_eval, data_root = get_subfolder_seq(dataset, split)
	det_str2id = {v: k for k, v in det_id2str.items()}

	# find results dir that contain detections in KITTI object format
	det_results = os.path.join(data_root, 'object/produced/results', subfolder, det_name, 'data')
	print('processing %s, %s, %s' % (dataset, det_name, split))
	if not is_path_exists(det_results):
		print('%s dir not exist' % det_results)
		return

	# find correspondences between detection frame ID and actual nuScenes sequence name
	det_corr_file = os.path.join(data_root, 'object/produced/correspondence', subfolder+'.txt')
	det_corr = load_correspondence(det_corr_file) 		# key sample_token, value frame
	trk_corr_dir  = os.path.join(data_root, 'tracking/produced/correspondence', subfolder)

	# save dir
	save_root = os.path.join('./data', dataset, 'detection')

	# loop through each sequence
	for seq in seq_eval:

		# create save data for each category
		save_file = dict()
		for cat in list(det_id2str.values()) + ['all']:
			save_file[cat] = os.path.join(save_root, '%s_%s_%s/%s.txt' % (det_name, cat, split, seq))
			mkdir_if_missing(save_file[cat]); save_file[cat] = open(save_file[cat], 'w')

		# for each frame of a sequence, find the synthesized frame ID in the nuKITTI object data
		trk_corr_file = os.path.join(trk_corr_dir, seq+'.txt')
		trk_corr = load_correspondence_inverse(trk_corr_file)
		for frame, sample_token in trk_corr.items():
			det_frame = det_corr[sample_token]

			# load detection results at this frame
			det_result_file = os.path.join(det_results, det_frame+'.txt')
			dets = read_label(det_result_file)

			# loop over each detection
			for obj in dets:
				type_tmp = det_str2id[obj.type]
				str_to_write = obj.convert_to_trk_input_str(frame, type_tmp)

				# save to the corresponding category and also an overall folder
				save_file[obj.type].write(str_to_write + '\n')
				save_file['all'].write(str_to_write + '\n')

		# close files
		for cat in save_file.keys():
			save_file[cat].close()

if __name__ == '__main__':	

	args = parse_args()
	combine_dets(args.dataset, args.split, args.det_name)