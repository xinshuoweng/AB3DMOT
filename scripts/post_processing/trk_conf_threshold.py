# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os, sys, argparse
from AB3DMOT_libs.utils import get_threshold
from xinshuo_io import load_txt_file, load_list_from_folder, mkdir_if_missing, fileparts

def parse_args():
    parser = argparse.ArgumentParser(description='AB3DMOT')
    parser.add_argument('--result_sha', type=str, default='pointrcnn_Car_test_thres', help='name of the result folder')
    parser.add_argument('--dataset', type=str, default='nuScenes', help='KITTI, nuScenes')
    parser.add_argument('--num_hypo', type=int, default=1, help='number of hypothesis used')
    args = parser.parse_args()
    return args

def conf_thresholding(data_dir, save_dir, score_threshold, num_hypo):
	
	# loop through all hypotheses
	for hypo_index in range(num_hypo):
		
		# collect all trajectories
		tracker_id_score = dict()
		eval_dir = os.path.join(data_dir, 'data_%d' % (hypo_index))
		seq_list, num_seq = load_list_from_folder(eval_dir)
		for seq_file in seq_list:
			seq_data, num_line = load_txt_file(seq_file)
			for data_line in seq_data:
				data_split = data_line.split(' ')
				score_tmp = float(data_split[-1])
				id_tmp = int(data_split[1])
				if id_tmp not in tracker_id_score.keys():
					tracker_id_score[id_tmp] = list()
				tracker_id_score[id_tmp].append(score_tmp)

		# collect the ID to remove based on the confidence
		to_delete_id = list()
		for track_id, score_list in tracker_id_score.items():
			average_score = sum(score_list) / float(len(score_list))
			if average_score < score_threshold:
				to_delete_id.append(track_id)

		# remove the ID in the data folder for tracking evaluation
		save_dir_tmp = os.path.join(save_dir, 'data_%d' % (hypo_index)); mkdir_if_missing(save_dir_tmp)
		for seq_file in seq_list:
			seq_name = fileparts(seq_file)[1]
			seq_file_save = os.path.join(save_dir_tmp, seq_name+'.txt'); seq_file_save = open(seq_file_save, 'w')

			seq_data, num_line = load_txt_file(seq_file)
			for data_line in seq_data:
				data_split = data_line.split(' ')
				id_tmp = int(float(data_split[1]))
				if id_tmp not in to_delete_id:
					seq_file_save.write(data_line + '\n')
			seq_file_save.close()

		# remove the ID in the trk with id folder for detection evaluation and tracking visualization
		trk_id_dir = os.path.join(data_dir, 'trk_withid_%d' % (hypo_index))
		seq_dir_list, num_seq = load_list_from_folder(trk_id_dir)
		save_dir_tmp = os.path.join(save_dir, 'trk_withid_%d' % (hypo_index))
		for seq_dir in seq_dir_list:
			frame_list, num_frame = load_list_from_folder(seq_dir)
			seq_name = fileparts(seq_dir)[1]
			save_frame_dir = os.path.join(save_dir_tmp, seq_name); mkdir_if_missing(save_frame_dir)
			for frame in frame_list:
				frame_index = fileparts(frame)[1]
				frame_file_save = os.path.join(save_frame_dir, frame_index+'.txt'); frame_file_save = open(frame_file_save, 'w')	
				frame_data, num_line = load_txt_file(frame)
				for data_line in frame_data:
					data_split = data_line.split(' ')
					id_tmp = int(data_split[-1])
					if id_tmp not in to_delete_id:
						frame_file_save.write(data_line + '\n')
				frame_file_save.close()

if __name__ == '__main__':
	
	# get config
	args = parse_args()
	result_sha = args.result_sha
	cat = result_sha.split('_')[1]
	num_hypo = args.num_hypo
	score_threshold = get_threshold(args.dataset)[cat]

	# get directories
	root_dir = os.path.join('./results', args.dataset)
	data_dir = os.path.join(root_dir, result_sha)
	save_dir = os.path.join(root_dir, result_sha+'_thres'); mkdir_if_missing(save_dir)

	# run thresholding
	conf_thresholding(data_dir, save_dir, score_threshold, num_hypo)