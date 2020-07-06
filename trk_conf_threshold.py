# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os, sys
from xinshuo_io import load_txt_file, load_list_from_folder, mkdir_if_missing, fileparts

def conf_thresholding(data_dir, save_dir, score_threshold):
	
	# collect all trajectories
	tracker_id_score = dict()
	eval_dir = os.path.join(data_dir, 'data')
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
	save_dir_tmp = os.path.join(save_dir, 'data'); mkdir_if_missing(save_dir_tmp)
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
	trk_id_dir = os.path.join(data_dir, 'trk_withid')
	seq_dir_list, num_seq = load_list_from_folder(trk_id_dir)
	save_dir_tmp = os.path.join(save_dir, 'trk_withid')
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
	if len(sys.argv) != 2:
		print('Usage: python trk_conf_threshold.py result_sha(e.g., pointrcnn_Car_test_thres)')
		sys.exit(1)

	result_sha = sys.argv[1]

	cat = result_sha.split('_')[1]
	if cat == 'Car':
		score_threshold = 2.917300
	elif cat == 'Pedestrian':
		score_threshold = 2.070726
	else: assert False, 'error'

	root_dir = './results'
	data_dir = os.path.join(root_dir, result_sha)
	save_dir = os.path.join(root_dir, result_sha+'_thres'); mkdir_if_missing(save_dir)

	conf_thresholding(data_dir, save_dir, score_threshold)
