# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# combine KITTI tracking results from two categories for submission

import os
from xinshuo_io import load_txt_file, mkdir_if_missing, save_txt_file

def combine_files(file_list, save_path):

	# collect all files
	data_all = list()
	for file_tmp in file_list:
		data, num_lines = load_txt_file(file_tmp)
		data_all += data

	# sort based on frame number
	data_all.sort(key = lambda x: int(x.split(' ')[0]))

	save_txt_file(data_all, save_path)

if __name__ == '__main__':
	
	root_dir = './results'
	split = 'test'
	seq_list = ['%04d' % tmp for tmp in range(0, 29)]
	method = 'pointrcnn'
	cat_list = ['Car', 'Pedestrian']		# no cyclist due to KITTI does not include in MOT benchmarks
	subset = ['%s_%s_%s_thres' % (method, cat, split) for cat in cat_list]

	# save path
	save_dir = os.path.join(root_dir, '%s_%s_thres' % (method, split), 'data'); mkdir_if_missing(save_dir)

	# merge
	for seq_tmp in seq_list:
		file_list_tmp = list()
		for subset_tmp in subset:
			file_tmp = os.path.join(root_dir, subset_tmp, 'data', seq_tmp+'.txt')
			file_list_tmp.append(file_tmp)

		save_path_tmp = os.path.join(save_dir, seq_tmp+'.txt')
		combine_files(file_list_tmp, save_path_tmp)