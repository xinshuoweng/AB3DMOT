# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import yaml
from easydict import EasyDict as edict
# from xinshuo_io import save_txt_file

def Config(filename):
    listfile1 = open(filename, 'r')
    listfile2 = open(filename, 'r')
    cfg = edict(yaml.safe_load(listfile1))
    settings_show = listfile2.read().splitlines()

    listfile1.close()
    listfile2.close()

    return cfg, settings_show

# def combine_files(file_list, save_path):

# 	# with open(save_path, 'w'):
# 	data_all = list()
# 	for file_tmp in file_list:
# 		data, num_lines = load_txt_file(file_tmp)
# 		data_all += data

# 	data_all.sort(key = lambda x: int(x.split(',')[0]))

# 	new_data_all = list()
# 	for data_line in data_all:
# 		data_tmp = data_line.split(',')
# 		frame = data_tmp[0]
# 		class_num = data_tmp[1]
# 		box_2d = data_tmp[2:6]
# 		conf = data_tmp[6]
# 		box_3d = data_tmp[7:14]
# 		alpha = data_tmp[14]

# 		new_data_line = ' '.join([frame, '-1', class_num, '-1', '-1', alpha] + box_2d + box_3d + [conf])
# 		new_data_all.append(new_data_line)

# 	save_txt_file(new_data_all, save_path)
