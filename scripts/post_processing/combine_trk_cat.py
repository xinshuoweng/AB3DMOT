# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# combine tracking results from different categories 

import os, argparse
from AB3DMOT_libs.io import combine_files
from AB3DMOT_libs.utils import find_all_frames, get_subfolder_seq, Config
from xinshuo_io import mkdir_if_missing, is_path_exists

def parse_args():
    parser = argparse.ArgumentParser(description='AB3DMOT')
    parser.add_argument('--det_name', type=str, default='pointrcnn', help='we provide pointrcnn on KITTI, megvii for nuScenes')
    parser.add_argument('--dataset', type=str, default='KITTI', help='nuScenes, KITTI')
    parser.add_argument('--split', type=str, default='val', help='train, val, test')
    parser.add_argument('--suffix', type=str, default='H1', help='additional string of the folder to be combined')
    parser.add_argument('--num_hypo', type=int, default=1, help='number of hypothesis to combine')
    args = parser.parse_args()
    return args

def combine_trk_cat(split, dataset, method, suffix, num_hypo):

	# load dataset-specific config
	file_path = os.path.dirname(os.path.realpath(__file__))
	root_dir = os.path.join(file_path, '../../results', dataset)
	_, det_id2str, _, seq_list, _ = get_subfolder_seq(dataset, split)

	# load config files
	config_path = os.path.join(file_path, '../../configs/%s.yml' % dataset)
	cfg, _ = Config(config_path)
	log = os.path.join(root_dir, '%s_%s_%s' % (method, split, suffix), 'combine_log.txt')
	mkdir_if_missing(log); log = open(log, 'w+')

	# source directory
	subset = ['%s_%s_%s_%s' % (method, cat, split, suffix) for cat in cfg.cat_list]

	# loop through all hypotheses
	for hypo_index in range(num_hypo):
		data_suffix = '_%d' % hypo_index
		frame_dict = find_all_frames(root_dir, subset, data_suffix, seq_list)

		############ merge for 3D MOT evaluation
		save_root = os.path.join(root_dir, '%s_%s_%s' % (method, split, suffix), 'data'+data_suffix); mkdir_if_missing(save_root)
		for seq_tmp in seq_list:
			file_list_tmp = list()

			# loop through each category
			for subset_tmp in subset:
				file_tmp = os.path.join(root_dir, subset_tmp, 'data'+data_suffix, seq_tmp+'.txt')
				file_list_tmp.append(file_tmp)

			save_path_tmp = os.path.join(save_root, seq_tmp+'.txt')
			combine_files(file_list_tmp, save_path_tmp)

		############ merge for trk_withid, for detection evaluation and 3D MOT visualization
		save_root = os.path.join(root_dir, '%s_%s_%s' % (method, split, suffix), 'trk_withid'+data_suffix)
		for seq_tmp in seq_list:
			
			save_dir = os.path.join(save_root, seq_tmp); mkdir_if_missing(save_dir)
			for frame_tmp in frame_dict[seq_tmp]:
				file_list_tmp = list()
				for subset_tmp in subset:
					file_tmp = os.path.join(root_dir, subset_tmp, 'trk_withid'+data_suffix, seq_tmp, frame_tmp+'.txt')
					if is_path_exists(file_tmp): file_list_tmp.append(file_tmp)

				save_path_tmp = os.path.join(save_dir, frame_tmp+'.txt')
				combine_files(file_list_tmp, save_path_tmp, sort=False)

if __name__ == '__main__':
	args = parse_args()
	combine_trk_cat(args.split, args.dataset, args.det_name, args.suffix, args.num_hypo)