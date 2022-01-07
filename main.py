# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

from __future__ import print_function
import matplotlib; matplotlib.use('Agg')
import os, numpy as np, time, sys, argparse
from AB3DMOT_libs.utils import load_detection, Config, get_subfolder_seq, initialize, get_saving_dir, get_frame_det, save_results
from xinshuo_io import load_list_from_folder, fileparts, mkdir_if_missing
from xinshuo_miscellaneous import get_timestring, print_log

def parse_args():

    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='AB3DMOT')
    parser.add_argument('config', type=str, default='KITTI', help='name of the detection folder')
    args = parser.parse_args()
    return args

def main_per_cat(cfg, cat, log, ID_start):

	# get data-cat-split specific path
	result_sha = '%s_%s_%s' % (cfg.det_method, cat, cfg.split)
	det_root = os.path.join('./data', cfg.dataset, result_sha)
	trk_root = os.path.join('./data', cfg.dataset, 'tracking')
	subfolder, det_id2str, hw, seq_eval = get_subfolder_seq(cfg.dataset, cfg.split)
	save_dir = os.path.join(cfg.save_root, result_sha+'_H%d'%cfg.hypothesis); mkdir_if_missing(save_dir)

	# create eval dir for each hypothesis 
	eval_dir_dict = dict()
	for index in range(cfg.hypothesis):
		eval_dir_dict[index] = os.path.join(save_dir, 'data_H%d' % index); mkdir_if_missing(eval_dir_dict[index]) 		

	# loop every sequence
	seq_count = 0
	total_time, total_frames = 0.0, 0
	for seq_name in seq_eval:
		seq_file = os.path.join(det_root, seq_name+'.txt')
		seq_dets, flag = load_detection(seq_file) 				# load detection
		if not flag: continue									# no detection

		# create folders for saving
		eval_file_dict, save_trk_dir = get_saving_dir(eval_dir_dict, seq_name, save_dir, cfg.hypothesis)	

		# initialize tracker
		tracker = initialize(cfg, trk_root, save_dir, subfolder, seq_name, cat, ID_start, hw, log)

		# loop over frame
		min_frame, max_frame = int(seq_dets[:, 0].min()), int(seq_dets[:, 0].max())
		for frame in range(min_frame, max_frame + 1):
			
			# logging
			print_str = 'processing %s %s: %d/%d, %d/%d   \r' % (result_sha, seq_name, seq_count, len(seq_eval), frame, max_frame)
			sys.stdout.write(print_str)
			sys.stdout.flush()

			# tracking by detection
			dets_frame = get_frame_det(seq_dets, frame)
			since = time.time()
			results, affi = tracker.track(dets_frame, frame, seq_name)		
			total_time += time.time() - since

			# saving results, loop over each hypothesis
			for hypo in range(cfg.hypothesis):
				save_trk_file = os.path.join(save_trk_dir[hypo], '%06d.txt' % frame); save_trk_file = open(save_trk_file, 'w')
				for result_tmp in results[hypo]:				# N x 15
					save_results(result_tmp, save_trk_file, eval_file_dict[hypo], det_id2str, frame, cfg.score_threshold)
				save_trk_file.close()

			total_frames += 1
		seq_count += 1

		for index in range(cfg.hypothesis): 
			eval_file_dict[index].close()
			ID_start = max(ID_start, tracker.ID_count[index])

	print('%s, %25s: %4.f seconds for %5d frames or %6.1f FPS, metric is %s = %.2f' % \
		(cfg.dataset, result_sha, total_time, total_frames, total_frames / total_time, tracker.metric, tracker.thres))
	
	return ID_start

def main(args):

	# load config files
	config_path = './configs/%s.yml' % args.config
	cfg, settings_show = Config(config_path)
	time_str = get_timestring()
	log = open(os.path.join(cfg.save_root, 'log_%s_%s_%s.txt' % (time_str, cfg.dataset, cfg.split)), 'w')
	for idx, data in enumerate(settings_show):
		print_log(data, log, display=False)

	# global ID counter used for all categories, not start from 0 for each category to prevent different 
	# categories of objects have the same ID. This allows visualization of all object categories together
	# without ID conflicting
	ID_start = 0 								

	# run tracking for each category
	for cat in cfg.cat_list:
		ID_start = main_per_cat(cfg, cat, log, ID_start)
	log.close()

if __name__ == '__main__':

	args = parse_args()
	
	# check arguments
	if len(sys.argv) != 2:
		print('Usage: python main.py config (e.g., kitti)')
		sys.exit(1)

	main(args)