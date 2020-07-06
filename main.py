# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

from __future__ import print_function
import matplotlib; matplotlib.use('Agg')
import os, numpy as np, time, sys
from AB3DMOT_libs.model import AB3DMOT
from xinshuo_io import load_list_from_folder, fileparts, mkdir_if_missing

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print('Usage: python main.py result_sha(e.g., pointrcnn_Car_test)')
		sys.exit(1)

	result_sha = sys.argv[1]
	save_root = './results'
	det_id2str = {1:'Pedestrian', 2:'Car', 3:'Cyclist'}

	seq_file_list, num_seq = load_list_from_folder(os.path.join('data/KITTI', result_sha))
	total_time, total_frames = 0.0, 0
	save_dir = os.path.join(save_root, result_sha); mkdir_if_missing(save_dir)
	eval_dir = os.path.join(save_dir, 'data'); mkdir_if_missing(eval_dir)
	seq_count = 0
	for seq_file in seq_file_list:
		_, seq_name, _ = fileparts(seq_file)
		eval_file = os.path.join(eval_dir, seq_name + '.txt'); eval_file = open(eval_file, 'w')
		save_trk_dir = os.path.join(save_dir, 'trk_withid', seq_name); mkdir_if_missing(save_trk_dir)

		mot_tracker = AB3DMOT() 
		seq_dets = np.loadtxt(seq_file, delimiter=',')          # load detections, N x 15
		
		# if no detection in a sequence
		if len(seq_dets.shape) == 1: seq_dets = np.expand_dims(seq_dets, axis=0) 	
		if seq_dets.shape[1] == 0:
			eval_file.close()
			continue

		# loop over frame
		min_frame, max_frame = int(seq_dets[:, 0].min()), int(seq_dets[:, 0].max())
		for frame in range(min_frame, max_frame + 1):
			# logging
			print_str = 'processing %s: %d/%d, %d/%d   \r' % (seq_name, seq_count, num_seq, frame, max_frame)
			sys.stdout.write(print_str)
			sys.stdout.flush()
			save_trk_file = os.path.join(save_trk_dir, '%06d.txt' % frame); save_trk_file = open(save_trk_file, 'w')

			# get irrelevant information associated with an object, not used for associationg
			ori_array = seq_dets[seq_dets[:, 0] == frame, -1].reshape((-1, 1))		# orientation
			other_array = seq_dets[seq_dets[:, 0] == frame, 1:7] 		# other information, e.g, 2D box, ...
			additional_info = np.concatenate((ori_array, other_array), axis=1)		

			dets = seq_dets[seq_dets[:,0] == frame, 7:14]            # h, w, l, x, y, z, theta in camera coordinate follwing KITTI convention
			dets_all = {'dets': dets, 'info': additional_info}

			# important
			start_time = time.time()
			trackers = mot_tracker.update(dets_all)
			cycle_time = time.time() - start_time
			total_time += cycle_time

			# saving results, loop over each tracklet			
			for d in trackers:
				bbox3d_tmp = d[0:7]       # h, w, l, x, y, z, theta in camera coordinate
				id_tmp = d[7]
				ori_tmp = d[8]
				type_tmp = det_id2str[d[9]]
				bbox2d_tmp_trk = d[10:14]
				conf_tmp = d[14]

				# save in detection format with track ID, can be used for dection evaluation and tracking visualization
				str_to_srite = '%s -1 -1 %f %f %f %f %f %f %f %f %f %f %f %f %f %d\n' % (type_tmp, ori_tmp,
					bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
					bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[6], conf_tmp, id_tmp)
				save_trk_file.write(str_to_srite)

				# save in tracking format, for 3D MOT evaluation
				str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (frame, id_tmp, 
					type_tmp, ori_tmp, bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
					bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[6], 
					conf_tmp)
				eval_file.write(str_to_srite)

			total_frames += 1
			save_trk_file.close()
		seq_count += 1
		eval_file.close()    
	print('Total Tracking took: %.3f for %d frames or %.1f FPS' % (total_time, total_frames, total_frames / total_time))