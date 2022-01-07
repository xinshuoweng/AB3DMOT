# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import numpy as np, os, copy
from AB3DMOT_libs.bbox_utils import convert_3dbox_to_8corner, associate_detections_to_trackers
from AB3DMOT_libs.kalman_filter import KF
from xinshuo_miscellaneous import print_log
from xinshuo_io import mkdir_if_missing

np.set_printoptions(suppress=True, precision=3)

class AB3DMOT(object):			  	# A baseline of 3D multi-object tracking
	def __init__(self, cfg, cat, calib=None, oxts=None, img_dir=None, vis_dir=None, hw=None, log=None, ID_init=0):                    
		self.get_param(cfg, cat)
		self.trackers = []
		self.frame_count = 0
		self.reorder = [3, 4, 5, 6, 2, 1, 0]
		self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
		self.ID_count = [ID_init]
		self.ego_com = cfg.ego_com

		# vis and log purposes
		self.calib = calib
		self.oxts = oxts
		self.img_dir = img_dir
		self.vis_dir = vis_dir
		self.vis = cfg.vis
		self.hw = hw
		self.log = log

	def get_param(self, cfg, cat):
		if cfg.dataset == 'KITTI':
			# # good parameters
			if cat == 'Car': 			metric, thres = 'iou', 0.01
			elif cat == 'Pedestrian': 	metric, thres = 'dist', 1 		
			elif cat == 'Cyclist': 		metric, thres = 'dist', 2

			# bad parameters
			# if cat == 'Car': 			metric, thres = 'dist', 6
			# elif cat == 'Pedestrian': 	metric, thres = 'dist', 1 		
			# elif cat == 'Cyclist': 		metric, thres = 'dist', 6
			# else: assert False, 'error'

			self.max_age = 2 		# max age will preserve the bbox does not appear no more than 2 frames, interpolate the detection
			self.min_hits = 3

		elif cfg.dataset == 'nuScenes':
			if cat == 'Car': 			metric, thres = 'dist', 10
			elif cat == 'Pedestrian': 	metric, thres = 'dist', 6 		# add negative due to it is the cost
			elif cat == 'Bicycle': 		metric, thres = 'dist', 6
			elif cat == 'Motorcycle':	metric, thres = 'dist', 10
			elif cat == 'Bus': 			metric, thres = 'dist', 10
			elif cat == 'Trailer': 		metric, thres = 'dist', 10
			elif cat == 'Truck': 		metric, thres = 'dist', 10
			else: assert False, 'error'
		else: assert False, 'no such dataset'

		# add negative due to it is the cost
		if metric == 'dist': thres *= -1	

		self.metric, self.thres = metric, thres

	def within_range(self, theta):
		# make sure the orientation is within a proper range

		if theta >= np.pi: theta -= np.pi * 2    # make the theta still in the range
		if theta < -np.pi: theta += np.pi * 2

		return theta

	def orientation_correction(self, theta_pre, theta_obs):
		# update orientation in propagated tracks and detected boxes so that they are within 90 degree
		
		# make the theta still in the range
		theta_pre = self.within_range(theta_pre)
		theta_obs = self.within_range(theta_obs)

		# if the angle of two theta is not acute angle, then make it acute
		if abs(theta_obs - theta_pre) > np.pi / 2.0 and abs(theta_obs - theta_pre) < np.pi * 3 / 2.0:     
			theta_pre += np.pi       
			theta_pre = self.within_range(theta_pre)

		# now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
		if abs(theta_obs - theta_pre) >= np.pi * 3 / 2.0:
			if theta_obs > 0: theta_pre += np.pi * 2
			else: theta_pre -= np.pi * 2

		return theta_pre, theta_obs

	def ego_motion_compensation(self, frame, trks):
		# inverse ego motion compensation, move trks from the last frame of coordinate to the current frame for matching
		from AB3DMOT_libs.kitti_oxts import get_ego_traj, egomotion_compensation_ID

		ego_xyz_imu, ego_rot_imu, left, right = get_ego_traj(self.oxts, frame, 1, 1, only_fut=True, inverse=True) 
		for index in range(len(self.trackers)):
			trk_tmp = trks[index]
			compensated = egomotion_compensation_ID(trk_tmp[:3].reshape((1, -1)), self.calib, \
				ego_rot_imu, ego_xyz_imu, left, right)
			trk_tmp[:3] = compensated
			self.trackers[index].kf.x[:3] = copy.copy(compensated).reshape((-1, 1))

		return trks

	def visualization(self, img, dets, trks, calib, hw, save_path, height_threshold=0):
		# visualize to verify if the ego motion compensation is done correctly
		# ideally, the ego-motion compensated tracks should overlap closely with detections
		import cv2 
		from PIL import Image
		from AB3DMOT_libs.bbox_utils import project_to_image, draw_box3d_image
		from xinshuo_visualization import random_colors

		dets, trks = copy.copy(dets), copy.copy(trks)
		img = np.array(Image.open(img))
		max_color = 20
		colors = random_colors(max_color)       # Generate random colors

		def vis_obj(obj, img, color_tmp=None, str_vis=None):
			depth = obj[2]
			if depth >= 2: 			# check in front of camera
				obj_8corner = convert_3dbox_to_8corner(obj)
				obj_pts_2d = project_to_image(obj_8corner, calib.P)
				img, draw = draw_box3d_image(img, obj_pts_2d, hw, color=color_tmp)

				# draw text
				if draw and str_vis is not None:
					x1, y1 = int(obj_pts_2d[4, 0]), int(obj_pts_2d[4, 1])
					cv2.putText(img, str_vis, (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_tmp, 2)
			return img

		# visualize all detections as yellow boxes
		for det_tmp in dets: img = vis_obj(det_tmp, img, (255, 255, 0))				# yellow for detection
		
		# visualize color-specific tracks
		count = 0
		ID_list = [tmp.id for tmp in self.trackers]
		for trk_tmp in trks: 
			ID_tmp = ID_list[count]
			color_float = colors[int(ID_tmp) % max_color]
			color_int = tuple([int(tmp * 255) for tmp in color_float])
			str_vis = '%d' % ID_tmp
			img = vis_obj(trk_tmp, img, color_int, str_vis)		# blue for tracklets
			count += 1
		
		img = Image.fromarray(img)
		img = img.resize((hw[1], hw[0]))
		img.save(save_path)

	def matching(self, dets, trks):
		# matching using the Hungarian algorithm

		dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets]
		if len(dets_8corner) > 0: dets_8corner = np.stack(dets_8corner, axis=0)
		else: dets_8corner = []
		trks_8corner = [convert_3dbox_to_8corner(trk_tmp) for trk_tmp in trks]
		if len(trks_8corner) > 0: trks_8corner = np.stack(trks_8corner, axis=0)
		matched, unmatched_dets, unmatched_trks, cost, affi = \
			associate_detections_to_trackers(dets_8corner, trks_8corner, self.metric, self.thres)
		print_log(matched, log=self.log, display=False)

		return matched, unmatched_dets, unmatched_trks, cost, affi

	def prediction(self):
		# get predicted locations from existing tracks

		trks = np.zeros((len(self.trackers), 7))         # N x 7 
		for t, trk in enumerate(trks):
			
			# propagate locations
			kf_tmp = self.trackers[t]
			kf_tmp.kf.predict()
			kf_tmp.kf.x[3] = self.within_range(kf_tmp.kf.x[3])

			# update statistics
			kf_tmp.time_since_update += 1 		
			trk[:] = kf_tmp.kf.x.reshape((-1))[:7]

		return trks

	def update(self, matched, unmatched_trks, dets, info):
		# update matched trackers with assigned detections
		
		for t, trk in enumerate(self.trackers):
			if t not in unmatched_trks:
				d = matched[np.where(matched[:, 1] == t)[0], 0]     # a list of index

				# update statistics
				trk.time_since_update = 0		# reset because just updated
				trk.hits += 1

				# update orientation in propagated tracks and detected boxes so that they are within 90 degree
				bbox3d = dets[d, :][0]
				trk.kf.x[3], bbox3d[3] = self.orientation_correction(trk.kf.x[3], bbox3d[3])

				# kalman filter update with observation
				trk.kf.update(bbox3d)
				trk.kf.x[3] = self.within_range(trk.kf.x[3])
				trk.info = info[d, :][0]

	def birth(self, dets, info, unmatched_dets):
		# create and initialise new trackers for unmatched detections

		for i in unmatched_dets:        			# a scalar of index
			trk = KF(dets[i, :], info[i, :], self.ID_count[0])
			self.trackers.append(trk)
			self.ID_count[0] += 1

	def output(self):
		# output exiting tracks that have been stably associated, i.e., >= min_hits
		# and also delete tracks that have appeared for a long time, i.e., >= max_age

		num_trks = len(self.trackers)
		results = []
		for trk in reversed(self.trackers):
			d = trk.kf.x[:7].reshape((7, ))     # bbox location self
			d = d[self.reorder_back]			# change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]

			if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):      
				results.append(np.concatenate((d, [trk.id + 1], trk.info)).reshape(1, -1)) # +1 as MOT benchmark requires positive
			num_trks -= 1

			# deadth, remove dead tracklet
			if (trk.time_since_update >= self.max_age): 
				self.trackers.pop(num_trks)

		return results

	def track(self, dets_all, frame, seq_name=None):
		"""
		Params:
		  	dets_all: dict
				dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
				info: a array of other info for each det
			frame:    str, frame number, used to query ego pose
		Requires: this method must be called once for each frame even with empty detections.
		Returns the a similar array, where the last column is the object ID.

		NOTE: The number of objects returned may differ from the number of detections provided.
		"""
		dets, info = dets_all['dets'], dets_all['info']         # dets: N x 7, float numpy array

		# reorder the data to put x,y,z in front to be compatible with the state transition matrix
		# where the constant velocity model is defined in the first three rows of the matrix
		dets = dets[:, self.reorder]					# reorder the data to [[x,y,z,theta,l,w,h], ...]
		self.frame_count += 1

		# logging
		print_str = '\n\n*****************************************\n\nprocessing frame %d' % frame
		if seq_name is not None: print_str += ', seq name %s' % seq_name
		print_log(print_str, log=self.log, display=False)

		# tracks propagation based on velocity
		trks = self.prediction()

		# ego motion compensation
		if (frame > 0) and (self.ego_com) and (self.oxts is not None):
			trks = self.ego_motion_compensation(frame, trks)

		# visualization
		if self.vis and (self.vis_dir is not None):
			img = os.path.join(self.img_dir, f'{frame:06d}.png')
			save_path = os.path.join(self.vis_dir, f'{frame:06d}.jpg'); mkdir_if_missing(save_path)
			self.visualization(img, dets, trks, self.calib, self.hw, save_path)

		# matching
		matched, unmatched_dets, unmatched_trks, cost, affi = self.matching(dets, trks)

		# update trks with matched detection measurement
		self.update(matched, unmatched_trks, dets, info)

		# create and initialise new trackers for unmatched detections
		self.birth(dets, info, unmatched_dets)

		# output existing valid tracks
		results = self.output()
		if len(results) > 0: results = [np.concatenate(results)]			# h,w,l,x,y,z,theta, ID, other info, confidence
		else:            	 results = [np.empty((0, 15))]

		# logging
		print_log('\ntop-1 cost selected', log=self.log, display=False)
		print_log(cost, log=self.log, display=False)
		for result_index in range(len(results)):
			print_log(results[result_index][:, :8], log=self.log, display=False)
			print_log('', log=self.log, display=False)

		return results, affi