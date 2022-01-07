# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import numpy as np
from filterpy.kalman import KalmanFilter

class KF(object):
	def __init__(self, bbox3D, info, ID):

		# constant velocity model
		self.kf = KalmanFilter(dim_x=10, dim_z=7)       
		self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
		                      [0,1,0,0,0,0,0,0,1,0],
		                      [0,0,1,0,0,0,0,0,0,1],
		                      [0,0,0,1,0,0,0,0,0,0],  
		                      [0,0,0,0,1,0,0,0,0,0],
		                      [0,0,0,0,0,1,0,0,0,0],
		                      [0,0,0,0,0,0,1,0,0,0],
		                      [0,0,0,0,0,0,0,1,0,0],
		                      [0,0,0,0,0,0,0,0,1,0],
		                      [0,0,0,0,0,0,0,0,0,1]])     

		self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
		                      [0,1,0,0,0,0,0,0,0,0],
		                      [0,0,1,0,0,0,0,0,0,0],
		                      [0,0,0,1,0,0,0,0,0,0],
		                      [0,0,0,0,1,0,0,0,0,0],
		                      [0,0,0,0,0,1,0,0,0,0],
		                      [0,0,0,0,0,0,1,0,0,0]])

		# self.kf.R[0:,0:] *= 10.   # measurement uncertainty
		self.kf.P[7:, 7:] *= 1000. 	# state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
		self.kf.P *= 10.

		# self.kf.Q[-1,-1] *= 0.01    # process uncertainty
		self.kf.Q[7:, 7:] *= 0.01
		self.kf.x[:7] = bbox3D.reshape((7, 1))

		self.time_since_update = 0
		self.id = ID
		self.hits = 1           		# number of total hits including the first detection
		self.info = info        		# other information associated