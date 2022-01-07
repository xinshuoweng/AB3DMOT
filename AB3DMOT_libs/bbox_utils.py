# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import numpy as np, copy, cv2
from numba import jit
from scipy.optimize import linear_sum_assignment

@jit          
def poly_area(x,y):
	""" Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
	return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

@jit         
def box3d_vol(corners):
	''' corners: (8,3) no assumption on axis direction '''
	a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
	b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
	c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
	return a*b*c

#################### option 1 for computing polygon overlap
from scipy.spatial import ConvexHull

@jit          
def convex_hull_intersection(p1, p2):
	""" Compute area of two convex hull's intersection area.
		p1,p2 are a list of (x,y) tuples of hull vertices.
		return a list of (x,y) for the intersection and its volume
	"""
	inter_p = polygon_clip(p1,p2)
	if inter_p is not None:
		hull_inter = ConvexHull(inter_p)
		return inter_p, hull_inter.volume
	else:
		return None, 0.0  

def polygon_clip(subjectPolygon, clipPolygon):
	""" Clip a polygon with another polygon.
	Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

	Args:
		subjectPolygon: a list of (x,y) 2d points, any polygon.
		clipPolygon: a list of (x,y) 2d points, has to be *convex*
	Note:
		**points have to be counter-clockwise ordered**

	Return:
		a list of (x,y) vertex point for the intersection polygon.
	"""
	def inside(p):
		return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])
 
	def computeIntersection():
		dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
		dp = [s[0] - e[0], s[1] - e[1]]
		n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
		n2 = s[0] * e[1] - s[1] * e[0] 
		n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
		return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]
 
	outputList = subjectPolygon
	cp1 = clipPolygon[-1]
 
	for clipVertex in clipPolygon:
		cp2 = clipVertex
		inputList = outputList
		outputList = []
		s = inputList[-1]
 
		for subjectVertex in inputList:
			e = subjectVertex
			if inside(e):
				if not inside(s): outputList.append(computeIntersection())
				outputList.append(e)
			elif inside(s): outputList.append(computeIntersection())
			s = e
		cp1 = cp2
		if len(outputList) == 0: return None
	return (outputList)

def iou3d(corners1, corners2):
	''' Compute 3D bounding box IoU, only working for object parallel to ground

	Input:
	    corners1: numpy array (8,3), assume up direction is negative Y
	    corners2: numpy array (8,3), assume up direction is negative Y
	Output:
	    iou: 3D bounding box IoU
	    iou_2d: bird's eye view 2D bounding box IoU

	todo (rqi): add more description on corner points' orders.
	'''
	# corner points are in counter clockwise order
	rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
	rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
	area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
	area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])

	# inter_area = shapely_polygon_intersection(rect1, rect2)
	_, inter_area = convex_hull_intersection(rect1, rect2)

	# try:
	#   _, inter_area = convex_hull_intersection(rect1, rect2)
	# except ValueError:
	#   inter_area = 0
	# except scipy.spatial.qhull.QhullError:
	#   inter_area = 0

	iou_2d = inter_area/(area1+area2-inter_area)
	ymax = min(corners1[0,1], corners2[0,1])
	ymin = max(corners1[4,1], corners2[4,1])
	inter_vol = inter_area * max(0.0, ymax-ymin)
	vol1 = box3d_vol(corners1)
	vol2 = box3d_vol(corners2)
	iou = inter_vol / (vol1 + vol2 - inter_vol)
	return iou, iou_2d

@jit          
def roty(t):
	''' Rotation about the y-axis. '''
	c = np.cos(t)
	s = np.sin(t)
	return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])
     
def convert_3dbox_to_8corner(bbox3d_input):
	''' Takes an object's 3D box with the representation of [x,y,z,theta,l,w,h] and 
	    convert it to the 8 corners of the 3D box
	    
	    Returns:
	        corners_3d: (8,3) array in in rect camera coord
	'''
	# compute rotational matrix around yaw axis
	bbox3d = copy.copy(bbox3d_input)

	R = roty(bbox3d[3])    

	# 3d bounding box dimensions
	l = bbox3d[4]
	w = bbox3d[5]
	h = bbox3d[6]

	# 3d bounding box corners
	x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
	y_corners = [0,0,0,0,-h,-h,-h,-h];
	z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];

	# rotate and translate 3d bounding box
	corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
	#print corners_3d.shape
	corners_3d[0,:] = corners_3d[0,:] + bbox3d[0]
	corners_3d[1,:] = corners_3d[1,:] + bbox3d[1]
	corners_3d[2,:] = corners_3d[2,:] + bbox3d[2]

	return np.transpose(corners_3d)

#################### comput IoU 3D
def dist3d(corners1, corners2):
	''' Compute 3D bounding box IoU, only working for object parallel to ground

	Input:
	    corners1: numpy array (8,3), assume up direction is negative Y
	    corners2: numpy array (8,3), assume up direction is negative Y
	Output:
	    dist:		distance of two bounding boxes in 3D space
	'''

	# compute center point based on 8 corners
	c1 = np.average(corners1, axis=0)
	c2 = np.average(corners2, axis=0)

	dist = np.linalg.norm(c1 - c2)

	return dist

def project_to_image(pts_3d, P):
    ''' Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P)) # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]

def check_outside_image(x, y, height, width):
    if x < 0 or x >= width: return True
    if y < 0 or y >= height: return True

def draw_box3d_image(image, qs, img_size=(900, 1600), color=(255,255,255), thickness=4):
    ''' Draw 3d bounding box in image
        qs: (8,2) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''

    # if 6 points of the box are outside the image, then do not draw
    pts_outside = 0
    for index in range(8):
        check = check_outside_image(qs[index, 0], qs[index, 1], img_size[0], img_size[1])
        if check: pts_outside += 1
    if pts_outside >= 6: return image, False

    # actually draw
    if qs is not None:
        qs = qs.astype(np.int32)
        for k in range(0,4):
           i,j=k,(k+1)%4
           cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA) # use LINE_AA for opencv3

           i,j=k+4,(k+1)%4 + 4
           cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

           i,j=k,k+4
           cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

    return image, True

def associate_detections_to_trackers(detections, trackers, metric, threshold, hypothesis=1):   
	"""
	Assigns detections to tracked object (both represented as bounding boxes)

	detections:  N x 8 x 3
	trackers:    M x 8 x 3

	Returns 3 lists of matches, unmatched_detections and unmatched_trackers
	"""

	if len(trackers) == 0: 
		return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 8, 3), dtype=int), 0, None
	aff_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

	for d, det in enumerate(detections):
		for t, trk in enumerate(trackers):
			if metric == 'iou':    aff_matrix[d, t] = iou3d(det, trk)[0]             # det: 8 x 3, trk: 8 x 3
			elif metric == 'dist': aff_matrix[d, t] = -dist3d(det, trk)              # det: 8 x 3, trk: 8 x 3		
			else: assert False, 'error'

	if hypothesis == 1:
		# matched_indices = linear_assignment(-aff_matrix)      # hougarian algorithm, compatible to linear_assignment in sklearn.utils
		row_ind, col_ind = linear_sum_assignment(-aff_matrix)      # hougarian algorithm
		matched_indices = np.stack((row_ind, col_ind), axis=1)
	else:
		cost_list, hun_list = best_k_matching(-aff_matrix, hypothesis)

	# compute cost
	cost = 0
	for row_index in range(matched_indices.shape[0]):
		cost -= aff_matrix[matched_indices[row_index, 0], matched_indices[row_index, 1]]
	# print(matched_indices.shape)
	# zxc

	unmatched_detections = []
	for d, det in enumerate(detections):
		if (d not in matched_indices[:, 0]): unmatched_detections.append(d)
	unmatched_trackers = []
	for t, trk in enumerate(trackers):
		if (t not in matched_indices[:, 1]): unmatched_trackers.append(t)

	# filter out matched with low IoU or high distance
	matches = []
	for m in matched_indices:
		if (aff_matrix[m[0], m[1]] < threshold):
			unmatched_detections.append(m[0])
			unmatched_trackers.append(m[1])
		else: matches.append(m.reshape(1, 2))
	if len(matches) == 0: 
		matches = np.empty((0, 2),dtype=int)
	else: matches = np.concatenate(matches, axis=0)

	return matches, np.array(unmatched_detections), np.array(unmatched_trackers), cost, aff_matrix