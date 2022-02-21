import numpy as np, copy
from numba import jit
from scipy.optimize import linear_sum_assignment
from .kitti_oxts import roty

# option 1 for computing polygon overlap
from scipy.spatial import ConvexHull

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

def PolyArea2D(pts):
    roll_pts = np.roll(pts, -1, axis=0)
    area = np.abs(np.sum((pts[:, 0] * roll_pts[:, 1] - pts[:, 1] * roll_pts[:, 0]))) * 0.5
    return area

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

#################### distance metric

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

def dist_ground(corners1, corners2):
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

def giou2d(corners1, corners2):
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    
    # compute intersection and union
    I = reca.intersection(recb).area
    U = box_a.w * box_a.l + box_b.w * box_b.l - I

    # compute the convex area
    all_corners = np.vstack((boxa_corners, boxb_corners))
    C = ConvexHull(all_corners)
    convex_corners = all_corners[C.vertices]
    convex_area = PolyArea2D(convex_corners)
    C = convex_area

    # compute giou
    return I / U - (C - U) / C

def giou3d(corners1, corners2):
    boxa_corners = np.array(BBox.box2corners2d(box_a))[:, :2]
    boxb_corners = np.array(BBox.box2corners2d(box_b))[:, :2]
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    ha, hb = box_a.h, box_b.h
    za, zb = box_a.z, box_b.z
    overlap_height = max(0, min((za + ha / 2) - (zb - hb / 2), (zb + hb / 2) - (za - ha / 2)))
    union_height = max((za + ha / 2) - (zb - hb / 2), (zb + hb / 2) - (za - ha / 2))
    
    # compute intersection and union
    I = reca.intersection(recb).area * overlap_height
    U = box_a.w * box_a.l * ha + box_b.w * box_b.l * hb - I

    # compute the convex area
    all_corners = np.vstack((boxa_corners, boxb_corners))
    C = ConvexHull(all_corners)
    convex_corners = all_corners[C.vertices]
    convex_area = PolyArea2D(convex_corners)
    C = convex_area * union_height

    # compute giou
    giou = I / U - (C - U) / C
    return giou

def compute_m_distance(dets, tracks, trk_innovation_matrix):
    """ compute l2 or mahalanobis distance
        when the input trk_innovation_matrix is None, compute L2 distance (euler)
        else compute mahalanobis distance
        return dist_matrix: numpy array [len(dets), len(tracks)]
    """
    euler_dis = (trk_innovation_matrix is None) # is use euler distance
    if not euler_dis:
        trk_inv_inn_matrices = [np.linalg.inv(m) for m in trk_innovation_matrix]
    dist_matrix = np.empty((len(dets), len(tracks)))

    for i, det in enumerate(dets):
        for j, trk in enumerate(tracks):
            if euler_dis:
                dist_matrix[i, j] = utils.m_distance(det, trk)
            else:
                dist_matrix[i, j] = utils.m_distance(det, trk, trk_inv_inn_matrices[j])
    return dist_matrix


def compute_iou_distance(dets, tracks, asso='iou'):
    iou_matrix = np.zeros((len(dets), len(tracks)))
    for d, det in enumerate(dets):
        for t, trk in enumerate(tracks):
            if asso == 'iou':
                iou_matrix[d, t] = utils.iou3d(det, trk)[1]
            elif asso == 'giou':
                iou_matrix[d, t] = utils.giou3d(det, trk)
    dist_matrix = 1 - iou_matrix
    return dist_matrix

################### matching method
     
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
			if metric == 'iou':    
				aff_matrix[d, t] = iou3d(det, trk)[0]             # det: 8 x 3, trk: 8 x 3
			elif metric == 'dist': 
				aff_matrix[d, t] = -dist3d(det, trk)              # det: 8 x 3, trk: 8 x 3		
			else: assert False, 'error'

	if hypothesis == 1:
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

def bipartite_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix):
    if asso == 'iou':
        dist_matrix = compute_iou_distance(dets, tracks, asso)
    elif asso == 'giou':
        dist_matrix = compute_iou_distance(dets, tracks, asso)
    elif asso == 'm_dis':
        dist_matrix = compute_m_distance(dets, tracks, trk_innovation_matrix)
    elif asso == 'euler':
        dist_matrix = compute_m_distance(dets, tracks, None)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    matched_indices = np.stack([row_ind, col_ind], axis=1)
    return matched_indices, dist_matrix

def greedy_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix):
    """ it's ok to use iou in bipartite
        but greedy is only for m_distance
    """
    matched_indices = list()
    
    # compute the distance matrix
    if asso == 'm_dis':
        distance_matrix = compute_m_distance(dets, tracks, trk_innovation_matrix)
    elif asso == 'euler':
        distance_matrix = compute_m_distance(dets, tracks, None)
    elif asso == 'iou':
        distance_matrix = compute_iou_distance(dets, tracks, asso)
    elif asso == 'giou':
        distance_matrix = compute_iou_distance(dets, tracks, asso)
    num_dets, num_trks = distance_matrix.shape

    # association in the greedy manner
    # refer to https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking/blob/master/main.py
    distance_1d = distance_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_trks, index_1d % num_trks], axis=1)
    detection_id_matches_to_tracking_id = [-1] * num_dets
    tracking_id_matches_to_detection_id = [-1] * num_trks
    for sort_i in range(index_2d.shape[0]):
        detection_id = int(index_2d[sort_i][0])
        tracking_id = int(index_2d[sort_i][1])
        if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[detection_id] == -1:
            tracking_id_matches_to_detection_id[tracking_id] = detection_id
            detection_id_matches_to_tracking_id[detection_id] = tracking_id
            matched_indices.append([detection_id, tracking_id])
    if len(matched_indices) == 0:
        matched_indices = np.empty((0, 2))
    else:
        matched_indices = np.asarray(matched_indices)
    return matched_indices, distance_matrix