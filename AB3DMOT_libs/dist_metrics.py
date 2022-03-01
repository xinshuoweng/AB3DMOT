import numpy as np, time
from numba import jit
from scipy.spatial import ConvexHull
from AB3DMOT_libs.box import Box3D

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

def compute_inter_2D(boxa_bottom, boxb_bottom):
	# computer intersection over union of two sets of bottom corner points

	_, I_2D = convex_hull_intersection(boxa_bottom, boxb_bottom)

	# a slower version
	# from shapely.geometry import Polygon
	# reca, recb = Polygon(boxa_bottom), Polygon(boxb_bottom)
	# I_2D = reca.intersection(recb).area

	return I_2D

def compute_height(box_a, box_b, inter=True):

	corners1 = Box3D.box2corners3d_camcoord(box_a) 	# 8 x 3
	corners2 = Box3D.box2corners3d_camcoord(box_b)	# 8 x 3
	
	if inter: 		# compute overlap height
		ymax = min(corners1[0, 1], corners2[0, 1])
		ymin = max(corners1[4, 1], corners2[4, 1])
		height = max(0.0, ymax - ymin)
	else:			# compute union height
		ymax = max(corners1[0, 1], corners2[0, 1])
		ymin = min(corners1[4, 1], corners2[4, 1])
		height = max(0.0, ymax - ymin)

	return height

def compute_bottom(box_a, box_b):
	# obtain ground corners and area, not containing the height

	corners1 = Box3D.box2corners3d_camcoord(box_a) 	# 8 x 3
	corners2 = Box3D.box2corners3d_camcoord(box_b)	# 8 x 3
	
	# get bottom corners and inverse order so that they are in the 
	# counter-clockwise order to fulfill polygon_clip
	boxa_bot = corners1[-5::-1, [0, 2]] 		# 4 x 2
	boxb_bot = corners2[-5::-1, [0, 2]]			# 4 x 2
		
	return boxa_bot, boxb_bot

def PolyArea2D(pts):
    roll_pts = np.roll(pts, -1, axis=0)
    area = np.abs(np.sum((pts[:, 0] * roll_pts[:, 1] - pts[:, 1] * roll_pts[:, 0]))) * 0.5
    return area

def convex_area(boxa_bottom, boxb_bottom):

	# compute the convex area
	all_corners = np.vstack((boxa_bottom, boxb_bottom))
	C = ConvexHull(all_corners)
	convex_corners = all_corners[C.vertices]
	convex_area = PolyArea2D(convex_corners)

	return convex_area

#################### distance metric

def iou(box_a, box_b, metric='giou_3d'):
	''' Compute 3D/2D bounding box IoU, only working for object parallel to ground

	Input:
		Box3D instances
	Output:
	    iou_3d: 3D bounding box IoU
	    iou_2d: bird's eye view 2D bounding box IoU

	box corner order is like follows
            1 -------- 0 		 top is bottom because y direction is negative
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7    
	
	rect/ref camera coord:
    right x, down y, front z
	'''	

	# compute 2D related measures
	boxa_bot, boxb_bot = compute_bottom(box_a, box_b)
	I_2D = compute_inter_2D(boxa_bot, boxb_bot)

	# only needed for GIoU
	if 'giou' in metric:
		C_2D = convex_area(boxa_bot, boxb_bot)

	if '2d' in metric:		 	# return 2D IoU/GIoU
		U_2D = box_a.w * box_a.l + box_b.w * box_b.l - I_2D
		if metric == 'iou_2d':  return I_2D / U_2D
		if metric == 'giou_2d': return I_2D / U_2D - (C_2D - U_2D) / C_2D

	elif '3d' in metric:		# return 3D IoU/GIoU
		overlap_height = compute_height(box_a, box_b)
		I_3D = I_2D * overlap_height	
		U_3D = box_a.w * box_a.l * box_a.h + box_b.w * box_b.l * box_b.h - I_3D
		if metric == 'iou_3d':  return I_3D / U_3D
		if metric == 'giou_3d':
			union_height = compute_height(box_a, box_b, inter=False)
			C_3D = C_2D * union_height
			return I_3D / U_3D - (C_3D - U_3D) / C_3D
	else:
		assert False, '%s is not supported' % space

def dist_ground(bbox1, bbox2):
	# Compute distance of bottom center in 3D space, NOT considering the difference in height

	c1 = Box3D.bbox2array(bbox1)[[0, 2]]
	c2 = Box3D.bbox2array(bbox2)[[0, 2]]
	dist = np.linalg.norm(c1 - c2)

	return dist

def dist3d_bottom(bbox1, bbox2):	
	# Compute distance of bottom center in 3D space, considering the difference in height / 2

	c1 = Box3D.bbox2array(bbox1)[:3]
	c2 = Box3D.bbox2array(bbox2)[:3]
	dist = np.linalg.norm(c1 - c2)

	return dist

def dist3d(bbox1, bbox2):
	# Compute distance of actual center in 3D space, considering the difference in height

	corners1 = Box3D.box2corners3d_camcoord(bbox1) 	# 8 x 3
	corners2 = Box3D.box2corners3d_camcoord(bbox2)	# 8 x 3

	# compute center point based on 8 corners
	c1 = np.average(corners1, axis=0)
	c2 = np.average(corners2, axis=0)

	dist = np.linalg.norm(c1 - c2)

	return dist

def diff_orientation_correction(diff):
    """
    return the angle diff = det - trk
    if angle diff > 90 or < -90, rotate trk and update the angle diff
    """
    if diff > np.pi / 2:  diff -= np.pi
    if diff < -np.pi / 2: diff += np.pi
    return diff

def m_distance(det, trk, trk_inv_innovation_matrix=None):
    
	# compute difference
    det_array = Box3D.bbox2array(det)[:7]
    trk_array = Box3D.bbox2array(trk)[:7] 		# (7, )
    diff = np.expand_dims(det_array - trk_array, axis=1) 	# 7 x 1

    # correct orientation
    corrected_yaw_diff = diff_orientation_correction(diff[3])
    diff[3] = corrected_yaw_diff

    if trk_inv_innovation_matrix is not None:
        dist = np.sqrt(np.matmul(np.matmul(diff.T, trk_inv_innovation_matrix), diff)[0][0])
    else:
        dist = np.sqrt(np.dot(diff.T, diff)) 		# distance along 7 dimension
    return dist