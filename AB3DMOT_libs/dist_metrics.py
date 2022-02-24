import numpy as np
from numba import jit
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
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

def iou(bbox1, bbox2, space='3D'):
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

	corners1 = Box3D.box2corners3d_camcoord(bbox1)
	corners2 = Box3D.box2corners3d_camcoord(bbox2)

	# corner points are in counter clockwise order, also only take x and z
	# because y is the negative height direction
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
	if space == '2D':
		return iou_2d
	elif space == '3D':
		ymax = min(corners1[0, 1], corners2[0, 1])
		ymin = max(corners1[4, 1], corners2[4, 1])
		inter_vol = inter_area * max(0.0, ymax - ymin)
		vol1 = box3d_vol(corners1)
		vol2 = box3d_vol(corners2)
		iou_3d = inter_vol / (vol1 + vol2 - inter_vol)
		return iou_3d
	else:
		assert False, '%s is not supported' % space

def giou2d(box_a, box_b):

	# obtain ground corners and area, not containing the height
	corners1 = Box3D.box2corners3d_camcoord(box_a) 	# 8 x 3
	corners2 = Box3D.box2corners3d_camcoord(box_b)	# 8 x 3
	boxa_bot = corners1[:4, [0, 2]] 		# 4 x 2
	boxb_bot = corners2[:4, [0, 2]]			# 4 x 2
	reca, recb = Polygon(boxa_bot), Polygon(boxb_bot)

	# compute intersection and union
	I = reca.intersection(recb).area
	U = box_a.w * box_a.l + box_b.w * box_b.l - I

	# compute the convex area
	all_corners = np.vstack((boxa_bot, boxb_bot))
	C = ConvexHull(all_corners)
	convex_corners = all_corners[C.vertices]
	convex_area = PolyArea2D(convex_corners)
	C = convex_area

	return I / U - (C - U) / C

def giou3d(box_a, box_b):

	# obtain ground corners and area, not containing the height
	corners1 = Box3D.box2corners3d_camcoord(box_a) 	# 8 x 3
	corners2 = Box3D.box2corners3d_camcoord(box_b)	# 8 x 3
	boxa_bot = corners1[:4, [0, 2]] 		# 4 x 2
	boxb_bot = corners2[:4, [0, 2]]			# 4 x 2
	reca, recb = Polygon(boxa_bot), Polygon(boxb_bot)

	# compute overlap height
	ymax = min(corners1[0, 1], corners2[0, 1])
	ymin = max(corners1[4, 1], corners2[4, 1])
	overlap_height = max(0.0, ymax - ymin)

	# compute IoU
	I = reca.intersection(recb).area * overlap_height
	U = box_a.w * box_a.l * box_a.h + box_b.w * box_b.l * box_b.h - I

	# compute the union height
	ymax = max(corners1[0, 1], corners2[0, 1])
	ymin = min(corners1[4, 1], corners2[4, 1])
	union_height = max(0.0, ymax - ymin)

	# compute the convex area
	all_corners = np.vstack((boxa_bot, boxb_bot))
	C = ConvexHull(all_corners)
	convex_corners = all_corners[C.vertices]
	convex_area = PolyArea2D(convex_corners)
	C = convex_area * union_height

	return I / U - (C - U) / C
	
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
    det_array = Box3D.bbox2array(det)[:7]
    trk_array = Box3D.bbox2array(trk)[:7]
    
    diff = np.expand_dims(det_array - trk_array, axis=1)
    corrected_yaw_diff = diff_orientation_correction(diff[3])
    diff[3] = corrected_yaw_diff

    if trk_inv_innovation_matrix is not None:
        result = \
            np.sqrt(np.matmul(np.matmul(diff.T, trk_inv_innovation_matrix), diff)[0][0])
    else:
        result = np.sqrt(np.dot(diff.T, diff))
    return result

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