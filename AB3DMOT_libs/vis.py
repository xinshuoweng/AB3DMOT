import numpy as np, cv2
from PIL import Image
from AB3DMOT_libs.box import Box3D
from xinshuo_visualization import random_colors

max_color = 30
colors = random_colors(max_color)       # Generate random colors

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

	def check_outside_image(x, y, height, width):
		if x < 0 or x >= width: return True
		if y < 0 or y >= height: return True

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
			image = cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA) # use LINE_AA for opencv3

			i,j=k+4,(k+1)%4 + 4
			image = cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

			i,j=k,k+4
			image = cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

	return image, True

def vis_obj(obj, img, calib, hw, color_tmp=None, str_vis=None):
	depth = obj.z
	if depth >= 2: 			# check in front of camera
		obj_8corner = Box3D.box2corners3d_camcoord(obj)
		obj_pts_2d = calib.project_rect_to_image(obj_8corner)
		img, draw = draw_box3d_image(img, obj_pts_2d, hw, color=color_tmp)

		# draw text
		if draw and obj_pts_2d is not None and str_vis is not None:
			x1, y1 = int(obj_pts_2d[4, 0]), int(obj_pts_2d[4, 1])
			img = cv2.putText(img, str_vis, (x1+5, y1-10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color_tmp, 2)
	return img

def vis_image_with_obj(img, objects_res, object_gt, calib, hw, save_path, height_threshold=0):
	img = np.array(Image.open(img))

	for obj in objects_res:
		depth = obj.z
		if depth >= 2:
			color_tmp = tuple([int(tmp * 255) for tmp in colors[obj.id % max_color]])
			box_tmp = obj.get_box3D()
			str_vis = 'ID: %d' % obj.id
			img = vis_obj(box_tmp, img, calib, hw['image'], color_tmp, str_vis)

	img = Image.fromarray(img)
	img = img.resize((hw['image'][1], hw['image'][0]))
	img.save(save_path)