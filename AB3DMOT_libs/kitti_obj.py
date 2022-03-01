import numpy as np, cv2, os
from AB3DMOT_libs.box import Box3D

# read object data in the KITTI detection format, one file per frame

def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object_3D(line) for line in lines]
    return objects

class Object_3D(object):
    # modified from https://github.com/kuixu/kitti_object_vis/blob/master/kitti_util.py, MIT license
    def __init__(self, label_file_line=None, obj_type=None, trunc=None, occ=None, alpha=None, \
        xmin=None, ymin=None, xmax=None, ymax=None, \
        h=None, w=None, l=None, x=None, y=None, z=None, ry=None, \
        s=None, id=None, velo_2d=None, velo_3d=None):

        # initialize
        self.type = obj_type
        self.trunc, self.occ, self.alpha = trunc, occ, alpha
        self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax
        self.h, self.w, self.l, self.x, self.y, self.z, self.ry = h, w, l, x, y, z, ry
        self.s = s       # score
        self.id = id     # identity
        self.velo_3d, self.velo_2d = velo_3d, velo_2d   # velocity

        # overwrite if the data file is provided
        if label_file_line is not None:
            data = label_file_line.split(' ')
            data[1:] = [float(x) for x in data[1:]]

            # extract label, truncation, occlusion
            self.type = data[0] # 'Car', 'Pedestrian', ...

            # truncated pixel ratio [0..1]
            # occlusion 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
            # object observation angle [-pi..pi]
            self.trunc, self.occ, self.alpha = data[1], int(data[2]), data[3] 

            # extract 2d bounding box in 0-based coordinates
            self.xmin, self.ymin, self.xmax, self.ymax = data[4], data[5], data[6], data[7]

            # extract 3d bounding box information
            self.h, self.w, self.l, self.x, self.y, self.z = \
                data[8], data[9], data[10], data[11], data[12], data[13] 
            
            self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
            
            # update score/ID
            if len(data) > 15: self.s = float(data[15])
            if len(data) > 16: self.id = int(data[16])
        
        # group some data
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])
        self.xyz = [self.x, self.y, self.z]  # location (x,y,z) in camera coord.
        self.wlh = [self.w, self.l, self.h]

    def get_box3D(self):
        return Box3D(self.x, self.y, self.z, self.h, self.w, self.l, self.ry, self.s)

    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % (self.type, self.trunc, self.occ, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % (self.h, self.w, self.l))
        print('3d bbox location xyz, ry: (%f, %f, %f), %f' % (self.xyz[0], self.xyz[1], self.xyz[2], self.ry))

    def convert_to_det_str(self):
        output_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' % \
            (self.type, self.trunc, self.occ, self.alpha, self.xmin, self.ymin, self.xmax, self.ymax,
            self.h, self.w, self.l, self.x, self.y, self.z, self.ry)
        
        if self.s is None: return output_str        # no score and id
        else:
            output_str = '%s %.2f' % (output_str, self.s)
            if self.id is None: return output_str   # with score, no id
            else:
                output_str = '%s %d' % (output_str, self.id)
                return output_str                   # with score and id

    def convert_to_trk_input_str(self, frame, type_id):
        # format follows the input data, i.e., the standard MOT pre-processed data

        assert self.s is not None, 'error'    
        return '%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f' % \
            (int(frame), type_id, self.xmin, self.ymin, self.xmax, self.ymax, self.s, \
            self.h, self.w, self.l, self.x, self.y, self.z, self.ry, self.alpha)

    def convert_to_trk_output_str(self, frame):
        # format follows the KITTI tracking results format

        assert self.s is not None, 'error'    
        assert self.id is not None, 'error'    
        return '%d %d %s %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' % \
            (int(frame), self.id, self.type, self.trunc, self.occ, self.alpha, \
            self.xmin, self.ymin, self.xmax, self.ymax, \
            self.h, self.w, self.l, self.x, self.y, self.z, self.ry, self.s)