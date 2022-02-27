import numpy as np, cv2, os
from AB3DMOT_libs.box import Box3D

class Object3d(object):
    ''' 3d object label '''
    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0] # 'Car', 'Pedestrian', ...
        self.truncation = data[1] # truncated pixel ratio [0..1]
        self.occlusion = int(data[2]) # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3] # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4] # left
        self.ymin = data[5] # top
        self.xmax = data[6] # right
        self.ymax = data[7] # bottom
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])
        
        # extract 3d bounding box information
        self.h = data[8]    # box height
        self.w = data[9]    # box width
        self.l = data[10]   # box length (in meters)
        self.x = data[11]
        self.y = data[12]
        self.z = data[13] 
        self.center = [self.x, self.y, self.z] # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        
        # additional information
        self.s = None       # score
        self.id = None      # identity

        if len(data) > 15: self.s = float(data[15])
        if len(data) > 16: self.id = int(data[16])

    def get_box3D(self):
        return Box3D(self.x, self.y, self.z, self.h, self.w, self.l, self.ry, self.s)

    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % (self.center[0], self.center[1], self.center[2], self.ry))

    def convert_to_str(self):
        return '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' % \
            (self.type, self.truncation, self.occlusion, self.alpha, self.xmin, self.ymin, self.xmax, self.ymax,
                self.h, self.w, self.l, self.center[0], self.center[1], self.center[2], self.ry)

def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects