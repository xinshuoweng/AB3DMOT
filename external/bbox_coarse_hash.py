""" Split the area into grid boxes
    BBoxes in different grid boxes without overlap cannot have overlap
"""
import numpy as np
from ..data_protos import BBox

class BBoxCoarseFilter:
    def __init__(self, grid_size, scaler=100):
        self.gsize = grid_size
        self.scaler = 100
        self.bbox_dict = dict()
    
    def bboxes2dict(self, bboxes):
        for i, bbox in enumerate(bboxes):
            grid_keys = self.compute_bbox_key(bbox)
            for key in grid_keys:
                if key not in self.bbox_dict.keys():
                    self.bbox_dict[key] = set([i])
                else:
                    self.bbox_dict[key].add(i)
        return
        
    def compute_bbox_key(self, bbox):
        corners = np.asarray(BBox.box2corners2d(bbox))
        min_keys = np.floor(np.min(corners, axis=0) / self.gsize).astype(np.int)
        max_keys = np.floor(np.max(corners, axis=0) / self.gsize).astype(np.int)
        
        # enumerate all the corners
        grid_keys = [
            self.scaler * min_keys[0] + min_keys[1],
            self.scaler * min_keys[0] + max_keys[1],
            self.scaler * max_keys[0] + min_keys[1],
            self.scaler * max_keys[0] + max_keys[1]
        ]
        return grid_keys
    
    def related_bboxes(self, bbox):
        """ return the list of related bboxes
        """ 
        result = set()
        grid_keys = self.compute_bbox_key(bbox)
        for key in grid_keys:
            if key in self.bbox_dict.keys():
                result.update(self.bbox_dict[key])
        return list(result)
    
    def clear(self):
        self.bbox_dict = dict()