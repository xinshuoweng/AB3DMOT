import numpy as np
from external.bbox_coarse_hash import BBoxCoarseFilter
from AB3DMOT_libs.dist_metrics import iou

def weird_bbox(bbox):
    if bbox.l <= 0 or bbox.w <= 0 or bbox.h <= 0: return True
    else:                                         return False

def nms(dets, threshold_low=0.1, threshold_high=1, threshold_yaw=0.3):
    """ keep the bboxes with overlap <= threshold
    """

    # define a course grid to speed up, i.e., avoiding considering objects that are far away
    dets_coarse_filter = BBoxCoarseFilter()
    dets_coarse_filter.bboxes2dict(dets)

    # get info, sort score and take care of high-score detections first
    yaws = np.asarray([det.ry for det in dets])
    scores = np.asarray([det.s for det in dets])
    order = np.argsort(scores)[::-1]        # from the greatest to the smallest

    # loop through each detection
    result_indexes = list()
    while order.size > 0:
        index = order[0]

        # remove bad boxes
        if weird_bbox(dets[index]):
            order = order[1:]
            continue

        # locate the related bboxes 
        filter_indexes = dets_coarse_filter.related_bboxes(dets[index])
        in_mask = np.isin(order, filter_indexes)        # find indexes that are still in the order list
        related_idxes = order[in_mask]                  # extract the absolute boxes indexes

        # compute the ious
        bbox_num = len(related_idxes)
        ious = np.zeros(bbox_num)
        for i, idx in enumerate(related_idxes):
            ious[i] = iou(dets[index], dets[idx], metric='iou_3d')

        # thresholding
        related_inds = np.where(ious > threshold_low)
        related_inds_vote = np.where(ious > threshold_high)
        order_vote = related_idxes[related_inds_vote]

        if len(order_vote) >= 2:

            # keep the bboxes with similar yaw
            if order_vote.shape[0] <= 2:
                score_index = np.argmax(scores[order_vote])
                median_yaw = yaws[order_vote][score_index]
            elif order_vote.shape[0] % 2 == 0:
                tmp_yaw = yaws[order_vote].copy()
                tmp_yaw = np.append(tmp_yaw, yaws[order_vote][0])
                median_yaw = np.median(tmp_yaw)
            else:
                median_yaw = np.median(yaws[order_vote])
            yaw_vote = np.where(np.abs(yaws[order_vote] - median_yaw) % (2 * np.pi) < threshold_yaw)[0]
            order_vote = order_vote[yaw_vote]
            
            # start weighted voting
            vote_score_sum = np.sum(scores[order_vote])
            det_arrays = list()
            for idx in order_vote:
                det_arrays.append(Box3D.bbox2array(dets[idx])[np.newaxis, :])
            det_arrays = np.vstack(det_arrays)
            avg_bbox_array = np.sum(scores[order_vote][:, np.newaxis] * det_arrays, axis=0) / vote_score_sum
            bbox = Box3D.array2bbox(avg_bbox_array)
            bbox.s = scores[index]
            result_indexes.append(index)
        else:
            result_indexes.append(index)

        # delete the overlapped bboxes
        delete_idxes = related_idxes[related_inds]
        # if len(delete_idxes) > 1: 
        #     print(delete_idxes)
        #     print(dets[delete_idxes[0]])
        #     print(dets[delete_idxes[1]])
        #     zxc
        in_mask = np.isin(order, delete_idxes, invert=True)
        order = order[in_mask]

    return result_indexes