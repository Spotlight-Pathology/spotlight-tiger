"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2
"""

import torch
from torchvision.ops import nms

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def apply_nms(res_targets, iou_threshold=0.5):
    """
    Apply NMS to a batch of FasterRCNN output boxes, according
    to their intersection-over-union (IoU).

    Works for only 1 object class.

    Parameters
    ----------
    res_targets: List[dict]
        {
            'boxes': FloatTensor[N, 4],
            'labels': Int64Tensor[N],  # the object predicted class
            'scores': FloatTensor[N]  # the object probabilities
        }
    iou_threshold: float
         Overlapping boxes with IoU > iou_threshold are discarded.

    Returns
    -------
    nms_targets: List[dict]
        The boxes remaining after NMS. Same structure as res_targets.
    """
    nms_targets = []
    for res_target in res_targets:
        boxes = res_target['boxes']
        if len(res_target["boxes"]) == 0:
            nms_targets.append(res_target)
            continue
        labels = res_target['labels']
        scores = res_target['scores']

        idx_keep = nms(boxes, scores, iou_threshold)
        nms_targets.append({
            'boxes': boxes[idx_keep, :],
            'labels': labels[idx_keep],
            'scores': scores[idx_keep]
        })
    return nms_targets
