"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2
"""

from train.postprocessing import nms
import torch
import numpy as np
from numpy.testing import assert_array_equal


def test_apply_nms():
    bboxes = [
        [0, 0, 4, 4],
        [0, 0, 5, 5]
    ]
    image_labels = [1] * len(bboxes)
    target = {
        'boxes': torch.FloatTensor(bboxes),
        'labels': torch.IntTensor(image_labels).to(torch.int64),
        'scores': torch.FloatTensor([0.7, 0.8])
    }
    res_targets = [target]
    out = nms.apply_nms(res_targets, iou_threshold=0.5)[0]
    out_boxes = out['boxes'].numpy()
    assert len(out_boxes) == 1
    assert_array_equal(out_boxes, np.array([[0, 0, 5, 5]]))
