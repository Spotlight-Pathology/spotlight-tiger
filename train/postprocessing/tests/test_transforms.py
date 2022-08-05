"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2
"""

import torch
import train.postprocessing.transforms as transforms


def test_get_area():
    img = torch.ones((3, 10, 10))
    area = transforms.get_area(img, img_resolution=1000)
    assert area == 100
