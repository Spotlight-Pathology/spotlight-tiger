"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2
"""

import train.segmentation.segmentation_loader as load
from train.preprocessing.tests.test_data import PREPROCESS_DATA_DIR as DATA_DIR
import os
from numpy.testing import assert_array_equal

LABEL_MAPPING = {
    0: 100,  # unlabelled
    1: 1,  # tumour (invasive)
    2: 2,  # stroma
    3: 1,  # tumour (invasive)
    4: 0,  # other (healthy)
    5: 0,  # other (necrosis)
    6: 2,  # stroma (inflammed)
    7: 0  # other (rest)
}


def test_create_roi_loader():
    roi_paths = [os.path.join(DATA_DIR, 'rotated_img.png')]
    maskfile = os.path.join(DATA_DIR, 'rotated_mask.png')
    labels = {
        roi_paths[0]: maskfile
    }
    loader = load.create_roi_loader(
        roi_paths,
        labels=labels,
        tile_size=224,
        tile_res=0.5,
        res_zero=0.5,
        batch_size=2,
        reshuffle=False,
        num_workers=0,
        label_conversion_map=LABEL_MAPPING,
        stride=200,
        padding=12
    )
    tiles, masks, origins, names = next(iter(loader))
    assert_array_equal(tiles.shape, (2, 3, 224, 224))
    assert_array_equal(masks.shape, (2, 224, 224))
    assert_array_equal(origins.shape, (2, 2))
    assert len(names) == 2
    assert names[0] == roi_paths[0]
