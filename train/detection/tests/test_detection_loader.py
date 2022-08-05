"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2
"""

import train.detection.detection_loader as loader
from train.detection.tests.test_data import DETECTION_TEST_DATA_DIR as DATA_DIR
import os
import torch


def test_DetectionsDataset():
    image_paths = [os.path.join(DATA_DIR, 'test_detection.png')]
    labels_json = os.path.join(DATA_DIR, 'testcoco.json')
    tempdir = os.path.join(DATA_DIR, 'tempdir')
    os.makedirs(tempdir, exist_ok=True)
    dataset = loader.DetectionsDataset(
        image_paths, labels_json,
        size=224,
        stride=224,
        tempdir=tempdir,
        augment=False
    )

    img, target, _ = dataset[0]
    assert img.size()[0] == 3
    boxes = torch.FloatTensor([
        [  # true lymphocyte annotation from tiger
            107,
            112,
            107 + 16,
            112 + 16
        ],
        [  # random bbox added for this test
            132,
            70,
            132 + 16,
            70 + 16
        ]
    ])
    assert len(dataset) == 1  # 1 tile with two targets
    assert len(target['boxes'].numpy()) == len(boxes.numpy())
    assert torch.equal(target['image_id'], torch.tensor(0))
    dataset.clean_up()
