"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2
"""

from train.preprocessing import transforms
from train.preprocessing.tests.test_data import PREPROCESS_DATA_DIR as DATA_DIR
import os
from skimage.io import imread
import torch
import numpy as np
import random
from numpy.testing import assert_array_equal
from train.utils import LABEL_MAPPING

IMG = imread(os.path.join(DATA_DIR, 'rotated_img.png'))
MASK = imread(os.path.join(DATA_DIR, 'rotated_mask.png'))


def test_augment():

    aug_img, aug_mask = transforms.augment(IMG, MASK)
    aug_only_img = transforms.augment(IMG)

    # Check that the mask is rotated/ flipped same way as the img
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(aug_img)
    # ax2.imshow(aug_mask)
    # plt.tight_layout()
    # plt.show()

    assert aug_mask.shape == aug_img.shape[slice(0, 2)]

    # check if same amount of pixels (dimensions could be rotated)
    pixels_aug_only_img = aug_only_img.shape[0] * aug_only_img.shape[1] * 3
    pixels_aug_img = aug_img.shape[0] * aug_img.shape[1] * 3
    assert pixels_aug_only_img == pixels_aug_img

    # assert we haven't changed the content of the mask
    labels_mask_before = np.sort(np.unique(MASK))
    labels_mask_after = np.sort(np.unique(aug_mask))
    assert_array_equal(labels_mask_after, labels_mask_before)


def test_resize_tiling():

    # check image larger than tile size
    # -------------------------------------------------------------------------
    full_res_img, full_res_mask, full_res_origins = transforms.resize_tiling(
        IMG, MASK, size=224, resolution=0.5
    )
    half_res_img, half_res_mask, half_res_origins = transforms.resize_tiling(
        IMG, MASK, size=224, resolution=1.
    )

    assert full_res_img.shape[slice(0, 3)] == full_res_mask.shape
    assert half_res_img.shape[slice(0, 3)] == half_res_mask.shape
    assert full_res_img.shape[slice(1, 3)] == half_res_img.shape[slice(1, 3)]

    # fewer tiles are generated at lower resolution
    assert len(full_res_img) > len(half_res_img)

    # check image smaller than tile size
    # -------------------------------------------------------------------------
    small = IMG[1500:1650, 1500:1650, :]
    small_mask = MASK[1500:1650, 1500:1650]
    res_small_img, res_small_mask, res_small_origin = transforms.resize_tiling(
        small, small_mask, size=224, resolution=0.5
    )
    assert res_small_mask.shape == (1, 224, 224)
    assert res_small_img.shape == (1, 224, 224, 3)
    # assert image and mask are padded with zeros
    assert_array_equal(res_small_img[0, 0, 0, :], [0, 0, 0])
    assert res_small_mask[0, 0, 0] == LABEL_MAPPING[0]  # padded areas ignored

    # Check
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(res_small_mask[0])
    # ax2.imshow(res_small_img[0])
    # plt.tight_layout()
    # plt.show()


def test_find_boxes_tile():
    bboxes = [
        [15, 15, 20, 20],
        [0, 0, 5, 5]
    ]
    tile_origin = [0, 0]
    size = 10
    idx, tile_boxes = transforms.find_boxes_tile(bboxes, tile_origin, size)
    assert idx == [1]
    assert tile_boxes == [(0, 0, 5, 5)]


def test_check_box_overlaps():
    bboxes = [
        [15, 15, 20, 20],
        [0, 0, 5, 5]
    ]
    size = 10
    fits = transforms.check_box_overlaps(bboxes[0], (size, size))
    assert not fits

    fits = transforms.check_box_overlaps(bboxes[1], (size, size))
    assert fits


def test_get_box_to_fit():
    bboxes = [
        [-5, -5, 5, 5],  # upper left
        [-5, 2, 5, 12],  # lower left
        [2, -5, 12, 5],  # upper right
        [2, 2, 12, 12]  # lower right
    ]
    size = 10
    fits = [transforms.get_box_to_fit(bbox, (size, size)) for bbox in bboxes]
    for fitbox in fits:
        assert transforms.check_box_overlaps(fitbox, (size, size), 1)


def test_tile_detections():
    image = np.ones((100, 100, 3))
    bboxes = [
        [15, 15, 20, 20],
        [0, 0, 5, 5]
    ]
    image_labels = [1] * len(bboxes)
    target = {
        'boxes': torch.FloatTensor(bboxes),
        'labels': torch.IntTensor(image_labels).to(torch.int64),
        'image_id': torch.tensor(1),
        'area': torch.tensor([25, 25]),
        'iscrowd': torch.IntTensor([0, 0])
    }

    size = 10
    stride = 10

    tiles, tile_targets = transforms.tile_detections(
        image, target, size, stride, include_empty=False
    )
    assert len(tiles) == 2
    assert [0, 0, 5, 5] in tile_targets[0]['boxes'].numpy()


def test_detection_resize_tiling():
    image = np.ones((100, 100, 3))
    bboxes = [
        [15, 15, 20, 20],
        [0, 0, 5, 5]
    ]
    image_labels = [1] * len(bboxes)
    target = {
        'boxes': torch.FloatTensor(bboxes),
        'labels': torch.IntTensor(image_labels).to(torch.int64),
        'image_id': torch.tensor(1),
        'area': torch.tensor([25, 25]),
        'iscrowd': torch.IntTensor([0, 0])
    }
    size = 10
    stride = 10
    tiles, tile_targets = transforms.detection_resize_tiling(
        image,
        target,
        size,
        stride,
    )
    assert len(tiles) == 2
    assert [0, 0, 5, 5] in tile_targets[0]['boxes'].numpy()


def test_augment_detections():
    image = np.ones((100, 100, 3)) * 150
    image[:5, :5, 0] = 240
    image = image.astype(np.uint8)
    bboxes = [
        [15, 15, 20, 20],
        [0, 0, 5, 5]
    ]
    image_labels = [1] * len(bboxes)
    target = {
        'boxes': torch.FloatTensor(bboxes),
        'labels': torch.IntTensor(image_labels).to(torch.int64),
        'image_id': torch.tensor(1),
        'area': torch.tensor([25, 25]),
        'iscrowd': torch.IntTensor([0, 0])
    }
    # check change in colour
    random.seed(15)
    transformed_img, transformed_target = transforms.augment_detections(
        image, target
    )
    assert transformed_img[-1, -1, -1] == 241

    # check horizontal flip with colour shift
    random.seed(1)
    transformed_img, transformed_target = transforms.augment_detections(
        image, target
    )
    assert transformed_img[0, -3, 0] == 196
    assert_array_equal(
        transformed_target['boxes'].numpy(),
        np.array(
            [[80., 15., 85., 20.],
             [95., 0., 100., 5.]]
        )
    )
