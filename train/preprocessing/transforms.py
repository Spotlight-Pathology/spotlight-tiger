"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2
"""

from shapely.geometry import Polygon
import albumentations as A
import numpy as np
import torch
import cv2
from train.preprocessing.tile_utils import build_tile_grid, read_tile_roi
from train.utils import LABEL_MAPPING

ZERO_RES = 0.5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def augment(img, mask=None, p=0.3):
    """
    Augment by rotations, flips and colour transformations.

    Parameters
    ----------
    img: ndarray, (h, w, c=3)
    mask: ndarray, (h, w)
        By default None and only the image is processed.
    p: float
        Probability of flips and rotations.
    """
    aug_transform = A.Compose([

        # the following are automatically applied to  both img and mask:
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
        A.RandomRotate90(p=p),

        # the following are automatically applied only to the img:
        A.CLAHE(p=p),  # Contrast Limited Adaptive Histogram Equalization
        A.RandomBrightnessContrast(p=p),
        A.RandomGamma(p=p),
        A.Downscale(
            scale_min=0.9, scale_max=0.99, p=p, interpolation=cv2.INTER_LINEAR
        )
    ])

    if mask is not None:
        transformed = aug_transform(image=img, mask=mask)
        return transformed['image'], transformed['mask']
    else:
        transformed = aug_transform(image=img)
        return transformed['image']


def augment_detections(img, target, p=0.5):
    """
    Augment by rotations, flips and colour transformations.

    Parameters
    ----------
    img: ndarray, (h, w, c=3)
    target: dict
        See format here:
        https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    p: float
        probability of transform being applied
    """
    augment_transform = A.Compose([
        # the following are automatically applied to  both img and target:
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
        A.RandomRotate90(p=p),

        # the following are automatically applied only to the img:
        A.CLAHE(p=p),  # Contrast Limited Adaptive Histogram Equalization
        A.RandomBrightnessContrast(p=p),
        A.RandomGamma(p=p),
        A.Downscale(scale_min=0.9, scale_max=0.99, p=p)
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        min_visibility=0.9,
        label_fields=['area', 'iscrowd', 'labels']
    ))

    bboxes = target['boxes'].numpy()
    areas = target['area'].numpy()
    iscrowd = target['iscrowd'].numpy()
    labels = target['labels'].numpy()

    transformed = augment_transform(
        image=img,
        bboxes=bboxes,
        area=areas,
        iscrowd=iscrowd,
        labels=labels,
    )
    aug_image = transformed['image']
    aug_target = {
        'boxes': torch.FloatTensor(transformed['bboxes']),
        'area': torch.tensor(transformed['area']),
        'iscrowd': torch.IntTensor(transformed['iscrowd']),
        'labels': torch.tensor(transformed['labels']).to(torch.int64),
        'image_id': target['image_id']
    }
    return aug_image, aug_target


def resize_tiling(
        img,
        mask=None,
        size=224,
        resolution=ZERO_RES,
        stride=None,
        res_zero=ZERO_RES,
        padding=0,  # pad the tile grid before tiling (used at inference)
        return_transformed_big_image=False
):
    """
    Generate image tiles at the specified resolution. Will pad images with
    `0` if they are smaller than the tile size specified.

    Can be used for training and/ or testing.

    Parameters
    ----------
    img: ndarray, (h, w, c=3)
    mask: ndarray, (h, w)
        By default None and only the image is processed.
    size: int
        The final image & mask size will be square (size, size).
    resolution: float
        The final resolution.
    stride: int
        The step when generating tiles. By default, stride is set equal to
        the tile size.
    res_zero: float
        The resolution at level 0 (microns/ pixel).
    padding: int
        Add padding around the ROI before creating the tile grid.
        Tiles extracted from the padded areas will == 0.
        Padding is given at res_zero.
    """
    if stride is None:
        stride = size

    downscale_factor = resolution / res_zero
    downscaled_height = int(img.shape[0] // downscale_factor)
    downscaled_width = int(img.shape[1] // downscale_factor)
    downscaled_padding = int(padding // downscale_factor)

    transform = A.Compose([
        A.Resize(
            height=downscaled_height,
            width=downscaled_width,
            interpolation=cv2.INTER_LINEAR
        ),
        A.PadIfNeeded(
            min_height=size,
            min_width=size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0
        )
    ])

    transform_mask = A.Compose([
        A.Resize(
            height=downscaled_height,
            width=downscaled_width,
            interpolation=cv2.INTER_NEAREST
        ),
        A.PadIfNeeded(
            min_height=size,
            min_width=size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0
        )
    ])

    if mask is not None:
        img = transform(image=img)['image']
        mask = transform_mask(image=mask)['image']

        # add the 'ignore' label to padded areas
        mask[np.sum(img, axis=-1) == 0] = LABEL_MAPPING[0]

        tile_origins = build_tile_grid(
            img.shape[slice(0, 2)],
            tile_size=size,
            stride=stride,
            padding=downscaled_padding
        )

        img_tiles, mask_tiles = [], []
        for origin in tile_origins:
            img_tiles.append(read_tile_roi(img, origin, size))
            mask_tiles.append(read_tile_roi(mask, origin, size))

        img_tiles = np.array(img_tiles)
        mask_tiles = np.array(mask_tiles)
        tile_origins = np.array(tile_origins)
        return img_tiles, mask_tiles, tile_origins

    else:
        transformed = transform(image=img)
        img = transformed['image']

        tile_origins = build_tile_grid(
            img.shape[slice(0, 2)],
            tile_size=size,
            stride=stride,
            padding=downscaled_padding
        )

        img_tiles = []
        for origin in tile_origins:
            img_tiles.append(read_tile_roi(img, origin, size))

        img_tiles = np.array(img_tiles)
        tile_origins = np.array(tile_origins)

        if return_transformed_big_image:
            return img_tiles, tile_origins, img
        else:
            return img_tiles, tile_origins


def transform_mask_labels(mask, mapping_dict,):
    """
    Convert original mask labels to relevant classes (tumour, stroma, other).

    Parameters
    ----------
    mask: np.array, (h, w)
    mapping_dict: {}
        Mapping from original labels to relevant classes:
        0 -> other
        1 -> tumour
        2 -> stroma
        100 -> unlabelled

    Returns
    -------
    mask: np.array, (h, w)
    """
    labels_mask = np.unique(mask)
    for label in labels_mask:
        mask[mask == label] = mapping_dict[label]
    return mask


def detection_resize_tiling(
        image,
        target,
        size,
        stride,
        include_empty=False
):
    """
    Transform for object detection that generates examples of shape
    (size, size) by padding, if needed, and tiling.

    Parameters
    ----------
    image: ndarray, (h, w, c=3)
    target: dict
        See the target dict as described here:
        https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    size: int
        The tile width
    stride: int
        The stride to use when tiling.
    include_empty: bool
        whether to include empty tiles (excluding tiles that only
        consist of padding)
    """
    padded_image, padded_target = pad_detections(image, target, size)
    tiles, tiled_targets = tile_detections(
        padded_image, padded_target, size, stride, include_empty=include_empty
    )
    return tiles, tiled_targets


def pad_detections(image, target, size):
    """Pad smaller images to minimum size."""
    resize_transform = A.Compose([
        A.PadIfNeeded(
            min_height=size,
            min_width=size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0
        )
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        min_visibility=0.9,
        label_fields=['area', 'iscrowd', 'labels']
    ))

    bboxes = target['boxes'].numpy()
    areas = target['area'].numpy()
    iscrowd = target['iscrowd'].numpy()
    labels = target['labels'].numpy()

    transformed = resize_transform(
        image=image,
        bboxes=bboxes,
        area=areas,
        iscrowd=iscrowd,
        labels=labels,
    )
    padded_image = transformed['image']
    padded_target = {
        'boxes': torch.FloatTensor(transformed['bboxes']),
        'area': torch.tensor(transformed['area']),
        'iscrowd': torch.IntTensor(transformed['iscrowd']),
        'labels': torch.tensor(transformed['labels']).to(torch.int64),
        'image_id': target['image_id']
    }
    return padded_image, padded_target


def tile_detections(image, target, size, stride, include_empty=True):
    """
    Parameters
    ----------
    image: ndarray, (h, w, c=3)
    target: dict
        See the target dict as described here:
        https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    size: int
        The tile width
    stride: int
        The stride to use when tiling.
    include_empty: bool
        If true, adds a dummy box of label 0 to include the empty tiles
        Excludes tiles that only include padding.
    """
    tile_origins = build_tile_grid(
        image.shape[slice(0, 2)],
        tile_size=size,
        stride=stride,
    )
    bboxes = target['boxes'].numpy()
    iscrowd = target['iscrowd'].numpy()
    labels = target['labels'].numpy()
    tiles, tile_targets = [], []

    for tile_origin in tile_origins:
        x, y = tile_origin
        tile = image[slice(y, y + size), slice(x, x + size)]

        if np.max(tile) == 0:
            # tile contains only padding, skip it
            continue

        # build target for tile
        idx_tile, bboxes_tile = find_boxes_tile(bboxes, tile_origin, size)
        areas_tile = [get_bbox_area(bbox) for bbox in bboxes_tile]
        iscrowd_tile = iscrowd[idx_tile]
        labels_tile = labels[idx_tile]

        # if targets are found and include lymphocytes
        if (len(bboxes_tile) > 0) & (1 in labels_tile.tolist()):
            tile_target = {
                'boxes': torch.FloatTensor(bboxes_tile),
                'area': torch.tensor(areas_tile),
                'iscrowd': torch.IntTensor(iscrowd_tile),
                'labels': torch.tensor(labels_tile).to(torch.int64),
                'image_id': target['image_id']
            }
            tiles.append(tile)
            tile_targets.append(tile_target)
        # if targets are found but do not include lymphocytes
        elif (len(bboxes_tile) > 0) & (1 not in labels_tile.tolist()):
            if include_empty:
                tile_target = {
                    'boxes': torch.FloatTensor(bboxes_tile),
                    'area': torch.tensor(areas_tile),
                    'iscrowd': torch.IntTensor(iscrowd_tile),
                    'labels': torch.tensor(labels_tile).to(torch.int64),
                    'image_id': target['image_id']
                }
                tiles.append(tile)
                tile_targets.append(tile_target)
            pass
        else:
            if include_empty:
                # if no targets are found in tile, add a dummy "0" type
                # object
                tile_target = {
                    'boxes': torch.FloatTensor([[1, 2, 3, 4]]),
                    'area': torch.tensor([4]),
                    'iscrowd': torch.IntTensor([0]),
                    'labels': torch.IntTensor([0]),
                    'image_id': target['image_id']
                }
                tiles.append(tile)
                tile_targets.append(tile_target)
            pass

    assert len(tiles) > 0
    return tiles, tile_targets


def get_bbox_area(bbox):
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    area = width * height
    return area


def find_boxes_tile(bboxes, tile_origin, size, min_overlap=0.5):
    """
    From a list of bboxes, keep only the ones found inside the tile area.
    """
    tile_boxes, idxs = [], []
    x, y = tile_origin
    for idx, bbox in enumerate(bboxes):
        x0, y0, x1, y1 = bbox
        relative_bbox = (x0 - x, y0 - y, x1 - x, y1 - y)
        if check_box_overlaps(relative_bbox, (size, size), min_overlap):
            relative_bbox = get_box_to_fit(relative_bbox, (size, size))
            tile_boxes.append(relative_bbox)
            idxs.append(idx)
    return idxs, tile_boxes


def get_box_to_fit(bbox, image_dimensions):
    """
    Modify the box so it fits exactly into the image.
    Parameters
    ----------
    bbox: (x0, y0, x1, y1)
    image_dimensions: tuple
        width, height
    """
    width, height = image_dimensions
    x0, y0, x1, y1 = bbox
    if x0 < 0:
        x0 = 0
        if x1 == x0:
            x1 += 1
    if y0 < 0:
        y0 = 0
        if y1 == y0:
            y1 += 1
    if x1 >= width:
        x1 = width - 1
        if x1 == x0:
            x0 -= 1
    if y1 >= height:
        y1 = height - 1
        if y1 == y0:
            y0 -= 1
    return x0, y0, x1, y1


def check_box_overlaps(bbox, image_dimensions, min_overlap=0.5):
    """
    Parameters
    ----------
    bbox: (x0, y0, x1, y1)
    image_dimensions: tuple
        width, height
    min_overlap: float
        Min percentage of bbox area that must overlap to consider
        the bbox overlapping.
    """
    width, height = image_dimensions
    x0, y0, x1, y1 = bbox
    img_poly = Polygon(
        [(0, 0), (0, height), (width, height), (width, 0)]
    )
    bbox_poly = Polygon(
        [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
    )
    fits = img_poly.intersects(bbox_poly)
    if fits:
        intersection = img_poly.intersection(bbox_poly).area
        fits = intersection / get_bbox_area(bbox) >= min_overlap
    return fits


def get_centroid(bbox):
    x0, y0, x1, y1 = bbox
    xc = (x1 - x0) / 2
    yc = (y1 - y0) / 2
    return x0 + xc, y0 + yc


def convert_bbox(bbox):
    """
    Convert bbox from (x, y, width, height)
    to (xo, yo, x1, y1).
    """
    x, y, width, height = bbox
    xo = x
    yo = y
    y1 = y + height
    x1 = x + width
    return xo, yo, x1, y1
