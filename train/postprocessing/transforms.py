"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2

Post processing transform after inference to write results.
"""
import torch
import numpy as np
from train.preprocessing.transforms import check_box_overlaps

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def threshold_outputs(res_targets, threshold: float):
    targets = []
    for target in res_targets:
        boxes = target['boxes']
        if len(boxes) == 0:
            targets.append(target)
            continue
        else:
            labels = target['labels']
            scores = target['scores']
            idx_keep = torch.where(scores >= threshold)[0]
            targets.append({
                'boxes': boxes[idx_keep, :],
                'labels': labels[idx_keep],
                'scores': scores[idx_keep]
            })
    return targets


def merge_tile_outputs(tile_outputs, tile_origins):
    """
    Merge outputs from tiles of a big ROI.

    Parameters
    ----------
    tile_outputs: list(dict)
        Each element is a dict with keys: "boxes", "labels", "scores"
        and represents predictions for a tile.
    tile_origins: ndarray, shape=(num_tiles, 2)
        The x, y tile origins.

    Returns
    -------
    merged_outputs: dict
        Merged boxes from all tiles in the big ROI, by shifting
        the box coordinates and concatenating.
        {
            'boxes': FloatTensor[N, 4],
            'labels': Int64Tensor[N],  # the object predicted class
            'scores': FloatTensor[N]  # the object probabilities
        }
    """
    tile_origins_h = np.hstack([tile_origins, tile_origins])
    tile_origins_h = torch.tensor(tile_origins_h).to(DEVICE)
    boxes, labels, scores = [], [], []
    for idx, tile in enumerate(tile_outputs):
        # shift coordinates to the ROI origin
        tile["boxes"] = tile["boxes"] + tile_origins_h[idx, :]
        boxes.append(tile["boxes"])
        labels.append(tile["labels"])
        scores.append(tile["scores"])

    boxes = torch.cat(boxes, dim=0)
    labels = torch.cat(labels, dim=0)
    scores = torch.cat(scores, dim=0)
    merged_outputs = {
        "boxes": boxes,
        "labels": labels,
        "scores": scores
    }
    return merged_outputs


def check_is_padded(image):
    """
    Check if an image has been padded.
    Parameters
    ----------
    image: ndarray, (h, w, c)

    Returns
    -------
    bool
    """
    is_top_pad = np.all(image[0, :, :] == 0)
    is_bottom_pad = np.all(image[-1, :, :] == 0)
    is_left_pad = np.all(image[:, 0, :] == 0)
    is_right_pad = np.all(image[:, -1, :] == 0)

    is_pad = (
        is_top_pad | is_bottom_pad | is_right_pad | is_left_pad
    )
    return is_pad


def convert_non_padded(outputs, image, unpadded_shape):
    """
    Convert detections bbox outputs to non-padded coordinates

    Parameters
    ----------
    outputs: dict
        {
            'boxes': FloatTensor[N, 4],
            'labels': Int64Tensor[N],  # the object predicted class
            'scores': FloatTensor[N]  # the object probabilities
        }
    image: ndarray (h, w, c)
    unpadded_shape: tuple
        The shape of the image before padding (h, w)
    """
    half_width = image.shape[1] // 2
    half_height = image.shape[0] // 2

    half_col = image[:, half_width, :]
    half_row = image[half_height, :, :]

    # get ind of 1st non-zero element
    img_ymin = np.argmin(np.all(half_col == 0, axis=1))
    img_xmin = np.argmin(np.all(half_row == 0, axis=1))
    pad_coords = torch.tensor((img_xmin, img_ymin, img_xmin, img_ymin))
    pad_coords = pad_coords.to(DEVICE)

    outputs["boxes"] = outputs["boxes"] - pad_coords
    idx_keep = []
    for idx, box in enumerate(outputs["boxes"]):
        fits = check_box_overlaps(
            box, (unpadded_shape[1], unpadded_shape[0]), min_overlap=0.5
        )
        if fits:
            idx_keep.append(idx)

    outputs["boxes"] = outputs["boxes"][idx_keep, :]
    outputs["labels"] = outputs["labels"][idx_keep]
    outputs["scores"] = outputs["scores"][idx_keep]
    return outputs


def get_area(img, img_resolution):
    """Ignore image areas padded with zeros and calculate area."""
    img = img.detach().cpu().numpy()
    tissue_mask = (img[0] != 0) | (img[1] != 0) | (img[2] != 0)
    pixels_tissue = np.sum(tissue_mask)
    area_pixel = (img_resolution / 1000) ** 2
    area = pixels_tissue * area_pixel
    return area
