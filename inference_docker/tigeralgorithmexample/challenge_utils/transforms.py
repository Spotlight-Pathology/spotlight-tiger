"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2

Post-processing transforms applied after inference
"""
import torch
import torchvision.transforms.functional as F
from torchvision.ops import nms
import numpy as np
import copy
from shapely.geometry import Polygon
from torch.utils.data import DataLoader
from .tile_utils import SimpleDataset
from typing import Union
import os
import torch.nn as nn
import xml.etree.ElementTree as ET
from skimage.draw import polygon2mask

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CentreFOVDownscaleWithAveraging:
    def __init__(
            self,
        tile_res,
        write_resolution,
        tile_size,
        padding,  # at res_zero
        res_zero
    ):
        """
        Centre crop the tile_mask and downscale to `write_resolution`.
        """
        self.padding = int(padding // (tile_res / res_zero))  # at tile res
        self.padding_zero = padding
        self.centre_size = tile_size - 2 * self.padding
        self.abs_downscaled_factor = write_resolution / res_zero
        self.downscaled_factor = write_resolution / tile_res
        self.downscaled_size = int(self.centre_size // self.downscaled_factor)

    def apply(self, y_prob, tile_origins):
        """
        Parameters
        ----------
        y_prob: torch.Tensor, (batch, c, h ,w)
        tile_origins: torch.Tensor, (2,)
        """
        # keep central fov
        y_centre = y_prob[
            :,
            :,
            self.padding: self.padding + self.centre_size,
            self.padding: self.padding + self.centre_size
        ]
        tile_origins = tile_origins + self.padding_zero

        # downscale
        y_prep = F.resize(y_centre, self.downscaled_size)

        # tile origins are always given at res zero
        # convert them to the dims of the write mask
        tile_origins = torch.div(
            tile_origins, self.abs_downscaled_factor, rounding_mode='floor'
        )
        y_prep = y_prep.permute(0, 2, 3, 1)

        tile_origins = tile_origins.detach().cpu().numpy()
        xs = tile_origins[:, 0]
        ys = tile_origins[:, 1]
        prep = []
        for t in range(len(y_prep)):
            prep.append(
                (y_prep[t], xs[t], ys[t])
            )

        return prep


class PostprocessDetections:
    def __init__(
            self,
            nms_iou,
            padding,  # res_zero
            tile_size,  # without padding
            tissue_mask_tile  # without padding
    ):
        """Post-process detections from tile to write in challenge."""

        self.nms_iou = nms_iou
        self.padding = padding
        self.tile_size = tile_size
        self.tissue_mask_tile = tissue_mask_tile

    def apply(self, model_output):
        """
        Works for model_ouput from a single tile.
        Steps:
        - applies NMS
        - get the box centroid
        - remove centroids that fall on background areas
        - remove centroids that fall in the padded area
        """
        # if not detections
        if len(model_output) == 0:
            return []
        elif len(model_output[0]['boxes']) == 0:
            return []

        outputs = apply_nms(model_output, self.nms_iou)[0]

        outputs['boxes'] = get_all_centroids(outputs['boxes'])
        outputs['boxes'] = outputs['boxes'] - self.padding

        # filter and remove detections
        valid_dets = [self.is_valid(*det) for det in outputs['boxes']]
        outputs['boxes'] = outputs['boxes'][valid_dets]
        outputs['scores'] = outputs['scores'][valid_dets]

        outputs = {k: v.cpu().numpy() for k, v in outputs.items()}
        xs = outputs['boxes'][:, 0]
        ys = outputs['boxes'][:, 1]
        scores = outputs['scores']
        return list(zip(xs, ys, scores))

    def is_valid(self, xc, yc):
        """
        x and y are box centroid coordinates relative to the tile origin.
        Checks the detection is inside the central FOV and not background.
        """
        x = int(copy.deepcopy(xc))
        y = int(copy.deepcopy(yc))
        if (x < 0) | (x >= self.tile_size):
            return False
        elif (y < 0) | (y >= self.tile_size):
            return False
        elif self.tissue_mask_tile[y, x] == 0:
            return False
        else:
            return True


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
        labels = res_target['labels']
        scores = res_target['scores']

        idx_keep = nms(boxes, scores, iou_threshold)
        nms_targets.append({
            'boxes': boxes[idx_keep, :],
            'labels': labels[idx_keep],
            'scores': scores[idx_keep]
        })
    return nms_targets


def get_all_centroids(bboxes):
    x0 = bboxes[:, 0]
    y0 = bboxes[:, 1]
    x1 = bboxes[:, 2]
    y1 = bboxes[:, 3]

    xc = ((x1 - x0) / 2) + x0
    yc = ((y1 - y0) / 2) + y0
    return torch.vstack((xc, yc)).T


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


def get_bbox_area(bbox):
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    area = width * height
    return area


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


def point_to_box(x, y, size):
    """Convert centerpoint to bounding box of fixed size"""
    return np.array([x - size, y - size, x + size, y + size])


def slide_nms(detection_writer, iou_threshold, box_size=4):
    """

    Parameters
    ----------
    detection_writer: []
        Each entry is a point given at global slide coords.
        {
            "point": [x, y, spacing],
            "probability": proba
        }
    iou_threshold: float
    box_size: int
        Recreates box sizes of twice this size (in microns).
    """
    detection_writer_data = detection_writer._data["points"]
    scores = [det["probability"] for det in detection_writer_data]
    scores = np.array(scores)

    centroids = [det["point"][:2] for det in detection_writer_data]
    centroids = np.array(centroids) * 1000  # convert to um
    boxes = np.array([point_to_box(c[0], c[1], box_size) for c in centroids])

    keep = batched_nms(boxes, scores, batch=10000, iou_threshold=iou_threshold)

    new_points = []
    for idx, point in enumerate(detection_writer_data):
        if idx in keep:
            new_points.append(point)
    detection_writer._data["points"] = new_points
    return detection_writer


def batched_nms(bboxes, bscores, batch, iou_threshold=0.5):
    """
    Apply NMS to the FasterRCNN output boxes, according
    to their intersection-over-union (IoU), but process them in batches
    so that it is memory efficient.

    Works for only 1 object class.

    Parameters
    ----------
    bboxes: np.array, (N, 4)
    bscores: np.array, (N,)
        the object probabilities
    iou_threshold: float
         Overlapping boxes with IoU > iou_threshold are discarded.
    batch: int

    Returns
    -------
    keep: [int64]
        The idx of boxes remaining after NMS.
    """
    # sort boxes by xo, then by yo
    # so that boxes closer together mostly end up on same batch
    order = np.array(range(len(bscores)))[:, np.newaxis]
    bboxes = np.hstack([bboxes, bscores[:, np.newaxis], order])
    bboxes = bboxes[np.lexsort((bboxes[:, 0], bboxes[:, 1]))]

    loader = DataLoader(dataset=SimpleDataset(bboxes), batch_size=batch)
    keep_idx = []

    for batch in loader:
        batch_boxes = batch[:, :4].to(DEVICE)
        batch_scores = batch[:, 4].to(DEVICE)
        batch_order = batch[:, 5].numpy().astype(int)

        keep_batch = nms(batch_boxes, batch_scores, iou_threshold)  # cuda
        keep_batch = keep_batch.cpu().numpy().astype(int)

        keep_idx.extend(batch_order[keep_batch].tolist())

    return keep_idx


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


class WriterMaskWithAveraging:
    def __init__(
            self,
            output_path: str,
            tile_size: int,
            dimensions: tuple,
            spacing: tuple,
            roi_origin: tuple  # (x, y)
    ):
        """
        Writer for combining tile segmentation inference results into a
        reconstructed big mask array. When tiles overlap, the logits are
        averaged.

        Parameters
        ----------
        output_path: str
            path to output file
        tile_size: int
            tile size used for writing image tiles
        dimensions: tuple
            dimensions of the output image (width, height)
        spacing: tuple
            base spacing (x resolution, y resolution) of the output image
        """
        if output_path is not None:
            if os.path.splitext(output_path)[1] != '.png':
                output_path = os.path.join(output_path, '.png')

        self.dimensions = dimensions  # (width, height)
        self.output_path = output_path
        self.image = torch.zeros(
            (
                int(dimensions[1]),
                int(dimensions[0]),
                3
            )).to(DEVICE)
        self.counter = torch.zeros(
            (
                int(dimensions[1]),
                int(dimensions[0])
            )).to(DEVICE)
        self.spacing = spacing
        self.tile_size = int(tile_size)
        self.roi_origin = roi_origin

    def write_segmentation(
        self, tile: np.ndarray, x: Union[int, float], y: Union[int, float]
    ):
        x = int(x) - self.roi_origin[0]
        y = int(y) - self.roi_origin[1]

        self.image[
            slice(y, y + self.tile_size),
            slice(x, x + self.tile_size)
        ] += tile

        self.counter[
            slice(y, y + self.tile_size),
            slice(x, x + self.tile_size)
        ] += 1

    def finish(self):
        self.image /= self.counter.unsqueeze(2)


def mask_to_labels(logit_mask, tissue_mask_roi):
    image = nn.Softmax(dim=2)(logit_mask)
    image = torch.argmax(image, dim=2)
    image = image.detach().cpu().numpy()
    image[image == 0] = 7
    return (image * tissue_mask_roi).astype(np.int)


def get_writer(
        writer_type: str,
        write_resolution,
        tile_resolution,
        res_zero,
        tile_padding,  # at res_zero
        tile_size,
        output_path,
        dimensions_zero,  # (width, height)
        roi_origin

):
    # calculate downscaling factors
    relative_factor = write_resolution / tile_resolution
    abs_factor = write_resolution / res_zero
    tile_padding = int(tile_padding // (tile_resolution / res_zero))
    write_tile_size = (tile_size - 2 * tile_padding) // relative_factor
    dimensions_zero = np.array(dimensions_zero)

    if writer_type == 'png-averaging':
        writer = WriterMaskWithAveraging(
            output_path,
            tile_size=write_tile_size,
            dimensions=dimensions_zero // abs_factor,
            spacing=(write_resolution, write_resolution),
            roi_origin=roi_origin
        )
    else:
        raise Exception('Writer not implemented.')

    return writer


def convert_xml_to_mask(
        input_file: str,
        out_resolution,
        in_resolution,
        wsi_dims_zero  # (width, height)
):
    """Convert xml annotation to mask array"""
    tree = ET.parse(input_file)
    root = tree.getroot()
    annotations = root[0]

    downscale_factor = out_resolution / in_resolution
    wsi_dims = tuple(np.array(wsi_dims_zero) // downscale_factor)
    wsi_dims = [int(d) for d in wsi_dims]

    img = np.zeros((wsi_dims[1], wsi_dims[0]))

    for annotation in annotations:
        coords = []
        ordering = []
        for coord in annotation[0]:
            coords.append(
                [float(coord.attrib['Y']), float(coord.attrib['X'])]
            )
            ordering.append(int(coord.attrib['Order']))

        inds = np.argsort(ordering)
        coords_prep = (np.array(coords) / downscale_factor).astype(int)
        coords_prep = coords_prep[inds, :]

        mask = polygon2mask((wsi_dims[1], wsi_dims[0]), coords_prep)
        img[mask] = 1

    return img
