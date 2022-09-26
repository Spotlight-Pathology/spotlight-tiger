"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2

Til score calculation
"""
import numpy as np
import json
from ..rw import open_multiresolutionimage_image
from .tile_utils import get_downscaled_img
from .transforms import convert_xml_to_mask
from pathlib import Path
from .utils import timing
import math

# the lymphocyte score cut-off used
THRESHOLD = 0.5


def get_til_stroma(
        tumour_bulk_xml: Path,
        dimensions,
        spacing: float,
        downscale,
        downscale_level,
        segmentation_path: Path
):
    """
    Read the tumour bulk xml and get a downscaled mask array
    of the tumour associated stroma by combining with the segmentation result.
    """
    tumour_bulk = convert_xml_to_mask(
        str(tumour_bulk_xml.resolve()),
        in_resolution=spacing,
        out_resolution=spacing * downscale,
        wsi_dims_zero=dimensions
    )
    segmentation_mask = get_downscaled_img(
        segmentation_path,
        level=downscale_level, downsample=downscale
    )
    til_stroma_mask = (tumour_bulk == 1) & (segmentation_mask == 2)
    return til_stroma_mask


def get_lymph_area_at_thresholds(
        detections_json,
        res_zero,
        thresholds,
        mask,
        mask_downsample
):
    """
    Calculate the area occupied by lymphocytes at each object score threshold.

    **Area is given as um^2.**
    Assumes lymphocytes non overlapping.

    Parameters
    ----------
    detections_json: Path
    downsample: float
    res_zero: float
    thresholds: []
        The thresholds to calculate the area for.
    mask: np.array
        The stroma bool mask - keep lymphocytes only where True
    mask_downsample: int
        The downsample factor of the mask.
    """
    detections_json = str(detections_json.resolve())
    with open(detections_json) as f:
        json_detections = json.load(f)['points']

    points, scores = [], []
    for item in json_detections:
        points.append(item["point"][:2])
        scores.append(float(item["probability"]))
    points = np.array(points) * 1000 / res_zero  # pixels
    points = points.astype(np.float32)
    scores = np.array(scores)

    points_mask = (points / mask_downsample).astype(int)
    is_stroma = []
    for point in points_mask:
        is_stroma.append(mask[point[1], point[0]])
    is_stroma = np.array(is_stroma)

    cell_area = 16 * 16  # pixels at res_zero
    area_pixel = res_zero * res_zero
    cell_area = cell_area * area_pixel  # um^2
    areas = []

    for thresh in thresholds:
        is_valid = (scores > thresh) & is_stroma
        thresh_points = points[np.where(is_valid)]
        areas.append(len(thresh_points) * cell_area)

    return np.array(areas)


def get_lymph_ratio_at_thresholds(
        tumour_bulk_xml: Path,
        segmentation_tif: Path,
        detections_json: Path,
        stroma_area: float = None,  # in um2
        downsample=64,
        downsample_level=6,
        thresholds=(THRESHOLD,)
):
    segmentation_wsi = open_multiresolutionimage_image(segmentation_tif)
    res_zero = segmentation_wsi.getSpacing()[0]  # um / pixel
    dimensions = segmentation_wsi.getDimensions()

    stroma_mask = get_til_stroma(
        tumour_bulk_xml,
        dimensions,
        spacing=res_zero,
        downscale=downsample,
        downscale_level=downsample_level,
        segmentation_path=segmentation_tif
    )

    lymph_area_at_thresholds = get_lymph_area_at_thresholds(
        detections_json=detections_json,
        res_zero=res_zero,
        thresholds=thresholds,
        mask=stroma_mask,
        mask_downsample=downsample
    )

    if stroma_area is None:
        # if not pre-specified, calculate it from whole mask
        res_out = res_zero * downsample  # um / pixel
        area_pixel_out = res_out ** 2
        stroma_area = np.sum(stroma_mask) * area_pixel_out

    ratios = (lymph_area_at_thresholds / stroma_area) * 100
    return ratios


@timing
def get_tils_score(
        segmentation_tif,
        tumour_bulk_xml,
        detections_json,
        stroma_area=None,
        thresholds=(THRESHOLD,)
):
    try:
        ratio = get_lymph_ratio_at_thresholds(
            tumour_bulk_xml=tumour_bulk_xml,
            segmentation_tif=segmentation_tif,
            detections_json=detections_json,
            thresholds=thresholds,
            stroma_area=stroma_area
        )[0]

    except Exception:
        # if for some reason we did not find any tumour bulk/ stroma
        # til score does not make sense - is undefined
        # set the value to the median
        ratio = 4.01

    if math.isnan(ratio):
        ratio = 4.01

    # convert to [0, 100]
    score = linear_transform(ratio)

    # restriction by Grand Challenge platform
    if score > 100:
        score = 100
    if score < 0:
        score = 0

    return score


def linear_transform(x):
    """Huber regression with l=1.9"""
    coef = 5.37118222
    intercept = -2.9716045232664814
    y = coef * x + intercept
    return int(y)
