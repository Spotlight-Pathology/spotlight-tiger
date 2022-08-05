"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2

Detection workflow
"""
from pathlib import Path
import openslide
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from .utils import timing

from ..rw import open_multiresolutionimage_image
from .tile_utils import (
    get_tile,
    resize_tiling,
    SimpleDataset,
    ROITileDatasetInference
)
from .transforms import (
    PostprocessDetections,
    check_is_padded,
    merge_tile_outputs,
    convert_non_padded,
    threshold_outputs,
    slide_nms
)


RES_ZERO = 0.5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@timing
def run_detection_l1(
        image_path,
        dimensions,
        tile_size,
        spacing,
        model_detection,
        detection_writer,
        mask_value_to_detect,
        mask_path,
        params_detection
):
    """
    Detect lymphocytes on the whole tissue region of the WSI for L1.

    Will tile the WSI into regions for processing and will carry out
    NMS on the whole slide before saving the detections .json.

    Parameters
    ----------
    image_path: Path
        Path to the WSI
    dimensions: tuple
        The (width, height) of the WSI at level 0.
    tile_size: int
        Size of regions to be processed and written to file one at a time.
        These regions are going to be tiled into smaller pieces before being
        fed into the detection model and results will be combined for
        the whole region.
    spacing: tuple
        The base resolution (microns / pixel).
    model_detection: torch model
    detection_writer: DetectionWriter
    mask_value_to_detect: int
        Keep only detections found withing this mask value. The mask is
        expected to be parsed as a pyramid .tif file.
    mask_path: Path
        Path to the tissue mask .tif.
    params_detection: dict
        The detection model configuration parameters.

    """
    print("Run detections inference on whole image...")
    mask_wsi = open_multiresolutionimage_image(mask_path)
    slide = openslide.OpenSlide(str(image_path.resolve()))

    # loop over image and get tiles that include tissue
    for y in tqdm(range(0, dimensions[1], tile_size)):
        for x in range(0, dimensions[0], tile_size):

            # if a big tile doesn't fit, add an offset from the end
            if x > (dimensions[0] - tile_size):
                x = dimensions[0] - tile_size

            if y > (dimensions[1] - tile_size):
                y = dimensions[1] - tile_size

            tissue_mask_tile = mask_wsi.getUCharPatch(
                startX=x, startY=y,
                width=tile_size, height=tile_size,
                level=0
            ).squeeze()

            tissue_mask_tile = tissue_mask_tile == mask_value_to_detect
            tissue_mask_tile = np.array(tissue_mask_tile).astype(int)

            if not np.any(tissue_mask_tile):
                continue

            # detections are pixel coordinates relative to roi origin
            detections = process_big_image_tile_to_detections(
                slide=slide,
                model=model_detection,
                roi_origin=(x, y),
                roi_size=tile_size,
                tile_size=params_detection['tile'],
                batch=30,
                tissue_mask_roi=tissue_mask_tile,
                tile_res=params_detection['resolution'],
                padding=params_detection['padding'],  # at res zero
                nms_iou=params_detection['nms_iou'],
                threshold=params_detection['threshold'],
                num_workers=0,
                prefetch_factor=2
            )

            detection_writer.write_detections(
                detections=detections, spacing=spacing, x_offset=x, y_offset=y
            )

    slide.close()

    print("Slide nms...")
    print(
        'Number of detections before nms',
        len(detection_writer._data["points"])
    )

    if len(detection_writer._data["points"]) > 0:
        detection_writer = slide_nms(
            detection_writer,
            params_detection["slide_nms_iou"]
        )
        print(
            'Number of detections after nms',
            len(detection_writer._data["points"])
        )

    print("Saving detections...")
    detection_writer.save()


@timing
def run_detection_l2(
        til_stroma,
        dimensions,
        tile_size,
        spacing,
        image_path,
        model_detection,
        detection_writer,
        params_detection
):
    """
    Detect lymphocytes on up to 180 randomly sampled ROIs from the
    tumour assoc stroma of the WSI for L2.

    Parameters
    ----------
    til_stroma: np.ndarray
        The mask of where the tumour associated stroma is found.
        Downscaled x64 from the level 0 resolution.
    dimensions: tuple
        The (width, height) of the WSI at level 0.
    tile_size: int
        Size of regions to be sampled, processed and written to file
        one at a time.
        These regions are going to be tiled into smaller pieces before being
        fed into the detection model and results will be combined for
        the whole region.
    spacing: tuple
        The base resolution (microns / pixel).
    image_path: Path
        The WSI path
    model_detection: torch model
    detection_writer: DetectionWriter
    params_detection: dict
        The detection model configuration parameters.

    Returns
    -------
    stroma_area: float
        The total tumour associated stroma area from the sampled ROIs
        in microns^2.
    """

    rois, stroma_area = sample_rois(
        mask_to_detect=til_stroma,
        mask_downscale=64,
        dimensions=dimensions,
        tile_size=tile_size,
        spacing_zero=spacing[0]
    )
    run_detection_til_stroma(
        roi_origins=rois,
        image_path=image_path,
        tile_size=tile_size,
        spacing=spacing,
        model_detection=model_detection,
        detection_writer=detection_writer,
        mask_to_detect=til_stroma,
        params_detection=params_detection,
        mask_downscale=64,
        mask_value_to_detect=1
    )

    return stroma_area


def sample_rois(
        mask_to_detect,
        mask_downscale,
        dimensions,  # of wsi at level zero
        tile_size,
        spacing_zero,
        mask_value_to_detect=1,  # keep rois found within this mask value,
        num_rois_max=180
):
    """Sample up to num_rois_max of ROI overlapping with tumour assoc stroma"""

    print('Sampling ROIs for detection...')
    area_pixel = spacing_zero ** 2

    # loop over image and get rois overlapping with mask
    roi_origins, mask_areas = [], []
    for y in tqdm(range(0, dimensions[1], tile_size)):
        for x in range(0, dimensions[0], tile_size):

            # if a big tile doesn't fit, skip it
            if x > (dimensions[0] - tile_size):
                continue

            if y > (dimensions[1] - tile_size):
                continue

            tissue_mask_tile = get_tile(
                x, y, tile_size,
                upscale=mask_downscale,
                mask=mask_to_detect
            )
            tissue_mask_tile = tissue_mask_tile == mask_value_to_detect
            tissue_mask_tile = np.array(tissue_mask_tile).astype(int)

            if not np.any(tissue_mask_tile):
                continue
            else:
                roi_origins.append([x, y])
                # Area is given as um^2
                mask_area = np.sum(tissue_mask_tile) * area_pixel
                mask_areas.append(mask_area)

    mask_areas = np.array(mask_areas)
    roi_origins = np.array(roi_origins)

    if len(roi_origins) <= num_rois_max:
        return roi_origins, np.sum(mask_areas)
    else:
        # sample randomly a subset of the valid rois
        indices = np.array(range(len(roi_origins)))
        np.random.seed(0)
        np.random.shuffle(indices)
        keep = indices[:num_rois_max]
        return roi_origins[keep], np.sum(mask_areas[keep])


def run_detection_til_stroma(
        roi_origins,
        image_path,
        tile_size,
        spacing,
        model_detection,
        detection_writer,
        mask_to_detect,
        params_detection,
        mask_downscale=64,
        mask_value_to_detect=1,
):
    """
    Run detection on a list of ROI.

    Will only detect lymphocytes on tumour assoc stroma, specified as a
    mask array in `mask_to_detect`.

    Parameters
    ----------
    roi_origins: []
        The list of (x, y) tuple coordinates of ROI upper left origins
        given at level 0.
    image_path: Path
        The path to WSI
    tile_size:
        The size of ROI at level 0.
    spacing: tuple
        The base resolution in microns / pixel.
    model_detection: torch model
    detection_writer: DetectionWriter
    mask_to_detect: np.ndarray
        The tumour assoc stroma mask.
    params_detection: dict
        The config parameters for the detection model.
    mask_downscale: int
        How many times was the tumour assoc mask downscaled compared to the
        base level resolution.
    mask_value_to_detect: int
        The value of the mask representing tumour assoc stroma.

    """
    print("Run detections inference on sampled ROIs...")
    slide = openslide.OpenSlide(str(image_path.resolve()))

    # loop over image and get tiles
    for (x, y) in tqdm(roi_origins):
        tissue_mask_tile = get_tile(
            x, y, tile_size,
            upscale=mask_downscale, mask=mask_to_detect
        )
        tissue_mask_tile = tissue_mask_tile == mask_value_to_detect
        tissue_mask_tile = np.array(tissue_mask_tile).astype(int)

        # detections are pixel coordinates relative to roi origin
        detections = process_big_image_tile_to_detections(
            slide=slide,
            model=model_detection,
            roi_origin=(x, y),
            roi_size=tile_size,
            tile_size=params_detection['tile'],
            batch=30,
            tissue_mask_roi=tissue_mask_tile,
            tile_res=params_detection['resolution'],
            padding=params_detection['padding'],  # at res zero
            nms_iou=params_detection['nms_iou'],
            threshold=params_detection['threshold'],
            num_workers=0,
            prefetch_factor=2
        )

        detection_writer.write_detections(
            detections=detections, spacing=spacing, x_offset=x, y_offset=y
        )

    slide.close()

    print("Slide nms...")
    print(
        'Number of detections before nms',
        len(detection_writer._data["points"])
    )

    if len(detection_writer._data["points"]) > 0:
        detection_writer = slide_nms(
            detection_writer,
            params_detection["slide_nms_iou"]
        )
        print(
            'Number of detections after nms',
            len(detection_writer._data["points"])
        )

    print("Saving detections...")
    detection_writer.save()


@timing
def process_big_image_tile_to_detections(
        slide,
        model,
        roi_origin,
        roi_size: int,
        tile_size: int,
        batch: int,
        tissue_mask_roi,
        tile_res,
        padding,  # at res zero
        nms_iou,
        stride=None,
        res_zero=RES_ZERO,
        threshold=0,
        num_workers=0,
        prefetch_factor=2
):
    """
    Process a big ROI from a slide to perform detection.

    Detections are returned as a list(zip(xs, ys, probabilities)),
    where xs, ys the centroid coordinates.

    To generate the detections for the big ROI only the central
    FOV is used (if padding > 0).
    """
    roi, origin = ROITileDatasetInference(
        slide=slide,
        roi_origin=roi_origin,
        roi_size=(roi_size, roi_size),
        tile_size=roi_size + 2 * padding,
        tile_res=tile_res,
        stride=stride,
        res_zero=res_zero,
        padding=padding
    )[0]

    roi = roi.to(DEVICE)
    outputs = tile_and_infer(
        roi,
        resize_by_tiling=tile_size,
        model=model,
        batch=batch,
        stride=tile_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor
    )

    outputs = threshold_outputs([outputs], threshold)

    postprocessing = PostprocessDetections(
        nms_iou=nms_iou,
        padding=padding,
        tile_size=roi_size,
        tissue_mask_tile=tissue_mask_roi
    )
    detections = postprocessing.apply(outputs)
    return detections


def tile_and_infer(
        image,
        resize_by_tiling: int,
        model,
        batch,
        stride,
        num_workers,
        prefetch_factor
):
    """
    Tile a big image and run inference per tile.
    Return the overall predictions for the image.

    Parameters
    ----------
    num_workers: int
        For parallelisation
    prefetch_factor: int
         Number of batches loaded in advance by each worker.
    image: torch.tensor (c, h, w)
    resize_by_tiling: int
    model: torch.nn.Module
    batch: int
        batch size to use after tiling for inference
    stride: int
        stride to use when tiling.

    Returns
    -------
    outputs: {"boxes", "labels", "scores"}
        The predictions for this image.
    """
    tiles, tile_origins, transformed = resize_tiling(
        image.cpu().permute(1, 2, 0).numpy(),
        size=resize_by_tiling,
        stride=stride,
        return_transformed_big_image=True
    )
    tiles = list(
        torch.tensor(tile).permute(2, 0, 1)
        for tile in tiles
    )
    dataset = DataLoader(
        SimpleDataset(tiles),
        batch_size=batch,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=prefetch_factor
    )
    tile_outputs = []
    for tile_batch in dataset:
        tile_batch = list(t.to(DEVICE) for t in tile_batch)
        with torch.no_grad():
            tile_outputs.extend(model(tile_batch))
    outputs = merge_tile_outputs(tile_outputs, tile_origins)

    if check_is_padded(transformed):
        outputs = convert_non_padded(
            outputs, transformed, image.cpu().permute(1, 2, 0).numpy().shape
        )
    return outputs
