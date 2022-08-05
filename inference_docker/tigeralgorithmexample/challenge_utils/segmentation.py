"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2

Segmentation workflow
"""

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
from .utils import timing
from .tile_utils import ROITileDatasetInference
from .transforms import (
    CentreFOVDownscaleWithAveraging,
    get_writer,
    mask_to_labels
)

RES_ZERO = 0.5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@timing
def run_segmentation(
        dimensions,
        tile_size,
        tissue_mask,
        slide,
        level,
        params_segmentation,
        models_segmentation,
        segmentation_writer
):
    """
    Segment the WSI.

    Parameters
    ----------
    dimensions: tuple
        The (width, height) of the WSI at level 0.
    tile_size: int
        Size of regions to be processed and written to file one at a time.
        These regions are going to be tiled into smaller pieces before being
        fed into the segmentation models and the whole region mask will be
        reconstructed before writing to file.
    tissue_mask: mir.MultiResolutionImage
        The tissue mask
    slide: openslide.Openslide
        The WSI
    level: int
        The pyramid resolution level of the resulting masks.
    params_segmentation: dict
        Config parameters for segmentation models.
    models_segmentation: dict
        Dict containing the segmentation models for the ensemble.
    segmentation_writer: SegmentationWriter

    """
    # loop over image and get the big tiles
    for y in tqdm(range(0, dimensions[1], tile_size)):
        y_len_fits = tile_size

        # if a big tile doesn't fit vertically, add an offset from the end
        if y > (dimensions[1] - tile_size):
            y_len_fits = dimensions[1] - y
            y = dimensions[1] - tile_size

        for x in range(0, dimensions[0], tile_size):
            x_len_fits = tile_size

            # if a big tile doesn't fit horizontally
            if x > (dimensions[0] - tile_size):
                x_len_fits = dimensions[0] - x
                x = dimensions[0] - tile_size

            tissue_mask_tile = tissue_mask.getUCharPatch(
                startX=x, startY=y,
                width=tile_size, height=tile_size,
                level=level
            ).squeeze()

            if not np.any(tissue_mask_tile):
                continue

            segmentation_masks = torch.zeros(
                (tile_size, tile_size, 3)
            ).to(DEVICE)

            # inference with each model
            for key in list(params_segmentation.keys()):
                segmentation_mask = process_big_image_tile_to_segmentation(
                    slide=slide,
                    model=models_segmentation[key],
                    roi_origin=(x, y),
                    roi_size=tile_size,
                    tile_size=params_segmentation[key]['tile'],
                    tile_res=params_segmentation[key]['resolution'],
                    batch_size=40,
                    padding=params_segmentation[key]['padding'],
                    stride=params_segmentation[key]['stride'],
                    num_workers=0,
                    prefetch_factor=2
                )
                segmentation_masks += segmentation_mask

            segmentation_masks = mask_to_labels(
                segmentation_masks, tissue_mask_tile
            )

            # take care of tiles at the edge of slide
            # mir writer cannot overwrite the same region once written
            if (x_len_fits < tile_size) & (y_len_fits == tile_size):
                # if fits vertically but not horizontally
                slide_edge_helper = np.zeros_like(segmentation_masks)
                slide_edge_helper[
                    :y_len_fits,
                    :x_len_fits,
                ] = segmentation_masks[-y_len_fits:, -x_len_fits:]
                segmentation_writer.write_segmentation(
                    tile=slide_edge_helper, x=dimensions[0] - x_len_fits, y=y
                )
            elif (y_len_fits < tile_size) & (x_len_fits == tile_size):
                # if fits horizontally but not vertically
                slide_edge_helper = np.zeros_like(segmentation_masks)
                slide_edge_helper[
                    :y_len_fits,
                    :x_len_fits,
                ] = segmentation_masks[-y_len_fits:, -x_len_fits:]
                segmentation_writer.write_segmentation(
                    tile=slide_edge_helper, x=x, y=dimensions[1] - y_len_fits
                )
            elif (y_len_fits < tile_size) & (x_len_fits < tile_size):
                # if fits neither horizontally nor vertically
                slide_edge_helper = np.zeros_like(segmentation_masks)
                slide_edge_helper[
                    :y_len_fits,
                    :x_len_fits,
                ] = segmentation_masks[-y_len_fits:, -x_len_fits:]
                segmentation_writer.write_segmentation(
                    tile=slide_edge_helper,
                    x=dimensions[0] - x_len_fits,
                    y=dimensions[1] - y_len_fits
                )
            else:
                # fully fits
                segmentation_writer.write_segmentation(
                    tile=segmentation_masks, x=x, y=y
                )

    slide.close()

    print("Saving segmentation...")
    # save segmentation
    segmentation_writer.save()


@timing
def process_big_image_tile_to_segmentation(
        slide,
        model,
        roi_origin,
        roi_size: int,
        tile_size,
        tile_res,
        batch_size,
        padding,  # at res zero
        stride=None,
        num_workers=0,
        prefetch_factor=2,
        res_zero=RES_ZERO
):
    """
    Process a big ROI from a slide to perform segmentation.
    This pads the ROI, tiles it, and runs batched inference.

    To generate the segmentation mask for the big ROI, only the central
    FOV is used per tile, and overlapping tiles are blended by averaging
    the model scores for each object class.
    """
    small_tiles_dataset = ROITileDatasetInference(
        slide=slide,
        roi_origin=roi_origin,
        roi_size=(roi_size, roi_size),
        tile_size=tile_size,
        tile_res=tile_res,
        stride=stride,
        res_zero=res_zero,
        padding=padding
    )
    small_tiles_loader = DataLoader(
        small_tiles_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # required for multiprocessing with cuda
        prefetch_factor=prefetch_factor
    )

    # get writer for this big roi
    writer = get_writer(
        writer_type='png-averaging',
        write_resolution=res_zero,
        tile_resolution=tile_res,
        res_zero=res_zero,
        tile_padding=padding,
        tile_size=tile_size,
        output_path=None,
        dimensions_zero=(roi_size, roi_size),
        roi_origin=roi_origin
    )

    postprocessing = CentreFOVDownscaleWithAveraging(
        tile_res, res_zero, tile_size, padding, res_zero
    )

    for i, (tiles, tile_origins) in enumerate(small_tiles_loader):

        tiles = tiles.to(torch.float)
        tiles = tiles.to(DEVICE)
        with torch.no_grad():
            y_prob = model(tiles)

        # postprocess tiles and write to mask
        y_prep = postprocessing.apply(y_prob, tile_origins)
        for (prep_tile, x, y) in y_prep:
            writer.write_segmentation(prep_tile, x, y)

    writer.finish()
    return writer.image
