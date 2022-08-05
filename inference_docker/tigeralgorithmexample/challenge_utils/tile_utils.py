"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2

Utils for WSI tiling.
"""
import numpy as np
import pandas as pd
from copy import deepcopy
from skimage.transform import resize
from torch.utils.data import Dataset
import albumentations as A
import cv2
import torch
from ..rw import open_multiresolutionimage_image

ZERO_RES = 0.5


def check_tile_fits(tile_coords, tile_size, slide_dimensions):
    """
    Check the tile fits fully in the slide.

    Parameters
    ----------
    tile_coords: tuple
        The x, y coordinate of the tile at level 0
    tile_size: int
        The size of the tile at level 0
    slide_dimensions: tuple
        The width, height of the slide at level 0
    """
    width, height = slide_dimensions
    fits_left = tile_coords[0] >= 0
    fits_up = tile_coords[1] >= 0
    fits_right = (tile_coords[0] + tile_size) < (width - 1)
    fits_down = (tile_coords[1] + tile_size) < (height - 1)
    return fits_left, fits_right, fits_up, fits_down


def read_tile(
        slide,
        tile_coords,
        tile_size,
        tile_resolution,
        full_res
):
    """
    Get a single tile of size tile_size, given the tile_coords,
    at the requested tile_resolution.

    If the tile doesn't fit fully, it will be padded with zeros.

    Parameters
    ----------
    slide: openslide.OpenSlide
    tile_coords: numpy.array
        (X, Y) coordinates of tile's upper left corner, at pyramid level 0
    tile_size: int
        The tile size (pixels)
    tile_resolution: float
        The extracted tile's resolution, in microns / pixel.
    full_res: float
        The resolution at level 0 (microns/ pixel).
    Returns
    -------
    tile: np.array, shape=(tile_size, tile_size, 3)
        RGB image of the tile
    """
    x, y = deepcopy(tile_coords)
    down_factor = tile_resolution // full_res
    tile_size_level_zero = int(tile_size * down_factor)

    # check if tile fits in level 0 dimensions
    width, height = slide.level_dimensions[0]

    # read the image from the part that fits
    x, y, tile_width, tile_height, place_h, place_v = dimensions_that_fit(
        tile_coords, tile_size_level_zero, width, height
    )

    tile_img = slide.read_region(
        location=(x, y),
        level=0,
        size=(tile_width, tile_height)
    )
    tile_img = np.array(tile_img.convert('RGB'))  # np.array from PIL image
    tile = np.zeros((tile_size_level_zero, tile_size_level_zero, 3))
    tile = tile.astype(np.uint8)

    # place image in padded tile
    tile[place_v, place_h, :] = tile_img

    # downscale to resolution specified
    tile = resize(tile, (tile_size, tile_size), anti_aliasing=True)
    return tile


def dimensions_that_fit(tile_coords, tile_size, im_width, im_height):
    """
    Given a tile, return the tile subset coordinates for the part of the tile
    that fits fully inside the ROI.

    Parameters
    ----------
    tile_coords: np.array
        the (x, y) tile coordinates of the tile's upper left corner.
    tile_size: int
        The tile size (pixels).
    im_width: int
        The size of the ROI we're extracting a tile from.
    im_height: int
        The height of the ROI.
    Returns
    -------
    x: int
        The x origin coordinate of the part of the tile that fully fits.
    y: int
        The y origin.
    tile_width: int
        The tile width, corrected so it fully fits in the slide.
    tile_height: int
        The tile height, corrected so it fully fits.
    place_h: slice
        The horizontal placement of the part of the tile that fully fits,
        related to the whole tile.
    place_v: slice
        The vertical placement of the part of the tile that fully fits,
        related to the whole tile.

    Example
    -------
     (0, 0, 5, 10) = dimensions_that_fit((-5, 0), 10, 100, 100)
    """
    x, y = deepcopy(tile_coords)
    tile_height = tile_size
    tile_width = tile_size

    fits_left, fits_right, fits_up, fits_down = check_tile_fits(
        tile_coords, tile_size, (im_width, im_height)
    )

    if not fits_right:
        # If tile does not fit to the right, read only the part that fits
        tile_width = int(im_width - 1 - tile_coords[0])

    if not fits_down:
        # If tile does not fit at the bottom, read only the part that fits
        tile_height = int(im_height - 1 - tile_coords[1])

    if not fits_left:
        tile_width = tile_width + tile_coords[0]  # tile_coord is negative
        x = 0

    if not fits_up:
        tile_height = tile_height + tile_coords[1]  # tile_coord is negative
        y = 0

    place_v = slice(0, tile_height)
    place_h = slice(0, tile_width)

    if not fits_up:
        place_v = slice(tile_size - tile_height, tile_size)
    if not fits_left:
        place_h = slice(tile_size - tile_width, tile_size)

    return x, y, tile_width, tile_height, place_h, place_v


def build_tile_grid(
        bbox_shape,
        tile_size,
        stride=None,
        bbox_origin=(0, 0),
        padding=0
):
    """
    For a bounding box with given width and height and origin coordinate,
    build a grid of tiles inside it.

    The bounding box can encompass an entire slide
    or a relevant region on the slide.

    If a tile does not fully fit, an offset from the end is added.

    Parameters
    ----------
    bbox_shape: tuple
        The (height, width) of the bounding box
    tile_size: int
        The tile size in pixels (assume square tiles)
    stride: int
        The step between one tile and the next (in pixels).
        If not set, stride is considered equal to tile_size.
    bbox_origin: tuple
        (X, Y) coordinates of the bounding box's origin
        By default equals (0, 0).
    padding: int
        By default equals 0. Add padding around the tiles before building grid.
        Ensure the central FOV fits fully inside the grid.


    Returns
    -------
    tile_origins: np.array, shape=(number_tiles, 2)
        X and Y dimensions of the tile origins (upper left corner of tile)
    """
    tile_size = tile_size - 2 * padding  # make sure the central FOV fits fully
    assert (tile_size <= bbox_shape[0]) & (tile_size <= bbox_shape[1])

    if stride is None:
        stride = tile_size

    assert stride <= tile_size  # otherwise we get gaps between tiles
    xo, yo = bbox_origin  # starting point of the grid
    tile_origins = []

    bbox_height, bbox_width = bbox_shape

    for x in range(xo, xo + bbox_width, stride):
        for y in range(yo, yo + bbox_height, stride):

            # if a tile doesn't fit, add an offset from the end
            if x > (xo + bbox_width - tile_size):
                x = xo + bbox_width - tile_size

            if y > (yo + bbox_height - tile_size):
                y = yo + bbox_height - tile_size

            tile_origins.append([x, y])

    tile_origins = pd.DataFrame(data=tile_origins, columns=['x', 'y'])
    tile_origins.drop_duplicates(inplace=True)

    tile_origins = tile_origins.values - padding  # shift coordinates
    return tile_origins


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
        A.Resize(height=downscaled_height, width=downscaled_width),
        A.PadIfNeeded(
            min_height=size,
            min_width=size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0
        )
    ])

    if mask is not None:
        transformed = transform(image=img, mask=mask)
        img, mask = transformed['image'], transformed['mask']

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


def read_tile_roi(
        roi,
        tile_coords,
        tile_size,

):
    """
    Get a single tile of size tile_size from a ROI np.array image, given the
    tile_coords.

    If the tile doesn't fit fully, it will be padded with zeros.

    Parameters
    ----------
    roi: np.array
        An RGB image representing a region of interest (ROI).
    tile_coords: np.array
        (X, Y) coordinates of tile's upper left corner, at the ROI resolution.
    tile_size: int
        The tile size (pixels)
    Returns
    -------
    tile: np.array, shape=(tile_size, tile_size, 3)
        RGB image of the tile
    """
    # check if tile fits
    height, width = roi.shape[0], roi.shape[1]
    if len(roi.shape) == 2:  # 2D image
        tile = np.zeros((tile_size, tile_size))
    elif len(roi.shape) == 3:
        tile = np.zeros((tile_size, tile_size, 3))
    else:
        raise Exception('Wrong input ROI dimensions.')
    tile = tile.astype(roi.dtype)

    # read the image from the part that fits
    x, y, tile_width, tile_height, place_h, place_v = dimensions_that_fit(
        tile_coords, tile_size, width, height
    )

    tile_img = roi[
        slice(y, y + tile_height),
        slice(x, x + tile_width)
    ]

    # place image in padded tile
    tile[place_v, place_h] = tile_img
    return tile


class SimpleDataset(Dataset):
    def __init__(self, tensor_list):
        self.tensor_list = tensor_list

    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, idx):
        return self.tensor_list[idx]


def get_tile(x, y, tile_size, upscale, mask):
    """
    Get an up-scaled tile from a mask,
    given the x, y, coordinates for the up-scaled mask.
    Essentially picks a tile from a mask and zooms in.
    """
    # get downscaled coords
    x_down = x // upscale
    y_down = y // upscale
    tile_size_down = tile_size // upscale

    tile = mask[
        y_down: y_down + tile_size_down,
        x_down: x_down + tile_size_down
    ]

    #  upscale mask tile with nearest neighbour interpolation
    tile = resize(
        tile,
        output_shape=(tile_size, tile_size),
        order=0
    )
    return tile


def get_downscaled_img(slide_path, downsample, level):
    """Get a downscaled thumbnail of a WSI"""
    tissue_mask = open_multiresolutionimage_image(path=slide_path)

    wsi_dims_zero = tissue_mask.getDimensions()
    wsi_dims = np.array(wsi_dims_zero) // downsample

    tissue_mask_img = tissue_mask.getUCharPatch(
        0, 0,
        int(wsi_dims[0]), int(wsi_dims[1]),
        level
    ).squeeze()
    return tissue_mask_img


class ROITileDatasetInference(Dataset):
    def __init__(
            self,
            slide,
            roi_origin: tuple,
            roi_size: tuple,  # (width, height)
            tile_size,
            tile_res,
            stride,
            res_zero,
            padding=0
    ):
        """
        Initialise a torch Dataset from a big ROI by tiling it
        into patches of tile_size.

        Assumes the ROI can fully fit into memory.

        Parameters
        ----------
        slide:  OpenSlide
        tile_size: int
        tile_res: float
            The tile resolution in microns/ pixel
        stride: int
            The step between tiles.
        res_zero: float
            The resolution at level 0 (microns/ pixel).
        padding: int or None
            Add padding around the ROI before creating the tile grid.
            Given at res_zero. Used at inference.

        """
        self.roi_origin = roi_origin
        self.roi_dimensions = roi_size
        self.tile_size = tile_size
        self.tile_res = tile_res
        self.res_zero = res_zero
        self.slide = slide
        downscale_factor = tile_res / res_zero

        tile_origins = build_tile_grid(
            bbox_shape=(self.roi_dimensions[1], self.roi_dimensions[0]),
            tile_size=int(tile_size * downscale_factor),  # at zero,
            stride=int(
                stride *
                downscale_factor) if stride is not None else None,
            padding=padding,  # at zero
            bbox_origin=self.roi_origin
        )
        self.tile_origins = np.array(tile_origins)

    def __len__(self):
        return len(self.tile_origins)

    def __getitem__(self, idx):

        tile = read_tile(
            self.slide,
            self.tile_origins[idx, :],
            tile_size=self.tile_size,
            tile_resolution=self.tile_res,
            full_res=self.res_zero
        )

        tile = torch.from_numpy(tile).float()
        tile = tile.permute(2, 0, 1)  # torch requires (c, h, w) format
        tile_origin = torch.from_numpy(np.array(self.tile_origins[idx, :]))

        return tile, tile_origin
