"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2

Tile utilities
"""
import numpy as np
import pandas as pd
from copy import deepcopy


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
