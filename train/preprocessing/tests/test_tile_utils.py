"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2
"""

from train.preprocessing import tile_utils as tu
import numpy as np
from numpy.testing import assert_array_equal


def test_read_tile_roi():
    roi = np.ones((100, 100, 3)) * 5
    tile_size = 10

    # tile fully fits
    tile_coords = np.array([0, 0])
    tile = tu.read_tile_roi(
        roi,
        tile_coords,
        tile_size
    )
    assert_array_equal(tile, np.ones((10, 10, 3)) * 5)

    # tile not fits upper left corner
    tile_coords = np.array([-5, -5])
    tile = tu.read_tile_roi(
        roi,
        tile_coords,
        tile_size
    )
    assert tile[0, 0, 0] == 0
    assert tile[-1, -1, 0] == 5

    # tile not fits upper right corner
    tile_coords = np.array([95, -5])
    tile = tu.read_tile_roi(
        roi,
        tile_coords,
        tile_size
    )
    assert tile[0, -1, 0] == 0
    assert tile[-1, 0, 0] == 5

    # tile not fits lower left corner
    tile_coords = np.array([- 5, 95])
    tile = tu.read_tile_roi(
        roi,
        tile_coords,
        tile_size
    )
    assert tile[-1, 0, 0] == 0
    assert tile[0, -1, 0] == 5

    # tile not fits lower right corner
    tile_coords = np.array([95, 95])
    tile = tu.read_tile_roi(
        roi,
        tile_coords,
        tile_size
    )
    assert tile[-1, -1, 0] == 0
    assert tile[0, 0, 0] == 5
