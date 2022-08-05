"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2
"""

import numpy as np
from train.detection import froc_fn


def test_get_froc_all_imgs():
    area_mm = 224 * 224 * 2
    probs = [
        np.array([0.5, 0.6]),
        np.array([0.5001, 0.6, 0.9]),
    ]

    preds = [
        np.array([
            [0, 1, 0.6],
            [1, 1, 0.6]
        ]),
        np.array([
            [0, 1, 0.5001],
            [1, 1, 0.6],
            [1, 2, 0.9]
        ]),
    ]

    gts = [
        np.array([
            [0, 0],
            [1, 2]
        ]),
        np.array([
            [10, 10],
            [1, 1],
            [1, 3]
        ]),
    ]

    froc_auc, _, _, _ = froc_fn.get_froc_all_imgs(probs, gts, preds, area_mm)
    assert np.round(froc_auc, 4) == 0.8

    # test case with no gts
    gts = [[], []]
    froc_auc, _, _, _ = froc_fn.get_froc_all_imgs(probs, gts, preds, area_mm)
    assert froc_auc == 0

    # test case with no preds
    preds = [[], []]
    gts = [
        np.array([
            [0, 0],
            [1, 2]
        ]),
        np.array([
            [10, 10],
            [1, 1],
            [1, 3]
        ])
    ]
    froc_auc, _, _, _ = froc_fn.get_froc_all_imgs(probs, gts, preds, area_mm)
    assert froc_auc == 0

    # case with no gts and no preds
    preds = [[], []]
    ts = [[], []]
    froc_auc, _, _, _ = froc_fn.get_froc_all_imgs(probs, gts, preds, area_mm)
    assert froc_auc == 0
