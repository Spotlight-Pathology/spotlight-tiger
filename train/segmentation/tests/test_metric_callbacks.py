"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2
"""

import train.segmentation.metric_callbacks as callbacks
import numpy as np
from numpy.testing import assert_array_equal
import torch


def helper_create_data():
    """create some masks and predictions"""
    label_ignore = 100
    batch = 5
    img_size = 3

    # create ground truth masks:
    # one pixel equals `1`, one pixel equals `2`, all others equal `0`
    # 1 pixel should be ignored
    masks = np.zeros((batch, img_size, img_size))
    masks[0, 0, 0] = 1
    masks[0, 1, 1] = 2
    masks[0, 2, 2] = label_ignore
    masks = torch.from_numpy(masks)

    # assume we predicted everything as `0`:
    yproba_img = np.array([
        [
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 0.99]],
        [
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]
        ],
        [
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.01]
        ]
    ])
    yproba = np.concatenate([yproba_img[np.newaxis, ]] * batch, axis=0)
    yproba = torch.from_numpy(yproba)
    loss = torch.Tensor([0.5])
    return masks, yproba, label_ignore, loss


def test_get_flat_batch():
    masks, yproba, label_ignore, loss = helper_create_data()
    n_examples, y, labels = callbacks.get_flat_batch(
        yproba, masks, label_ignore
    )
    assert n_examples == 44
    assert_array_equal(y[0], (1., 0., 0.))
    assert labels[0] == 1


def test_History_update():
    history = callbacks.History()
    history.start_epoch()
    masks, yproba, label_ignore, loss = helper_create_data()

    # first batch
    history.update(masks, yproba, loss, label_ignore=label_ignore)
    assert history.n_examples == 3 * 3 * 5 - 1
    assert history.n_classes == 3
    assert history.is_first_batch == False
    assert_array_equal(history.tp, (3 * 3 * 5 - 3, 0, 0))
    assert_array_equal(history.fp, (2, 0, 0))
    assert_array_equal(history.tn, (0, 3 * 3 * 5 - 2, 3 * 3 * 5 - 2))
    assert_array_equal(history.fn, (0, 1, 1))
    assert_array_equal(history.preds, (3 * 3 * 5 - 1, 0, 0))
    assert_array_equal(history.trues, (3 * 3 * 5 - 3, 1, 1))
    assert history.loss == [0.5]

    # second batch
    history.update(masks, yproba, loss, label_ignore=label_ignore)
    assert history.n_examples == (3 * 3 * 5 - 1) * 2
    assert history.n_classes == 3
    assert history.is_first_batch == False
    assert_array_equal(history.tp, ((3 * 3 * 5 - 3) * 2, 0, 0))
    assert_array_equal(history.fp, (2 * 2, 0, 0))
    assert_array_equal(
        history.tn,
        (0, (3 * 3 * 5 - 2) * 2, (3 * 3 * 5 - 2) * 2)
    )
    assert_array_equal(history.fn, (0, 2, 2))
    assert_array_equal(history.preds, ((3 * 3 * 5 - 1) * 2, 0, 0))
    assert_array_equal(history.trues, ((3 * 3 * 5 - 3) * 2, 2, 2))
    assert history.loss == [[0.5], [0.5]]


def test_History_calculate_metrics():
    history = callbacks.History()
    history.start_epoch()
    masks, yproba, label_ignore, loss = helper_create_data()
    history.update(masks, yproba, loss, label_ignore=label_ignore)

    history.calculate_metrics(epoch=0)
    metrics = history.metrics
    assert metrics['accuracy'] == 42 / 44
