"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2

Based on the challenge froc metric:
https://github.com/DIAGNijmegen/pathology-tiger-algorithm-example/blob/main/evaluations/eval_det_froc_snippet.py
"""
import bisect
from evalutils.evalutils import score_detection
import numpy as np

TARGET_FPS = [10, 20, 50, 100, 200, 300]


def compute_scores(gt_coords, pred_coords, dist_thresh, thresholds):
    """
    Computes the overall tps, fns, fps per threshold given a list of gt and
    predictions for multiple slides
    gtcoords:[[x,y]], predcoords: [[x,y,prob]]
    """
    assert len(gt_coords) > 0
    assert len(pred_coords) > 0
    n_thresh = len(thresholds)
    tps = np.zeros((n_thresh))
    fns = np.zeros((n_thresh))
    fps = np.zeros((n_thresh))
    for i, thresh in enumerate(thresholds):
        for gt, pred in zip(gt_coords, pred_coords):
            if len(pred) == 0:
                fns[i] += len(gt)
            elif len(gt) == 0:
                thresh_pred = pred[np.where(pred[:, 2] >= thresh)[0], :2]
                fps[i] += len(thresh_pred)
            else:
                thresh_pred = pred[np.where(pred[:, 2] >= thresh)[0], :2]
                det_score = score_detection(
                    ground_truth=gt,
                    predictions=thresh_pred,
                    radius=dist_thresh
                )
                tps[i] += det_score.true_positives
                fns[i] += det_score.false_negatives
                fps[i] += det_score.false_positives
    return tps, fns, fps


def compute_froc_score(
        tprs: list,
        fps: list,
        target_fps: list,
        interpolate_edges=True,
        verbose=True
):
    """
    Compute the average sensitivity at predefined false positive rates.

    Parameters
    ----------
    tprs: []
        List of true positive rates for different thresholds.
    fps: []
        List of (average) false positives for different thresholds.
    target_fps: []
        Target fps for score calculation.

    Returns
    -------
    froc_score: float
        Computed FROC score.
    """
    if interpolate_edges:
        # If starts after 0, add 0-entry
        if fps[0] != 0:
            fps.insert(0, 0)
            tprs.insert(0, 0)

        # if ends before one of the target fps, add missing
        # (horizontal interpolation)
        for t_fp in target_fps:
            if t_fp > max(fps):
                fps.append(t_fp)
                tprs.append(tprs[-1])

    n_thresh = len(tprs)
    if verbose:
        print(
            'computing froc score with %d thresholds and average fps: %s' % (
                n_thresh, str(target_fps))
        )

    target_sensitivities = []
    for target_fp in target_fps:

        target_index = bisect.bisect_right(fps, target_fp) - 1
        target_index = min(target_index, n_thresh - 1)
        target_sensitivities.append(tprs[target_index])

    froc_score = sum(target_sensitivities) / len(target_fps)
    return froc_score


def get_froc_all_imgs(probs, gts, preds, area_mm, verbose=True):
    probs = np.sort(np.unique(np.hstack(probs)))
    # from big to small, important for bisect to get the selected tprs
    thresholds = probs[::-1]

    tps, fns, fps = compute_scores(
        gts, preds, dist_thresh=8, thresholds=thresholds
    )

    if sum(tps) == 0:
        froc = 0
        tprs = []
        av_fps = []
        modified_fps = []
    else:
        tprs = [tp / (tp + fn) for tp, fn in zip(tps, fns)]
        # area_mm is the tissue area  in mm2
        av_fps = [fp / area_mm for fp in fps]
        froc = compute_froc_score(tprs, av_fps, TARGET_FPS, verbose)

        modified_fps = [fp * area_mm for fp in av_fps]

    return froc, tprs, av_fps, modified_fps
