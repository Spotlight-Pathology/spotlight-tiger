"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2
"""
import copy
import pandas as pd
import numpy as np
import torch
import os
from train.detection import froc_fn
from train.preprocessing.transforms import get_centroid
import matplotlib.pyplot as plt

RES_ZERO = 0.5  # microns / pixel


def remove_other_targets(targets):
    """Keep only targets of label 1."""
    prep_targets = copy.deepcopy(targets)
    for target in prep_targets:
        id_keep = [
            t for (t, label) in enumerate(target['labels'].detach().cpu())
            if label.item() == 1
        ]
        target['labels'] = target['labels'][id_keep]
        target['boxes'] = target['boxes'][id_keep, :]
        target['area'] = target['area'][id_keep]
        target['iscrowd'] = target['iscrowd'][id_keep]
    return prep_targets


def remove_other_output(target: dict):
    """Keep only predictions of label 1."""
    id_keep = [
        t for (t, label) in enumerate(target['labels'].detach().cpu())
        if label.item() == 1
    ]
    target['labels'] = target['labels'][id_keep]
    target['boxes'] = target['boxes'][id_keep, :]
    target['scores'] = target['scores'][id_keep]
    return target


def unpack_detection_results(targets, targets_pred):
    """
    Parameters
    ----------
    targets: dict, len=batch_size
        See the target dict as described here:
        https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    targets_pred: dict, len=batch_size
        Maps the image_id to the predicted boxes.
        Example:
        {0: {
            'boxes': FloatTensor[N, 4],
            'labels': Int64Tensor[N],  # the object predicted class
            'scores': FloatTensor[N]  # the object probabilities
            }
        }

    """
    probs, gts, preds = [], [], []
    img_ids = [target['image_id'].item() for target in targets]
    for img_count, img_id in enumerate(img_ids):
        targets_pred_im = targets_pred[img_id]
        probs.append(targets_pred_im['scores'].numpy())

        targets_im = [t for t in targets if t['image_id'] == img_id][0]
        gts.append(np.array([
            get_centroid(bbox)
            for bbox in targets_im['boxes'].cpu().numpy()
        ]))

        if len(targets_pred_im['boxes'].numpy()) > 0:
            preds.append(np.array([
                get_centroid(bbox) + (probs[img_count][i],)
                for i, bbox in enumerate(targets_pred_im['boxes'].numpy())
            ]))
        else:
            preds.append(np.array([]))
    return probs, gts, preds


class History:
    """
    Helper class to accumulate metrics over batches/ epochs.
    """

    def __init__(self, mode):
        self.im_paths = None
        self.sizes = None
        self.areas = None
        self.mode = mode
        self.last_model = None
        self.last_optimizer = None
        self.best_optimizer = None
        self.loss = []
        self.loss_classifier = []
        self.loss_box_reg = []
        self.loss_objectness = []
        self.loss_rpn_box_reg = []
        self.best_model = None
        self.best_metrics = None
        self.metrics = None
        self.gts = []
        self.preds = []
        self.area_mm = 0
        self.probs = []
        # at object threshold 0.5:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.is_epoch_started = False
        self.plateau_tracker = []
        self.plateau_counter = 0

        self.tprs = None
        self.av_fps = None
        self.fps = None

    def __str__(self):
        return self.metrics.__str__()

    def start_epoch(self):
        if self.mode == 'val':
            self.gts = []
            self.preds = []
            self.area_mm = 0
            self.areas = []
            self.probs = []
            self.sizes = []
            self.im_paths = []
            self.tp = 0
            self.fp = 0
            self.fn = 0
            self.tprs = None
            self.av_fps = None

        elif self.mode == 'train':
            self.loss = []
            self.loss_classifier = []
            self.loss_box_reg = []
            self.loss_objectness = []
            self.loss_rpn_box_reg = []

        self.is_epoch_started = True

    def update(
            self,
            loss=None,
            targets=None,
            targets_pred=None,
            areas=None,
            sizes=None,
            im_paths=None
    ):
        """
        Update history from this batch.
        """
        assert self.is_epoch_started
        if self.mode == 'val':
            probs, gts, preds = unpack_detection_results(
                targets, targets_pred
            )
            self.probs.extend(probs)
            self.gts.extend(gts)
            self.preds.extend(preds)
            self.area_mm += sum(areas)
            self.areas.extend(areas)
            self.sizes.extend(sizes)
            self.im_paths.extend(im_paths)

            tp, fn, fp = froc_fn.compute_scores(
                gts, preds, dist_thresh=8, thresholds=[0.5]
            )
            self.tp += tp[0]
            self.fn += fn[0]
            self.fp += fp[0]

        elif self.mode == 'train':
            self.loss.append(sum(loss for loss in loss.values()).item())
            self.loss_classifier.append(loss['loss_classifier'].item())
            self.loss_box_reg.append(loss['loss_box_reg'].item())
            self.loss_objectness.append(loss['loss_objectness'].item())
            self.loss_rpn_box_reg.append(loss['loss_rpn_box_reg'].item())

    def check_plateau(self):
        if self.mode == 'train':
            loss_last_epoch = self.metrics['loss']

            if len(self.plateau_tracker) > 0:
                loss_id_to_compare = -1 - self.plateau_counter
                loss_to_compare = self.plateau_tracker[loss_id_to_compare]
                not_decreased = loss_last_epoch >= loss_to_compare
                if not_decreased:
                    self.plateau_counter += 1
                    print('Plateau counter: ' + str(self.plateau_counter))
                else:
                    self.plateau_counter = 0

            self.plateau_tracker.append(loss_last_epoch)

    def calculate_metrics(self, epoch: int, is_epoch_end=False):

        if self.mode == 'val':
            if is_epoch_end:

                num_gts = 0
                for gts in self.gts:
                    num_gts += len(gts)
                targets_density = num_gts / self.area_mm

                num_preds = 0
                for preds in self.preds:
                    num_preds += len(preds)
                preds_density = num_preds / self.area_mm

                try:
                    froc_score, tprs, av_fps, fps = froc_fn.get_froc_all_imgs(
                        self.probs, self.gts, self.preds, self.area_mm
                    )
                except Exception:
                    # if no detections predicted, froc is set to nan
                    froc_score = float("Nan")
                    tprs, av_fps = [], []

                self.metrics = {
                    'froc': froc_score,
                    'epoch': epoch,
                    'targets per mm2': targets_density,
                    'preds per mm2': preds_density,
                    'accuracy': accuracy(self.tp, self.fp, self.fn),
                    'precision': precision(self.tp, self.fp),
                    'recall': recall(self.tp, self.fn),
                    'f1': f1(self.tp, self.fp, self.fn)
                }

                # keep values to plot froc curve
                self.tprs = tprs
                self.av_fps = av_fps
                self.fps = fps

        elif self.mode == 'train':
            self.metrics = {
                'loss': np.mean(self.loss),
                'loss_classifier': np.mean(self.loss_classifier),
                'loss_box_reg': np.mean(self.loss_box_reg),
                'loss_objectness': np.mean(self.loss_objectness),
                'loss_rpn_box_reg': np.mean(self.loss_rpn_box_reg),
                'epoch': epoch,
            }

    def keep_best(self, model, optimizer, metric_of_interest='f1'):
        """
        Keep track of the metrics + model of the best epoch so far
        by comparing the metric_of_interest (loss by default).

        Assumes that higher values of the metric are better.
        """
        # if 1st epoch
        if not self.best_metrics:
            self.best_metrics = self.metrics
            self.best_model = copy.deepcopy(model)
            self.best_optimizer = copy.deepcopy(optimizer)

        # if current epoch did not improve the metric of interest
        elif self.best_metrics[
            metric_of_interest
        ] > self.metrics[metric_of_interest]:
            pass

        else:
            self.best_metrics = self.metrics
            self.best_model = copy.deepcopy(model)
            self.best_optimizer = copy.deepcopy(optimizer)

    def keep_last(self, model, optimizer):
        """
        Keep track of the model and optimizer so far.
        """
        self.last_model = copy.deepcopy(model)
        self.last_optimizer = copy.deepcopy(optimizer)

    def save_best_checkpoint(self, save_dir, prefix):
        """
        Dump the best checkpoint and metrics locally.
        """
        os.makedirs(save_dir, exist_ok=True)

        metrics_file = os.path.join(save_dir, prefix + '_best_metrics.csv')
        pd.DataFrame(self.best_metrics, index=[0]).to_csv(metrics_file)

        model_file = os.path.join(save_dir, 'best_checkpoint.tar')
        torch.save({
            'epoch': self.best_metrics['epoch'],
            'model_state_dict': self.best_model.state_dict(),
            'optimizer_state_dict': self.best_optimizer.state_dict(),
        },
            model_file
        )

    def save_last_checkpoint(self, save_dir):
        """
        Dump the last checkpoint and metrics locally.
        """
        os.makedirs(save_dir, exist_ok=True)

        model_file = os.path.join(save_dir, 'last_checkpoint.tar')
        torch.save({
            'model_state_dict': self.last_model.state_dict(),
            'optimizer_state_dict': self.last_optimizer.state_dict(),
        },
            model_file
        )

    def save_metric_plots(self, save_dir):
        plt.plot(self.av_fps, self.tprs, "b")
        plt.xlabel('average fps')
        plt.ylabel('tpr')
        figure_name = os.path.join(save_dir, "froc_plot.png")
        plt.savefig(figure_name)

        # save csv of plot
        curve_points = pd.DataFrame()
        curve_points['av_fps'] = self.av_fps
        curve_points['tpr'] = self.tprs
        curve_points["fps"] = self.fps
        curve_points['area_mm'] = self.area_mm
        curve_points_filename = os.path.join(save_dir, 'curve.csv')
        curve_points.to_csv(curve_points_filename)


def precision(tp, fp):
    return tp / (tp + fp) if tp > 0 else 0


def recall(tp, fn):
    return tp / (tp + fn) if tp > 0 else 0


def accuracy(tp, fp, fn):
    # also known as "average precision"
    return tp / (tp + fp + fn) if tp > 0 else 0


def f1(tp, fp, fn):
    # also known as "dice coefficient"
    return (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0
