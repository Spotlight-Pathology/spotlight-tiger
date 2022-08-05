"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2
"""

import copy
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import os


def get_flat_batch(yproba, masks, label_ignore=None):
    """
    Flatten the yproba and masks to a list of pixels for a batch.

    Parameters
    ----------
    yproba: torch.Tensor, size = (batch, n_classes, h, w)
        Output of the last Linear layer of the network after Softmax.
    masks: torch.Tensor, size = (batch, h, w)
        The int mask labels.
    label_ignore: int
        A label to ignore when calculating metrics.

    Returns
    -------
    n_examples: int,
        The numbers of pixels, excluding the ignored label.
    y: np.array, shape=(n_examples, n_classes)
    labels: np.array, shape=(n_examples,)
    """
    yproba = yproba.detach().cpu().permute(0, 2, 3, 1).numpy()  # (b, h, w, c)
    masks = masks.detach().cpu().numpy()

    if label_ignore is None:
        select = np.ones_like(masks).astype(bool)
    else:
        select = masks != label_ignore

    n_examples = np.sum(select)
    all_pixels = masks.shape[0] * masks.shape[1] * masks.shape[2]
    select = select.reshape(all_pixels)
    y = yproba.reshape(all_pixels, -1)[select, :]
    labels = masks.reshape(all_pixels)[select]
    return n_examples, y, labels


class History:
    """
    Helper class to accumulate metrics over batches/ epochs.
    """

    def __init__(self):
        self.last_model = None
        self.last_optimizer = None
        self.best_optimizer = None
        self.loss = []
        self.best_model = None
        self.best_metrics = None
        self.metrics = None
        self.n_examples = None
        self.trues = None
        self.preds = None
        self.fn = None
        self.tn = None
        self.fp = None
        self.tp = None
        self.n_classes = None
        self.is_first_batch = True

    def __str__(self):
        return self.metrics.__str__()

    def start_epoch(self):
        self.is_first_batch = True

    def update(self, masks, yproba, loss, label_ignore=None):
        """
        Update history from this batch.

        Parameters
        ----------
        masks: torch.Tensor, size = (batch, h, w)
            The int labels.
        yproba: torch.Tensor, size = (batch, n_classes, h, w)
            Output of the last Linear layer of the network after Softmax.
        loss: torch.Tensor, size=1
            The loss
        label_ignore: int
            A label to ignore when calculating metrics.
        """
        if self.is_first_batch:
            self.is_first_batch = False
            self.n_classes = yproba.size()[1]
            assert self.n_classes > 1

            n_examples, y, labels = get_flat_batch(yproba, masks, label_ignore)

            self.n_examples = n_examples
            tp, fp, tn, fn, preds, trues = get_batch_metrics(labels, y)
            self.tp = tp
            self.fp = fp
            self.tn = tn
            self.fn = fn
            self.preds = preds
            self.trues = trues
            self.loss = [loss.detach().cpu().numpy()]
        else:
            n_examples, y, labels = get_flat_batch(yproba, masks, label_ignore)

            tp, fp, tn, fn, preds, trues = get_batch_metrics(labels, y)
            self.tp += tp
            self.fp += fp
            self.tn += tn
            self.fn += fn
            self.preds += preds
            self.trues += trues
            self.n_examples += n_examples
            self.loss.append(loss.detach().cpu().numpy())

    def calculate_metrics(self, epoch: int):
        self.metrics = {
            'accuracy': get_accuracy(self.tp, self.n_examples),
            'loss': np.mean(self.loss), 'epoch': epoch,
        }
        precisions = get_precision(self.tp, self.preds)
        recalls = get_recall(self.tp, self.trues)
        dices = get_dice(self.tp, self.fp, self.fn)

        for i in range(self.n_classes):
            self.metrics['precision_{}'.format(i)] = precisions[i]
            self.metrics['recall_{}'.format(i)] = recalls[i]
            self.metrics['dice_{}'.format(i)] = dices[i]

        ts_dice = np.mean([self.metrics['dice_1'], self.metrics['dice_2']])
        self.metrics['ts_dice'] = ts_dice

    def keep_best(self, model, optimizer, metric_of_interest='loss'):
        """
        Keep track of the metrics + model of the best epoch so far
        by comparing the metric_of_interest (loss by default).

        Assumes that lower values of the metric are better.
        """
        # if 1st epoch
        if not self.best_metrics:
            self.best_metrics = self.metrics
            self.best_model = copy.deepcopy(model)
            self.best_optimizer = copy.deepcopy(optimizer)

        # if current epoch did not improve the metric of interest
        elif self.best_metrics[
            metric_of_interest
        ] <= self.metrics[metric_of_interest]:
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
            'loss': self.best_metrics['loss']
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


def get_batch_metrics(labels, y):
    """
    Multiclass tp, fp, tn, fn for batch.

    Parameters
    ----------
    labels: np.array, size = (batch)
        The int labels.
    y: np.array, size = (batch, n_classes)
        Output of the last Softmax + Linear layer of the network.

    Returns
    -------
    tp: array(int), shape = (n_classes,) true positives
    fp: array(int), shape = (n_classes,) false positives
    tn: array(int), shape = (n_classes,) true negatives
    fn: array(int), shape = (n_classes,) false negatives
    preds: array(int), shape = (n_classes,) total predicted for class
    trues: array(int), shape = (n_classes,) total true for class
    """
    n_classes = y.shape[1]
    y_categorical = np.argmax(y, axis=1)
    tp, fp, tn, fn, preds, trues = [], [], [], [], [], []
    for i in range(n_classes):
        preds.append(np.sum(y_categorical == i))
        trues.append(np.sum(labels == i))
        tp.append(np.sum((y_categorical == i) & (labels == i)))
        fp.append(np.sum((y_categorical == i) & (labels != i)))
        tn.append(np.sum((y_categorical != i) & (labels != i)))
        fn.append(np.sum((y_categorical != i) & (labels == i)))

    tp = np.array(tp)
    fp = np.array(fp)
    tn = np.array(tn)
    fn = np.array(fn)
    preds = np.array(preds)
    trues = np.array(trues)
    return tp, fp, tn, fn, preds, trues


def get_precision(tp, preds, eps=1e-5):
    """
    Parameters
    ----------
    tp: list(int), the tp per class
    preds: list(int), the total predicted per class
    eps: float
        epsilon to deal with the cases where no examples were predicted
        positive

    Returns
    -------
    precision_metric: list(float), len = n_classes
        The precisions per class
    """
    precision_metric = []
    for i in range(len(tp)):
        if preds[i] <= eps:
            precision_metric.append(0)
        else:
            precision_metric.append(tp[i] / preds[i])
    return precision_metric


def get_recall(tp, trues, eps=1e-5):
    """
    Parameters
    ----------
    tp: list(int), the tp per class
    trues: list(int), the total true examples for each class
    eps: float
        epsilon to deal with the cases were no examples were positive

    Returns
    -------
    precision_metric: list(float), len = n_classes
        The precisions per class
    """
    recall_metric = []
    for i in range(len(tp)):
        if trues[i] <= eps:
            recall_metric.append(0)
        else:
            recall_metric.append(tp[i] / trues[i])
    return recall_metric


def get_accuracy(tp, n_examples):
    """
    Parameters
    ----------
    tp: list(int) the tp per class
    n_examples: int

    Returns
    -------
    accuracy: float
        The overall accuracy over all classes.
    """
    correct = np.sum(tp)
    accuracy = correct / n_examples
    return accuracy


def get_dice(tp, fp, fn):
    """
    See: en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient#Formula
    Calculate boolean dice coefficient.

    Returns
    -------
    dice: list(float), len=n_classes
        The dice for each class.
    """
    dice = []
    for i in range(len(tp)):
        dice_num = 2 * tp[i]
        dice_denom = (2 * tp[i]) + fp[i] + fn[i]
        dice.append(dice_num / dice_denom)
    return dice


def get_auc(labels, y_proba):
    """
    Multiclass AUC score (one-vs-rest).

    Parameters
    ----------
    labels: array(int), shape = (n_examples)
        The int labels.
    y_proba: array(int), shape = (n_examples, n_classes)
        Predicted probabilities.

    Returns
    -------
    auc_metric: float
        The auc_metric.
    """
    try:
        auc_metric = roc_auc_score(labels, y_proba, multi_class='ovr')
    except ValueError:
        auc_metric = float("NaN")
    return auc_metric
