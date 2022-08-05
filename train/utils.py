"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2
"""
import os
from typing import List
import sys
import random

LABEL_MAPPING = {
    0: 100,  # unlabelled
    1: 1,  # tumour (invasive)
    2: 2,  # stroma
    3: 0,  # tumour (in-situ)
    4: 0,  # other (healthy)
    5: 0,  # other (necrosis)
    6: 2,  # stroma (inflammed)
    7: 0,  # other (rest),
    8: 0,  # other (non-tumour-bulk)
    100: 100  # padded area (ignore)
}


def traverse_dir(root):
    all_files = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            all_files.append(os.path.join(path, name))
    return all_files


def find_files(subset, filetype, data_dir):
    files = [
        f for f in traverse_dir(data_dir)
        if (subset in f) & (f.endswith(filetype))
    ]
    return files


def get_roi_to_mask_mapping(roi_paths: List[str], mask_paths: List[str]):
    """
    Get a `labels` dict that maps each ROI file path to the corresponding
    mask file path.
    The masks have the same filename as the ROI images.
    """
    labels = {}
    for roipath in roi_paths:
        mask_filename = os.path.basename(roipath)
        maskfilepath = [p for p in mask_paths if mask_filename in p]
        assert len(maskfilepath) == 1, roipath
        labels[roipath] = maskfilepath[0]
    return labels


def print_batch(i, n_examples, history, epoch):
    """
    Print updates for batch i.
    """
    print_every = n_examples // 10
    print_every = print_every if print_every > 5 else 2

    sys.stdout.write("\rBatch %i" % i)
    sys.stdout.flush()

    if i % print_every == (print_every - 1):
        history.calculate_metrics(epoch)
        print(history)


def get_split(
        filepaths,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=10
):
    """
    Split the dataset into train/ validation/ test.
    Make sure there is no overlap of slides between train/ validation/ test.
    """
    assert (train_ratio + val_ratio + test_ratio) == 1
    random.seed(seed)
    unique_slides = sorted(list(set(
        [os.path.basename(p).split('_[')[0] for p in filepaths]
    )))
    random.shuffle(unique_slides)

    num_train = int(round(train_ratio * len(unique_slides)))
    num_val = int(round(val_ratio * len(unique_slides)))

    train_slides = unique_slides[:num_train]
    val_slides = unique_slides[num_train:num_train + num_val]
    test_slides = unique_slides[num_train + num_val:]

    train_paths, val_paths, test_paths = [], [], []
    for s in train_slides:
        train_paths.extend([p for p in filepaths if s in p])
    for s in val_slides:
        val_paths.extend([p for p in filepaths if s in p])
    for s in test_slides:
        test_paths.extend([p for p in filepaths if s in p])

    return train_paths, val_paths, test_paths
