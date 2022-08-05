"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2
"""

import argparse
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from shutil import rmtree

from train.utils import find_files, print_batch
from train.utils import get_split
from train.detection.detection_loader import create_detection_loader, SimpleDataset
from train.detection.setup_train import setup_training
from train.detection.metric_callbacks import History, remove_other_targets
from train.preprocessing.transforms import resize_tiling
from train.postprocessing.nms import apply_nms
from train.postprocessing.transforms import (
    threshold_outputs,
    merge_tile_outputs,
    check_is_padded,
    convert_non_padded,
    get_area
)

RES_ZERO = 0.5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_parser():
    """
    Flags for training setup.
    """
    parser = argparse.ArgumentParser(__doc__)

    # dataset
    parser.add_argument('--tile', '-t', type=int, default=256)
    parser.add_argument('--stride', type=int, default=None)

    # training params
    parser.add_argument('--scheduler', type=str, default='y')
    parser.add_argument('--scheduler-plateau-epochs', type=int, default=3)
    parser.add_argument('--early-stopping-epochs', type=int, default=10)
    parser.add_argument('--augment', type=str, default='y')
    parser.add_argument('--augment_p', type=float, default=0.3)

    # whether to tile the input images or use them as they are
    parser.add_argument('--resize-by-tiling', type=str, default='y')
    # whether to include regions without any lymphocytes
    parser.add_argument('--include-empty', type=str, default='y')
    parser.add_argument('--balance-empty-weight', type=float, default=0.1)

    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--anchors', type=tuple, default=(6.0, 12., 23.))
    parser.add_argument('--freeze-backbone', type=str, default='n')
    parser.add_argument('--fpn', type=str, default='y')
    parser.add_argument('--nms', type=str, default='y')
    parser.add_argument('--nms_iou', type=float, default=0.5)
    # discard detections with score below this threshold
    parser.add_argument('--threshold', type=float, default=0.0)

    parser.add_argument('--batch', '-b', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--max-epochs', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--num-workers', type=int, default=2)

    # get only a subset of images, used to quickly test the code
    parser.add_argument('--subset', type=int, default=None)

    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--labels-json', type=str, required=True)
    parser.add_argument('--logs', type=str, required=True, help='log dir')

    return parser


def train_steps(
        model,
        dataloader,
        optimizer,
        epoch,
        history: History,
):

    model.train()
    history.start_epoch()

    for i, (images, targets, im_paths) in enumerate(dataloader):

        # zero the parameter gradients after each batch
        optimizer.zero_grad()
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        losses.backward()

        history.update(loss=loss_dict)
        print_batch(i, len(dataloader), history, epoch)
        optimizer.step()

    history.calculate_metrics(epoch, is_epoch_end=True)
    return history


def val_steps(
        model,
        dataloader,
        optimizer,
        epoch: int,
        history: History,
        nms,
        nms_iou=None,  # only used if nms == True
        resize_by_tiling=None,
        resize_by_tiling_batch=None,
        resize_by_tiling_stride=None,
        threshold=0
):
    model.eval()
    history.start_epoch()

    for i, (images, targets, im_paths) in enumerate(dataloader):
        # zero the parameter gradients after each batch
        optimizer.zero_grad()
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        prep_targets = remove_other_targets(targets)

        if resize_by_tiling is not None:
            outputs = []
            for image in images:
                outputs.append(
                    tile_and_infer(
                        image,
                        resize_by_tiling,
                        model,
                        batch=resize_by_tiling_batch,
                        stride=resize_by_tiling_stride
                    )
                )

        else:
            with torch.no_grad():
                outputs = model(images)

        if nms:
            outputs = apply_nms(outputs, nms_iou)

        outputs = threshold_outputs(outputs, threshold)

        outputs = [
            {k: v.cpu() for k, v in t.items()}
            for t in outputs
        ]
        res = {
            target["image_id"].item(): output
            for target, output in zip(prep_targets, outputs)
        }

        # save image info
        areas = [get_area(img, img_resolution=RES_ZERO) for img in images]
        sizes = [img.detach().cpu().size() for img in images]

        history.update(
            targets=prep_targets,
            targets_pred=res,
            areas=areas,
            sizes=sizes,
            im_paths=im_paths
        )

    history.calculate_metrics(epoch, is_epoch_end=True)
    history.keep_best(model, optimizer)
    history.keep_last(model, optimizer)
    return history


def train(
        train_data,
        val_data,
        model,
        optimizer,
        max_epochs,
        save_dir,
        nms,
        nms_iou=None,
        scheduler=None,
        scheduler_plateau_epochs=None,
        early_stopping_epochs=None,
        resize_by_tiling=None,
        resize_by_tiling_batch=None,
        resize_by_tiling_stride=None,
        threshold=0
):
    dataloader = {'train': train_data, 'val': val_data}
    train_history, val_history = History('train'), History('val')

    for epoch in range(max_epochs):
        print('\nEpoch {}/{}'.format(epoch, max_epochs - 1))
        print('-' * 10)

        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print('\n' + phase)
            if phase == 'train':
                train_history = train_steps(
                    model,
                    dataloader[phase],
                    optimizer,
                    epoch,
                    train_history,
                )
            else:
                val_history = val_steps(
                    model,
                    dataloader[phase],
                    optimizer,
                    epoch,
                    val_history,
                    nms,
                    nms_iou,
                    resize_by_tiling,
                    resize_by_tiling_batch,
                    resize_by_tiling_stride,
                    threshold=threshold
                )

        if scheduler is not None:
            train_history.check_plateau()
            if train_history.plateau_counter >= scheduler_plateau_epochs:
                scheduler.step()
                print("New lr = " + str(scheduler.get_last_lr()))
            if train_history.plateau_counter >= early_stopping_epochs:
                print('Training stopped on plateau.')
                break

    val_history.save_best_checkpoint(save_dir, prefix='val')
    val_history.save_last_checkpoint(save_dir)
    print('\n Finished Training')
    return val_history


def tile_and_infer(
        image,
        resize_by_tiling: int,
        model,
        batch,
        stride
):
    """
    Tile a big image and run inference per tile.
    Return the overall predictions for the image.

    Parameters
    ----------
    image: torch.tensor (c, h, w)
    resize_by_tiling: int
    model: torch.nn.Module
    batch: int
        batch size to use after tiling for inference
    stride: int
        stride to use when tiling.

    Returns
    -------
    outputs: {"boxes", "labels", "scores"}
        The predictions for this image.
    """
    tiles, tile_origins, transformed = resize_tiling(
        image.cpu().permute(1, 2, 0).numpy(),
        size=resize_by_tiling,
        stride=stride,
        return_transformed_big_image=True
    )
    tiles = list(
        torch.tensor(tile).permute(2, 0, 1)
        for tile in tiles
    )
    dataset = DataLoader(SimpleDataset(tiles), batch_size=batch)
    tile_outputs = []
    for tile_batch in dataset:
        tile_batch = list(t.to(DEVICE) for t in tile_batch)
        with torch.no_grad():
            tile_outputs.extend(model(tile_batch))
    outputs = merge_tile_outputs(tile_outputs, tile_origins)

    if check_is_padded(transformed):
        outputs = convert_non_padded(
            outputs, transformed, image.cpu().permute(1, 2, 0).numpy().shape
        )
    return outputs


def main(args):

    print(DEVICE)

    # setup dataset
    # ---------------------------------------------------------------------
    roi_paths = find_files('images', '.png', args.data_dir)

    assert len(roi_paths) > 0

    for repeat in range(args.repeats):
        print("CV Repeat " + str(repeat))
        train_roi_paths, val_roi_paths, _ = get_split(
            roi_paths, seed=repeat,
            train_ratio=0.8, val_ratio=0.2, test_ratio=0.0
        )

        # save the train/val split as artifact
        save_dir = os.path.join(args.logs, "repeat_{}".format(repeat))
        os.makedirs(save_dir, exist_ok=True)
        df_write_train = pd.DataFrame(columns=['Files'], data=train_roi_paths)
        df_write_val = pd.DataFrame(columns=['Files'], data=val_roi_paths)

        train_file = os.path.join(save_dir, 'train_files.csv')
        val_file = os.path.join(save_dir, 'val_files.csv')

        df_write_train.to_csv(train_file)
        df_write_val.to_csv(val_file)

        # create temp dirs to store tiles while training
        tempdir_tiles_train = os.path.join(save_dir, 'temp_tiles', 'train')
        tempdir_tiles_val = os.path.join(save_dir, 'temp_tiles', 'val')
        os.makedirs(tempdir_tiles_train, exist_ok=True)
        os.makedirs(tempdir_tiles_val, exist_ok=True)

        if args.stride is None:
            args.stride = args.tile - 10

        print('Preparing dataset training...\n')
        train_data = create_detection_loader(
            image_paths=train_roi_paths,
            labels_json=args.labels_json,
            batch_size=args.batch,
            reshuffle=True,
            num_workers=args.num_workers,
            size=args.tile,  # tile the images to this size
            stride=args.stride,
            tempdir=tempdir_tiles_train,
            subset=args.subset,
            augment=(args.augment == 'y'),
            augment_p=args.augment_p,
            resize_by_tiling=(args.resize_by_tiling == 'y'),
            include_empty=(args.include_empty == 'y'),
            balance_empty=True,
            balance_empty_weight=args.balance_empty_weight,
        )

        print('Preparing dataset validation...\n')
        val_data = create_detection_loader(
            image_paths=val_roi_paths,
            labels_json=args.labels_json,
            batch_size=args.batch,
            reshuffle=False,
            num_workers=args.num_workers,
            size=args.tile,  # tile the images to this size
            stride=args.tile,
            tempdir=tempdir_tiles_val,
            subset=args.subset,
            resize_by_tiling=False,
            include_empty=(args.include_empty == 'y'),
        )

        # start training
        # ---------------------------------------------------------------------
        model, optimizer, scheduler = setup_training(args)

        train(
            train_data,
            val_data,
            model,
            optimizer,
            args.max_epochs,
            save_dir,
            nms=(args.nms == 'y'),
            nms_iou=args.nms_iou,
            scheduler=scheduler if (args.scheduler == 'y') else None,
            scheduler_plateau_epochs=args.scheduler_plateau_epochs,
            early_stopping_epochs=args.early_stopping_epochs,
            resize_by_tiling=args.tile,
            resize_by_tiling_batch=args.batch,
            resize_by_tiling_stride=args.tile,
            threshold=args.threshold
        )

        # remove temp tiles
        rmtree(os.path.join(save_dir, 'temp_tiles'))


if __name__ == '__main__':
    args = build_parser().parse_args()
    main(args)
