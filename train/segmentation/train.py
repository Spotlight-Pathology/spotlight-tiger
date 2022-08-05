"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2
"""

import argparse
import os
import torch
import torch.nn as nn
from shutil import rmtree
import pandas as pd
import h5py as h5

from train.utils import (
    find_files,
    print_batch,
    get_roi_to_mask_mapping,
    get_split,
    LABEL_MAPPING
)
from train.segmentation.segmentation_loader import create_roi_loader
from train.segmentation.setup_train import setup_training
from train.segmentation.metric_callbacks import History

RES_ZERO = 0.5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_parser():
    """
    Flags for training setup.
    """
    parser = argparse.ArgumentParser(__doc__)

    # dataset
    parser.add_argument('--n-classes', type=int, default=3)
    parser.add_argument('--freeze-encoder', type=str, default="y")
    # learning rate warmup
    parser.add_argument('--warmup', type=str, default="y")
    parser.add_argument('--batch', '-b', type=int, default=10)
    parser.add_argument('--tile', '-t', type=int, default=224)
    parser.add_argument('--resolution', type=float, default=0.5)
    parser.add_argument('--tile-stride', type=int, default=None)

    # training params
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--augment', type=str, default='y')
    parser.add_argument('--augment_p', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--max-epochs', type=int, default=2)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--weight_decay', type=float, default=5 * 1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    # class weights
    parser.add_argument('--clw', nargs='+', type=float, default=(1., 1., 1.))
    parser.add_argument('--balance-dataset', type=str, default='n')

    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--logs', type=str, required=True, help='log dir')

    # get only a subset of images, used to quickly test the code
    parser.add_argument('--subset', type=int, default=None)

    return parser


def train_steps(
        model,
        dataloader,
        criterion,
        optimizer,
        epoch,
        history: History,
):

    model.train()
    history.start_epoch()

    for i, (tiles, masks, _, _) in enumerate(dataloader):

        # zero the parameter gradients after each batch
        optimizer.zero_grad()

        tiles = tiles.to(torch.float)  # float 32 for speed
        masks = masks.to(torch.long)

        tiles, masks = tiles.to(DEVICE), masks.to(DEVICE)
        Y_logit = model(tiles)

        if torch.isnan(Y_logit).any() or torch.isinf(Y_logit).any():
            print('invalid Y_proba detected at iteration ', i)

        if torch.isnan(masks).any() or torch.isinf(masks).any():
            print('invalid masks detected at iteration ', i)

        loss = criterion(Y_logit, masks)
        loss.backward()

        Y_prob = nn.Softmax(dim=1)(Y_logit)
        history.update(masks, Y_prob, loss, label_ignore=LABEL_MAPPING[0])
        print_batch(i, len(dataloader), history, epoch)
        optimizer.step()

    history.calculate_metrics(epoch)
    return history


def val_steps(
        model,
        dataloader,
        criterion,
        optimizer,
        epoch: int,
        history: History,
        save_dir: str = None
):
    model.eval()
    history.start_epoch()

    for i, (tiles, masks, _, _) in enumerate(dataloader):
        # zero the parameter gradients after each batch
        optimizer.zero_grad()

        tiles = tiles.to(torch.float)
        masks = masks.to(torch.long)

        tiles, masks = tiles.to(DEVICE), masks.to(DEVICE)
        with torch.no_grad():
            Y_logit = model(tiles)

        if torch.isnan(Y_logit).any() or torch.isinf(Y_logit).any():
            print('invalid Y_proba detected at iteration ', i)

        if torch.isnan(masks).any() or torch.isinf(masks).any():
            print('invalid masks detected at iteration ', i)

        if torch.equal(masks, (torch.ones_like(masks) * 100).to(torch.long)):
            print(' all values in mask area 100')

        loss = criterion(Y_logit, masks)

        if torch.isnan(loss):
            tiles = tiles.detach().cpu().numpy()
            Y_logit = Y_logit.detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            with h5.File(os.path.join(save_dir, 'debug.h5'), 'w') as f:
                f.create_dataset('tiles', data=tiles)
                f.create_dataset('logits', data=Y_logit)
                f.create_dataset('masks', data=masks)

            raise ValueError('val loss is nan')

        Y_prob = nn.Softmax(dim=1)(Y_logit)
        history.update(masks, Y_prob, loss, label_ignore=LABEL_MAPPING[0])
        print_batch(i, len(dataloader), history, epoch)

    history.calculate_metrics(epoch)
    history.keep_best(model, optimizer)
    history.keep_last(model, optimizer)
    return history


def train(
        train_data,
        val_data,
        model,
        optimizer,
        criterion,
        max_epochs,
        save_dir,
        scheduler
):
    dataloader = {'train': train_data, 'val': val_data}
    train_history, val_history = History(), History()

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
                    criterion,
                    optimizer,
                    epoch,
                    train_history,
                )
            else:
                val_history = val_steps(
                    model,
                    dataloader[phase],
                    criterion,
                    optimizer,
                    epoch,
                    val_history,
                    save_dir
                )

        scheduler.step()

    val_history.save_best_checkpoint(save_dir, prefix='val')
    val_history.save_last_checkpoint(save_dir)
    print('\n Finished Training')
    return val_history


def main(args):

    # setup dataset
    # ---------------------------------------------------------------------
    roi_paths = find_files('images', '.png', args.data_dir)
    mask_paths = find_files('masks', '.png', args.data_dir)

    # get label mapping
    labels = get_roi_to_mask_mapping(roi_paths, mask_paths)

    assert len(roi_paths) > 0
    assert len(mask_paths) > 0

    if not args.tile_stride:
        args.tile_stride = args.tile

    for repeat in range(args.repeats):
        print("CV Repeat " + str(repeat))
        train_roi_paths, val_roi_paths, _ = get_split(
            roi_paths, seed=repeat,
            train_ratio=0.75, val_ratio=0.25, test_ratio=0.0
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

        print('Preparing dataset training...\n')
        train_data = create_roi_loader(
            train_roi_paths,
            labels=labels,
            tile_size=args.tile,
            tile_res=args.resolution,
            stride=args.tile_stride,
            batch_size=args.batch,
            reshuffle=True,
            num_workers=args.num_workers,
            label_conversion_map=LABEL_MAPPING,
            tempdir=tempdir_tiles_train,
            enable_augment=(args.augment == 'y'),
            augment_p=args.augment_p,
            res_zero=RES_ZERO,
            subset=args.subset,
            enable_tiling=True,
            balance_weights=(args.balance_dataset == 'y')
        )

        print('Preparing dataset validation...\n')
        val_data = create_roi_loader(
            val_roi_paths,
            labels=labels,
            tile_size=args.tile,
            tile_res=args.resolution,
            stride=args.tile,
            batch_size=args.batch,
            reshuffle=False,
            num_workers=args.num_workers,
            label_conversion_map=LABEL_MAPPING,
            tempdir=tempdir_tiles_val,
            res_zero=RES_ZERO,
            subset=args.subset,
            enable_tiling=True
        )

        # start training
        # ---------------------------------------------------------------------
        model, optimizer, loss, scheduler = setup_training(
            args, label_ignore=LABEL_MAPPING[0]
        )

        train(
            train_data,
            val_data,
            model,
            optimizer,
            loss,
            args.max_epochs,
            save_dir,
            scheduler
        )

        # delete temp tiles
        rmtree(os.path.join(save_dir, 'temp_tiles'))


if __name__ == '__main__':

    args = build_parser().parse_args()
    main(args)
