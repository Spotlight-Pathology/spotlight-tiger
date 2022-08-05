"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2
"""

import torch
from train.segmentation.segmentation_model import Unet
from torch.optim import Adam, SGD, lr_scheduler
from warmup_scheduler import GradualWarmupScheduler

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_training(args, label_ignore):
    """
    Setup model, optimizers and losses for Unet multiclass.
    """
    # setup model
    # ---------------------------------------------------------------------
    model = Unet(
        args.backbone,
        num_classes=args.n_classes,
        in_channels=3,
        model_size=args.tile,
        freeze_encoder=(args.freeze_encoder == 'y')
    )

    # move model to CUDA before constructing optimizers
    # https://pytorch.org/docs/stable/optim.html
    model.to(DEVICE)

    # setup optimizer
    # ---------------------------------------------------------------------
    params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'Adam':
        optimizer = Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = SGD(
            params,
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError('Parser arg optimizer only supports SGD, Adam.')

    # setup loss
    # ---------------------------------------------------------------------
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.Tensor(args.clw).to(DEVICE),
        ignore_index=label_ignore  # ignored/ does not contribute to gradient
    )

    # setup lr scheduler
    # ---------------------------------------------------------------------
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    if args.warmup == 'y':
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler
        )

    return model, optimizer, criterion, scheduler
