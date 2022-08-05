"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2
"""

import torch
from torch.optim import Adam, SGD, lr_scheduler
from train.detection.detection_model import get_model_instance_detection_original

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_training(args):
    """
    Setup model, optimizers for object detection.
    """
    # setup model
    # ---------------------------------------------------------------------
    model = get_model_instance_detection_original(
        num_classes=args.num_classes,
        anchors_per_featmap=args.anchors,
        freeze_backbone=(args.freeze_backbone == 'y'),
    )

    # move model to CUDA before constructing optimizers
    # https://pytorch.org/docs/stable/optim.html
    model.to(DEVICE)

    # setup optimizer
    # ---------------------------------------------------------------------
    params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'Adam':
        optimizer = Adam(params, lr=args.lr, weight_decay=5 * 1e-4)
    elif args.optimizer == 'SGD':
        optimizer = SGD(
            params,
            lr=args.lr,
            momentum=0.9,
            weight_decay=5 * 1e-4
        )
    else:
        raise ValueError('Parser arg optimizer only supports SGD, Adam.')

    # setup lr scheduler
    # ---------------------------------------------------------------------
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    return model, optimizer, scheduler


def load_model(model_path, params):
    """
    Build the detection model and load the weights from the checkpoint file
    found in model_path.
    """
    checkpoint = torch.load(
        model_path,
        map_location=torch.device(DEVICE)
    )

    if "num_classes" in list(params.keys()):
        num_classes = int(params["num_classes"])
    else:
        num_classes = 2

    model = get_model_instance_detection_original(
        num_classes,
        eval(params['anchors']),
        freeze_backbone=(params['freeze_backbone'] == 'y'),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(DEVICE)
    return model
