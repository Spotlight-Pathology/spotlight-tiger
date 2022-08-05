"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2
"""

import torch
from .detection.detection_model import get_model_instance_detection_original
from .segmentation.segmentation_model import Unet
from .detection import DETECTION_MODEL_PATH
from .segmentation import SEGMENTATION_MODEL_PATHS


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define parameters used to build the models
PARAMS_DETECTION = {
    "anchors": "(6.0, 12.0, 23.0)",
    "resolution": 0.5,
    "nms_iou": 0.4,
    "tile": 256,
    "slide_nms_iou": 0.4,
    "threshold": 0.0,
    'padding': 0

}
PARAMS_SEGMENTATION_L1 = {
    "0f233026de1a40f081cb26ef1221a133": {
        "backbone": "resnet101",
        "tile": 224,
        "resolution": 0.5,
        'stride': 200,
        'write_size': 1024,
        'padding': 10
    },
    "19ce0b18cc92413482966bd7b7fbd837": {
        "backbone": "densenet121",
        "tile": 224,
        "resolution": 2.0,
        'stride': 200,
        'write_size': 1024,
        'padding': 40
    }
}

PARAMS_SEGMENTATION_L2 = {
    "0f233026de1a40f081cb26ef1221a133": {
        "backbone": "resnet101",
        "tile": 224,
        "resolution": 0.5,
        'stride': 200,
        'write_size': 2048,
        'padding': 10
    },
    "19ce0b18cc92413482966bd7b7fbd837": {
        "backbone": "densenet121",
        "tile": 224,
        "resolution": 2.0,
        'stride': 200,
        'write_size': 2048,
        'padding': 40
    }
}


def load_model_detection(
        params=PARAMS_DETECTION,
        model_path=DETECTION_MODEL_PATH
):
    """
    Build the detection model based on experiment params
    and load the weights from the checkpoint file found in model_path.
    """
    checkpoint = torch.load(
        model_path,
        map_location=torch.device(DEVICE)
    )
    model = get_model_instance_detection_original(
        2,
        eval(params['anchors']),
        pretrained=False
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(DEVICE)
    return model


def load_models_segmentation(
        params=PARAMS_SEGMENTATION_L1,
        model_paths=SEGMENTATION_MODEL_PATHS
):
    """
    Build model based on experiment params
    and load the weights from the checkpoint file found in model_path.
    """
    models = {}
    for key in list(params.keys()):
        model_path = [m for m in model_paths if key in m][0]
        checkpoint = torch.load(
            model_path,
            map_location=torch.device(DEVICE)
        )
        model = Unet(
            params[key]['backbone'],
            num_classes=3,
            in_channels=3,
            activation='identity'
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(DEVICE)
        models[key] = model
    return models
