"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2

A pretrained Faster-RCNN implementation
"""
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model_instance_detection_original(
    num_classes=2,
    anchors_per_featmap=(6.0, 12., 23., 46.),
    pretrained=True
):
    """Replace only the head of the FasterRCNN model."""
    # load a model pre-trained on COCO with FPN and RPN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=pretrained
    )

    # create an anchor_generator for the FPN with 5 feature maps
    # get 1 anchor per image location with aspect ratio 1
    anchor_generator = AnchorGenerator(
        sizes=([anchors_per_featmap for _ in range(5)]),
        aspect_ratios=([(1.0,) for _ in range(5)])
    )
    model.rpn.anchor_generator = anchor_generator
    model.rpn.head = RPNHead(
        model.backbone.out_channels,
        anchor_generator.num_anchors_per_location()[0]
    )

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
