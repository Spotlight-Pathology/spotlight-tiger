"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2

Pytorch segmentation model based on Unet.

Uses this package:
https://pypi.org/project/segmentation-models-pytorch/#examples
"""
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
import torch.nn as nn


def get_softmax_activation():
    return nn.Softmax(dim=1)


class Unet(nn.Module):
    """
    Get an Unet with custom backbone encoder.
    The encoder will be pretrained on Imagenet and frozen.
    """

    def __init__(
            self,
            encoder: str,
            num_classes: int,
            in_channels=3,
            model_size=224,
            activation='softmax'
    ):
        """
        Parameters
        ----------
        encoder: str
            Valid encoders: https://smp.readthedocs.io/en/latest/encoders.html
        num_classes: int
        in_channels: int
        activation: str,
            An activation function to apply after the final convolution layer.
            Available options are “sigmoid”, “softmax”, “logsoftmax”, “tanh”,
            “identity”, callable and None. Default is softmax.
        """
        super().__init__()

        # The segmentation_models_pytorch package does now allow specifying
        # the dim in softmax, but this is required in latest pytorch versions
        # Define custom callable to get around this:
        if activation == 'softmax':
            activation = get_softmax_activation

        self.prep_data = PrepareDataPretrained(model_size)

        self.model_unet = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
        for param in self.model_unet.encoder.parameters():
            param.requires_grad = False  # freeze entire encoder

    def forward(self, x):
        x = self.prep_data(x)
        return self.model_unet(x)


class PrepareDataPretrained(nn.Module):
    def __init__(self, model_size=None):
        """
        nn.Module to resize and normalise the data to match the format
        expected by pytorch pretrained models.

        Parameters
        ----------
        model_size: int
            The input size required by the model.
            This transform will resize the input to match the model_size.
            By default = None, then no resizing takes place.
        """
        super().__init__()

        steps = []
        if model_size is not None:
            steps.append(transforms.Resize(size=(model_size, model_size)))
        steps.append(
            transforms.Normalize(
                # from imagenet dataset
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )

        self.prepare = nn.Sequential(*steps)

    def forward(self, x):
        return self.prepare(x)
