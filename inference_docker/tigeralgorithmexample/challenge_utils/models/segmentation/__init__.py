"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2
"""

import os
from shutil import copyfile

modeldir = "/home/user/pathology-tiger-algorithm/tigeralgorithmexample/challenge_utils/models/segmentation/model_files"

pretrained_weigths = [
    os.path.join(modeldir, f)
    for f in os.listdir(modeldir)
    if f.endswith('.pth')
]

hubdir = "/home/user/.cache/torch/hub/checkpoints/"
os.makedirs(hubdir, exist_ok=True)

for pweight in pretrained_weigths:
    copyfile(
        pweight,
        os.path.join(hubdir, os.path.basename(pweight))
    )

SEGMENTATION_MODEL_PATHS = [
    os.path.join(modeldir, f)
    for f in os.listdir(modeldir)
    if f.endswith('.tar')
]

print(SEGMENTATION_MODEL_PATHS)
