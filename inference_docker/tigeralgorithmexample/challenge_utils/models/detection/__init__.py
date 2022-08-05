"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2
"""

import os
from shutil import copyfile

modeldir = "/home/user/pathology-tiger-algorithm/tigeralgorithmexample/challenge_utils/models/detection/model_files"

pretrained_weigths = [
    os.path.join(modeldir, f)
    for f in os.listdir(modeldir)
    if f.endswith('.pth')
][0]

os.makedirs("/home/user/.cache/torch/hub/checkpoints/", exist_ok=True)
copyfile(
    pretrained_weigths,
    "/home/user/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth"
)

DETECTION_MODEL_PATH = [
    os.path.join(modeldir, f)
    for f in os.listdir(modeldir)
    if f.endswith('.tar')
][0]

print(DETECTION_MODEL_PATH)
