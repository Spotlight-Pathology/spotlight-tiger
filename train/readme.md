# The train module

Training scripts used to build the detection and segmentation models.

## Preparing the environment
- A GPU is not necessary to run the training scripts, but is strongly advised.
- Create a new virtual environment with python=3.7
- In the virtual environment, install all required packages:
```
pip install -r requirements.txt
```
- (Optionally) run the tests to make sure everything is set up correctly:
```
pytest train/
```

## Training the models

The training scripts will run CV by randomly splitting the data into training
and validation on each iteration (the `repeats` argument controls how many splits
will be carried out). Only ROI level annotations were used for training.

### Detection 
To train the detection model, the entrypoint is the [detection/train.py](detection/train.py):
```
python detection/train.py --data-dir "tissue-cells"  --labels-json "tiger-coco.json" --logs "output-dir" --max-epochs 50 --repeats 2 
```
 The `data-dir` should be in the same format as the `tissue-cells` directory
of the [TIGER challenge](https://tiger.grand-challenge.org/Data/). The `labels-json`
argument requires the full path to the coco style json labels. 

The `logs` directory is where all outputs, logs and models will be saved during
training.

For a list of all additional configuration options see the  `build_parser` 
function of `train.py`


### Segmentation
To train the segmentation model, the entrypoint is the [segmentation/train.py](segmentation/train.py):
```
python segmentation/train.py --data-dir "roi-level-annotations" --logs "output-dir" --max-epochs 50 --repeats 2 
```
 The `data-dir` should be in the same format as the `roi-level-annotations` directory
of the [TIGER challenge](https://tiger.grand-challenge.org/Data/). 

The `logs` directory is where all outputs, logs and models will be saved during
training.

For a list of all additional configuration options see the  `build_parser` 
function of `train.py`





