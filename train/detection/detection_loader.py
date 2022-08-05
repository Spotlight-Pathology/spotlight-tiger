"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2
"""

from torch.utils.data import DataLoader, Dataset, Subset
import torch
from skimage.io import imread
from skimage import img_as_float32, img_as_ubyte
import json
import os
from shutil import rmtree
import h5py as h5
import uuid
from train.preprocessing.transforms import (
    detection_resize_tiling,
    get_box_to_fit,
    convert_bbox,
    augment_detections
)


class SimpleDataset(Dataset):
    def __init__(self, tensor_list):
        self.tensor_list = tensor_list

    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, idx):
        return self.tensor_list[idx]


class DetectionsDataset(Dataset):
    def __init__(
            self,
            image_paths,
            labels_json,
            size: int,
            stride: int,
            tempdir: str,
            augment: bool,
            augment_p: float = None,
            resize_by_tiling: bool = True,
            include_empty: bool = False,
            balance_empty: bool = False,
            balance_empty_weight: float = 0.1,
    ):
        with open(labels_json) as f:
            json_dict = json.load(f)
            self.labels = json_dict['annotations']
            all_images = json_dict['images']

        # map images to the correct filepaths
        self.images = []
        for image in all_images:
            old_path = os.path.basename(image['file_name'])
            new_path = [p for p in image_paths if old_path in p]
            if len(new_path) == 1:
                image['file_name'] = new_path[0]
                self.images.append(image)
        assert len(self.images) == len(image_paths)

        # create targets per image
        self.num_has_targets = 0
        self.num_empty = 0
        self.include_empty = include_empty
        self.resize_by_tiling = resize_by_tiling
        self.balance_empty = balance_empty
        self.augment = augment
        self.augment_p = augment_p
        self.data = []
        self.create_targets()
        self.size = size
        self.stride = stride
        self.tempdir = tempdir
        self.tile_data = []

        if self.resize_by_tiling:
            for datapoint in self.data:
                image = imread(datapoint[0])

                tiles, tile_targets = detection_resize_tiling(
                    image,
                    target=datapoint[1],
                    size=self.size,
                    stride=self.stride,
                    include_empty=self.include_empty
                )
                self.save_tiles(tiles, tile_targets, datapoint[0])

        else:
            for datapoint in self.data:
                image = imread(datapoint[0])
                self.save_tiles([image], [datapoint[1]], datapoint[0])

        if self.balance_empty:
            self.weights = self.make_weights_for_balanced_density(
                balance_empty_weight
            )

        # count how many dataset items have lymphocytes
        for tdata in self.tile_data:
            targets = tdata[1]
            has_targets = torch.max(targets["labels"]).item() > 0
            if has_targets:
                self.num_has_targets += 1
            else:
                self.num_empty += 1

    def save_tiles(self, tiles, tile_targets, im_name):
        """Save tiles temporarily as h5 to avoid memory overload."""
        for i, tile in enumerate(tiles):
            tileh5 = os.path.join(
                self.tempdir,
                os.path.basename(im_name) + str(uuid.uuid4()) + '.h5'
            )
            self.tile_data.append([tileh5, tile_targets[i]])

            with h5.File(tileh5, 'w') as f:
                f.create_dataset(
                    'tile', data=tile,
                    compression='gzip',
                )

    def create_targets(self):
        """
        Create targets per image by combining all bboxes for each image.
        """
        for image in self.images:

            image_id = image['id']
            image_labels = [
                bbox for bbox in self.labels
                if bbox['image_id'] == image_id
            ]

            img_path = image['file_name']
            img_shape = imread(img_path).shape
            img_dims = (img_shape[1], img_shape[0])

            # add to dataset only if there are some detections in image
            if len(image_labels) > 0:
                target = {}
                # boxes converted to x0, y0, x1, y1 and reshaped to fit in img
                target['boxes'] = torch.FloatTensor(
                    [
                        get_box_to_fit(convert_bbox(bbox['bbox']), img_dims)
                        for bbox in image_labels
                    ]
                )
                target['labels'] = torch.IntTensor([1 for _ in image_labels])
                target['labels'] = target['labels'].to(torch.int64)
                target['image_id'] = torch.tensor(image_id)
                target['area'] = torch.tensor(
                    [bbox['area'] for bbox in image_labels]
                )
                target['iscrowd'] = torch.IntTensor(
                    [bbox['iscrowd'] for bbox in image_labels]
                )
                self.data.append([img_path, target])
                self.num_has_targets += 1
            else:
                if self.include_empty:
                    # if no targets are found in image, add a dummy "0" type
                    # object
                    target = {}
                    target['boxes'] = torch.FloatTensor([[1, 2, 3, 4]])
                    target['labels'] = torch.IntTensor([0])
                    target['labels'] = target['labels'].to(torch.int64)
                    target['image_id'] = torch.tensor(image_id)
                    target['area'] = torch.tensor([4])
                    target['iscrowd'] = torch.IntTensor([0])

                    self.data.append([img_path, target])
                    self.num_empty += 1
                pass

    def make_weights_for_balanced_density(self, weight_empty=0.1):
        weight_has_targets = 1 - weight_empty
        weights = []

        data = self.tile_data

        for idx, (_, target) in enumerate(data):
            has_targets = torch.max(target["labels"]).item() > 0
            if has_targets:
                weights.append(weight_has_targets)
            else:
                weights.append(weight_empty)
        weights = torch.DoubleTensor(weights)
        return weights

    def __getitem__(self, idx):
        im_path, target = self.tile_data[idx]
        with h5.File(im_path, 'r') as f:
            img = f.get('tile')[:]

        img = img_as_ubyte(img)
        if self.augment:  # img must be uint8
            img, target = augment_detections(img, target, self.augment_p)

        img = img_as_float32(img)
        img = torch.tensor(img).permute(2, 0, 1)
        return img, target, im_path

    def __len__(self):
        return len(self.tile_data)

    def clean_up(self):
        rmtree(self.tempdir)


def create_detection_loader(
        image_paths,
        labels_json: str,
        batch_size: int,
        reshuffle: bool,
        num_workers: int,
        size: int = None,
        stride: int = None,
        tempdir: str = None,
        subset: int = None,
        augment: bool = False,
        augment_p: float = None,
        resize_by_tiling: bool = True,
        include_empty: bool = False,
        balance_empty: bool = False,
        balance_empty_weight: float = 0.1,
):
    """
    Create a dataloader for cell detection.
    """
    dataset = DetectionsDataset(
        image_paths=image_paths,
        labels_json=labels_json,
        size=size,
        stride=stride,
        tempdir=tempdir,
        augment=augment,
        augment_p=augment_p,
        resize_by_tiling=resize_by_tiling,
        include_empty=include_empty,
        balance_empty=balance_empty,
        balance_empty_weight=balance_empty_weight,
    )

    if (subset is not None) & (not balance_empty):
        indices = torch.randperm(subset)
        dataset = Subset(dataset, indices=indices)

    if balance_empty:
        # will create a dataset of same length by weighted
        # random sampling WITH replacement
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            dataset.weights,
            len(dataset) if subset is None else subset
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,  # required for multiprocessing with cuda
            prefetch_factor=2,
            collate_fn=collate_fn,
            sampler=sampler
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=reshuffle,  # this determines if we will reshuffle
            num_workers=num_workers,
            pin_memory=True,  # required for multiprocessing with cuda
            prefetch_factor=2,
            collate_fn=collate_fn
        )
    return loader


def collate_fn(batch):
    return tuple(zip(*batch))
