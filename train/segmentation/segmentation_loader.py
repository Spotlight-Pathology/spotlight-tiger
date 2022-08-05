"""
Authors: Spotlight Pathology Ltd
License: Apache License V.2
"""

from torch.utils.data import Dataset, DataLoader, Subset
import torch
import numpy as np
from skimage.io import imread
from skimage.util import img_as_float32
import uuid
import h5py as h5
import os
from train.preprocessing.transforms import (
    resize_tiling,
    transform_mask_labels,
    augment
)
from train.utils import LABEL_MAPPING


class ROITileDataset(Dataset):
    def __init__(
            self,
            roi_paths,
            labels,
            tile_size,
            tile_res,
            stride,
            label_conversion_map,
            res_zero,
            padding=0,
            tempdir=None,
            enable_augment=False,
            enable_tiling=True,
            augment_p=0.3

    ):
        """
        Initialise a torch Dataset from ROIs by tiling them into patches.
        Assumes each ROI can fully fit into memory.

        When enable_tiling = False, then the whole image is returned.
        No padding or downscaling is applied and tempdir must be None.

        Parameters
        ----------
        roi_paths: [str]
        labels: dict
            Mapping from each roi_path to the corresponding mask_path.
        tile_size: int
        tile_res: float
            The tile resolution in microns/ pixel
        stride: int
            The step between tiles.
        label_conversion_map: {}
            Mapping from original labels to relevant classes.
        res_zero: float
            The max ROI resolution.
        padding: int
            0 by default. Will pad the ROI with zeros before extracting
            the tiles. Used for correct inference at the edges.
            Given at zero res.
        tempdir: str
            Dir to save the tiles during dataset initialisation and read
            them back during training.
        enable_augment: bool
            whether to perform augmentation
        enable_tiling: bool
        """
        self.data = []
        if tempdir is not None:
            self.tempdir = tempdir
            os.makedirs(self.tempdir, exist_ok=True)
        else:
            self.tempdir = None
        self.enable_augment = enable_augment
        self.label_conversion_map = label_conversion_map
        self.augment_p = augment_p
        self.enable_tiling = enable_tiling
        counter = 0

        for roi_path in roi_paths:
            roi = imread(roi_path)[..., :3]
            mask = imread(labels[roi_path])
            downscale_factor = tile_res / res_zero

            if self.enable_tiling:
                tiles, mask_tiles, tile_origins = resize_tiling(
                    roi, mask,
                    size=tile_size,
                    resolution=tile_res,
                    stride=stride,
                    res_zero=res_zero,
                    padding=padding  # at zero
                )

                for t, _ in enumerate(tiles):
                    # check the tile does not include only ignored labels
                    only_ignore = mask_tiles[t] == 0
                    only_pad = mask_tiles[t] == 100
                    ignored_labels = only_ignore | only_pad
                    if (1 - ignored_labels).any():
                        tile_dict = {
                            'tile': tiles[t],
                            'mask': mask_tiles[t],
                            # at zero
                            'tile_origin': (
                                tile_origins[t] * downscale_factor
                            ).astype(int)
                        }

                        if self.tempdir is None:
                            tile_dict['roi_path'] = roi_path
                            self.data.append(tile_dict)

                        else:
                            # write tile to h5
                            hex = uuid.uuid4().hex
                            tileh5 = os.path.join(self.tempdir, hex + '.h5')
                            with h5.File(tileh5, 'w') as f:
                                for k, v in tile_dict.items():
                                    dst = f.create_dataset(
                                        k, data=v, compression='gzip',
                                    )
                                    dst.attrs['roi_path'] = roi_path
                            self.data.append(tileh5)
                    else:
                        pass
            else:
                assert self.tempdir is None
                # check the image does not include only ignored labels
                only_ignore = mask == 0
                only_pad = mask == 100
                ignored_labels = only_ignore | only_pad
                if (1 - ignored_labels).any():
                    self.data.append(
                        {
                            'tile': roi,
                            'mask': mask,
                            'tile_origin': (int(0), int(0)),
                            'roi_path': roi_path
                        }
                    )
                else:
                    pass

    def __len__(self):
        return len(self.data)

    def get_weights(self):
        """Weights calculated before transforming labels."""
        # count how pixels of each label exist
        counts = {}
        for tile_idx in self.data:
            if self.tempdir is None:
                mask = tile_idx['mask']
            else:
                with h5.File(tile_idx, 'r') as f:
                    mask = f.get('mask')[:]

            labels_this_mask = np.unique(mask)
            for label in labels_this_mask:
                if label in counts.keys():
                    counts[label] += np.sum(mask == label)
                else:
                    counts[label] = 0

        counts.pop(LABEL_MAPPING[0], None)  # pop any 100 (padding)
        counts.pop(0, None)  # pop any 0 (unlabelled areas before conversion)
        total = sum([counts[key] for key in counts])
        for (key, value) in counts.items():
            counts[key] = value / total
        print(counts)

        # get weight per image
        # weights are given to over-sample dcis and healthy glands
        weights = []
        for tile_idx in self.data:
            if self.tempdir is None:
                mask = tile_idx['mask']
            else:
                with h5.File(tile_idx, 'r') as f:
                    mask = f.get('mask')[:]

            labels_this_mask = [
                la for la in np.unique(mask)
                if la in counts.keys()
            ]
            if 3 in labels_this_mask:  # dcis
                weight = 2
            elif 4 in labels_this_mask:  # healthy
                weight = 2
            else:
                weight = 1
            weights.append(weight)

        return weights

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int
            The member of the Dataset to retrieve in this
            iteration.

        Returns
        -------
        tile: torch.Tensor
            Image of a tile as (c, h, w), should be float32
        mask: torch.Tensor
            Mask for the tile (h, w)
        tile_origin: torch.Tensor
            The tile's (x, y) origin coordinates at resolution level 0.
        roi_path: str
            The filepath of the ROI.
        """
        tile_idx = self.data[idx]

        if self.tempdir is None:
            tile = tile_idx['tile']
            mask = tile_idx['mask']
            tile_origin = tile_idx['tile_origin']
            roi_path = tile_idx['roi_path']

        else:
            with h5.File(tile_idx, 'r') as f:
                tile = f.get('tile')[:]
                mask = f.get('mask')[:]
                tile_origin = f.get('tile_origin')[:]
                roi_path = f.get('tile').attrs['roi_path']

        mask = transform_mask_labels(mask, self.label_conversion_map)

        if self.enable_augment:
            # needs uint8
            tile, mask = augment(tile, mask=mask, p=self.augment_p)

        tile = img_as_float32(tile)
        tile = torch.from_numpy(tile).permute(2, 0, 1)
        mask = torch.from_numpy(mask)

        return tile, mask, tile_origin, roi_path


def create_roi_loader(
        roi_paths,
        labels: dict,
        tile_size: int,
        tile_res: float,
        stride: int,
        batch_size: int,
        reshuffle: bool,
        num_workers: int,
        res_zero: float,
        label_conversion_map: dict,
        tempdir: str = None,
        enable_augment=False,
        padding=0,
        subset=None,
        enable_tiling=True,
        augment_p=0.3,
        balance_weights=False,
):
    """Create a dataloader from the ROI tiles."""
    all_roi_dataset = ROITileDataset(
        roi_paths,
        labels,
        tile_size,
        tile_res,
        stride,
        label_conversion_map,
        padding=padding,
        res_zero=res_zero,
        tempdir=tempdir,
        enable_augment=enable_augment,
        enable_tiling=enable_tiling,
        augment_p=augment_p
    )

    if (subset is not None) & (not balance_weights):
        indices = torch.randperm(subset)
        all_roi_dataset = Subset(all_roi_dataset, indices=indices)

    if balance_weights:
        # will create a dataset of same length by weighted
        # random sampling WITH replacement
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            all_roi_dataset.get_weights(),
            len(all_roi_dataset) if subset is None else subset
        )

        loader = DataLoader(
            all_roi_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,  # required for multiprocessing with cuda
            prefetch_factor=2,
            sampler=sampler
        )
    else:
        loader = DataLoader(
            all_roi_dataset,
            batch_size=batch_size,
            shuffle=reshuffle,  # this determines if we will reshuffle
            num_workers=num_workers,
            pin_memory=True,  # required for multiprocessing with cuda
            prefetch_factor=2
        )
    return loader
