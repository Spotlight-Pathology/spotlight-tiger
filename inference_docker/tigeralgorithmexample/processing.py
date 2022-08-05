"""
Adapted from: https://github.com/DIAGNijmegen/pathology-tiger-algorithm-example

Authors: Spotlight Pathology Ltd
License: Apache License V.2

Entrypoint to process a slide (logic for all steps carried out in L1 & L2).
"""

import numpy as np
import openslide
from time import time
import GPUtil

from .gcio import (
    TMP_DETECTION_OUTPUT_PATH,
    TMP_SEGMENTATION_OUTPUT_PATH,
    TMP_TILS_SCORE_PATH,
    TMP_BULK_OUTPUT_PATH,
    TMP_FOLDER,
    copy_data_to_output_folders,
    get_image_path_from_input_folder,
    get_tissue_mask_path_from_input_folder,
    initialize_output_folders,
)
from .rw import (
    READING_LEVEL,
    DetectionWriter,
    SegmentationWriter,
    TilsScoreWriter,
    open_multiresolutionimage_image,
)

from .challenge_utils.models.load_models import (
    load_model_detection,
    load_models_segmentation,
    PARAMS_SEGMENTATION_L1,
    PARAMS_SEGMENTATION_L2,
    PARAMS_DETECTION
)


from .challenge_utils.concave_hull import concave_hull
from .challenge_utils.tils_score import get_tils_score, get_til_stroma
from .challenge_utils.tile_utils import get_downscaled_img
from .challenge_utils.segmentation import run_segmentation
from .challenge_utils.detection import run_detection_l1, run_detection_l2

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RES_ZERO = 0.5


def check_if_l2(tissue_mask_path):
    """
    If returns True, then we are surely in L2.
    If not, we don't know if L1 or L2.

    If we know for sure a WSI is from L2 - then only detect lymphocytes in
    the tumour bulk area. Otherwise, detect lymphocytes everywhere.
    """
    downsample = 64
    level = 6

    res_zero = open_multiresolutionimage_image(
        tissue_mask_path
    ).getSpacing()[0]
    res_out = res_zero * downsample  # um / pixel

    area_pixel_out = res_out ** 2  # sq. um

    tissue_mask_img = get_downscaled_img(
        slide_path=tissue_mask_path,
        downsample=downsample,
        level=level
    )
    area_pixels = np.sum(tissue_mask_img == 1)
    area = area_pixels * area_pixel_out / (1000 ** 2)  # sq. mm

    # if roi mask larger than 10 sq mm then we are in L2
    is_l2 = area > 10
    print(area)
    return is_l2


def process():
    """Processes a test slide"""
    start_time = time()
    print(DEVICE)

    print('GPU usage at the start of this script: ')
    GPUtil.showUtilization()

    # load our models
    models_segmentation = load_models_segmentation()
    model_detection = load_model_detection()

    level = READING_LEVEL

    initialize_output_folders()

    # get input paths
    image_path = get_image_path_from_input_folder()
    tissue_mask_path = get_tissue_mask_path_from_input_folder()

    print(f'Processing image: {image_path}')
    print(f'Processing with mask: {tissue_mask_path}')

    # open images
    image = open_multiresolutionimage_image(path=image_path)
    tissue_mask = open_multiresolutionimage_image(path=tissue_mask_path)

    # check if L2
    is_l2 = check_if_l2(tissue_mask_path)
    print("Are we in L2? " + str(is_l2))

    if is_l2:
        PARAMS_SEGMENTATION = PARAMS_SEGMENTATION_L2
    else:
        PARAMS_SEGMENTATION = PARAMS_SEGMENTATION_L1

    tile_size = list(PARAMS_SEGMENTATION.values())[0]['write_size']

    # get image info
    dimensions = image.getDimensions()
    spacing = image.getSpacing()
    print(spacing)

    # create writers
    print(f"Setting up writers")
    segmentation_writer = SegmentationWriter(
        TMP_SEGMENTATION_OUTPUT_PATH,
        tile_size=tile_size,
        dimensions=dimensions,
        spacing=spacing,
    )
    detection_writer = DetectionWriter(TMP_DETECTION_OUTPUT_PATH)
    tils_score_writer = TilsScoreWriter(TMP_TILS_SCORE_PATH)

    slide = openslide.OpenSlide(str(image_path.resolve()))

    print("Segmenting image...")
    run_segmentation(
        dimensions=dimensions,
        tile_size=tile_size,
        tissue_mask=tissue_mask,
        slide=slide,
        level=level,
        params_segmentation=PARAMS_SEGMENTATION,
        models_segmentation=models_segmentation,
        segmentation_writer=segmentation_writer
    )

    # write bulk tumour area
    concave_hull(
        input_file=TMP_SEGMENTATION_OUTPUT_PATH,
        output_dir=str(TMP_FOLDER.resolve()),
        input_level=6  # downscaled x64
    )
    if TMP_BULK_OUTPUT_PATH.is_file():
        # if we found a tumour bulk
        til_stroma = get_til_stroma(
            TMP_BULK_OUTPUT_PATH,
            dimensions,
            spacing[0],
            downscale=64,
            downscale_level=6,
            segmentation_path=TMP_SEGMENTATION_OUTPUT_PATH
        )
    else:
        # get a blank mask
        wsi_width = dimensions[0] // 64
        wsi_height = dimensions[1] // 64
        til_stroma = np.zeros((wsi_height, wsi_width)).astype(int)

    if is_l2:
        # run detections only in randomly sampled ROIs of tumour assoc stroma
        stroma_area = run_detection_l2(
            til_stroma=til_stroma,
            dimensions=dimensions,
            tile_size=tile_size,
            spacing=spacing,
            image_path=image_path,
            model_detection=model_detection,
            detection_writer=detection_writer,
            params_detection=PARAMS_DETECTION
        )

        print("Compute tils score...")
        tils_score = get_tils_score(
            segmentation_tif=TMP_SEGMENTATION_OUTPUT_PATH,
            tumour_bulk_xml=TMP_BULK_OUTPUT_PATH,
            detections_json=TMP_DETECTION_OUTPUT_PATH,
            stroma_area=stroma_area  # area just from rois
        )
    else:
        # we are not sure if L1 or L2
        # so run detections in all tissue areas
        run_detection_l1(
            image_path=image_path,
            dimensions=dimensions,
            tile_size=tile_size,
            spacing=spacing,
            model_detection=model_detection,
            detection_writer=detection_writer,
            mask_path=tissue_mask_path,
            mask_value_to_detect=1,
            params_detection=PARAMS_DETECTION
        )

        print("Compute tils score...")
        # get the til score by filtering TILs from tumour assoc stroma
        # this will still be calculated in L1, but will be gibberish
        tils_score = get_tils_score(
            segmentation_tif=TMP_SEGMENTATION_OUTPUT_PATH,
            tumour_bulk_xml=TMP_BULK_OUTPUT_PATH,
            detections_json=TMP_DETECTION_OUTPUT_PATH
        )

    tils_score_writer.set_tils_score(tils_score=tils_score)

    print("Saving...")
    # save tils score
    tils_score_writer.save()

    print("Copy data...")
    # save data to output folder
    copy_data_to_output_folders()

    print("Completed!")
    end_time = time()
    print('Total time: {}'.format(end_time - start_time))
