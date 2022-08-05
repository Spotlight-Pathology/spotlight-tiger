"""
Adapted from: https://github.com/DIAGNijmegen/pathology-tiger-algorithm-example

Authors: Spotlight Pathology Ltd
License: Apache License V.2

GrandChallenge Input/Output (gcio)

In this file settings concerning folders and paths for reading and writing o
n GrandChallenge are defined. Note that these settings are mostly specific to
the GrandChallenge Tiger Challenge.

"""
from pathlib import Path
from shutil import copy


# Grand Challenge paths
GRAND_CHALLENGE_SEGMENTATION_OUTPUT_PATH = Path(
    "/output/images/breast-cancer-segmentation-for-tils/segmentation.tif"
)
GRAND_CHALLENGE_DETECTION_OUTPUT_PATH = Path(
    "/output/detected-lymphocytes.json"
)
GRAND_CHALLENGE_TILS_SCORE_PATH = Path(
    "/output/til-score.json"
)
GRAND_CHALLENGE_BULK_OUTPUT_PATH = Path(
    "/output/segmentation.xml"
)

# Temporary directory
TMP_FOLDER = Path("/home/user/tmp")

# Temp input paths
TMP_INPUT_PATH = (TMP_FOLDER / Path('/input_slide.tif').name)
TMP_INPUT_MASK_PATH = (TMP_FOLDER / Path('/input_mask.tif').name)

# Temp output paths
TMP_SEGMENTATION_OUTPUT_PATH = TMP_FOLDER / \
    GRAND_CHALLENGE_SEGMENTATION_OUTPUT_PATH.name

TMP_DETECTION_OUTPUT_PATH = TMP_FOLDER / \
    GRAND_CHALLENGE_DETECTION_OUTPUT_PATH.name

TMP_BULK_OUTPUT_PATH = TMP_FOLDER / \
    GRAND_CHALLENGE_BULK_OUTPUT_PATH.name

TMP_TILS_SCORE_PATH = TMP_FOLDER / \
    GRAND_CHALLENGE_TILS_SCORE_PATH.name

# Grand Challenge folders were input files can be found
GRAND_CHALLENGE_IMAGE_FOLDER = Path("/input/")
GRAND_CHALLENGE_MASK_FOLDER = Path("/input/images/")

# Grand Challenge suffixes for required files
GRAND_CHALLENGE_IMAGE_SUFFIX = ".tif"
GRAND_CHALLENGE_MASK_SUFFIX = ".tif"


def initialize_output_folders():
    """
    This function initialize all output folders for Grand Challenge output
    as well as temporary folder
    """
    GRAND_CHALLENGE_SEGMENTATION_OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )
    GRAND_CHALLENGE_DETECTION_OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )
    GRAND_CHALLENGE_TILS_SCORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TMP_FOLDER.mkdir(parents=True, exist_ok=True)


def _get_file_from_folder(folder: Path, suffix: str) -> Path:
    """
    Gets this first file in a folder with the specified suffix.
    This is used because the docker is meant to process one sample at a time.

    Parameters
    ----------
    folder: Path
        folder to search for files
    suffix: str
        suffix for file  to search for

    Returns
    -------
    Path
        path to file

    """
    return list(Path(folder).glob("*" + suffix))[0]


def get_image_path_from_input_folder() -> Path:
    """
    Gets a test image which needs to be processed for the Tiger Challenge.

    Returns
    -------
    Path
        Path to multiresolution image from the test set.
    """
    grand_challenge_input_path = _get_file_from_folder(
        GRAND_CHALLENGE_IMAGE_FOLDER, GRAND_CHALLENGE_IMAGE_SUFFIX
    )
    copy(grand_challenge_input_path, TMP_INPUT_PATH)
    return TMP_INPUT_PATH


def get_tissue_mask_path_from_input_folder() -> Path:
    """
    Gets the tissue mask for the corresponding test image that needs to be
    processed.

    Returns
    -------
    Path
        Path to tissue mask.
    """
    grand_challenge_mask_path = _get_file_from_folder(
        GRAND_CHALLENGE_MASK_FOLDER,
        GRAND_CHALLENGE_MASK_SUFFIX
    )
    copy(grand_challenge_mask_path, TMP_INPUT_MASK_PATH)
    return TMP_INPUT_MASK_PATH


def copy_data_to_output_folders():
    """Copies all temporary files to the (mandatory) output files/folders."""

    # copy segmentation tif to grand challenge
    copy(
        TMP_SEGMENTATION_OUTPUT_PATH,
        GRAND_CHALLENGE_SEGMENTATION_OUTPUT_PATH
    )
    # copy detections json to grand challenge
    copy(
        TMP_DETECTION_OUTPUT_PATH,
        GRAND_CHALLENGE_DETECTION_OUTPUT_PATH
    )
    # copy tils score json to grand challenge
    copy(
        TMP_TILS_SCORE_PATH,
        GRAND_CHALLENGE_TILS_SCORE_PATH
    )

    # # copy tumour bulk mask
    # copy(TMP_BULK_OUTPUT_PATH, GRAND_CHALLENGE_BULK_OUTPUT_PATH)

    print('Check segmentation file exists')
    print(GRAND_CHALLENGE_SEGMENTATION_OUTPUT_PATH.is_file())
    print('Check detection file exists')
    print(GRAND_CHALLENGE_DETECTION_OUTPUT_PATH.is_file())
    print('Check tils score file exists')
    print(GRAND_CHALLENGE_TILS_SCORE_PATH.is_file())
