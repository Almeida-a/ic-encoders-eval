import os

import cv2
import numpy
from pydicom import dcmread
from pydicom.tag import BaseTag


def parse_dcm(filepath: str):
    """

    :param filepath:
    :return:
    """
    # Validate input
    assert os.path.exists(filepath), f"\"{filepath}\" is not a dicom file!"

    # Read file
    file_data = dcmread(filepath)

    # Extract metadata for output file naming
    modality = file_data.file_meta[BaseTag(0x0002_0002)]  # FIXME This returns the uid, not the modality name
    body_part = file_data[BaseTag(0x0018_0015)]
    # TODO continue testing from here on
    # Read the pixel data
    img_array = file_data.pixel_array

    # Only accept 8bit depths for now
    assert img_array.dtype == numpy.int8, f"Invalid pixel datatype," \
                                          f" can only read int8, but " \
                                          f"\"{img_array.dtype}\" provided"

    cv2.imwrite(img_array, f"{modality.value}_{body_part.value}.png")


def main():
    # Specify the directory where the dicom files are
    raw_dataset: str = "images/dataset_raw/"
    dirs = []

    # Get all dicom files (hardcoded)
    for suffix in ("CT/IMAGES/", "MGT/IMAGES/"):
        for filename in os.listdir(raw_dataset + suffix):
            dirs.append(raw_dataset + suffix + filename)

    # Call a function to parse each dicom file
    for dcm_file in dirs:
        parse_dcm(filepath=dcm_file)


if __name__ == "__main__":
    main()
