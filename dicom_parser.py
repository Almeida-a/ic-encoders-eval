"""Parse the .dcm dataset into standalone images dataset.

    Extracts the image from the DICOM file (assuming only one frame is present)
    and writes it in the dataset {parameters.DATASET_PATH} in the image format {parameters.LOSSLESS_EXTENSION}.
"""

import os

import cv2
from pydicom import dcmread
from pydicom.tag import BaseTag

import metrics
from parameters import LOSSLESS_EXTENSION, DATASET_PATH


def parse_dcm(filepath: str):
    """ Parse the .dcm file into ".{LOSSLESS_EXTENSION}".

    :param filepath: Path to the DICOM file.
    """
    # Validate input
    assert os.path.exists(filepath), f"\"{filepath}\" is not a dicom file!"

    # Read file
    file_data = dcmread(filepath)

    # Extract metadata for output file naming
    modality = file_data[BaseTag(0x0008_0060)]
    body_part = file_data[BaseTag(0x0018_0015)]
    stored_bits = file_data[BaseTag(0x0028_0101)]

    # Read the pixel data
    img_array = file_data.pixel_array

    # Skip images with unsupported characteristics
    # Only accept mono{chrome,frame} images for now
    if len(img_array.shape) != 2:
        print(f"Multi-frame images are unsupported for now")
        return
    # Only accept 12bit depths
    if stored_bits.value > 12:
        print("Invalid pixel depth,"
              f" can only read 12, but \"{stored_bits.value}\" provided")
        return

    # Set image path where it will be written on
    out_img_path: str = DATASET_PATH + f"{modality.value.replace(' ', '')}_" \
                                       f"{body_part.value}"

    # Basename repetition marker
    rep = [0, f"{LOSSLESS_EXTENSION}"]

    # Loop until finding unique name
    while os.path.exists(f"{out_img_path}{rep[1]}"):
        # Increment repetition marker
        rep[0] += 1
        rep[1] = f"_{rep[0]}{LOSSLESS_EXTENSION}"

    out_img_path += f"{rep[1]}"

    # Write image
    cv2.imwrite(out_img_path, img_array, (cv2.IMWRITE_PNG_COMPRESSION, 0))

    # Assert no information loss within the written image
    saved_img_array = cv2.imread(out_img_path, cv2.IMREAD_UNCHANGED)
    if metrics.ssim(img_array, saved_img_array) != 1:  # TODO optimize this verification
        # Remove written image
        os.remove(out_img_path)
        # Warn the user of the issue
        print(f"Quality loss accidentally applied to the image \"{out_img_path}\"!")


if __name__ == "__main__":
    # Specify the directory where the dicom files are
    raw_dataset: str = "images/dataset_dicom/"
    dirs = []

    # Get all dicom files (hardcoded)
    for suffix in ("MGT/IMAGES/", "CT/IMAGES/"):
        for filename in os.listdir(raw_dataset + suffix):
            dirs.append(raw_dataset + suffix + filename)

    # Call a function to parse each dicom file
    for dcm_file in dirs:
        parse_dcm(filepath=dcm_file)
