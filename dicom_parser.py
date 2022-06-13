"""Parse the .dcm dataset into standalone images dataset.

    Extracts the image from the DICOM file (assuming only one frame is present)
    and writes it in the dataset {parameters.DATASET_PATH} in the image format {parameters.LOSSLESS_EXTENSION}.
"""

import os
from typing import List, Tuple

import apng
import cv2
import numpy as np
from PIL import Image, ImageSequence
from pydicom import dcmread, FileDataset
from pydicom.tag import BaseTag

from parameters import DATASET_PATH, LOSSLESS_EXTENSION

# TODO Go get Dicom files w/ various color-spaces

MODALITY_TAG = BaseTag(0x0008_0060)
BODY_PART_TAG = BaseTag(0x0018_0015)
STORED_BITS_TAG = BaseTag(0x0028_0101)
PHOTOMETRIC_INTERPRETATION_TAG = BaseTag(0x0028_0004)  # ColourSpace
SAMPLES_PER_PIXEL_TAG = BaseTag(0x0028_0002)
PIXEL_DATA_TAG = BaseTag(0xfeff_00e0)
PIXEL_DATA_TAG_2 = BaseTag(0x7fe0_0010)


def parse_dcm(filepath: str):
    """Parse the .dcm file into ".{LOSSLESS_EXTENSION}".

    Example: CT_HEAD.png

    :param filepath: Path to the DICOM file.
    """
    # Validate input
    assert os.path.exists(filepath), f"\"{filepath}\" is not a dicom file!"

    # Read file
    file_data: FileDataset = dcmread(filepath)

    # Extract metadata for output file naming
    modality = file_data[MODALITY_TAG]
    body_part = file_data[BODY_PART_TAG]
    bps = file_data[STORED_BITS_TAG]  # bits per sample TODO be careful, since the bpp at the main pipeline is based on
    #                                                      the max pixel value, different from how it is obtained here
    samples_per_pixel = file_data[SAMPLES_PER_PIXEL_TAG]
    color_space = file_data[PHOTOMETRIC_INTERPRETATION_TAG]

    # Read the pixel data
    img_array = file_data.pixel_array

    # Set image path where it will be written on
    attributes = '_'.join([str(elem.value) for elem in (color_space, samples_per_pixel, bps)])
    out_img_path: str = DATASET_PATH + f"{modality.value.replace(' ', '')}_{body_part.value}_{attributes}"

    repetition_id = 0

    is_og_name_not_available: bool = os.path.exists(f"{out_img_path}{LOSSLESS_EXTENSION}")

    # Enter loop if there is an already existing file with a new name
    while is_og_name_not_available:
        # Increment repetition marker
        repetition_id += 1
        # Loop until finding unique name
        if not os.path.exists(f"{out_img_path}_{repetition_id}{LOSSLESS_EXTENSION}"):
            break

    if repetition_id > 0:
        out_img_path = f"{out_img_path}_{repetition_id}"

    out_img_path += LOSSLESS_EXTENSION

    single_channel: bool = samples_per_pixel.value == 1

    if len(img_array.shape) <= 2 or not single_channel and len(img_array.shape) == 3:
        saved_img_array = write_single_frame(img_array, out_img_path)
    else:
        saved_img_array = write_multi_frame(out_img_path, img_array, single_channel)

    if (img_array == saved_img_array).all() is False:
        # Remove written image
        os.remove(out_img_path)
        # Warn the user of the issue
        raise AssertionError(f"Quality loss accidentally applied to the image \"{out_img_path}\"!")


def to_np_array(apng_img: apng.APNG) -> np.ndarray:

    frames_arr = []

    for frame, control in apng_img.frames:
        tmp_fname = "tmp.png"
        frame.save(tmp_fname)
        frames_arr.append(
            cv2.imread(tmp_fname, cv2.IMREAD_UNCHANGED)
        )
        os.remove(tmp_fname)

    return np.array(frames_arr)


def im_write_multi_apng(file_name: str, img_array: np.ndarray) -> Tuple[bool, np.ndarray]:

    sub_frames_fn_list = []

    for i in range(img_array.shape[0]):
        frame: np.ndarray = img_array[i]

        sub_frame_fname = f"tmp{i}.png"
        sub_frames_fn_list.append(sub_frame_fname)

        cv2.imwrite(sub_frame_fname, frame, params=[cv2.IMWRITE_TIFF_COMPRESSION, 1])

    apng_img = apng.APNG.from_files(sub_frames_fn_list)
    apng_img.save(file_name)

    for i in range(img_array.shape[0]):
        os.remove(f"tmp{i}.png")

    return True, to_np_array(apng_img)


def write_multi_frame(out_img_path: str, img_array: np.ndarray, is_single_channel: bool):
    """Write a multi-frame image to the specified path

    Warning: his function works for .tiff formats, might not work, however for other formats.

    :param is_single_channel: Boolean flag indicating if image is in rgb colorspace
    :param img_array: Image matrix. Image color-spaces can be supported according to the tiff format
    :param out_img_path: Relative path where the image should be written to
    :return: Matrix structure of the written image
    """

    # Pre-condition(s)
    assert is_single_channel and len(img_array.shape) == 3 or\
           not is_single_channel and len(img_array.shape) == 4, "Image is single frame!"

    out_img_path_tiff, out_img_path_apng = [out_img_path.replace(".png", format_) for format_ in (".tiff", ".apng")]

    # Save to tiff
    assert cv2.imwritemulti(out_img_path_tiff, img_array) is True, "Image writing (multi-frame) failed."
    frames: List[Image] = ImageSequence.all_frames(Image.open(out_img_path_tiff))
    saved_img_array_tiff = np.array([np.array(frames[i]) for i in range(img_array.shape[0])])

    status_, saved_img_array_apng = im_write_multi_apng(out_img_path_apng, img_array)
    assert status_ is True, "Error writing apng image"

    assert (saved_img_array_tiff == saved_img_array_apng).all(), "Quality loss in either apng " \
                                                                         "or tiff multi-frame image files."

    return saved_img_array_tiff


def write_single_frame(img_array: np.ndarray, out_img_path: str) -> np.ndarray:
    """Write a single frame image to the specified path

    Warning: his function works for png formats, might not work, however, for other formats.

    :param img_array: Image matrix. Multiple color spaces are accepted, as long as supported by png format
    :param out_img_path: Relative path where the image should be written to
    :return: Matrix structure of the written image
    """

    # Encode and write image to dataset folder
    assert cv2.imwrite(out_img_path, img_array, params=[cv2.IMWRITE_TIFF_COMPRESSION, 1]) \
           is True, "Image writing (single frame) failed"
    # Assert no information loss within the written image
    saved_img_array: np.ndarray = cv2.imread(out_img_path, cv2.IMREAD_UNCHANGED)
    return saved_img_array


def exec_shell(command):
    return_code = os.system(command)
    assert return_code == 0, f"Problem executing \"{command}\", code {return_code}"


if __name__ == "__main__":
    # Specify the directory where the dicom files are
    raw_dataset: str = "images/dataset_dicom/"
    dirs: List[str] = []

    # Get all dicom files (hardcoded)
    for filename in os.listdir(raw_dataset):
        dirs.append(raw_dataset + filename)

    # Empty the images/dataset directory
    for file in os.listdir("images/dataset"):
        os.remove(f"images/dataset/{file}")

    # Call a function to parse each dicom file
    for dcm_file in dirs:
        parse_dcm(filepath=dcm_file)
