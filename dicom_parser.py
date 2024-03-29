"""Parse the .dcm dataset into standalone images' dataset.

    Extracts the image from the DICOM file (assuming only one frame is present)
    and writes it in the dataset {parameters.DATASET_PATH} in the image format {parameters.LOSSLESS_EXTENSION}.
"""
import os
from argparse import ArgumentParser, Namespace
from pathlib import PosixPath

import cv2
import numpy as np
from pydicom import dcmread, FileDataset, DataElement
from pydicom.pixel_data_handlers import convert_color_space
from pydicom.tag import BaseTag
from pydicom.valuerep import VR

import util
from custom_apng import write_apng
from parameters import PathParameters, LOSSLESS_EXTENSION, PREFIX

MODALITY_TAG = BaseTag(0x0008_0060)
BODY_PART_TAG = BaseTag(0x0018_0015)
BITS_ALLOCATED_TAG = BaseTag(0x0028_0100)
STORED_BITS_TAG = BaseTag(0x0028_0101)
PHOTOMETRIC_INTERPRETATION_TAG = BaseTag(0x0028_0004)  # ColourSpace
SAMPLES_PER_PIXEL_TAG = BaseTag(0x0028_0002)
PIXEL_DATA_TAG = BaseTag(0xfeff_00e0)
PIXEL_DATA_TAG_2 = BaseTag(0x7fe0_0010)
NUMBER_OF_FRAMES_TAG = BaseTag(0x0028_0008)


def parse_dcm(filepath: str):
    """Parse the .dcm file into ".{LOSSLESS_EXTENSION}".

    Example: CT_HEAD.png

    :param filepath: Path to the DICOM file.
    """
    # Validate input
    assert os.path.exists(filepath), f"\"{filepath}\" is not a dicom file!"

    # Read file
    file_data: FileDataset = dcmread(filepath)

    body_part, bps, color_space, modality, samples_per_pixel = extract_attributes(file_data)

    single_channel: bool = samples_per_pixel.value == 1

    # Read the pixel data
    img_array = file_data.pixel_array
    if not single_channel:
        img_array = convert_color_space(img_array, color_space.value, "RGB")

    number_of_frames = get_number_of_frames(file_data, img_array.shape, single_channel)

    # Set image path where it will be written on
    attributes = '_'.join([str(elem) for elem in (color_space.value.replace("_", ""),
                                                  samples_per_pixel.value, bps.value, number_of_frames)])
    out_img_path: str = f"{PathParameters.DATASET_PATH}{modality.value.replace(' ', '')}_{body_part.value}_{attributes}"

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

    if len(img_array.shape) <= 2 or not single_channel and len(img_array.shape) == 3:
        saved_img_array = write_single_frame(img_array, out_img_path)
    else:
        saved_img_array = write_multi_frame(out_img_path, img_array, single_channel)

    if (img_array == saved_img_array).all() is False:
        # Remove written image
        os.remove(out_img_path)
        # Warn the user of the issue
        raise AssertionError(f"Quality loss accidentally applied to the image \"{out_img_path}\"!")


def get_number_of_frames(file_data: FileDataset, img_shape: tuple, single_channel: bool) -> int:
    """Extract number of frames of image from dicom file

    @param file_data:
    @param img_shape:
    @param single_channel:
    @return:
    """
    ndim = len(img_shape)

    number_of_frames = file_data.get(NUMBER_OF_FRAMES_TAG)
    if number_of_frames is not None:
        number_of_frames = number_of_frames.value
    elif single_channel and ndim == {3, 4}:
        number_of_frames = img_shape[0]
    else:
        number_of_frames = 1
    return number_of_frames


def extract_attributes(file_data: FileDataset) \
        -> tuple[DataElement, DataElement, DataElement, DataElement, DataElement]:
    """Extract main attributes from dicom file

    @param file_data:
    @return:
    """
    # Extract metadata for output file naming
    if file_data.get(BODY_PART_TAG) is None:
        file_data.add_new(BODY_PART_TAG, VR.CS, "NA")
    body_part: DataElement = file_data[BODY_PART_TAG]
    modality: DataElement = file_data[MODALITY_TAG]
    bps: DataElement = file_data[STORED_BITS_TAG]
    samples_per_pixel: DataElement = file_data[SAMPLES_PER_PIXEL_TAG]
    color_space: DataElement = file_data[PHOTOMETRIC_INTERPRETATION_TAG]
    return body_part, bps, color_space, modality, samples_per_pixel


def write_multi_frame(out_img_path: str, img_array: np.ndarray, is_single_channel: bool):
    """Write a multi-frame image to the specified path

    Warning: his function works for .tiff formats, might not work, however for other formats.

    :param is_single_channel: Boolean flag indicating if image is in rgb colorspace
    :param img_array: Image matrix. Image color-spaces can be supported according to the tiff format
    :param out_img_path: Relative path where the image should be written to
    :return: Matrix structure of the written image
    """

    # Pre-condition(s)
    assert is_single_channel and len(img_array.shape) == 3 or \
           not is_single_channel and len(img_array.shape) == 4, "Image is single frame!"

    out_img_path_tiff, out_img_path_apng = [out_img_path.replace(".png", format_) for format_ in (".tiff", ".apng")]

    # Save to apng
    status_, saved_img_array_apng = write_apng(out_img_path_apng, img_array)
    assert status_ is True, "Error writing apng image"

    return saved_img_array_apng


def write_single_frame(img_array: np.ndarray, out_img_path: str) -> np.ndarray:
    """Write a single frame image to the specified path

    Warning: his function works for png formats, might not work, however, for other formats.

    :param img_array: Image matrix. Multiple color spaces are accepted, as long as supported by png format
    :param out_img_path: Relative path where the image should be written to
    :return: Matrix structure of the written image
    """

    # Encode and write image to dataset folder
    # Assert no information loss within the written image
    assert cv2.imwrite(out_img_path, img_array) is True, "Image writing (single frame) failed"
    return cv2.imread(out_img_path, cv2.IMREAD_UNCHANGED)


def exec_shell(command: str):
    """Executes a command w/ sub-shell

    Also, ensures its correct operation w/ the assert feature

    @param command: Command to be executed
    """
    return_code = os.system(command)
    assert return_code == 0, f"Problem executing \"{command}\", code {return_code}"


def run_parsing(dicom_path: PosixPath):
    """Main function for this module

    @param dicom_path: The directory with the dicom files (no need to be an actual DICOM_DIR)
    """
    print("Pre-processing dicom dataset into .(a)png", end="...")

    dicom_path_str = str(dicom_path)

    if not os.path.exists(dicom_path):
        os.makedirs(dicom_path)
    if not os.path.exists(PathParameters.DATASET_PATH):
        os.makedirs(PathParameters.DATASET_PATH)

    # Empty the images/dataset directory
    for file in os.listdir(PathParameters.DATASET_PATH):
        os.remove(PathParameters.DATASET_PATH + file)

    files: list[str] = list(os.listdir(dicom_path_str))
    # Call a function to parse each dicom file
    if dicom_path.is_dir():
        for dcm_file in files:
            if not util.is_file_a_dicom(f"{dicom_path_str}/{dcm_file}"):
                print(f"File '{dcm_file}' is not dicom. Skipping...")
                continue
            parse_dcm(filepath=f"{dicom_path_str}/{dcm_file}")
    elif dicom_path.is_file():
        if util.is_file_a_dicom(dicom_path_str):
            parse_dcm(filepath=dicom_path_str)
    else:
        print(f"Path '{dicom_path.name}' is not a directory nor dicom file!")
        return
    print("Done!")


def main(args: Namespace):
    dicom_dirs = args.input

    if path := args.path:
        path = path.replace('~', os.environ['HOME'])
        PathParameters.DATASET_PATH = f'{path}/'

    if args.tmp:
        PathParameters.DATASET_PATH = PREFIX + PathParameters.DATASET_PATH

    for dicom_path in dicom_dirs:
        run_parsing(dicom_path=dicom_path)


if __name__ == '__main__':
    parser = ArgumentParser(description=f"Preprocess dicom files into lossless "
                                        f"'{LOSSLESS_EXTENSION}' format images")

    parser.add_argument('--tmp', action='store_true',
                        help='Store output files inside /tmp/ directory')
    parser.add_argument('--path', action='store', nargs='?',
                        help='Define the path for the output files (beware if you choose '
                             'this option along with --tmp, /tmp/ will be appended as prefix anyways)')
    parser.add_argument('input', action='store', nargs='+', type=PosixPath,
                        help='Folder or dicom files to serve as input for the experience')

    _args = parser.parse_args()

    main(args=_args)
