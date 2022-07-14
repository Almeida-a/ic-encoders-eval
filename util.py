"""Miscellaneous utility functions

"""

import os
import time
from subprocess import Popen, PIPE

import cv2

from custom_apng import get_apng_frames_resolution, get_apng_depth
from parameters import DATASET_COMPRESSED_PATH


def construct_djxl(decoded_path, target_image):
    """Creates the command for decoding an image to jxl

    Given the provided configurations

    @param decoded_path: Output path
    @param target_image: Input path
    @return: Command
    """

    return f"djxl {target_image} {decoded_path}"


def construct_davif(decoded_path: str, input_path: str):
    """Creates the command for decoding an image from avif

    Given the provided configurations

    @param decoded_path: Output path
    @param input_path: Input path
    @return: Command
    """
    command: str = f"avif_decode {input_path} {decoded_path}"
    return command


def construct_dwebp(decoded_path: str, input_path: str, additional_options: str = ""):
    """Creates the command for decoding an image from webp

    Given the provided configurations

    @param additional_options: More options
    @param decoded_path: Output path
    @param input_path: Input path
    @return: Command
    """
    command: str = f"dwebp -v {input_path} {additional_options} -o {decoded_path}"
    return command


def construct_cwebp(effort, output_path, quality, target_image):
    """Creates the command for encoding an image to webp

    Given the provided configurations

    @param effort: Effort setting
    @param output_path: Output
    @param quality: Quality setting
    @param target_image: Input
    @return: Void
    """
    return f"cwebp -quiet -v -q {quality} -m {effort} {target_image} -o {output_path}"


def construct_cavif(output_path, quality, speed, target_image):
    """Creates the command for encoding an image to avif

    Given the provided configurations

    @param output_path: Output
    @param quality: Quality setting
    @param speed: Effort setting
    @param target_image: Input
    @return: Void
    """
    command: str = f"cavif -o {output_path} " \
                   f"--quality {quality} --speed {speed} --quiet " \
                   f"{os.path.abspath(target_image)}"
    return command


def construct_cjxl(distance, effort, output_path, target_image):
    """Creates the command for encoding an image to jxl

    Given the provided configurations

    @param distance: Quality setting
    @param effort: Effort setting
    @param output_path: Output
    @param target_image: Input
    @return: Void
    """
    command: str = f"cjxl {target_image} {output_path} " \
                   f"--distance={distance} --effort={effort} --quiet"
    return command


def timed_command(stdin: str) -> float:
    """Runs a given command on a subshell and records its execution time

    Note: Execution timeout implemented to 60 seconds

    @param stdin: Used to run the subshell command
    @return: Time it took for the command to run (in seconds)
    """
    # Execute command and time the CT
    start = time.time()
    p = Popen(stdin, shell=True, stdout=PIPE, stderr=PIPE)
    _, stderr = p.communicate(timeout=180)
    ct = time.time() - start  # or extract_webp_ct(stderr)
    # Check for errors
    return_code: int = p.returncode
    if return_code != 0:
        print(f"Error code {return_code}, executing:"
              f"\nStdIn -> {stdin}"
              f"\nStdErr -> {stderr}")
        exit(1)

    return ct


def total_pixels(target_image: str) -> int:
    """Counts the number of pixels on an image

    Count method: height * height

    @param target_image: Input image path
    @return: Number of pixels
    """

    # Parameter checking
    assert os.path.exists(target_image), f"Image at \"{target_image}\" does not exist!"

    # Extract resolution
    if target_image.endswith("apng"):
        height, width = get_apng_frames_resolution(target_image)
        depth = get_apng_depth(target_image)

        return height * width * depth

    height, width = cv2.imread(target_image).shape[:2]
    # Number of pixels in the image
    pixels = height * width
    return pixels


def original_basename(intended_abs_filepath: str) -> str:
    """Get an original filename given the absolute path

    Example - give path/to/filename.txt -> it already exists -> return path/to/filename_1.txt

    @param intended_abs_filepath: Absolute path to a file not yet written (w/o the file's extension)
    @return: Same path, with basename of the file changed to an original one
    """
    # Separate path from extension
    extension: str = intended_abs_filepath.split(".")[-1]
    intended_abs_filepath: str = ".".join(intended_abs_filepath.split(".")[:-1])

    suffix: str = ""
    counter: int = 0

    while os.path.exists(f"{intended_abs_filepath + suffix}.{extension}"):
        counter += 1
        suffix = f"_{counter}"

    return f"{intended_abs_filepath + suffix}.{extension}"


def rm_encoded():
    """Removes compressed files from a previous (probably unsuccessful) execution

    """
    for file in os.listdir(DATASET_COMPRESSED_PATH):
        os.remove(os.path.abspath(DATASET_COMPRESSED_PATH + file))
