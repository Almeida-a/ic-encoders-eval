"""Miscellaneous utility functions

"""

import os
import re
import subprocess
import time
from subprocess import Popen, PIPE

import cv2

from custom_apng import get_apng_frames_resolution, get_apng_depth
from parameters import DATASET_COMPRESSED_PATH, DEPTH


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
    return f"avif_decode {input_path} {decoded_path}"


def construct_dwebp(decoded_path: str, input_path: str, additional_options: str = ""):
    """Creates the command for decoding an image from webp

    Given the provided configurations

    @param additional_options: More options
    @param decoded_path: Output path
    @param input_path: Input path
    @return: Command
    """
    return f"dwebp -v {input_path} {additional_options} -o {decoded_path}"


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
    return f"cavif -o {output_path} " f"--quality {quality} --speed {speed} --quiet " f"{os.path.abspath(target_image)}"


def construct_cjxl(distance, effort, output_path, target_image):
    """Creates the command for encoding an image to jxl

    Given the provided configurations

    @param distance: Quality setting
    @param effort: Effort setting
    @param output_path: Output
    @param target_image: Input
    @return: Void
    """
    return f"cjxl {target_image} {output_path} " f"--distance={distance} --effort={effort} --quiet"


def number_lgt_regex(expr: str) -> str:
    """Parses lt/gt condition expression into a regex that validates such a number

    @note function name could be read as - number lesser or greater than regular expression

    @param expr: Bigger/Lesser than condition - e.g.: "<122", ">9".
    @return: Regex that validates a number which passes the expression
    """

    bigger_than = re.compile(r">\d+")
    lesser_than = re.compile(r"<\d+")

    number = int(expr[1:])
    digits_count = len(expr) - 1
    last_digit = number % 10

    if bigger_than.fullmatch(expr) is not None:
        return fr"((\d{{{digits_count + 1}}}\d*)|(\d{{{digits_count - 1}}}[{last_digit + 1}-9]))"
        # return re.compile(fr"""\
        # (\d{{{digits_count + 1}}}\d*)\                  # More digits than the number
        # |\                                              # or
        # (\d{{{digits_count - 1}}}[{last_digit + 1}-9])  # Same #digits, but last one is greater than number % 10
        # """, re.VERBOSE).pattern
    elif lesser_than.fullmatch(expr) is not None:
        return fr"((\d{{1,{digits_count - 1}}})|(\d{{{digits_count - 1}}}[0-{last_digit-1}]))"
        # return fr"""\
        # (\d{{1,{digits_count - 1}}})\                   # Less digits than the original number
        # |\                                              # or
        # (\d{{{digits_count - 1}}}[0-{last_digit-1}])    # Same #digits, but last one is lesser than number % 10
        # """

    raise AssertionError(f"This is not a lt/gt expression! '{expr}'")


def timed_command(stdin: str) -> float:
    """Runs a given command on a subshell and records its execution time

    @todo Use subprocess.run(stdin, check=True) instead

    Note: Execution timeout implemented to 60 seconds

    @param stdin: Used to run the subshell command
    @return: Time it took for the command to run (in seconds)
    """
    # Execute command and time the CT
    start = time.time()
    subprocess.run(stdin, shell=True, check=True, timeout=180)
    return time.time() - start


def total_pixels(target_image: str) -> int:
    """Counts the number of pixels of an image

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

    return cv2.imread(target_image, cv2.IMREAD_UNCHANGED).size


def rename_duplicate(intended_abs_filepath: str) -> str:
    """Finds an original filename if it is already taken

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
    """Removes compressed files from a previous execution

    """
    for file in os.listdir(DATASET_COMPRESSED_PATH):
        os.remove(os.path.abspath(DATASET_COMPRESSED_PATH + file))


def dataset_img_info(target_image: str, keyword: int) -> str:
    """Extract the information present in the names of the pre-processed dataset

    @param target_image: Path to the image which name is to be evaluated
    @param keyword: parameter defined attribute (id) to extract
    @return: Attribute value for the image
    """
    retval = os.path.basename(target_image).split("_")[keyword]
    if keyword == DEPTH:
        return retval.replace(".apng", "")
    return retval
