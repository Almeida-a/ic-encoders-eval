"""Miscellaneous utility functions

"""

import os

import cv2
import numpy as np
from apng import APNG, PNG
from numpy import ndarray


def get_apng_frames_resolution(filename: str) -> tuple[int, int]:
    """

    @param filename: APNG image file name
    @return: height, width
    """

    apng_img = APNG.open(filename)

    first_frame, _ = apng_img.frames[0]

    tmp: ndarray = read_apng_frame(first_frame)

    return tmp.shape[:2]


def read_apng_frame(frame: PNG) -> ndarray:
    """

    @param frame: Single frame from an APNG
    @return:
    """
    tmp_fname = "tmp.png"
    frame.save(tmp_fname)
    tmp = cv2.imread(tmp_fname, cv2.IMREAD_UNCHANGED)
    os.remove(tmp_fname)
    return tmp


def to_np_array(apng_img: APNG) -> ndarray:
    """

    @param apng_img: APNG image parsed object
    @return: Numpy ndarray with the images contents
    """
    frames_arr: list[ndarray] = [read_apng_frame(frame) for frame, _ in apng_img.frames]

    return np.array(frames_arr)


def read_apng(filename: str) -> ndarray:
    """

    @param filename: .apng image file name - E.g.: "image.png"
    @return: Numpy ndarray with the images contents
    """
    return to_np_array(
        APNG.open(filename)
    )


def get_apng_depth(target_image: str) -> int:
    """

    @param target_image: Image to be evaluated
    @return: Number of frames in the multi-frame image
    """
    return len(APNG.open(target_image).frames)
