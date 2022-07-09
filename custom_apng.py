"""APNG procedures - read/write to/from numpy ndarray

"""
import os

import apng
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


def staple_pngs(output_path, *pngs: str) -> bool:
    """Joins multiple png images into animated png format

    @param output_path: apng output file name
    @param pngs: png file names list
    @return: Status code
    """
    frames = np.array([cv2.imread(img, cv2.IMREAD_UNCHANGED) for img in pngs])

    return write_apng(output_path, frames)[0]


def write_apng(file_name: str, img_array: np.ndarray) -> tuple[bool, np.ndarray]:
    """Writes the contents of a ndarray to an apng file

    @param file_name: file name of the apng file
    @param img_array: Contents to be written to apng format
    @return: Status code, ndarray with the apng contents (used to check if no change has occurred)
    """
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
