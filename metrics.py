""" Image quality loss metrics functions

 This script serves as a project local library.
 Includes functions used to determine, evaluated by a known metric
 standard, the loss of quality (after a lossy compression process).

 Supported metrics:
  * MSE
  * PSNR
  * Mean-SSIM

"""

import math
from typing import Tuple

from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np


def are_images_comparable(img1_: np.ndarray, img2_: np.ndarray, same_dtype: bool = False) -> Tuple[bool, str]:
    """ Checks if images are compatible for comparison.

    :param img1_: Image 1
    :param img2_: Image 2
    :param same_dtype: Check for data type compatibility
    :return: Compatibility status and error message, if non-compatible
    """

    if img2_.shape != img1_.shape:
        return False, f"Images have different shapes! {img1_.shape}, {img2_.shape}"

    if img2_.dtype != img1_.dtype and same_dtype:
        return False, f"Images have different data types! {img1_.dtype}, {img2_.dtype}"

    return True, ""


def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """ Calculates the Mean Squared Error between two images.

    Reference https://www.statology.org/mean-squared-error-python/
    :param img1: Image 1
    :param img2: Image 2
    :return: MSE between the "images" in float type
    """

    # Images must be comparable
    comparable, error_msg = are_images_comparable(img1, img2)
    assert comparable, error_msg

    return np.square(np.subtract(img1, img2)).mean()


def psnr(img1_: np.ndarray, img2_: np.ndarray) -> float:
    """ Calculates the PSNR value of the difference between the images

    Credits: https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/

    :param img1_: Image 1.
    :param img2_: Image 2.
    :return: PSNR score.
    """
    comparable, error_msg = are_images_comparable(img1_, img2_)
    assert comparable, error_msg

    mean_squared_error = mse(img1_, img2_)

    # No noise -> PSNR max
    if mean_squared_error == 0:
        return 100.

    # Get the largest bit depth of the images (extract from np.dtype)
    bit_depth = max([int(str(img.dtype).split("int")[-1]) for img in (img1_, img2_)])

    max_pixel = 2 ** bit_depth

    return 20 * math.log10(max_pixel / math.sqrt(mean_squared_error))


if __name__ == '__main__':
    img1_path: str = "images/miles_morales_night_spark-wallpaper-1920x1080.jpg"
    img2_path: str = "images/winter_season_8-wallpaper-1920x1080.jpg"

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Convert to grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    print(f"MSE = {round(mse(img1, img2), 1)}")
    print(f"PSNR = {round(psnr(img1, img2), 1)}")
    print(f"SSIM = {round(ssim(img1, img2), 1)}")
