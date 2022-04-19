import math
from typing import Tuple

from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np


def comparable_images(img1: np.ndarray, img2: np.ndarray, same_dtype: bool = False) -> Tuple[bool, str]:
    if img2.shape != img1.shape:
        return False, f"Images have different shapes! {img1.shape}, {img2.shape}"

    if img2.dtype != img1.dtype and same_dtype:
        return False, f"Images have different data types! {img1.dtype}, {img2.dtype}"

    return True, ""


def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Reference https://www.statology.org/mean-squared-error-python/
    :param img1:
    :param img2:
    :return: MSE between the "images" in float type
    """

    # Images must be comparable
    comparable, error_msg = comparable_images(img1, img2)
    assert comparable, error_msg

    return np.square(np.subtract(img1, img2)).mean()


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    From: https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
    :param img1: An Image
    :param img2: Another Image
    :return: PSNR score
    """
    comparable, error_msg = comparable_images(img1, img2)
    assert comparable, error_msg

    mean_squared_error = mse(img1, img2)

    # No noise -> PSNR max
    if mean_squared_error == 0:
        return 100.

    bit_depth = 8
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
