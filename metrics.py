""" Image quality loss metrics functions

 This script serves as a project local library.
 Includes functions used to determine, evaluated by a known metric
 standard, the loss of quality (after a lossy compression process).

 Supported metrics:
  * MSE
  * PSNR
  * Mean-SSIM

"""
import sys
from typing import Tuple, Callable, Any

import numpy as np
from numpy import ndarray
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr


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


def custom_ssim(img1_: np.ndarray, img2_: np.ndarray, color: bool = False) -> float:
    """

    @param color:
    @param img1_: Image 1
    @param img2_: Image 2
    @return: SSIM value measuring the visibility between the two images
    """

    if color is False:
        color = img1_.shape[-1] in (3, 4)
    kwargs = dict()

    if color:
        kwargs["channel_axis"] = -1

    return metric_router(img1_, img2_, ssim, **kwargs)


def custom_mse(img1_: np.ndarray, img2_: np.ndarray) -> float:
    """ Calculates the Mean Squared Error between two images.

    Reference https://www.statology.org/mean-squared-error-python/
    :param img1_: Image 1
    :param img2_: Image 2
    :return: MSE between the "images" in float type
    """

    return metric_router(img1_, img2_, mse)


def custom_psnr(img1_: np.ndarray, img2_: np.ndarray, bits_per_sample: int | None = None) -> float:
    """ Calculates the PSNR value of the difference between the images

    Calculates each frame at a time, for multi-frame images

    @param img1_: Image 1.
    @param img2_: Image 2.
    @param bits_per_sample:
    @return: PSNR score.
    """

    max_pixel = 2 ** bits_per_sample

    return metric_router(img1_, img2_, psnr, data_range=max_pixel)


def metric_router(img1_: ndarray, img2_: ndarray, metric_func: Callable, **kwargs) -> float:
    """Calls upon the metric function to calculate the value

    If multi-frame, calculates per frame and averages the result
    Used skimage lib

    @param img1_:
    @param img2_:
    @param metric_func: Function that calculates the metric per frame
    @param kwargs: Specific parameter for the metric
    @return:
    """
    assert callable(metric_func), f"Object \"{metric_func}\" is not a function!"
    if metric_func == mse and kwargs != dict():
        raise Warning(f"Keyword arguments: \"{kwargs}\" are not used in MSE metric!")
    elif metric_func == psnr:
        for key in kwargs.keys():
            if key != "data_range":
                kwargs.pop(key)
                raise Warning(f"Keyword argument: \"{key}\" is not used in the PSNR metric!")
    elif metric_func == ssim:
        for key in kwargs.keys():
            if key != "channel_axis":
                kwargs.pop(key)
                raise Warning(f"Keyword argument: \"{key}\" is not used in the SSIM metric!")

    comparable, error_msg = are_images_comparable(img1_, img2_)
    assert comparable, error_msg

    ndim = len(img1_.shape)

    if ndim == 2 or ndim == 3 and img1_.shape[-1] in (3, 4):
        # Single frame image (gray-scaled or colored)
        return metric_func(img1_, img2_, **kwargs)
    elif ndim == 4 or ndim == 3:
        # Multi-frame image
        return np.asarray(
            [mse(img1_[i], img2_[i]) for i in range(img1_.shape[0])]
        ).mean()
    else:
        raise AssertionError(f"Strange image shape: {img1_.shape}")
