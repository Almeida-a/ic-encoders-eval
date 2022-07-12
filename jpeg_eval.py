"""The goal is to map the JPEG encoded images' CR into MSSIM/PSNR/MSE

In other words, we will try to associate the compression ratio of the JPEG encoded images with the SSIM/PSNR/MSE
    values of them regarding their original versions.
More specifically, the purpose is to generate a json file that sums up statistics in the following structure:
    - quality N (aggregated images encoded w/ -quality == N)
        - cr: min,max,avg,stddev
        - mse: min,max,avg,stddev
        - psnr: min,max,avg,stddev
        - ssim: min,max,avg,stddev

This is a project secondary experiment.

"""

import json
import os
from subprocess import Popen, PIPE
from typing import Dict, List

import numpy as np
import pandas as pd
from pydicom import dcmread

import metrics
import parameters
from parameters import JPEG_EVAL_RESULTS_FILE

QUALITY_SPREAD: int = 1
# Quality settings
QUALITY_VALUES: np.ndarray = np.linspace(1, 100, QUALITY_SPREAD)
# Where the raw/processed results of the experiment are written on


def images(ext: str) -> str:
    """Generator function that yields images to be studied

    :param ext: Extension used to filter the images (the ones that don't have that format)
    :return: Sequence of image files path to be processed for the experiment
    """

    possible_ext = (".jpg", ".png", ".dcm")

    if ext not in possible_ext:
        raise AssertionError(f"Unsupported extension: {ext}")

    prefix: str = "images/dataset"

    if ext == ".dcm":
        prefix += "_dicom"
    prefix += "/"

    for file in os.listdir(prefix):
        if file.endswith(ext) or ext == ".dcm":
            # If we are looking for the dicom files, the extensions might not be there, but we know they are
            #   in that prefix/dir, thus no verification needed
            yield prefix+file


def compress_n_compare():
    """ Compress and extract jpeg statistic

    The purpose is to compress a set of uncompressed images in JPEG
    Compression is performed using multiple quality configurations
    For each quality configuration, compute the metrics of the resulting image (CR, PSNR, SSIM versus the dataset)

    :return:
    """

    # Compress using the above quality parameters
    # Save the compression ratio in a dataframe
    df = pd.DataFrame(data=dict(fname=[], cr=[], mse=[], psnr=[], ssim=[]))
    for file_path in images(".dcm"):
        file_name: str = os.path.basename(file_path)
        for quality in QUALITY_VALUES:
            quality = int(quality)

            encoded_target_path = "images/dataset/tmp.dcm"

            # Read input uncompressed image file
            img: np.ndarray = dcmread(file_path).pixel_array

            # Encode input file
            command = f"dcmcjpeg +ee +q {quality} {file_path} {encoded_target_path}"
            exec_cmd(command)

            # Read encoded image file
            img_encoded = dcmread(encoded_target_path)

            assert img_encoded.pixel_array.dtype == img.dtype,\
                f"Unexpected loss of bit depth to {img_encoded.pixel_array.dtype}!"

            # Get the compressed image size
            img_encoded_size = len(img_encoded.PixelData)

            # Calculate dataset_compressed bitstream size
            og_image_bit_depth = int(dcmread(file_path).pixel_array.dtype.name.split("uint")[1])
            cr = img.size * og_image_bit_depth / img_encoded_size

            # Calculate the SSIM between the images
            mse, psnr, ssim = (
                metric(img, img_encoded.pixel_array) for metric in (metrics.custom_mse, metrics.custom_psnr, metrics.custom_ssim)
            )

            # Write to dataframe
            df = pd.concat([pd.DataFrame(dict(
                fname=[f"{file_name}_q{quality}"],
                cr=[cr],
                mse=[mse],
                psnr=[psnr],
                ssim=[ssim]
            )), df])
    df.to_csv(f"{JPEG_EVAL_RESULTS_FILE}.csv", index=False)


def exec_cmd(command):
    p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
    p.communicate(timeout=15)
    assert p.returncode == 0, f"Error status {p.returncode} executing the following command: \"{command}\""


def squeeze_stats(csv_file_path: str):
    """Reads csv with raw data and writes json w/ more readable statistics

    :param csv_file_path: File with raw data
    :return:
    """
    # Check validity of argument
    if not os.path.exists(csv_file_path):
        raise AssertionError(f"Path: {csv_file_path} does not contain a file!")
    if not csv_file_path.endswith(".csv"):
        raise AssertionError(f"Argument file {csv_file_path} format is not csv")

    df = pd.read_csv(csv_file_path)

    # For common quality settings, calculate stats for CR and SSIM (min/max/avg/dev)
    stats: Dict[str, Dict[str, Dict[str, float]]] = dict()
    for quality_val in QUALITY_VALUES:
        # Cast quality_val to int
        quality_val = int(quality_val)
        # Get df rows where filename.endswith("*_q{qv}")
        sub_df = pd.DataFrame(data={title: [] for title in df.keys()})
        for index, row in df.iterrows():
            if row["fname"].endswith(f"_q{quality_val}"):
                sub_df.loc[len(sub_df.index)] = row
        # Set up new entry in stat dict
        stats[f"q{quality_val}"] = dict()

        for metric in (*parameters.qmetrics.keys(), "cr"):
            # Write stats in dict
            metric_vals: List[float] = sub_df[metric]
            stats[f"q{quality_val}"][metric] = dict(
                min=min(metric_vals), max=max(metric_vals),
                avg=np.mean(metric_vals), stddev=np.std(metric_vals)
            )

    # Store the results in a json file
    out_file = open(f"{JPEG_EVAL_RESULTS_FILE}.json", "w")
    json.dump(stats, out_file, indent=6)


if __name__ == '__main__':
    compress_n_compare()
    squeeze_stats(f"{JPEG_EVAL_RESULTS_FILE}.csv")
