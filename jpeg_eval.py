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

import os
from subprocess import Popen, PIPE
from typing import Any

import numpy as np
import pandas as pd
from numpy import ndarray
from pydicom import dcmread

import dicom_parser
import metrics
from dicom_parser import extract_attributes
from parameters import PathParameters, QUALITY_TOTAL_STEPS, MINIMUM_JPEG_QUALITY, ResultsColumnNames
from squeeze import squeeze_data

QUALITY_SPREAD: int = 1
# Quality settings
QUALITY_VALUES: np.ndarray = np.linspace(MINIMUM_JPEG_QUALITY, 100, QUALITY_TOTAL_STEPS) \
    .astype(np.ubyte)

# Alias
R = ResultsColumnNames


def compress_n_compare():
    """Compress and extract jpeg statistic

    The purpose is to compress a set of uncompressed images in JPEG
    Compression is performed using multiple quality configurations
    For each quality configuration, compute the metrics of the resulting image (CR, PSNR, SSIM versus the dataset)

    :return:
    """

    # Compress using the above quality parameters
    # Save the compression ratio in a dataframe
    results = pd.DataFrame(data=dict(filename=[], cr=[], mse=[], psnr=[], ssim=[]))

    for file_name in os.listdir(PathParameters.DATASET_DICOM_PATH):
        file_path: str = PathParameters.DATASET_DICOM_PATH + file_name

        dcm_data = dcmread(file_path)
        # Read input uncompressed image file
        uncompressed_img: ndarray = dcm_data.pixel_array

        print(f"Evaluating {file_name}", end="...")
        for quality in QUALITY_VALUES:

            body_part, bits_per_sample, color_space, modality, samples_per_pixel = extract_attributes(dcm_data)
            bits_allocated = dcm_data.get(dicom_parser.BITS_ALLOCATED_TAG)
            nframes: int = dicom_parser.get_number_of_frames(dcm_data, uncompressed_img.shape,
                                                             single_channel=samples_per_pixel == 1)

            encoded_target_path = f"{PathParameters.DATASET_PATH}{modality.value}_{body_part.value}" \
                                  f"_{color_space.value.replace('_', '')}_{samples_per_pixel.value}" \
                                  f"_{bits_per_sample.value}_{nframes}.dcm"

            # Encode input file
            command = f"dcmcjpeg +ee +q {quality} {file_path} {encoded_target_path}"
            if exec_cmd(command) is False:
                print("Skipped because of error at encoding.")
                continue

            # Read encoded image file
            img_encoded = dcmread(encoded_target_path)

            encoded_pixel_array: ndarray = img_encoded.pixel_array

            if not compatible_datatypes(encoded_pixel_array, uncompressed_img):
                print(f"Unexpected loss of bit depth from {uncompressed_img.dtype} to {encoded_pixel_array.dtype}!")
                break

            # Get the compressed image size (bytes)
            uncompressed_img_size: float | Any = uncompressed_img.size * (bits_allocated.value / 8)
            img_encoded_size: int = len(img_encoded.PixelData)

            # Calculate CR
            cr = uncompressed_img_size / img_encoded_size

            # Calculate the SSIM between the images
            mse, ssim = (
                metric(uncompressed_img, encoded_pixel_array) for metric in (metrics.custom_mse, metrics.custom_ssim)
            )
            psnr = metrics.custom_psnr(uncompressed_img, encoded_pixel_array, bits_per_sample=bits_per_sample.value)

            # Write to dataframe
            suffix = f'_q{quality}.jpeg'
            file_name = os.path.basename(encoded_target_path).replace('.dcm', suffix)

            # Ensure file name is unique (add id if need be)
            if file_name in list(results["filename"].values):
                file_name = file_name.replace(suffix, f"_1{suffix}")
                i = 1
                while file_name in results["filename"].values:
                    file_name = file_name.replace(f"_{i}{suffix}", f"_{i+1}{suffix}")
                    i += 1

            results = pd.concat([pd.DataFrame(dict(
                filename=[f"{file_name}"],
                cr=[cr],
                mse=[mse],
                psnr=[psnr],
                ssim=[ssim]
            )), results])

        print("Done!")

    for generated_dcm in filter(lambda file: file.endswith(".dcm"), os.listdir(PathParameters.DATASET_PATH)):
        os.remove(PathParameters.DATASET_PATH+generated_dcm)

    results.to_csv(f"{PathParameters.JPEG_EVAL_RESULTS_PATH}.csv", index=False)


def compatible_datatypes(ndarray1, ndarray2):
    return ndarray1.dtype == ndarray2.dtype


def exec_cmd(command) -> bool:
    """

    @param command:
    """
    p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
    _, _ = p.communicate(timeout=15)
    if p.returncode != 0:
        print(f"\nError status {p.returncode} executing the following command: \"{command}\"."
              f" Hint: Run it again to debug")
        return False
    return True


def check_deps():
    """Verifies the existence of dependencies

    """
    print("Searching for the dcmcjpeg tool... ", end="")
    if os.system("which dcmcjpeg") != 0:
        print("dcmtk not found!")
        exit(1)


if __name__ == '__main__':
    check_deps()

    compress_n_compare()
    squeeze_data(PathParameters.JPEG_EVAL_RESULTS_PATH)
