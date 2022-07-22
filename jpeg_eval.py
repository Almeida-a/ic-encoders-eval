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

import numpy as np
import pandas as pd
from numpy import ndarray
from pydicom import dcmread

import dicom_parser
import metrics
from dicom_parser import extract_attributes
from parameters import JPEG_EVAL_RESULTS_FILE, QUALITY_TOTAL_STEPS, MINIMUM_JPEG_QUALITY, DATASET_PATH, \
    DATASET_DICOM_PATH, ResultsColumnNames
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
    df = pd.DataFrame(data=dict(filename=[], cr=[], mse=[], psnr=[], ssim=[]))

    for file_name in os.listdir(DATASET_DICOM_PATH):
        file_path: str = DATASET_DICOM_PATH + file_name

        print(f"Evaluating {file_name}", end="...")
        for quality in QUALITY_VALUES:
            dcm_data = dcmread(file_path)

            # Read input uncompressed image file
            uncompressed_img: ndarray = dcm_data.pixel_array

            body_part, bits_per_sample, color_space, modality, samples_per_pixel = extract_attributes(dcm_data)
            nframes: int = dicom_parser.get_number_of_frames(dcm_data, uncompressed_img.shape,
                                                             single_channel=samples_per_pixel == 1)

            encoded_target_path = f"{DATASET_PATH}{modality.value}_{body_part.value}" \
                                  f"_{color_space.value.replace('_', '')}_{samples_per_pixel.value}" \
                                  f"_{bits_per_sample.value}_{nframes}.dcm"

            # Encode input file
            command = f"dcmcjpeg +ee +q {quality} {file_path} {encoded_target_path}"
            exec_cmd(command)

            # Read encoded image file
            img_encoded = dcmread(encoded_target_path)

            encoded_pixel_array: ndarray = img_encoded.pixel_array

            assert encoded_pixel_array.dtype == uncompressed_img.dtype, \
                f"Unexpected loss of bit depth from {uncompressed_img.dtype} to {encoded_pixel_array.dtype}!"

            # Get the compressed image size
            img_encoded_size = len(img_encoded.PixelData)

            # Calculate dataset_compressed bitstream size
            og_image_bit_depth = int(uncompressed_img.dtype.name.split("uint")[1])
            cr = uncompressed_img.size * og_image_bit_depth / img_encoded_size

            # Calculate the SSIM between the images
            mse, ssim = (
                metric(uncompressed_img, encoded_pixel_array) for metric in (metrics.custom_mse, metrics.custom_ssim)
            )
            psnr = metrics.custom_psnr(uncompressed_img, encoded_pixel_array, bits_per_sample=bits_per_sample.value)

            # Write to dataframe
            suffix = f'_q{quality}.jpeg'
            file_name = os.path.basename(encoded_target_path).replace('.dcm', suffix)

            # Ensure file name is unique (add id if need be)
            if file_name in list(df["filename"].values):
                file_name = file_name.replace(suffix, f"_1{suffix}")
                i = 1
                while file_name in df["filename"].values:
                    file_name = file_name.replace(f"_{i}{suffix}", f"_{i+1}{suffix}")
                    i += 1

            df = pd.concat([pd.DataFrame(dict(
                filename=[f"{file_name}"],
                cr=[cr],
                mse=[mse],
                psnr=[psnr],
                ssim=[ssim]
            )), df])

        print("Done!")

    for generated_dcm in filter(lambda file: file.endswith(".dcm"), os.listdir(DATASET_PATH)):
        os.remove(DATASET_PATH+generated_dcm)

    df.to_csv(f"{JPEG_EVAL_RESULTS_FILE}.csv", index=False)


def exec_cmd(command):
    """

    @param command:
    """
    p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
    p.communicate(timeout=15)
    if p.returncode != 0:
        print(f"Error status {p.returncode} executing the following command: \"{command}\"")
        exit(1)


def check_deps():
    """Verifies the existence of dependencies

    """
    if os.system("which dcmcjpeg") != 0:
        print(f"dcmtk not found!")
        exit(1)


if __name__ == '__main__':
    check_deps()

    compress_n_compare()
    squeeze_data(JPEG_EVAL_RESULTS_FILE)
