import json
import os
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

import metrics

QUALITY_STEPS: int = 20
# Quality settings
QUALITY_VALUES: np.ndarray = np.linspace(1, 100, QUALITY_STEPS)
# Where the raw/processed results of the experiment are written on
DEFAULT_OUT_FILE = "results"  # .csv or .json


def images(ext: str) -> str:
    """
    Generator function
    :return: Sequence of image files path to be processed in experience
    """

    possible_ext = (".jpg", ".png")

    if ext not in possible_ext:
        raise AssertionError(f"Unsupported extension: {ext}")

    prefix: str = "images/dataset/"

    for file in os.listdir(prefix):
        if file.endswith(ext):
            yield prefix+file


def compress_n_compare():

    """
    The purpose is to compress a set of uncompressed images in JPEG
    Compression is performed using multiple quality configurations
    For each quality configuration, compute the SSIM of the resulting image (versus the dataset)
    :return:
    """

    # Extension of the images to be processed
    ext: str = ".png"

    # Compress using the above quality parameters
    # Save the compression ratio in a dataframe
    df = pd.DataFrame(data=dict(fname=[], original_size=[], compressed_size=[], CR=[], SSIM=[]))
    for file_path in images(ext):
        file_name: str = ".".join(file_path.split("/")[-1].split(".")[:-1])
        for quality in QUALITY_VALUES:
            quality = int(quality)

            # Read input file
            img = cv2.imread(file_path)

            # Compress input file
            comp_img = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            comp_img_bytes: bytes = np.array(comp_img[1]).tobytes()

            # Calculate dataset_compressed bitstream size
            cr = int(img.size / (len(comp_img_bytes) * 8))

            # Decode JPEG bitstream
            buffer: np.array = np.asarray(bytearray(comp_img_bytes), dtype=np.uint8)
            img_c = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

            # Calculate the SSIM between the images
            ssim = metrics.ssim(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
            )

            # Write to dataframe
            df = pd.concat([pd.DataFrame(dict(
                fname=[f"{file_name}_q{quality}"],
                original_size=[img.size],
                compressed_size=[len(comp_img_bytes) * 8],
                CR=[cr],
                SSIM=ssim
            )), df], ignore_index=True)
    df.to_csv("results.csv", index=False)


def comparison_stats(csv_file_path: str):
    # Check validity of argument
    if not os.path.exists(csv_file_path):
        raise AssertionError(f"Path: {csv_file_path} does not contain a file!")
    if not csv_file_path.endswith(".csv"):
        raise AssertionError(f"Argument file {csv_file_path} format is not csv")
    # Read csv
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
        # Write stats in dict
        cr_list: List[float] = sub_df["CR"]
        stats[f"q{quality_val}"]["CR"] = {
            "min": min(cr_list), "max": max(cr_list), "avg": np.mean(cr_list), "dev": np.std(cr_list)
        }
        ssim_list: List[float] = sub_df["SSIM"]
        stats[f"q{quality_val}"]["SSIM"] = {
            "min": min(ssim_list), "max": max(ssim_list), "avg": np.mean(ssim_list), "dev": np.std(ssim_list)
        }

    # Store the results in a json file
    out_file = open(DEFAULT_OUT_FILE+".json", "w")
    json.dump(stats, out_file, indent=6)


if __name__ == '__main__':
    # compress_n_compare()
    comparison_stats(DEFAULT_OUT_FILE + ".csv")
