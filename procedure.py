import json
import os.path
import time
from subprocess import Popen, PIPE
from typing import List, Dict, Union, Tuple

import cv2
import numpy as np
import pandas as pd

import metrics

LOSSLESS_EXTENSION: str = ".png"
RESULTS_FILE: str = "procedure_results"
"""
    Codecs' versions
        cjxl, djxl -> v0.6.1
        cwebp, dwebp -> v0.4.1
        cavif -> v1.3.4
        avif_encode -> 0.2.2
            (not sure, can't check w/ -V, I ran cargo install at 4 Apr 2022)
"""


def check_codecs():
    """
    Checks if all codecs are present in current machine
    Exits program if one does not exist
    :return:
    """
    if os.system("which cavif") != 0 or os.system("which avif_decode") != 0:
        print("AVIF codec not found!")
        exit(1)
    elif os.system("which djxl") != 0 or os.system("which cjxl"):
        print("JPEG XL codec not found!")
        exit(1)
    elif os.system("which cwebp") != 0 or os.system("which dwebp") != 0:
        print("WebP codec not found!")
        exit(1)
    print("All codecs are available!")


def encode_jxl(target_image: str, distance: float, effort: int, output_path: str) -> float:
    """
    Encoder used: github.com/libjxl/libjxl/ -> build/tools/cjxl
    :param target_image: Path to image targeted for compression encoding
    :param distance: Quality setting as set by cjxl (butteraugli distance)
    :param effort: --effort level parameter as set by cjxl
    :param output_path: Path where the dataset_compressed image should go to
    :return: Time taken to compress
    """

    # Construct the encoding command
    command: str = f"cjxl {target_image} {output_path} " \
                   f"--distance={distance} --effort={effort} --quiet"

    # Execute jxl and measure the time it takes
    # TODO (low prio.) consider better methods of acquiring the CT
    start = time.time()
    # Try to compress
    if os.system(command) != 0:
        print(f"Error executing the following command:\n {command}")
        exit(1)
    comp_t: float = time.time() - start

    return comp_t


def encode_webp(target_image: str, quality: int, effort: int, output_path: str) -> float:
    """
    Encodes an image using the cwebp tool
    :param target_image: path/to/image.ext, where the extension needs to be supported by cwebp
    :param quality: Quality loss level (1 to 100), -q option of the tool
    :param effort: Effort of the encoding process (-m option of the tool)
    :param output_path: Directory where the compression
    :return: Compression time, in s
    """

    # Command to be executed for the compression
    command: str = f"cwebp -v -q {quality} -m {effort} {target_image} -o {output_path}"

    # Execute command (and check status)
    p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
    _, stderr = p.communicate()

    # Extract compression time (read from stdout)
    time_line: str = str(stderr)[2:-1].split(r"\n")[2]
    comp_t = float(time_line[-6:-1])

    return comp_t


def decode_compare(target_image: str) -> Tuple[float, float, float, float]:
    """

    :param target_image:
    :return: DT(s), MSE, PSNR, SSIM
    """

    extension: str = target_image.split(".")[-1]
    # Same directory, same name, .png
    out_path: str = ".".join(target_image.split(".")[:-1]) + LOSSLESS_EXTENSION
    # Decoding time
    dt: float = -1.

    if extension == "jxl":
        # Construct decoding command
        command: str = f"djxl {target_image} {out_path} --quiet"
        # Execute command and record time
        start = time.time()
        if os.system(command) != 0:
            raise AssertionError(f"Error at executing the following command\n => '{command}'")
        dt = time.time() - start
    elif extension == "avif":
        # Construct decoding command
        command: str = f"avif_decode {target_image} {out_path}"
        # Execute command and record time
        start = time.time()
        if os.system(command) != 0:
            raise AssertionError(f"Error at executing the following command\n => '{command}'")
        dt = time.time() - start
    elif extension == "webp":
        # Construct decoding command
        command: str = f"dwebp -v {target_image} -o {out_path}"
        # Execute command (and check status)
        p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
        _, stderr = p.communicate()
        # Extract DT from output
        time_line: str = str(stderr).split(r"\n")[0]
        dt = float(time_line[-7:-1])
    else:
        print(f"Unsupported extension for decoding: {extension}")
        exit(1)

    # Read the output image w/ opencv
    out_image = cv2.imread(out_path)
    # Get original lossless image
    og_image_path = "".join(target_image.split("_compressed"))
    og_image_path = "/".join(og_image_path.split("/")[:-1]) + LOSSLESS_EXTENSION
    og_image = cv2.imread(og_image_path)

    # Evaluate the quality of the resulting image
    mse: float = metrics.mse(og_image, out_image)
    psnr: float = metrics.psnr(og_image, out_image)

    # Convert to grayscale in order to calculate the MSSIM
    og_image = cv2.cvtColor(og_image, cv2.COLOR_BGR2GRAY)
    out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2GRAY)
    ssim: float = metrics.ssim(og_image, out_image)

    return dt, mse, psnr, ssim


def encode_avif(target_image: str, quality: int, speed: int, output_path: str) -> float:
    """

    :param target_image:
    :param quality:
    :param speed:
    :param output_path: Directory where the dataset_compressed file
    :return: Compression time, in seconds
    """
    # Construct the command
    command: str = f"cavif -o {output_path} " \
                   f"--quality {quality} --speed {speed} --quiet " \
                   f"{os.path.abspath(target_image)}"

    # Execute jxl and measure the time it takes
    # TODO consider better methods of acquiring the CT
    start = time.time()
    # Try to compress
    if os.system(command) != 0:
        print(f"Error executing the following command:\n {command}")
        exit(1)
    comp_t: float = time.time() - start

    return comp_t


def image_to_dir(dataset_path: str, target_image: str) -> str:
    """

    :param dataset_path: Path to the dataset folder (dataset for compressed is a sibling folder)
    :param target_image: Path to the image from the original dataset
    :return: Path to the folder in the dataset_compressed where the compressed form of
        target_image should be stored
    """
    # Do a "cd ../dataset_compressed"
    path = "/".join(os.path.abspath(dataset_path).split("/")[:-1]) + "/dataset_compressed"
    # Return path + image.path.basename
    folder = path + "/" + os.path.basename(target_image) + "/"
    # Mkdir if folder does not exist
    if not os.path.exists(folder):
        os.system(f"mkdir {folder}")
    return folder


def bulk_compress(dataset_path: str, jxl: bool = True, avif: bool = True, webp: bool = True):
    """
    Compresses all raw images in the dataset folder, each encoding done
        with a series of parameters.
    Each coded image will be within a folder named `dataset_path`_compressed/
    These images will be organized the same way the raw images are organized in the `dataset_path` folder
    In the _compressed/ folder, instead of imageX.raw (for each image),
        we will have imageX/qY-eZ.{jxl,avif,webp}, where:
         Y is the quality setting for the encoder (q1_0 for quality=1.0, for example)
         Z is the encoding effort (e.g.: e3 for effort=3).
    When the compression is finished, the function should write a .csv file
        with information on the encoding time of each encoding process carried out.
    Structure example of csv file:
        Head -> filename; CT_ms
        Data -> q90e3.avif; 900
                q1_0e5.jxl; 1000
    :param webp:
    :param avif:
    :param jxl:
    :param dataset_path: Path to the dataset folder
    :return:
    """
    # Save all images path relative to dataset_path
    image_list = [dataset_path + str(i) for i in range(1, 6)]

    # Set quality parameters to be used in compression
    # How many configurations are expected (evenly spaced in the range)
    spread: int = 3
    quality_param_jxl: np.ndarray = np.linspace(.0, 3.0, spread)
    quality_param_avif = range(1, 101, int(100 / spread))
    quality_param_webp = range(1, 101, int(100 / spread))

    # Set effort/speed parameters for compression (common step)
    step: int = 4
    effort_jxl = range(1, 9, step)
    speed_avif = range(0, 11, step)
    effort_webp = range(0, 7, step)

    # Encode (to target path) and record time of compression
    ct: float  # Record time of compression
    stats = pd.DataFrame(
        data=dict(filename=[], ct=[], dt=[], mse=[], psnr=[], ssim=[])
    )

    # JPEG XL
    if jxl is True:
        for target_image in image_list:
            for quality in quality_param_jxl:
                for effort in effort_jxl:
                    # Construct output file total path
                    outfile_name: str = f"q{quality}-e{effort}.jxl"
                    output_path = image_to_dir(dataset_path, target_image) + outfile_name

                    # Add wildcard for now because the extensions are missing
                    ct = encode_jxl(target_image=target_image + LOSSLESS_EXTENSION,
                                    distance=quality, effort=effort,
                                    output_path=output_path)

                    # Decode and collect stats to stats df
                    dt, mse, psnr, ssim = decode_compare(output_path)
                    # Remove generated (png and jxl) images
                    os.remove(output_path)
                    os.remove(".".join(output_path.split(".")[:-1]) + LOSSLESS_EXTENSION)

                    # Append new stats do dataframe
                    row = dict(filename=outfile_name, ct=ct, dt=dt, mse=mse, psnr=psnr, ssim=ssim)
                    stats.append(row, ignore_index=True)

                    print(f"Finished analysing image \"{outfile_name}\".")

    # AVIF
    if avif is True:
        for target_image in image_list:
            for quality in quality_param_avif:
                for speed in speed_avif:
                    # Construct output file total path
                    outfile_name: str = f"q{quality}-s{speed}.avif"
                    output_path = image_to_dir(dataset_path, target_image) + outfile_name

                    # Add wildcard for now because the extensions are missing
                    ct = encode_avif(target_image=target_image + LOSSLESS_EXTENSION,
                                     quality=quality, speed=speed, output_path=output_path)

                    # Decode and collect stats to stats df
                    dt, mse, psnr, ssim = decode_compare(output_path)
                    # Remove generated (png and avif) images
                    os.remove(output_path)
                    os.remove(".".join(output_path.split(".")[:-1]) + LOSSLESS_EXTENSION)

                    # Append new stats do dataframe
                    row = dict(filename=outfile_name, ct=ct, dt=dt, mse=mse, psnr=psnr, ssim=ssim)
                    stats.append(row, ignore_index=True)

                    print(f"Finished analysing image \"{outfile_name}\".")

    # WebP
    if webp is True:
        for target_image in image_list:
            for quality in quality_param_webp:
                for effort in effort_webp:
                    # Construct output file total path
                    outfile_name: str = f"q{quality}-e{effort}.webp"
                    output_path = image_to_dir(dataset_path, target_image) + outfile_name

                    # Add wildcard for now because the extensions are missing
                    ct = encode_webp(target_image=target_image + LOSSLESS_EXTENSION,
                                     quality=quality, effort=effort, output_path=output_path)

                    # Decode and collect stats to stats df
                    dt, mse, psnr, ssim = decode_compare(output_path)
                    # Remove generated (png and webp) images
                    os.remove(output_path)
                    os.remove(".".join(output_path.split(".")[:-1]) + LOSSLESS_EXTENSION)

                    # Append new stats do dataframe
                    row = dict(filename=outfile_name, ct=ct, dt=dt, mse=mse, psnr=psnr, ssim=ssim)
                    stats.append(row, ignore_index=True)

                    print(f"Finished analysing image \"{outfile_name}\".")

    # Save csv files
    stats.to_csv(RESULTS_FILE + ".csv", index=False)


def resume_stats():
    """

    :return:
    """
    # Read csv to df
    df = pd.read_csv(RESULTS_FILE + ".csv")

    # Aggregate the results to a dict
    resume: Dict[str, Dict[str, Dict[str, float]]] = dict()
    for unique_fname in tuple(set(df["fname"])):
        # Dataframe containing only the data with one of the unique_fnames
        fname_df = df[df["fname"] == unique_fname]
        # Gather statistics
        resume[unique_fname] = {
            "ct": dict(min=min(fname_df["ct"]), max=max(fname_df["ct"]),
                       avg=np.average(fname_df["ct"]),
                       std=np.std(fname_df["ct"])),
            "dt": dict(min=min(fname_df["dt"]), max=max(fname_df["dt"]),
                       avg=np.average(fname_df["dt"]),
                       std=np.std(fname_df["dt"])),
            "mse": dict(min=min(fname_df["mse"]), max=max(fname_df["mse"]),
                        avg=np.average(fname_df["mse"]),
                        std=np.std(fname_df["mse"])),
            "psnr": dict(min=min(fname_df["psnr"]), max=max(fname_df["psnr"]),
                         avg=np.average(fname_df["psnr"]),
                         std=np.std(fname_df["psnr"])),
            "ssim": dict(min=min(fname_df["ssim"]), max=max(fname_df["ssim"]),
                         avg=np.average(fname_df["ssim"]),
                         std=np.std(fname_df["ssim"]))
        }

    # Save dict to a json
    out_file = open(RESULTS_FILE + ".json", "w")
    json.dump(resume, out_file, indent=4)


if __name__ == '__main__':
    check_codecs()
    bulk_compress("images/dataset/", jxl=True, avif=True, webp=True)
    resume_stats()
