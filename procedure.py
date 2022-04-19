import json
import os.path
import re
import time
from subprocess import Popen, PIPE
from typing import List, Dict, Tuple

import cv2
import numpy as np
import pandas as pd

import metrics
from parameters import LOSSLESS_EXTENSION, PROCEDURE_RESULTS_FILE, DATASET_PATH

"""
    Codecs' versions
        cjxl, djxl -> v0.6.1
        cwebp, dwebp -> v0.4.1
        cavif -> v1.3.4
        avif_decode -> 0.2.2
            (not sure, can't check w/ -V, I ran cargo install at 4 Apr 2022)
"""
# TODO record time for all compressions (normalize C/DS retrieval)
# TODO low priority: implement timeout for
#  sub-shell calls embedding it in the command


def check_codecs():
    """
    Checks if all codecs are present in current machine
    Exits program if one does not exist

    :return:
    """
    print("Looking for the codecs...\n")
    if os.system("which cavif") != 0 or os.system("which avif_decode") != 0:
        print("AVIF codec not found!")
        exit(1)
    elif os.system("which djxl") != 0 or os.system("which cjxl") != 0:
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
    :return: Compression speed, in MP/s
    """

    # Construct the encoding command
    command: str = f"cjxl {target_image} {output_path} " \
                   f"--distance={distance} --effort={effort}"

    # Execute and measure the time it takes
    p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
    _, stderr = p.communicate()
    # Check if encoding process was successful
    if p.returncode != 0:
        print(f"Error executing the following command:\n\t\"{command}\"")
        exit(1)
    return extract_jxl_cs(stderr)


def encode_avif(target_image: str, quality: int, speed: int, output_path: str) -> float:
    """

    :param target_image: Input/Original image
    :param quality:
    :param speed:
    :param output_path: Directory where the dataset_compressed file
    :return: Compression speed, in MP/s
    """
    # Extract resolution
    height, width = cv2.imread(target_image).shape[:2]
    # Number of pixels in the image
    pixels = height * width

    # Construct the command
    command: str = f"cavif -o {output_path} " \
                   f"--quality {quality} --speed {speed} --quiet " \
                   f"{os.path.abspath(target_image)}"

    # Execute jxl and measure the time it takes
    # TODO refactor all c/ds to be calculated using time.time in order to normalize the measurements
    #   Also: use Popen timeout option to specify the command timeout
    start = time.time()
    # Try to compress
    if os.system(command) != 0:
        print(f"Error executing the following command:\n {command}")
        exit(1)
    comp_t: float = time.time() - start

    # Parse to compression speed
    cs = pixels / (comp_t * 1e6)

    return cs


def encode_webp(target_image: str, quality: int, effort: int, output_path: str) -> float:
    """
    Encodes an image using the cwebp tool
    :param target_image: path/to/image.ext, where the extension needs to be supported by cwebp
    :param quality: Quality loss level (1 to 100), -q option of the tool
    :param effort: Effort of the encoding process (-m option of the tool)
    :param output_path: Directory where the compression
    :return: Compression speed, in MP/s
    """

    # Command to be executed for the compression
    command: str = f"cwebp -v -q {quality} -m {effort} {target_image} -o {output_path}"

    # Execute command (and check status)
    p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
    _, stderr = p.communicate()

    height, width = extract_resolution(stderr, " x ")[0:2]
    pixels = int(height) * int(width)

    return pixels / (extract_webp_ct(stderr) * 1e6)


def extract_jxl_ds(stderr: bytes) -> float:
    # Extract
    ds: str = re.findall(r"\d+.\d+ MP/s", str(stderr, "utf-8"))[0]
    return float(ds[:-5])


def extract_webp_dt(stderr: bytes) -> float:
    # Extract
    ds: str = re.findall(r"Time to decode picture: \d+.\d+s", str(stderr, "utf-8"))[0]\
        .split("Time to decode picture: ")[-1]
    return float(ds[:-1])


def extract_resolution(stderr: bytes, sep: str = "x") -> List[str]:
    # Extract
    res: str = re.findall(rf"\d+{sep}\d+", str(stderr))[0]
    return res.split(sep)


def decode_compare(target_image: str) -> Tuple[float, float, float, float]:
    """

    :param target_image:
    :return: DT(s), MSE, PSNR, SSIM
    """

    extension: str = target_image.split(".")[-1]
    # Same directory, same name, .png
    out_path: str = ".".join(target_image.split(".")[:-1]) + LOSSLESS_EXTENSION
    # Decoding speed

    dt: float = -1.
    ds: float = -1.

    if extension == "jxl":
        # Construct decoding command
        command: str = f"djxl {target_image} {out_path}"
        # Execute command and record time
        p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
        _, stderr = p.communicate()
        if p.returncode != 0:
            raise AssertionError(f"Error at executing the following command\n => '{command}'")
        # Extract decoding speed from stderr

        ds = extract_jxl_ds(stderr)
    elif extension == "avif":
        # Construct decoding command
        command: str = f"avif_decode {target_image} {out_path}"
        # Execute command and record time
        start = time.time()
        if os.system(command) != 0:
            raise AssertionError(f"Error at executing the following command\n => '{command}'")
        # Decoding time
        dt = time.time() - start
    elif extension == "webp":
        # Construct decoding command
        command: str = f"dwebp -v {target_image} -o {out_path}"
        # Construct command
        p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
        # Issue command
        _, stderr = p.communicate()
        # Check for errors
        if p.returncode != 0:
            raise AssertionError(f"Error at executing"
                                 f" the following command\n => '{command}'")
        # Extract decoding time
        dt = extract_webp_dt(stderr)
    else:
        print(f"Unsupported extension for decoding: {extension}")
        exit(1)

    # Read the output image w/ opencv
    out_image = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
    og_image = get_og_image(compressed=target_image)
    pixels = og_image.shape[0] * og_image.shape[1]

    # Calculate decoding speed
    if ds == -1.:
        # Calculate decoding speed in MP/s
        ds = pixels / (dt * 1e6)

    # Evaluate the quality of the resulting image
    mse: float = metrics.mse(og_image, out_image)
    psnr: float = metrics.psnr(og_image, out_image)

    # Convert to grayscale in order to calculate the MSSIM
    ssim: float = metrics.ssim(
        og_image, out_image.astype(np.uint16)
    )

    return ds, mse, psnr, ssim


def extract_jxl_cs(stderr: bytes):
    # Find decoding speed in the output pool
    ds = re.findall(r"\d+.\d+ MP/s", str(stderr))[1][:-5]
    return ds


def extract_webp_ct(stderr: bytes) -> float:
    # Extract DT from output
    dt: str = re.findall(r"Time to encode picture: \d+.\d+s", str(stderr))[0] \
        .split("Time to encode picture: ")[-1]
    # Cast to float
    dt: float = float(dt[:-1])
    return dt


def get_og_image(compressed) -> np.ndarray:
    # Get original lossless image
    og_image_path = "".join(compressed.split("_compressed"))
    og_image_path = "_".join(og_image_path.split("_")[:-1]) + LOSSLESS_EXTENSION
    og_image = cv2.imread(og_image_path, cv2.IMREAD_UNCHANGED)
    return og_image


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


def bulk_compress(jxl: bool = True, avif: bool = True, webp: bool = True):
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
    :return:
    """

    if all(codec is False for codec in (jxl, avif, webp)):
        return

    # Save all images path relative to dataset_path
    image_list = os.listdir(DATASET_PATH)

    # Set quality parameters to be used in compression
    # How many configurations are expected (evenly spaced in the range)
    spread: int = 6
    quality_param_jxl: np.ndarray = np.linspace(.0, 3.0, spread)
    quality_param_avif = range(1, 101, int(100 / spread))
    quality_param_webp = range(1, 101, int(100 / spread))

    # Set effort/speed parameters for compression (common step)
    effort_jxl = (7,)
    speed_avif = (4,)
    effort_webp = (4,)

    # Encode (to target path) and record time of compression
    ct: float  # Record time of compression
    stats = pd.DataFrame(
        data=dict(filename=[], cs=[], ds=[], mse=[], psnr=[], ssim=[])
    )

    # JPEG XL
    if jxl is True:
        for target_image in image_list:
            for quality in quality_param_jxl:
                for effort in effort_jxl:

                    outfile_name, output_path = get_output_path(
                        DATASET_PATH, effort, quality, target_image, "jxl"
                    )

                    # Print image analysis
                    print(f"Started analysing image \"{outfile_name}\".")

                    # Add wildcard for now because the extensions are missing
                    cs = encode_jxl(target_image=DATASET_PATH + target_image,
                                    distance=quality, effort=effort,
                                    output_path=output_path)

                    # Decode and collect stats to stats df
                    stats = finalize(cs, outfile_name, output_path, stats)

    # AVIF
    if avif is True:
        for target_image in image_list:
            for quality in quality_param_avif:
                for speed in speed_avif:
                    # Construct output file total path
                    outfile_name, output_path = get_output_path(
                        DATASET_PATH, speed, quality, target_image, "avif"
                    )

                    # Print the progress being made
                    print(f"Finished analysing image \"{outfile_name}\".")

                    # Add wildcard for now because the extensions are missing
                    cs = encode_avif(target_image=DATASET_PATH + target_image,
                                     quality=quality, speed=speed, output_path=output_path)

                    # Decode and collect stats to stats df
                    stats = finalize(cs, outfile_name, output_path, stats)

    # WebP
    if webp is True:
        for target_image in image_list:
            for quality in quality_param_webp:
                for effort in effort_webp:
                    # Construct output file total path
                    outfile_name, output_path = get_output_path(
                        DATASET_PATH, effort, quality, target_image, "webp"
                    )

                    # Print the progress being made
                    print(f"Started analysing image \"{outfile_name}\".")

                    # Add wildcard for now because the extensions are missing
                    cs = encode_webp(target_image=DATASET_PATH + target_image,
                                     quality=quality, effort=effort, output_path=output_path)

                    # Decode and collect stats to stats df
                    stats = finalize(cs, outfile_name, output_path, stats)

    # TODO If procedure results file already exists, rename new file to filename+_1 or _n
    # Save csv files
    stats.to_csv(PROCEDURE_RESULTS_FILE + ".csv", index=False)


def finalize(cs, outfile_name, output_path, stats) -> pd.DataFrame:
    """
    Decodes the target file, removes the codec generated files and saves metadata to the provided dataframe

    :param cs: Compression time of the provided compressed image file
    :param outfile_name: Basename of the provided file
    :param output_path: Path to the provided file
    :param stats: Dataframe holding the data regarding the compressions
    :return: Updated Dataframe
    """
    ds, mse, psnr, ssim = decode_compare(output_path)
    # Remove generated (png and jxl) images
    os.remove(output_path)
    os.remove(".".join(output_path.split(".")[:-1]) + LOSSLESS_EXTENSION)
    # Append new stats do dataframe
    row = pd.DataFrame(
        dict(filename=[outfile_name], cs=[cs], ds=[ds], mse=[mse], psnr=[psnr], ssim=[ssim])
    )
    stats = pd.concat([stats, row])
    return stats


def get_output_path(dataset_path: str, effort: int, quality: float, target_image: str, format: str):
    # Construct output file total path
    outfile_name: str = target_image.split(LOSSLESS_EXTENSION)[0] \
                        + "_" + f"q{quality}-e{effort}.{format}"
    # Trim trailing slash "/"
    trimmed: list = dataset_path.split("/")
    trimmed.remove("")
    trimmed: str = "/".join(trimmed)
    output_path: str = trimmed + "_compressed/" + outfile_name
    return outfile_name, output_path


def resume_stats():
    """
    Digests raw compression stats into condensed stats, which are min/max/avg/std

    :return:
    """
    # Read csv to df
    df = pd.read_csv(PROCEDURE_RESULTS_FILE + ".csv")

    # Aggregate the results to a dict
    resume: Dict[str, Dict[str, Dict[str, float]]] = dict()
    # df["filename"] but without the "modality_bodypart_" part
    # TODO add another high level separation for modality
    settings_list: list = [elem.split("_")[-1] for elem in df["filename"]]
    for settings in tuple(set(settings_list)):
        # Dataframe containing only the data associated to the settings at hand
        fname_df = df.copy()
        for i, row in fname_df.iterrows():
            # If row does not point to specific setting, drop it from df
            if not row["filename"].endswith(settings):
                fname_df = fname_df.drop(i)
        # Gather statistics
        resume[settings] = {
            "cs": dict(min=fname_df["cs"].min(), max=fname_df["cs"].max(),
                       avg=np.average(fname_df["cs"]),
                       std=np.std(fname_df["cs"])),
            "ds": dict(min=min(fname_df["ds"]), max=max(fname_df["ds"]),
                       avg=np.average(fname_df["ds"]),
                       std=np.std(fname_df["ds"])),
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
    out_file = open(PROCEDURE_RESULTS_FILE + ".json", "w")
    json.dump(resume, out_file, indent=4)


if __name__ == '__main__':
    # Workaround to a local issue
    os.environ["PATH"] = f"{os.path.expanduser('~')}" \
                         f"/libwebp-0.4.1-linux-x86-64/bin:" + os.environ["PATH"]

    check_codecs()

    bulk_compress(jxl=True, avif=True, webp=True)
    resume_stats()
