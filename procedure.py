import json
import os.path
import re
import time
from subprocess import Popen, PIPE
from typing import List, Dict, Tuple, Union

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


def check_codecs():
    """ Checks if all codecs are present in current machine

    Exits program if one does not exist
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
    """ Encodes an image using the cjxl program

    :param target_image: Path to image targeted for compression encoding
    :param distance: Quality setting as set by cjxl (butteraugli --distance)
    :param effort: --effort level parameter as set by cjxl
    :param output_path: Path where the dataset_compressed image should go to
    :return: Compression speed, in MP/s
    """
    pixels = total_pixels(target_image)

    # Construct the encoding command
    command: str = f"cjxl {target_image} {output_path} " \
                   f"--distance={distance} --effort={effort} --quiet"

    # Run and extract ct
    ct = timed_command(command)

    return pixels / (ct * 1e6)


def encode_avif(target_image: str, quality: int, speed: int, output_path: str) -> float:
    """ Encodes an image using the cavif(-rs) program

    :param target_image: Input/Original image
    :param quality: --quality configuration
    :param speed: --speed configuration
    :param output_path: Directory where the dataset_compressed file
    :return: Compression speed, in MP/s
    """

    pixels = total_pixels(target_image)

    # Construct the command
    command: str = f"cavif -o {output_path} " \
                   f"--quality {quality} --speed {speed} --quiet " \
                   f"{os.path.abspath(target_image)}"

    ct = timed_command(command)

    # Parse to compression speed MP/s
    cs = pixels / (ct * 1e6)

    return cs


def encode_webp(target_image: str, quality: int, effort: int, output_path: str) -> float:
    """ Encodes an image using the cwebp program

    :param target_image: path/to/image.ext, where the extension needs to be supported by cwebp
    :param quality: Quality loss level (1 to 100), -q option of the tool
    :param effort: Effort of the encoding process (-m option of the tool)
    :param output_path: Directory where the compression
    :return: Compression speed, in MP/s
    """

    # Command to be executed for the compression
    command: str = f"cwebp -quiet -v -q {quality} -m {effort} {target_image} -o {output_path}"

    ct = timed_command(command)

    # Get number of pixels in the image
    pixels = total_pixels(target_image)

    return pixels / (ct * 1e6)


def timed_command(stdin: str) -> float:
    """ Runs a given command on a subshell and records its execution time

    Note: Execution timeout implemented to 60 seconds

    :param stdin: Used to run the subshell command
    :return: Time it took for the command to run (in seconds)
    """
    # Execute command and time the CT
    start = time.time()
    p = Popen(stdin, shell=True, stdout=PIPE, stderr=PIPE)
    _, stderr = p.communicate(timeout=60)
    ct = time.time() - start  # or extract_webp_ct(stderr)
    # Check for errors
    return_code: int = p.returncode
    if return_code != 0:
        print(f"Error code {return_code}, executing:"
              f"\nStdIn -> {stdin}"
              f"\nStdErr -> {stderr}")
        exit(1)

    return ct


def total_pixels(target_image: str) -> int:
    """ Counts the number of pixels on an image

    Count method: height * height

    :param target_image: Input image path
    :return: Number of pixels
    """

    # Parameter checking
    assert os.path.exists(target_image), f"Image at \"{target_image}\" does not exist!"

    # Extract resolution
    height, width = cv2.imread(target_image).shape[:2]
    # Number of pixels in the image
    pixels = height * width
    return pixels


def extract_jxl_ds(stderr: bytes) -> float:
    """ Extract decompression speed from the djxl output

    :param stderr: Console output
    :return: Decompression speed, in MP/s
    """

    ds: str = re.findall(r"\d+.\d+ MP/s", str(stderr, "utf-8"))[0]

    return float(ds[:-5])


def extract_webp_dt(stderr: bytes) -> float:
    """ Extract decompression speed from the dwebp output

    :param stderr: Console output
    :return: Decompression speed, in MP/s
    """
    # Extract
    ds: str = re.findall(r"Time to decode picture: \d+.\d+s", str(stderr, "utf-8"))[0]\
        .split("Time to decode picture: ")[-1]
    return float(ds[:-1])


def extract_resolution(stderr: bytes, sep: str = "x") -> List[str]:
    """ Extract resolution from dwebp or djxl

    :param stderr: Console output
    :param sep: Separator string - e.g.: 1920x1080, sep="x"
    :return:
    """

    res: str = re.findall(rf"\d+{sep}\d+", str(stderr))[0]
    return res.split(sep)


def decode_compare(target_image: str) -> Tuple[float, float, float, float, float]:
    """ Decodes the image and returns the process' metadata

    :param target_image: Path to the image to be decoded
    :return: CR, DT(MP/s), MSE, PSNR, SSIM regarding the compression applied to the image
    """

    # Get original image
    og_image_path = get_og_image(compressed=target_image, only_path=True)

    pixels = total_pixels(og_image_path)

    cr: float = os.path.getsize(og_image_path) / os.path.getsize(target_image)

    extension: str = target_image.split(".")[-1]

    # Same directory, same name, .png
    out_path: str = ".".join(target_image.split(".")[:-1]) + LOSSLESS_EXTENSION

    if extension == "jxl":
        # Construct decoding command
        command: str = f"djxl {target_image} {out_path}"

        dt = timed_command(command)

        # Compute decoding speed (MP/s)
        ds = pixels / (dt * 1e6)
    elif extension == "avif":
        # Construct decoding command
        command: str = f"avif_decode {target_image} {out_path}"

        dt = timed_command(command)

        # Compute decoding speed (MP/s)
        ds = pixels / (dt * 1e6)
    elif extension == "webp":
        # Construct decoding command
        command: str = f"dwebp -v {target_image} -o {out_path}"

        dt = timed_command(command)

        # Compute decoding speed (MP/s)
        ds = pixels / (dt * 1e6)
    else:
        raise AssertionError(f"Unsupported extension for decoding: {extension}")

    # Read the output image w/ opencv
    out_image = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
    og_image = cv2.imread(og_image_path, cv2.IMREAD_UNCHANGED)

    # Evaluate the quality of the resulting image
    mse: float = metrics.mse(og_image, out_image)
    psnr: float = metrics.psnr(og_image, out_image)
    ssim: float = metrics.ssim(og_image, out_image.astype(np.uint16))

    if mse == 0:
        raise AssertionError("Images are equal")

    return cr, ds, mse, psnr, ssim


def extract_jxl_cs(stderr: bytes) -> str:
    """ Extracts compression speed from a cjxl output

    :param stderr: Console output
    :return: Compression speed, in MP/s
    """

    # Find decoding speed in the output pool
    ds = re.findall(r"\d+.\d+ MP/s", str(stderr))[1][:-5]
    return ds


def extract_webp_ct(stderr: bytes) -> float:
    """ Extracts compression speed from a cjxl output

    :param stderr: Console output
    :return: Compression speed, in MP/s
    """

    # Extract DT from output
    dt: str = re.findall(r"Time to encode picture: \d+.\d+s", str(stderr))[0] \
        .split("Time to encode picture: ")[-1]
    # Cast to float
    dt: float = float(dt[:-1])
    return dt


def get_og_image(compressed, only_path: bool = False) -> Union[np.ndarray, str]:
    """ Extract info on the original image given the compressed one

    :param compressed: Path to compressed image
    :param only_path: Optional flag - Set to true to return only the path to the og image
    :return: Original image's path or contents
    """
    # Get original lossless image
    og_image_path = "".join(compressed.split("_compressed"))
    og_image_path = "_".join(og_image_path.split("_")[:-1]) + LOSSLESS_EXTENSION

    if only_path is True:
        return og_image_path
    og_image = cv2.imread(og_image_path, cv2.IMREAD_UNCHANGED)
    return og_image


def image_to_dir(dataset_path: str, target_image: str) -> str:
    """ Not used anymore

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
    """ Compress and analyse. Outputs analysis to ".csv".

    Fetches lossless images from the DATASET_PATH directory (defined in parameters.py).
    For each image, analyze:
        * Compression / Decompression speed (MP/s);
        * Compression Ratio;
        * Quality loss evaluating parameters such as MSE, PSNR and MSSIM.
    This is done for each of the chosen compression algorithms, in order to evaluate it.
    The analysis results are then stored in a csv file named ${parameters.PROCEDURE_RESULTS_FILE}.csv.

    The csv table entries will be an identifier of the image/format/settings used
    for that instance - MODALITY_BODY-PART_NUM-TAG_q{float}_e{int}.FORMAT e.g.: CT_HEAD_12_q3.0_e7.jxl


    :param webp: If true, toggle WEBP eval (default=True)
    :param avif: If true, toggle AVIF eval (default=True)
    :param jxl: If true, toggle JPEG XL eval (default=True)
    """

    if all(codec is False for codec in (jxl, avif, webp)):
        return

    # Save all images path relative to dataset_path
    image_list = os.listdir(DATASET_PATH)

    # Set quality parameters to be used in compression
    # How many configurations are expected (evenly spaced in the range)
    spread: int = 2
    quality_param_jxl: np.ndarray = np.linspace(.0, 3.0, spread)
    quality_param_avif = range(1, 101, int(100 / spread))
    quality_param_webp = range(1, 101, int(100 / spread))

    # Set effort/speed parameters for compression (common step)
    effort_jxl = (7,)
    speed_avif = (4,)
    effort_webp = (4,)

    # Record time of compression
    ct: float

    stats = pd.DataFrame(
        data=dict(filename=[], cs=[], ds=[], cr=[], mse=[], psnr=[], ssim=[])
    )

    # JPEG XL evaluation
    if jxl is True:
        for target_image in image_list:
            for quality in quality_param_jxl:
                for effort in effort_jxl:

                    # Set output path of compressed
                    outfile_name, output_path = get_output_path(
                        dataset_path=DATASET_PATH, effort=effort,
                        quality=quality, target_image=target_image, format_="jxl"
                    )

                    # Print image analysis
                    print(f"Started analysing image \"{outfile_name}\"...", end="")

                    # Add wildcard for now because the extensions are missing
                    cs = encode_jxl(target_image=DATASET_PATH + target_image,
                                    distance=quality, effort=effort,
                                    output_path=output_path)

                    # Decode and collect stats to stats df
                    stats = finalize(cs, outfile_name, output_path, stats)

                    # Print when finished
                    print("Done!")

    # AVIF
    if avif is True:
        for target_image in image_list:
            for quality in quality_param_avif:
                for speed in speed_avif:
                    # Construct output file total path
                    outfile_name, output_path = get_output_path(
                        dataset_path=DATASET_PATH, effort=speed,
                        quality=quality, target_image=target_image, format_="avif"
                    )

                    # Print the progress being made
                    print(f"Started analysing image \"{outfile_name}\"...", end="")

                    # Add wildcard for now because the extensions are missing
                    cs = encode_avif(target_image=DATASET_PATH + target_image,
                                     quality=quality, speed=speed, output_path=output_path)

                    # Decode and collect stats to stats df
                    stats = finalize(cs, outfile_name, output_path, stats)

                    # Print when finished
                    print("Done!")

    # WebP
    if webp is True:
        for target_image in image_list:
            for quality in quality_param_webp:
                for effort in effort_webp:
                    # Construct output file total path
                    outfile_name, output_path = get_output_path(
                        dataset_path=DATASET_PATH, effort=effort, quality=quality,
                        target_image=target_image, format_="webp"
                    )

                    # Print the progress being made
                    print(f"Started analysing image \"{outfile_name}\"... ", end="")

                    # Add wildcard for now because the extensions are missing
                    cs = encode_webp(target_image=DATASET_PATH + target_image,
                                     quality=quality, effort=effort, output_path=output_path)

                    # Decode and collect stats to stats df
                    stats = finalize(cs, outfile_name, output_path, stats)

                    # Print when finished
                    print("Done!")

    # Save csv files
    # If procedure results file already exists, new file renamed to filename+_1 or _n
    stats.to_csv(
        original_basename(f"{PROCEDURE_RESULTS_FILE}.csv"), index=False
    )


def original_basename(intended_abs_filepath: str) -> str:
    """ Get an original filename given the absolute path

    Example - give path/to/filename.txt -> it already exists -> return path/to/filename_1.txt

    :param intended_abs_filepath: Absolute path to a file not yet written (w/o the file's extension)
    :return: Same path, with basename of the file changed to an original one
    """
    # Separate path from extension
    extension: str = intended_abs_filepath.split(".")[-1]
    intended_abs_filepath: str = ".".join(intended_abs_filepath.split(".")[:-1])

    suffix: str = ""
    counter: int = 0

    while os.path.exists(f"{intended_abs_filepath + suffix}.{extension}"):
        counter += 1
        suffix = f"_{counter}"

    return f"{intended_abs_filepath + suffix}.{extension}"


def finalize(cs: float, outfile_name: str, output_path: str, stats: pd.DataFrame) -> pd.DataFrame:
    """ Decode, collect eval data and remove compressed image.

    Decodes the target image file, deletes it and saves metadata to the provided dataframe

    :param cs: Compression time of the provided compressed image file
    :param outfile_name: Basename of the provided file (compressed image)
    :param output_path: Path to the provided file (compressed image)
    :param stats: Dataframe holding the data regarding the compressions
    :return: Updated Dataframe
    """
    cr, ds, mse, psnr, ssim = decode_compare(output_path)
    # Remove generated (png and jxl) images
    os.remove(output_path)
    os.remove(".".join(output_path.split(".")[:-1]) + LOSSLESS_EXTENSION)
    # Append new stats do dataframe
    row = pd.DataFrame(
        dict(filename=[outfile_name], cs=[cs], ds=[ds], cr=[cr], mse=[mse], psnr=[psnr], ssim=[ssim])
    )
    stats = pd.concat([stats, row])
    return stats


def get_output_path(dataset_path: str, target_image: str, effort: int, quality: float, format_: str) -> Tuple[str, str]:
    """ Compute the output path of the compressed version of the target image.

    :param dataset_path: Dataset containing the target image.
    :param target_image: Input, original and lossless image.
    :param effort: Effort/speed configuration with which the image compression process will be set to.
    :param quality: Quality configuration with which the image compression process will be set to.
    :param format_: Image compression algorithm's extension, e.g.: jxl
    :return: Output file basename and path
    """

    # Construct output file total path
    outfile_name: str = target_image.split(LOSSLESS_EXTENSION)[0] \
                        + "_" + f"q{quality}-e{effort}.{format_}"
    # Trim trailing slash "/"
    trimmed: list = dataset_path.split("/")
    trimmed.remove("")
    trimmed: str = "/".join(trimmed)
    output_path: str = trimmed + "_compressed/" + outfile_name
    return outfile_name, output_path


def squeeze_data():
    """ Digests raw compression stats into condensed stats.

    Condensed stats:
     * are min/max/avg/std per modality, per body-part and per encoding format.
     * data is saved under {parameters.PROCEDURE_RESULTS_FILE}.json
    """

    # Read csv to df
    df = pd.read_csv(PROCEDURE_RESULTS_FILE + ".csv")

    # Aggregate the results to a dict
    resume: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = dict()
    # df["filename"] but without the "modality_bodypart_" part
    settings_list: list = [elem.split("_")[-1] for elem in df["filename"]]
    # df["filename"] but without the "modality_bodypart_setting" part
    modality_list: list = [elem.split("_")[0] for elem in df["filename"]]

    for settings in tuple(set(settings_list)):
        for modality in tuple(set(modality_list)):
            # Dataframe containing only the data associated to the settings at hand
            fname_df = df.copy()
            for i, row in fname_df.iterrows():
                # If row does not point to specific setting and modality, drop it from df
                if not row["filename"].endswith(settings) \
                        or not row["filename"].startswith(modality):
                    fname_df = fname_df.drop(i)
            # Create settings and modality entry if none exists
            if resume.get(settings) is None:
                resume[settings] = dict()
            if resume[settings].get(modality) is None:
                resume[settings][modality] = dict()

            # Gather statistics
            for metric in df.keys():
                # Brownfield solution to excluding the filename key
                if metric == "filename":
                    continue

                resume[settings][modality][metric] = dict(
                    min=fname_df[metric].min(), max=fname_df[metric].max(),
                    avg=np.average(fname_df[metric]),
                    std=np.std(fname_df[metric])
                )

    # Save dict to a json
    out_file = open(original_basename(f"{PROCEDURE_RESULTS_FILE}.json"), "w")
    json.dump(resume, out_file, indent=4)


if __name__ == '__main__':
    # Workaround to a local issue
    #   For some reason, the subshell doesn't recognize the cwebp tool
    os.environ["PATH"] = f"{os.path.expanduser('~')}" \
                         f"/libwebp-0.4.1-linux-x86-64/bin:" + os.environ["PATH"]

    check_codecs()

    bulk_compress(jxl=True, avif=True, webp=True)
    squeeze_data()
