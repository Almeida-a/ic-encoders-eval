"""Main pipeline's module

    Evaluates performance for: avif-webp-jxl

"""


import itertools
import os.path
from argparse import ArgumentParser
from functools import partial
from pathlib import PosixPath
from typing import Callable

import cv2
import numpy as np
import pandas as pd

import custom_apng
import metrics
from parameters import PathParameters, LOSSLESS_EXTENSION, SAMPLES_PER_PIXEL, \
    BITS_PER_SAMPLE, MINIMUM_WEBP_QUALITY, MINIMUM_AVIF_QUALITY, QUALITY_TOTAL_STEPS, MAXIMUM_JXL_DISTANCE
from squeeze import squeeze_data
from util import construct_djxl, construct_davif, construct_dwebp, construct_cwebp, construct_cavif, construct_cjxl, \
    timed_command, total_pixels, rename_duplicate, rm_encoded, dataset_img_info, mkdir_if_not_exists

"""
    Codecs' versions
        cjxl, djxl -> v0.7.0 -- ae95f45
        cwebp, dwebp -> v1.2.1
        cavif -> v1.3.4
        avif_decode -> 0.2.2
"""


def check_codecs():
    """ Checks if all codecs are present in current machine

    Exits program if one does not exist
    """
    print("Looking for the codecs...")
    if os.system("which cavif avif_decode") != 0:
        print("AVIF codec not found!")
        exit(1)
    elif os.system("which djxl cjxl") != 0:
        print("JPEG XL codec not found!")
        exit(1)
    elif os.system("which cwebp dwebp") != 0:
        print("WebP codec not found!")
        exit(1)
    print("All codecs are available!")


def encode_jxl(target_image: str, output_path: str, distance: float = 1.0,
               effort: int = 7, lossless: bool = False) -> float:
    """Encodes an image using the cjxl program

    @param target_image: Path to image targeted for compression encoding
    @param distance: Quality setting as set by cjxl (butteraugli --distance)
    @param effort: --effort level parameter as set by cjxl
    @param output_path: Path where the dataset_compressed image should go to
    @param lossless: Toggle lossless mode
    @return: Compression speed, in MP/s
    """
    pixels = total_pixels(target_image)

    # Construct the encoding command
    command = construct_cjxl(target_image, output_path, distance, effort, lossless=lossless)

    # Run and extract ct
    ct = timed_command(command)

    return pixels / (ct * 1e6)


def encode_avif(target_image: str, output_path: str, quality: int = 80,
                speed: int = 4, lossless: bool = False) -> float:
    """Encodes an image using the cavif(-rs) program

    @param target_image: Input/Original image
    @param output_path: Encoded file output path
    @param quality: --quality configuration
    @param speed: --speed configuration
    @param lossless: Toggle lossless encoding mode
    @return: Compression speed, in MP/s
    """

    if target_image.endswith(".apng"):
        encode_part = partial(encode_avif, quality=quality, speed=speed)
        return custom_multiframe_encoding(encode_part, "avif", output_path, target_image)

    pixels = total_pixels(target_image)

    # Construct the command
    command = construct_cavif(target_image, output_path, quality, speed, lossless=lossless)

    ct = timed_command(command)

    return pixels / (ct * 1e6)


def encode_webp(target_image: str, output_path: str, quality: int = 75,
                effort: int = 4, lossless: bool = False) -> float:
    """Encodes an image using the cwebp program

    @param target_image: path/to/image.ext, where the extension needs to be supported by cwebp
    @param quality: Quality loss level (1 to 100), -q option of the tool
    @param effort: Effort of the encoding process (-m option of the tool)
    @param output_path: Directory where the compression
    @param lossless: Toggle lossless encoding mode
    @return: Compression speed, in MP/s
    """

    if target_image.endswith(".apng"):
        encode_part = partial(encode_webp, quality=quality, effort=effort)

        return custom_multiframe_encoding(encode_part, "webp", output_path, target_image)

    # Command to be executed for the compression
    command = construct_cwebp(target_image, output_path, quality, effort, lossless=lossless)

    ct = timed_command(command)

    # Get number of pixels in the image
    pixels = total_pixels(target_image)

    return pixels / (ct * 1e6)


def custom_multiframe_encoding(encode_part: partial, format_: str, output_path: str, input_image: str) -> float:
    """Encodes a multi-frame apng image file into multiple frames

    This is for formats which don't support multiple frame images w/ >8 bits per sample

    @param encode_part:
    @param format_: Which encoder to be used to
    @param output_path:
    @param input_image:
    @return:
    """
    print("Custom encoding multi-frame image.")

    multiframe_img = custom_apng.read_apng(input_image)
    cs_list: list[float] = []
    for i, frame in enumerate(multiframe_img):
        # Write the multiple frames as .png in DATASET_COMPRESSED_PATH
        frame_name = output_path.replace(f".{format_}", f"-{i}.png")
        cv2.imwrite(frame_name, frame)
        # Call this function for every written png frame (and save cs for each one)
        cs_list.append(
            encode_part(
                target_image=frame_name,
                output_path=output_path.replace(f".{format_}", f"-{i}.{format_}")
            )
        )
        print(f"Encoded frame {i}")
        # Delete lossless frame
        os.remove(frame_name)
    # Compute the appropriate cs and return it
    return np.array(cs_list).mean()


def decode_compare(encoded_path: str, og_image_path) -> tuple[float, float, float, float, float]:
    # sourcery skip: collection-into-set
    """Decodes the image and returns the process' metadata

    @param encoded_path: Path to the encoded image (to be decoded)
    @param og_image_path: Path to the original, lossless image
    @return: CR, DS(MP/s), MSE, PSNR and SSIM regarding the compression applied to the image
    """

    encoded_extension: str = encoded_path.split(".")[-1]
    decoded_extension = og_image_path.split(".")[-1]

    pixels = total_pixels(og_image_path)

    if encoded_extension == "jxl" or og_image_path.endswith(".png"):
        cr: float = os.path.getsize(og_image_path) / os.path.getsize(encoded_path)
    elif encoded_extension in ("webp", "avif") and og_image_path.endswith(".apng"):
        frames_list = filter(
            lambda file: file.endswith(encoded_extension),
            os.listdir(PathParameters.DATASET_COMPRESSED_PATH)
        )
        cr = os.path.getsize(og_image_path) / np.sum(
            [os.path.getsize(os.path.abspath(PathParameters.DATASET_COMPRESSED_PATH + frame)) for frame in frames_list]
        )
    else:
        raise AssertionError("Bad state (bug).")

    match encoded_extension:
        case "jxl":

            decoded_path: str = encoded_path.replace("jxl", decoded_extension)

            # Construct decoding command
            command = construct_djxl(decoded_path, encoded_path)

            dt = timed_command(command)

            # Compute decoding speed (MP/s)
            ds = pixels / (dt * 1e6)
        case "avif":

            # Same directory, same name, .png
            decoded_path: str = encoded_path.replace("avif", decoded_extension)

            if og_image_path.endswith(".apng"):
                # Execute decode command for all frames and collect DTs
                dt = custom_multiframe_decoding(decoded_path, encoded_extension)

            elif og_image_path.endswith(".png"):
                # Construct decoding command
                command = construct_davif(decoded_path, encoded_path)

                dt = timed_command(command)

                if dataset_img_info(encoded_path, SAMPLES_PER_PIXEL) == "1":
                    # AVIF output is always RGB/YCbCr
                    transcode_gray(decoded_path)
            else:
                raise AssertionError(f"Illegal input image format: '{og_image_path}'")

            # Compute decoding speed (MP/s)
            ds = pixels / (dt * 1e6)

        case "webp":

            decoded_path: str = encoded_path.replace("webp", decoded_extension)

            if og_image_path.endswith(".apng"):
                # Execute decode command for all frames and collect DTs
                dt = custom_multiframe_decoding(decoded_path, encoded_extension)

            elif og_image_path.endswith(".png"):
                # Construct decoding command
                command = construct_dwebp(decoded_path, encoded_path)

                dt = timed_command(command)

                if dataset_img_info(encoded_path, SAMPLES_PER_PIXEL) == "1":
                    # WebP output is always RGB/YCbCr
                    transcode_gray(decoded_path)
            else:
                raise AssertionError(f"Illegal input image format: '{og_image_path}'")

            # Compute decoding speed (MP/s)
            ds = pixels / (dt * 1e6)

        case _:
            raise AssertionError(f"Unsupported extension for decoding: '{encoded_extension}'")

    # Read the output images w/ opencv
    if decoded_path.endswith("apng"):
        decoded_image = custom_apng.read_apng(decoded_path)
        og_image = custom_apng.read_apng(og_image_path)
    else:
        decoded_image = cv2.imread(decoded_path, cv2.IMREAD_UNCHANGED)
        og_image = cv2.imread(og_image_path, cv2.IMREAD_UNCHANGED)

    decoded_image = decoded_image.astype(np.uint16)
    og_image = og_image.astype(np.uint16)

    # Evaluate the quality of the resulting image
    mse: float = metrics.custom_mse(og_image, decoded_image)
    if mse != .0:
        psnr: float = metrics.custom_psnr(og_image, decoded_image,
                                          bits_per_sample=int(dataset_img_info(og_image_path, BITS_PER_SAMPLE)))
    else:
        psnr = float("inf")
    ssim: float = metrics.custom_ssim(og_image, decoded_image,
                                      is_colorized=int(dataset_img_info(og_image_path, SAMPLES_PER_PIXEL)) > 1)

    return cr, ds, mse, psnr, ssim


def custom_multiframe_decoding(decoded_path: str, encoded_extension: str):
    """Decodes all frames present in the dataset_compressed folder and outputs apng

    Those frames are encoded by either avif or webp

    @param decoded_path: APNG output file name
    @param encoded_extension: reference to which codec is being used
    @return: Sum of all frames' decoding time
    """
    print("Custom decoding multi-frame image.")

    match encoded_extension:
        case "avif":
            custom_command: Callable[[str, str], str] = construct_davif
        case "webp":
            custom_command = construct_dwebp
        case _:
            raise AssertionError(f"Illegal format used: '{encoded_extension}'")

    # Execute decode command for all frames and collect DTs (decoding times)
    dt = 0
    frame_names: list[str] = []
    for i, frame in enumerate(os.listdir(PathParameters.DATASET_COMPRESSED_PATH)):
        print(f"Decoded frame {i}")
        frame_names.append(
            PathParameters.DATASET_COMPRESSED_PATH + frame.replace(f".{encoded_extension}", ".png")
        )
        dt += timed_command(
            custom_command(decoded_path=frame_names[-1], input_path=PathParameters.DATASET_COMPRESSED_PATH + frame)
        )

        # Either avif or webp store images in multichannel
        if dataset_img_info(PathParameters.DATASET_COMPRESSED_PATH + frame, SAMPLES_PER_PIXEL) == "1":
            transcode_gray(frame_names[-1])

    # Staple output png frames
    assert custom_apng.staple_pngs(decoded_path, *frame_names) is True, "Error joining PNGs"

    return dt


def transcode_gray(img_path):
    """Transcodes an image to gray using opencv

    @param img_path: Path to the image to be transcoded
    """
    # Cavif's .avif files only store in RGB/YCbCr format
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    assert img_gray is not None, f"Error reading decode output: {img_path}"
    assert cv2.imwrite(img_path, img_gray) is True, "Error writing" \
                                                    "gray version for rgb (gray original) image."


# TODO break into 3 functions, one for each format
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


    @param webp: If true, toggle WEBP eval (default=True)
    @param avif: If true, toggle AVIF eval (default=True)
    @param jxl: If true, toggle JPEG XL eval (default=True)
    """

    if all(codec is False for codec in (jxl, avif, webp)):
        return

    # Save all images path relative to dataset_path
    image_list = os.listdir(PathParameters.DATASET_PATH)

    quality_param_jxl = (0.0,)
    quality_param_avif = (80,)
    quality_param_webp = (75,)

    if not args.lossless:
        # Set quality parameters to be used in compression
        # How many configurations are expected (evenly spaced in the range)
        quality_param_jxl = np.linspace(.0, MAXIMUM_JXL_DISTANCE, QUALITY_TOTAL_STEPS)
        quality_param_avif = np.linspace(MINIMUM_AVIF_QUALITY, 100, QUALITY_TOTAL_STEPS).astype(np.ubyte)
        quality_param_webp = np.linspace(MINIMUM_WEBP_QUALITY, 100, QUALITY_TOTAL_STEPS).astype(np.ubyte)

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
    if jxl:
        format_ = "jxl"
        for target_image in image_list:

            if not target_image.endswith(".apng") and not target_image.endswith(".png"):
                continue

            for quality, effort in itertools.product(quality_param_jxl, effort_jxl):
                # Set output path of compressed
                outfile_name, output_path = get_output_path(dataset_path=PathParameters.DATASET_PATH,
                                                            target_image=target_image, effort=effort, quality=quality,
                                                            format_=format_, lossless=args.lossless)

                # Print image analysis
                print(f"Started analysing image '{outfile_name}'", end="...")

                # Add wildcard for now because the extensions are missing
                cs = encode_jxl(target_image=PathParameters.DATASET_PATH + target_image, output_path=output_path,
                                distance=quality, effort=effort, lossless=args.lossless)

                # Decode and collect stats to stats df
                stats = finalize(cs, outfile_name, output_path, stats, PathParameters.DATASET_PATH + target_image)

                # Print when finished
                print("Done!")

    # AVIF
    if avif and not args.lossless:
        format_ = "avif"
        for target_image in image_list:

            if not any(target_image.endswith(accepted) for accepted in (".png", ".apng")):
                continue

            for quality, speed in itertools.product(quality_param_avif, speed_avif):
                # Construct output file total path
                outfile_name, output_path = get_output_path(dataset_path=PathParameters.DATASET_PATH,
                                                            target_image=target_image, effort=speed, quality=quality,
                                                            format_=format_, lossless=args.lossless)

                # Print the progress being made
                print(f"Started analysing image '{outfile_name}'", end="...")

                cs = encode_avif(target_image=PathParameters.DATASET_PATH + target_image, output_path=output_path,
                                 quality=quality, speed=speed, lossless=args.lossless)

                # Decode and collect stats to stats df
                stats = finalize(cs, outfile_name, output_path, stats, PathParameters.DATASET_PATH + target_image)

                # Print when finished
                print("Done!")

    # WebP
    if webp:
        format_ = "webp"
        for target_image in image_list:

            if not any(target_image.endswith(accepted) for accepted in (".png", ".apng")):
                continue

            for quality, effort in itertools.product(quality_param_webp, effort_webp):
                # Construct output file total path
                outfile_name, output_path = get_output_path(dataset_path=PathParameters.DATASET_PATH,
                                                            target_image=target_image, effort=effort, quality=quality,
                                                            format_=format_, lossless=args.lossless)

                # Print the progress being made
                print(f"Started analysing image \"{outfile_name}\"... ", end="")

                # Add wildcard for now because the extensions are missing
                cs = encode_webp(target_image=PathParameters.DATASET_PATH + target_image, output_path=output_path,
                                 quality=quality, effort=effort, lossless=args.lossless)

                # Decode and collect stats to stats df
                stats = finalize(cs, outfile_name, output_path, stats, PathParameters.DATASET_PATH + target_image)

                # Print when finished
                print("Done!")

    # Create directory if it doesn't exist
    mkdir_if_not_exists(PathParameters.PROCEDURE_RESULTS_PATH, regard_parent=True)

    # Save csv files
    # If procedure results file already exists, new file renamed to filename+_1 or _n
    stats.to_csv(
        rename_duplicate(f"{PathParameters.PROCEDURE_RESULTS_PATH}.csv"), index=False
    )


def finalize(cs: float, outfile_name: str, encoded_path: str, stats: pd.DataFrame, og_img_path: str) -> pd.DataFrame:
    """Decode, collect eval data and remove compressed image.

    Decodes the target image file, deletes it and saves metadata to the provided dataframe

    @param og_img_path: Path to the original image
    @param cs: Compression time of the provided compressed image file
    @param outfile_name: Basename of the provided file (compressed image)
    @param encoded_path: Path to the provided file (compressed image)
    @param stats: Dataframe holding the data regarding the compressions
    @return: Updated Dataframe
    """
    cr, ds, mse, psnr, ssim = decode_compare(encoded_path, og_img_path)
    # Remove generated (png and jxl/avif/webp) images
    rm_encoded()
    # Append new stats do dataframe
    row = pd.DataFrame(
        dict(filename=[outfile_name], cs=[cs], ds=[ds], cr=[cr], mse=[mse], psnr=[psnr], ssim=[ssim])
    )
    stats = pd.concat([stats, row])
    return stats


def get_output_path(dataset_path: str, target_image: str, effort: int,
                    quality: float, format_: str, lossless: bool) -> tuple[str, str]:
    """ Compute the output path of the compressed version of the target image.

    @param lossless:
    @param dataset_path: Dataset containing the target image.
    @param target_image: Input, original and lossless image.
    @param effort: Effort/speed configuration with which the image compression process will be set to.
    @param quality: Quality configuration with which the image compression process will be set to.
    @param format_: Image compression algorithm's extension, e.g.: jxl
    @return: Output file basename and path
    """

    quality_mode: str = "LOSSLESS" if lossless else f"q{quality}"

    # Construct output file total path
    outfile_name: str = f"{target_image.split(LOSSLESS_EXTENSION)[0]}_" + f"{quality_mode}-e{effort}.{format_}"

    # Trim trailing slash "/"
    trimmed: list = dataset_path.split("/")
    trimmed.remove("")
    trimmed: str = "/".join(trimmed)
    output_path: str = f"{trimmed}_compressed/{outfile_name}"
    return outfile_name, output_path


def main():

    if args.outdir:
        PathParameters.PROCEDURE_RESULTS_PATH = f"{args.outdir}/{PathParameters.PROCEDURE_RESULTS_PATH}"

    check_codecs()

    # Create paths
    if not os.path.exists(PathParameters.DATASET_PATH):
        os.makedirs(PathParameters.DATASET_PATH)
    if not os.path.exists(PathParameters.DATASET_COMPRESSED_PATH):
        os.makedirs(PathParameters.DATASET_COMPRESSED_PATH)
    rm_encoded()

    bulk_compress(jxl=True, avif=True, webp=True)
    squeeze_data(PathParameters.PROCEDURE_RESULTS_PATH)


if __name__ == '__main__':

    parser = ArgumentParser("(De)compress the provided dataset and capture metrics into data files.")

    parser.add_argument("--output", dest='outdir', action='store', nargs='?', type=PosixPath, default=".",
                        help='Specify output directory of metric result files.')
    parser.add_argument("--lossless", action="store_true", default="false")

    args = parser.parse_args()

    main()
