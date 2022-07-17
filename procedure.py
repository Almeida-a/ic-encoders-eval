"""Main pipeline's module

    Evaluates performance for: avif-webp-jxl

"""

import json
import os.path
from functools import partial
from typing import Callable

import cv2
import numpy as np
import pandas as pd

import custom_apng
import metrics
from parameters import LOSSLESS_EXTENSION, PROCEDURE_RESULTS_FILE, DATASET_PATH, SAMPLES_PER_PIXEL, \
    DATASET_COMPRESSED_PATH, BITS_PER_SAMPLE, MODALITY, DEPTH, MINIMUM_WEBP_QUALITY, MINIMUM_AVIF_QUALITY, \
    QUALITY_TOTAL_STEPS, MAXIMUM_JXL_DISTANCE
from util import construct_djxl, construct_davif, construct_dwebp, construct_cwebp, construct_cavif, construct_cjxl, \
    timed_command, total_pixels, original_basename, rm_encoded

"""
    Codecs' versions
        cjxl, djxl -> v0.6.1
        cwebp, dwebp -> v0.4.1
        cavif -> v1.3.4
        avif_decode -> 0.2.2
"""


def check_codecs():
    """ Checks if all codecs are present in current machine

    Exits program if one does not exist
    """
    print("Looking for the codecs...")
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
    """Encodes an image using the cjxl program

    @param target_image: Path to image targeted for compression encoding
    @param distance: Quality setting as set by cjxl (butteraugli --distance)
    @param effort: --effort level parameter as set by cjxl
    @param output_path: Path where the dataset_compressed image should go to
    @return: Compression speed, in MP/s
    """
    pixels = total_pixels(target_image)

    # Construct the encoding command
    command = construct_cjxl(distance, effort, output_path, target_image)

    # Run and extract ct
    ct = timed_command(command)

    return pixels / (ct * 1e6)


def encode_avif(target_image: str, quality: int, speed: int, output_path: str) -> float:
    """Encodes an image using the cavif(-rs) program

    @param target_image: Input/Original image
    @param quality: --quality configuration
    @param speed: --speed configuration
    @param output_path: Encoded file output path
    @return: Compression speed, in MP/s
    """

    if target_image.endswith(".apng"):
        encode_part = partial(encode_avif, quality=quality, speed=speed)
        format_ = "avif"

        return custom_multiframe_encoding(encode_part, format_, output_path, target_image)

    pixels = total_pixels(target_image)

    # Construct the command
    command = construct_cavif(output_path, quality, speed, target_image)

    ct = timed_command(command)

    # Parse to compression speed MP/s
    cs = pixels / (ct * 1e6)

    return cs


def encode_webp(target_image: str, quality: int, effort: int, output_path: str) -> float:
    """Encodes an image using the cwebp program

    @param target_image: path/to/image.ext, where the extension needs to be supported by cwebp
    @param quality: Quality loss level (1 to 100), -q option of the tool
    @param effort: Effort of the encoding process (-m option of the tool)
    @param output_path: Directory where the compression
    @return: Compression speed, in MP/s
    """

    if target_image.endswith(".apng"):
        format_ = "webp"
        encode_part = partial(encode_webp, quality=quality, effort=effort)

        return custom_multiframe_encoding(encode_part, format_, output_path, target_image)

    # Command to be executed for the compression
    command = construct_cwebp(effort, output_path, quality, target_image)

    ct = timed_command(command)

    # Get number of pixels in the image
    pixels = total_pixels(target_image)

    return pixels / (ct * 1e6)


def custom_multiframe_encoding(encode_part, format_, output_path, input_image) -> float:
    """Encodes a multi-frame apng image file into multiple frames

    This is for formats which don't support multiple frame images w/ >8 bits per sample

    @param encode_part:
    @param format_:
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
        # Recursively call this function for every written png frame (and save cs for each one)
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
    """ Decodes the image and returns the process' metadata

    @param og_image_path:
    @param encoded_path: Path to the image to be decoded
    @return: CR, DS(MP/s), MSE, PSNR, SSIM regarding the compression applied to the image
    """

    encoded_extension: str = encoded_path.split(".")[-1]
    decoded_extension = og_image_path.split(".")[-1]

    pixels = total_pixels(og_image_path)

    if encoded_extension == "jxl" or og_image_path.endswith(".png"):
        cr: float = os.path.getsize(og_image_path) / os.path.getsize(encoded_path)
    elif encoded_extension in ("webp", "avif") and og_image_path.endswith(".apng"):
        frames_list = filter(
            lambda file: file.endswith(encoded_extension),
            os.listdir(DATASET_COMPRESSED_PATH)
        )
        cr = os.path.getsize(og_image_path) / np.sum(
            [os.path.getsize(os.path.abspath(DATASET_COMPRESSED_PATH + frame)) for frame in frames_list]
        )
    else:
        raise AssertionError("Bad state (use debugger)")

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
            raise AssertionError(f"Unsupported extension for decoding: {encoded_extension}")

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
                                      color=int(dataset_img_info(og_image_path, SAMPLES_PER_PIXEL)) > 1)

    return cr, ds, mse, psnr, ssim


def custom_multiframe_decoding(decoded_path, encoded_extension):
    """Decodes all frames present in the dataset_compressed folder and outputs apng

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

    # Execute decode command for all frames and collect DTs
    dt = 0
    frame_names: list[str] = []
    for i, frame in enumerate(os.listdir(DATASET_COMPRESSED_PATH)):
        print(f"Decoded frame {i}")
        frame_names.append(
            DATASET_COMPRESSED_PATH + frame.replace(f".{encoded_extension}", ".png")
        )
        dt += timed_command(
            custom_command(decoded_path=frame_names[-1], input_path=DATASET_COMPRESSED_PATH + frame)
        )

        # Either avif or webp store images in multichannel
        if dataset_img_info(DATASET_COMPRESSED_PATH + frame, SAMPLES_PER_PIXEL) == "1":
            transcode_gray(frame_names[-1])

    # Staple output png frames
    assert custom_apng.staple_pngs(decoded_path, *frame_names) is True, "Error joining PNGs"

    return dt


def dataset_img_info(target_image: str, keyword: int) -> str:
    """Extract the information present in the names of the pre-processed dataset

    @param target_image: Path to the image which name is to be evaluated
    @param keyword: parameter defined attribute (id) to extract
    @return: Attribute value for the image
    """
    retval = os.path.basename(target_image).split("_")[keyword]
    if keyword == DEPTH:
        return retval.replace(".apng", "")
    return retval


def transcode_gray(img_path):
    """Transcodes an image to gray using opencv

    @param img_path: Path to the image to be transcoded
    """
    # Cavif's .avif files only store in RGB/YCbCr format
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    assert img_gray is not None, f"Error reading decode output: {img_path}"
    assert cv2.imwrite(img_path, img_gray) is True, "Error writing" \
                                                    "gray version for rgb (gray original) image."


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
    image_list = os.listdir(DATASET_PATH)

    # Set quality parameters to be used in compression
    # How many configurations are expected (evenly spaced in the range)
    quality_param_jxl: np.ndarray = np.linspace(.0, MAXIMUM_JXL_DISTANCE, QUALITY_TOTAL_STEPS)
    quality_param_avif = range(MINIMUM_AVIF_QUALITY, 101, int(100 / QUALITY_TOTAL_STEPS))
    quality_param_webp = range(MINIMUM_WEBP_QUALITY, 101, int(100 / QUALITY_TOTAL_STEPS))

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

            if not target_image.endswith(".apng") and not target_image.endswith(".png"):
                continue

            for quality in quality_param_jxl:
                for effort in effort_jxl:
                    # Set output path of compressed
                    outfile_name, output_path = get_output_path(
                        dataset_path=DATASET_PATH, effort=effort,
                        quality=quality, target_image=target_image, format_="jxl"
                    )

                    # Print image analysis
                    print(f"Started analysing image \"{outfile_name}\"", end="...")

                    # Add wildcard for now because the extensions are missing
                    cs = encode_jxl(target_image=DATASET_PATH + target_image,
                                    distance=quality, effort=effort,
                                    output_path=output_path)

                    # Decode and collect stats to stats df
                    stats = finalize(cs, outfile_name, output_path, stats, DATASET_PATH + target_image)

                    # Print when finished
                    print("Done!")

    # AVIF
    if avif is True:
        for target_image in image_list:

            if not any([target_image.endswith(accepted) for accepted in (".png", ".apng")]):
                continue

            for quality in quality_param_avif:
                for speed in speed_avif:
                    # Construct output file total path
                    outfile_name, output_path = get_output_path(
                        dataset_path=DATASET_PATH, effort=speed,
                        quality=quality, target_image=target_image, format_="avif"
                    )

                    # Print the progress being made
                    print(f"Started analysing image \"{outfile_name}\"", end="...")

                    cs = encode_avif(target_image=DATASET_PATH + target_image,
                                     quality=quality, speed=speed, output_path=output_path)

                    # Decode and collect stats to stats df
                    stats = finalize(cs, outfile_name, output_path, stats, DATASET_PATH + target_image)

                    # Print when finished
                    print("Done!")

    # WebP
    if webp is True:
        for target_image in image_list:

            if not any([target_image.endswith(accepted) for accepted in (".png", ".apng")]):
                continue

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
                    stats = finalize(cs, outfile_name, output_path, stats, DATASET_PATH + target_image)

                    # Print when finished
                    print("Done!")

    # Save csv files
    # If procedure results file already exists, new file renamed to filename+_1 or _n
    stats.to_csv(
        original_basename(f"{PROCEDURE_RESULTS_FILE}.csv"), index=False
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


def get_output_path(dataset_path: str, target_image: str, effort: int, quality: float, format_: str) -> tuple[str, str]:
    """ Compute the output path of the compressed version of the target image.

    @param dataset_path: Dataset containing the target image.
    @param target_image: Input, original and lossless image.
    @param effort: Effort/speed configuration with which the image compression process will be set to.
    @param quality: Quality configuration with which the image compression process will be set to.
    @param format_: Image compression algorithm's extension, e.g.: jxl
    @return: Output file basename and path
    """

    # Construct output file total path
    outfile_name: str = target_image.split(LOSSLESS_EXTENSION)[0] + "_" + f"q{quality}-e{effort}.{format_}"
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
    ordered_proc_res = list(filter(
        lambda file: file.startswith(PROCEDURE_RESULTS_FILE) and file.endswith(".csv"),
        os.listdir()
    ))
    ordered_proc_res.sort()
    latest_procedure_results = ordered_proc_res[-1]

    df = pd.read_csv(latest_procedure_results)

    # Aggregate the results to a dict
    resume = dict()

    for _, filename in enumerate(df["filename"]):
        settings = filename.split("_")[-1]
        modality = dataset_img_info(filename, MODALITY)
        depth = dataset_img_info(filename, DEPTH)
        samples_per_pixel = dataset_img_info(filename, SAMPLES_PER_PIXEL)
        bits_per_sample = dataset_img_info(filename, BITS_PER_SAMPLE)

        # Dataframe containing only the data associated to the settings/characteristics at hand
        fname_df = df.copy()
        for i, row in fname_df.iterrows():
            # If row does not fit, drop it from df
            other_filename = row["filename"]
            fits: bool = (
                    other_filename.endswith(settings)
                    and dataset_img_info(other_filename, MODALITY) == modality
                    and dataset_img_info(other_filename, DEPTH) == depth
                    and dataset_img_info(other_filename, SAMPLES_PER_PIXEL) == samples_per_pixel
                    and dataset_img_info(other_filename, BITS_PER_SAMPLE) == bits_per_sample
            )
            if not fits:
                fname_df = fname_df.drop(i)

        # Create settings and modality entry if they don't exist
        if resume.get(settings) is None:
            resume[settings] = dict()
        if resume[settings].get(modality) is None:
            resume[settings][modality] = dict()
        if resume[settings][modality].get(depth) is None:
            resume[settings][modality][depth] = dict()
        if resume[settings][modality][depth].get(samples_per_pixel) is None:
            resume[settings][modality][depth][samples_per_pixel] = dict()
        if resume[settings][modality][depth][samples_per_pixel].get(bits_per_sample) is None:
            resume[settings][modality][depth][samples_per_pixel][bits_per_sample] = dict()
        else:
            # Entry has already been filled, skip to avoid re-doing the operations
            continue

        entry = resume[settings][modality][depth][samples_per_pixel][bits_per_sample]

        # Gather statistics
        for metric in df.keys():
            # Brownfield solution to excluding the filename key
            if metric in ["filename", "size"]:
                continue

            mean = np.mean(fname_df[metric])
            std = np.std(fname_df[metric]) if mean != float("inf") else 0.

            entry[metric] = dict(
                min=fname_df[metric].min(), max=fname_df[metric].max(),
                avg=mean, std=std
            )
        entry["size"] = fname_df.shape[0]

    # Save dict to a json
    out_file = open(original_basename(f"{PROCEDURE_RESULTS_FILE}.json"), "w")
    json.dump(resume, out_file, indent=4)


if __name__ == '__main__':
    check_codecs()

    rm_encoded()

    bulk_compress(jxl=True, avif=True, webp=True)
    squeeze_data()
