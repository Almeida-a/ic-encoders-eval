import os.path
import time
from subprocess import Popen, PIPE
from typing import List, Dict, Union

import numpy as np
import pandas as pd


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
                   f"--distance={distance} --effort={effort}"

    # Execute jxl and measure the time it takes
    # TODO consider better methods of acquiring the CT
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


def encode_avif(target_image: str, quality: int, speed: int, output_path: str) -> float:
    """

    :param target_image:
    :param quality:
    :param speed:
    :param output_path: Directory where the dataset_compressed file
    :return: Compression time, in seconds
    """
    # Filename without the extension
    filename: str = ".".join(os.path.basename(target_image).split(".")[:-1])

    command: str = f"cavif -o {output_path} " \
                   f"--quality {quality} --speed={speed} {os.path.abspath(target_image)}"

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
    ext: str = ".png"

    # Save all images path relative to dataset_path
    image_list = [dataset_path+str(i) for i in range(1, 6)]

    # Set quality parameters to be used in compression
    spread: int = 5  # TODO explain
    quality_param_jxl: np.ndarray = np.linspace(.0, 3.0, spread)
    quality_param_avif = range(1, 101, int(100/spread))
    quality_param_webp = range(1, 101, int(100/spread))

    # Set effort/speed parameters for compression (common step)
    step: int = 3
    effort_jxl = range(1, 9, step)
    speed_avif = range(0, 11, step)
    effort_webp = range(0, 7, step)

    # Encode (to target path) and record time of compression
    ct: float  # Record time of compression
    time_record: Dict[str, List[Union[float, str]]] = dict(filename=[], ct=[])

    # JPEG XL
    if jxl is True:
        for target_image in image_list:
            for quality in quality_param_jxl:
                for effort in effort_jxl:
                    # Construct output file total path
                    outfile_name: str = f"q{quality}-{effort}.jxl"
                    output_path = image_to_dir(dataset_path, target_image) + outfile_name

                    # Add wildcard for now because the extensions are missing
                    ct = encode_jxl(target_image=target_image+ext,
                                    distance=quality, effort=effort, output_path=output_path)

                    time_record["filename"].append(outfile_name)
                    time_record["ct"].append(ct)

    # AVIF
    if avif is True:
        for target_image in image_list:
            for quality in quality_param_avif:
                for speed in speed_avif:
                    # Construct output file total path
                    outfile_name: str = f"q{quality}-{speed}.avif"
                    output_path = image_to_dir(dataset_path, target_image) + outfile_name

                    # Add wildcard for now because the extensions are missing
                    ct = encode_avif(target_image=target_image+ext,
                                     quality=quality, speed=speed, output_path=output_path)

                    time_record["filename"].append(outfile_name)
                    time_record["ct"].append(ct)

    # WebP
    if webp is True:
        for target_image in image_list:
            for quality in quality_param_webp:
                for effort in effort_webp:
                    # Construct output file total path
                    outfile_name: str = f"q{quality}-{effort}.webp"
                    output_path = image_to_dir(dataset_path, target_image) + outfile_name

                    # Add wildcard for now because the extensions are missing
                    ct = encode_webp(target_image=target_image+ext,
                                     quality=quality, effort=effort, output_path=output_path)

                    time_record["filename"].append(outfile_name)
                    time_record["ct"].append(ct)

    # Save csv files
    df = pd.DataFrame.from_dict(data=time_record)
    df.to_csv("bulk_results.csv", index=False)


if __name__ == '__main__':
    bulk_compress("images/dataset/", jxl=False, webp=False)
