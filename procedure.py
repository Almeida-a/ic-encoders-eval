from typing import List, Dict, Union

import pandas as pandas


def encode_jxl(target_image: str, distance: float, effort: int, output_path: str) -> float:
    """

    :param target_image: Path to image targeted for compression encoding
    :param distance: Quality setting as set by cjxl (butteraugli distance)
    :param effort: --effort level parameter as set by cjxl
    :param output_path: Path where the compressed image should go to
    :return: Time taken to compress
    """
    pass


def bulk_compress(dataset_path: str):
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
    :param dataset_path: Path to the dataset folder
    :return:
    """

    # TODO save all images path relative to dataset_path
    image_list: List[str] = [...]  # TODO read dataset and infer from there
    # Take out the extensions
    for i in range(len(image_list)):
        image_list[i] = ".".join(image_list[i].split(".")[:-1])

    # TODO Set quality parameters to be used in compression
    quality_param_jxl: List[float] = []
    quality_param_avif: List[float] = []
    quality_param_webp: List[float] = []

    # TODO Set effort/speed parameters for compression
    effort_jxl: List[int] = []
    effort_avif: List[int] = []
    effort_webp: List[int] = []

    # TODO encode (to target path) and record time of compression
    ct: float  # Record time of compression
    time_record: Dict[str, List[Union[float, str]]] = dict(filename=[], ct=[])

    # JPEG XL
    for target_image in image_list:
        for quality in quality_param_jxl:
            for effort in effort_jxl:
                # Construct output file total path
                outfile_name: str = target_image + "/" + f"q{quality}-{effort}.jxl"
                output_path = dataset_path + "compressed/" + outfile_name

                # Add wildcard for now because the extensions are missing
                ct = encode_jxl(target_image=dataset_path+target_image+".*",
                                distance=quality, effort=effort, output_path=output_path)

                time_record["filename"].append(outfile_name)
                time_record["ct"].append(ct)

    # TODO AVIF
    ...

    # TODO WebP
    ...

    # TODO Save csv files
    df = pandas.Dataframe(data=time_record)
    df.to_csv()


if __name__ == '__main__':
    bulk_compress("")
