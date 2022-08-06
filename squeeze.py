"""Module used to process the exhaustive compression results into a simplified version

Aggregates the results by quality-effort configuration.
    For each one, provides statistics (min/max/avg/dev) upon each metric.

"""

import json
import os
from pathlib import PosixPath

import numpy as np
import pandas as pd

from parameters import PathParameters, MODALITY, DEPTH, SAMPLES_PER_PIXEL, BITS_PER_SAMPLE, BODYPART
from util import dataset_img_info, rename_duplicate, mkdir_if_not_exists


def squeeze_data(results_path: str = PathParameters.PROCEDURE_RESULTS_PATH):
    """Digests raw compression stats into condensed stats.

    Condensed stats:
     * are min/max/avg/std per modality, per body-part and per encoding format.
     * data is saved under {parameters.PROCEDURE_RESULTS_FILE}.json
    """

    results_parent_dir = str(PosixPath(results_path).parent)
    ordered_proc_res = sorted(filter(
        lambda file: file.startswith(results_path) and file.endswith(".csv"),
        (f'{results_parent_dir}/' + file for file in os.listdir(results_parent_dir))
    ))

    latest_procedure_results = ordered_proc_res[-1]

    file_data = pd.read_csv(latest_procedure_results)

    # Aggregate the results to a dict
    resume = {}

    file_data = file_data.set_index("filename")  # allows using df.filter

    used_regexs: list[str] = []

    for filename in file_data.index.values:
        settings = filename.split("_")[-1]
        modality = dataset_img_info(filename, MODALITY)
        body_part = dataset_img_info(filename, BODYPART)
        depth = dataset_img_info(filename, DEPTH)
        samples_per_pixel = dataset_img_info(filename, SAMPLES_PER_PIXEL)
        bits_per_sample = dataset_img_info(filename, BITS_PER_SAMPLE)

        # Dataframe containing only the data associated to the settings/characteristics at hand
        regex = fr"{modality}_{body_part}_\w+_{samples_per_pixel}_{bits_per_sample}_{depth}(.apng)?(_\d+)?_{settings}"

        # Avoid re-doing the same operations
        if regex in used_regexs:
            continue

        used_regexs.append(regex)

        filter_df = file_data.filter(axis="index", regex=regex)

        # Create nested dictionary with groupings/filters TODO make this pretty
        if resume.get(settings) is None:
            resume[settings] = {}
        if resume[settings].get(modality) is None:
            resume[settings][modality] = {}
        if resume[settings][modality].get(body_part) is None:
            resume[settings][modality][body_part] = {}
        if resume[settings][modality][body_part].get(depth) is None:
            resume[settings][modality][body_part][depth] = {}
        if resume[settings][modality][body_part][depth].get(samples_per_pixel) is None:
            resume[settings][modality][body_part][depth][samples_per_pixel] = {}
        if resume[settings][modality][body_part][depth][samples_per_pixel].get(bits_per_sample) is None:
            resume[settings][modality][body_part][depth][samples_per_pixel][bits_per_sample] = {}
        else:
            # Entry has already been filled, skip to avoid re-doing the operations
            continue

        entry = resume[settings][modality][body_part][depth][samples_per_pixel][bits_per_sample]

        # Gather statistics
        for metric in file_data.keys():

            mean = np.mean(filter_df[metric])
            std = np.std(filter_df[metric]) if mean != float("inf") else 0.

            entry[metric] = dict(
                min=filter_df[metric].min(), max=filter_df[metric].max(),
                avg=mean, std=std
            )
        entry["size"] = filter_df.shape[0]

    mkdir_if_not_exists(results_path, regard_parent=True)

    # Save dict to a json
    with open(rename_duplicate(f"{results_path}.json"), "w") as out_file:
        json.dump(resume, out_file, indent=4)


if __name__ == '__main__':
    squeeze_data(PathParameters.PROCEDURE_RESULTS_PATH)
