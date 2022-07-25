"""Module used to process the exhaustive compression results into a simplified version

Aggregates the results by quality-effort configuration.
    For each one, provides statistics (min/max/avg/dev) upon each metric.

"""

import json
import os

import numpy as np
import pandas as pd

from parameters import PROCEDURE_RESULTS_FILE, MODALITY, DEPTH, SAMPLES_PER_PIXEL, BITS_PER_SAMPLE
from util import dataset_img_info, rename_duplicate


def squeeze_data(results_file: str = PROCEDURE_RESULTS_FILE):
    """Digests raw compression stats into condensed stats.

    Condensed stats:
     * are min/max/avg/std per modality, per body-part and per encoding format.
     * data is saved under {parameters.PROCEDURE_RESULTS_FILE}.json
    """

    # Read csv to df
    ordered_proc_res = list(filter(
        lambda file: file.startswith(results_file) and file.endswith(".csv"),
        os.listdir()
    ))
    ordered_proc_res.sort()
    latest_procedure_results = ordered_proc_res[-1]

    file_data = pd.read_csv(latest_procedure_results)

    # Aggregate the results to a dict
    resume = dict()

    file_data = file_data.set_index("filename")  # allows using df.filter

    used_regexs: list[str] = []

    for _, filename in enumerate(file_data.index.values):
        settings = filename.split("_")[-1]
        modality = dataset_img_info(filename, MODALITY)
        depth = dataset_img_info(filename, DEPTH)
        samples_per_pixel = dataset_img_info(filename, SAMPLES_PER_PIXEL)
        bits_per_sample = dataset_img_info(filename, BITS_PER_SAMPLE)

        # Dataframe containing only the data associated to the settings/characteristics at hand
        regex = fr"{modality}_\w+_\w+_{samples_per_pixel}_{bits_per_sample}_{depth}(.apng)?(_\d+)?_{settings}"

        # Avoid re-doing the same operations
        if regex in used_regexs:
            continue

        used_regexs.append(regex)

        filter_df = file_data.filter(axis="index", regex=regex)

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
        for metric in file_data.keys():

            mean = np.mean(filter_df[metric])
            std = np.std(filter_df[metric]) if mean != float("inf") else 0.

            entry[metric] = dict(
                min=filter_df[metric].min(), max=filter_df[metric].max(),
                avg=mean, std=std
            )
        entry["size"] = filter_df.shape[0]

    # Save dict to a json
    with open(rename_duplicate(f"{results_file}.json"), "w") as out_file:
        json.dump(resume, out_file, indent=4)


if __name__ == '__main__':
    squeeze_data(PROCEDURE_RESULTS_FILE)
