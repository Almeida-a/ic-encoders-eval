"""Module used to process the exhaustive compression results into a simplified version

Aggregates the results by quality-effort configuration.
    For each one, provides statistics (min/max/avg/dev) upon each metric.

"""

import json
import os

import numpy as np
import pandas as pd

from parameters import PROCEDURE_RESULTS_FILE, MODALITY, DEPTH, SAMPLES_PER_PIXEL, BITS_PER_SAMPLE
from util import dataset_img_info, original_basename


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
            # If row does not fit, drop it from df TODO perf optimization: store the settings groups, saving
            #                                       complexity O(n**2) for memory space
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
    with open(original_basename(f"{results_file}.json"), "w") as out_file:
        json.dump(resume, out_file, indent=4)


if __name__ == '__main__':
    squeeze_data(PROCEDURE_RESULTS_FILE)
