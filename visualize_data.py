"""
    Function names convention:
        y_per_x -> produces a 2D graph:
         y being a dependent variable (ssim, psnr, mse, ds, cs)
         x being a controlled/independent variable (effort, quality)
        fixed variables -> modality, compression format

"""
import json
import re
from typing import Optional

import matplotlib.pyplot as plt
import pandas
import pandas as pd

from parameters import PROCEDURE_RESULTS_FILE, MODALITY, DEPTH, SAMPLES_PER_PIXEL, BITS_PER_SAMPLE
from util import dataset_img_info

MARGIN = .1  # ylim margin. e.g.: 10% margin

BARS_COLOR = "#007700"

METRICS_DESCRIPTION = dict(
    ds="Decompression Speed", cs="Compression Speed", cr="Compression Ratio",
    ssim="Structure Similarity", psnr="Peak Signal to Noise Ratio", mse="Mean Squared Error"
)
UNITS = dict(
    ds="MP/s", cs="MP/s"
)


def draw_lines(x: list[float], y: list[float],
               x_label: str = "", y_label: str = "", title: str = ""):
    """Draws a graph given a list of x and y values

    @param x: Independent axis vector
    @param y: Dependent axis vector
    @param x_label:
    @param y_label:
    @param title:
    """

    # Plot the lists' points
    plt.plot(x, y)

    if x_label != "":
        # Label to the x axis
        plt.xlabel(x_label)

    if y_label != "":
        # Label to the y axis
        plt.ylabel(y_label)

    if title != "":
        # Title to the graph
        plt.title(title)

    plt.show()


def search_dataframe(df: pd.DataFrame, key: str, value: str) -> pd.DataFrame:
    """Search the dataframe for rows containing particular attribute value

    @param df: Dataframe to be searched over
    @param key: Attribute key element
    @param value: Attribute value
    @return: All rows matching the attribute value
    """
    return df.loc[df[key] == value]


def metric_per_image(modality: str, metric: str, compression_format: str,
                     raw_data_fname: str = PROCEDURE_RESULTS_FILE + ".csv"):
    """Given a data file, display statistics of kind metric = f(quality)

    Line graph portraying metric = f(quality), given the modality and format

    @param raw_data_fname: input data - file containing the procedure results
    @param modality: Used to filter the global data
    @param compression_format: Used to filter the global data
    @param metric: Evaluates a characteristic of a file compression (e.g.: ratio, similarity, speed)
    """
    assert raw_data_fname.endswith(".csv"), f"Data source must be a csv file! Found \"{raw_data_fname}\"."

    modality = modality.upper()

    # Read file into dataframe
    df: pd.DataFrame = pandas.read_csv(raw_data_fname)

    # Get the file names w/ the given modality, format and effort/speed value
    filtered_fnames_x = list(filter(
        lambda filename: re.fullmatch(
            pattern=rf"{modality}_[\w_]+q\d+(.\d+)?-e\d.{compression_format}",
            string=filename
        ) is not None,
        tuple(df.filename)
    ))

    ssim_y = [
        float(search_dataframe(df, key="filename", value=filename)[metric].values[0])
        for filename in filtered_fnames_x
    ]

    # Draw graph
    draw_lines(x=list(range(len(ssim_y))), y=ssim_y, y_label=metric, title=f"Modality: {modality},"
                                                                           f" Format: {compression_format}")


def draw_bars(keys: list, values: list, errs: list = None, x_label: str = "", y_label: str = "", title: str = ""):
    """Draw a histogram using matplotlib

    @param keys: Name of each bar
    @param values: Height of each bar
    @param errs: Error / Standard deviation of each bar
    @param x_label: Label for the x-axis
    @param y_label: Label for the y-axis
    @param title: Graph title
    """
    # Sort the bars
    dict_unsorted = {keys[i]: (values[i], errs[i]) for i in range(len(keys))}
    dict_sorted = dict(sorted(dict_unsorted.items(), key=lambda x: x[0], reverse=False))
    keys = dict_sorted.keys()
    values, errs = [[value[i] for value in dict_sorted.values()] for i in (0, 1)]

    plt.bar(keys, values, yerr=errs, color=BARS_COLOR)
    plt.xlabel(x_label.upper())
    plt.ylabel(y_label.upper())
    plt.title(title.upper())

    min_: float = min([values[i] - errs[i] for i in range(len(values))])

    maxes = [values[i] + errs[i] for i in range(len(values))]
    maxes.remove(float("inf")) if float("inf") in maxes else None
    max_: float = max(maxes)

    min_ -= (max_ - min_) * MARGIN
    max_ += (max_ - min_) * MARGIN

    plt.ylim(ymax=max_, ymin=min_)

    plt.show()


def metric_per_quality(compression_format: str, modality: Optional[str] = None,
                       metric: Optional[str] = None, depth: Optional[str] = None,
                       spp: Optional[str] = None, bps: Optional[str] = None,
                       raw_data_fname: str = PROCEDURE_RESULTS_FILE + ".json"):
    """Draws bar graph for metric results (mean + std error) per quality setting

    @param modality:
    @param metric:
    @param depth:
    @param spp:
    @param bps:
    @param compression_format:
    @param raw_data_fname:
    """

    modality = modality.upper()
    metric = metric.lower()
    compression_format = compression_format.lower()

    with open(raw_data_fname) as f:
        data: dict = json.load(f)

    # Initialize x, y and err lists, respectively
    qualities, avg, std = [], [], []
    histogram = dict()

    for key, value in data.items():
        if not key.endswith(compression_format):
            continue

        stats: dict[str, float] = get_stats(value, bps=bps, depth=depth,
                                            metric=metric, modality=modality, spp=spp)

        # histogram[quality] = metric.mean
        histogram[key.split("-")[0]] = stats["avg"]

        qualities.append(key.split("-")[0])
        avg.append(stats["avg"])
        std.append(stats["std"])

    unit = f"({UNITS.get(metric)})" if UNITS.get(metric) is not None else ""
    draw_bars(qualities, avg, std, x_label="Quality values", y_label=f"{METRICS_DESCRIPTION[metric]} {unit}",
              title=f"{modality} images, {compression_format} format, depth={depth}, spp={spp}, bps={bps}")


def get_stats(data: dict, modality: Optional[str], depth: Optional[str],
              spp: Optional[str], bps: Optional[str],
              metric: str) -> dict[str, float]:
    """Extract stats from the procedure_results*.json, allowing wildcard queries

    @param data: Main data to be queried
    @param modality: Modality of the medical image
    @param depth: Number of frames of the image
    @param spp: Samples per pixel
    @param bps: Bits per sample
    @param metric: Evaluation parameter
    @return: Dictionary containing statistics (min/max/avg/dev)
    """
    keys: dict[str, list | tuple] = dict()

    keys["modality"] = (*data.keys(),) if modality is None else (modality,)

    keys["depth"] = []
    for key_modality_ in keys["modality"]:
        keys["depth"].extend(
            data[key_modality_].keys() if depth is None else [depth]
        )
    keys["depth"] = tuple(set(keys["depth"]))

    keys["spp"] = []
    for key_modality_ in keys["modality"]:
        for key_depth_ in keys["depth"]:
            keys["spp"].extend(
                [data[key_modality_][key_depth_].keys()]
                if spp is None else [spp]
            )
    keys["spp"] = tuple(set(keys["spp"]))

    keys["bps"] = []
    for key_modality_ in keys["modality"]:
        for key_depth_ in keys["depth"]:
            for key_spp_ in keys["spp"]:
                keys["bps"].extend(
                    data[key_modality_][key_depth_][key_spp_].keys()
                    if bps is None else [bps]
                )
    keys["bps"] = tuple(set(keys["bps"]))

    result_stats: dict[str, float] = dict(min=float("inf"), max=.0, avg=.0, std=.0)

    total_images: int = 0

    for modality_ in keys["modality"]:
        for depth_ in keys["depth"]:
            for spp_ in keys["spp"]:
                for bps_ in keys["bps"]:
                    stat = data[modality_][depth_][spp_][bps_]

                    result_stats["avg"] += stat[metric]["avg"] * stat["size"]
                    result_stats["min"] = min(result_stats["min"], stat[metric]["min"])
                    result_stats["max"] = max(result_stats["max"], stat[metric]["max"])

                    total_images += stat["size"]

    if total_images == 0:
        print("No data found!")
        exit(0)

    result_stats["avg"] /= total_images

    return result_stats


def metric_per_metric(x_metric: str, y_metric: str, modality: Optional[str],
                      depth: Optional[str], spp: Optional[str], bps: Optional[str],
                      compression_format: Optional[str], raw_data_fname: str):
    """Pair metrics with metrics and show relationship using a line graph

    @param x_metric:
    @param y_metric:
    @param modality:
    @param depth:
    @param spp:
    @param bps:
    @param compression_format:
    @param raw_data_fname:
    """
    x_metric, y_metric = x_metric.lower(), y_metric.lower()
    modality = modality.upper()

    df = pd.read_csv(raw_data_fname)

    # Filter rows
    for i, row in df.iterrows():
        row_filename_ = row["filename"]
        if modality not in [dataset_img_info(row_filename_, MODALITY), None]\
                or depth not in [dataset_img_info(row_filename_, DEPTH), None]\
                or spp not in [dataset_img_info(row_filename_, SAMPLES_PER_PIXEL), None]\
                or bps not in [dataset_img_info(row_filename_, BITS_PER_SAMPLE), None]\
                or (row_filename_.endswith(f".{compression_format}") or row_filename_ is None):
            df = df.drop(index=i)

    x, y = df[x_metric], df[y_metric]

    zipped = list(zip(x, y))
    zipped = sorted(zipped, key=lambda elem: elem[0])  # Sort by the x-axis

    x, y = list(zip(*zipped))

    draw_lines(x, y, x_label=METRICS_DESCRIPTION[x_metric], y_label=METRICS_DESCRIPTION[y_metric],
               title=f"{modality} images, {compression_format} format, depth={depth}, spp={spp}, bps={bps}")


if __name__ == '__main__':
    EVALUATE = "metric"

    match EVALUATE:
        case "image":
            metric_per_image(modality="CT", metric="ds", compression_format="jxl")  # for now, displays a line graph
        case "quality":
            metric_per_quality(modality="CT", metric="ds", depth="1", spp="1", bps="12",
                               compression_format="jxl",
                               raw_data_fname=f"{PROCEDURE_RESULTS_FILE}_2.json")

        case "metric":
            metric_per_metric(x_metric="ssim", y_metric="cs",
                              modality="CT", depth="1", spp="1", bps="12",
                              compression_format="avif",
                              raw_data_fname=f"{PROCEDURE_RESULTS_FILE}_1.csv")
        case _:
            print("Invalid settings!")
            exit(1)
