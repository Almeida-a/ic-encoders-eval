"""
    Function names convention:
        y_per_x -> produces a 2D graph:
         y being a dependent variable (ssim, psnr, mse, ds, cs)
         x being a controlled/independent variable (effort, quality)
        fixed variables -> modality, compression format

"""
import itertools
import json
import os
import re
from datetime import datetime
from enum import Enum
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd

import util
from parameters import PROCEDURE_RESULTS_FILE, JPEG_EVAL_RESULTS_FILE

WILDCARD_REGEX = r"[a-zA-Z0-9]+"

WILDCARD: str = "*"

MARGIN = .1  # ylim margin. e.g.: 10% margin

BARS_COLOR = "#007700"

METRICS_DESCRIPTION = dict(
    ds="Decompression Speed", cs="Compression Speed", cr="Compression Ratio",
    ssim="Structural Similarity", psnr="Peak Signal to Noise Ratio", mse="Mean Squared Error"
)
UNITS = dict(
    ds="MP/s", cs="MP/s"
)


class GraphMode(Enum):
    """Defines types of data visualization

    """
    METRIC = 1
    QUALITY = 2
    IMAGE = 3


class Pipeline(Enum):
    """Enum class: Identifies the pipelines

    Whether if it's the main pipeline, which evaluates the recent compression formats,
        or the jpeg, which evaluates that format.

    """
    MAIN = 1
    JPEG = 2


class ImageCompressionFormat(Enum):
    """Enum class: Identifies the Image Compression Formats

    """
    JXL = 1
    AVIF = 2
    WEBP = 3
    JPEG = 4


def draw_lines(x: list[float], y: list[float],
               x_label: str = "", y_label: str = "", title: str = "", filename: str = ""):
    """Draws a graph given a list of x and y values

    @param x: Independent axis vector
    @param y: Dependent axis vector
    @param x_label:
    @param y_label:
    @param title:
    @param filename: Name of plot file. Leave empty if you don't want it written
    """

    fig = plt.figure()

    # Plot the lists' points
    plt.plot(x, y, marker=".", linestyle='')

    if x_label != "":
        # Label to the x-axis
        plt.xlabel(x_label)

    if y_label != "":
        # Label to the y-axis
        plt.ylabel(y_label)

    if title != "":
        # Title to the graph
        plt.title(title)

    if not filename:
        plt.show()
    else:
        save_fig(fig, filename)


def search_dataframe(df: pd.DataFrame, key: str, value: str) -> pd.DataFrame:
    """Search the dataframe for rows containing particular attribute value

    @param df: Dataframe to be searched over
    @param key: Attribute key element
    @param value: Attribute value
    @return: All rows matching the attribute value
    """
    return df.loc[df[key] == value]


def metric_per_image(modality: str, metric: str, compression_format: str,
                     raw_data_fname: str = PROCEDURE_RESULTS_FILE + ".csv",
                     save: bool = False):
    """Given a data file, display statistics of kind metric = f(quality)

    Line graph portraying metric = f(quality), given the modality and format

    @todo Maybe use filter_data to narrow a bit the dataframe
    @param raw_data_fname: input data - file containing the procedure results
    @param modality: Used to filter the global data
    @param compression_format: Used to filter the global data
    @param metric: Evaluates a characteristic of a file compression (e.g.: ratio, similarity, speed).
    @param save: Whether to save or not the chart as file.
    """
    assert raw_data_fname.endswith(".csv"), f"Data source must be a csv file! Found \"{raw_data_fname}\"."

    modality = modality.upper()
    compression_format = compression_format.lower()

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

    y = [
        float(search_dataframe(df, key="filename", value=filename)[metric].values[0])
        for filename in filtered_fnames_x
    ]

    filename: str = f"{modality.lower()}_{compression_format.lower()}" if save else ""

    # Draw graph
    draw_lines(x=list(range(len(y))), y=y, y_label=metric,
               title=f"Modality: {modality}, Format: {compression_format}", filename=filename)


def draw_bars(keys: list, values: list, errs: list = None, x_label: str = "", y_label: str = "",
              title: str = "", filename: str = ""):
    """Draw a histogram using matplotlib

    @param keys: Name of each bar
    @param values: Height of each bar
    @param errs: Error / Standard deviation of each bar
    @param x_label: Label for the x-axis
    @param y_label: Label for the y-axis
    @param title: Graph title
    @param filename: save file name
    """
    # Sort the bars
    dict_unsorted = {keys[i]: (values[i], errs[i]) for i in range(len(keys))}
    dict_sorted = dict(sorted(dict_unsorted.items(), key=lambda x: x[0], reverse=False))
    keys = dict_sorted.keys()
    values, errs = [[value[i] for value in dict_sorted.values()] for i in (0, 1)]

    fig = plt.figure()

    plt.bar(keys, values, yerr=errs, color=BARS_COLOR)
    plt.xlabel(x_label.upper())
    plt.ylabel(y_label.upper())
    plt.title(title.upper())

    min_: float = min(values[i] - errs[i] for i in range(len(values)))

    maxes = [values[i] + errs[i] for i in range(len(values))]
    maxes.remove(float("inf")) if float("inf") in maxes else None
    max_: float = max(maxes)

    min_ -= (max_ - min_) * MARGIN
    max_ += (max_ - min_) * MARGIN

    plt.ylim(
        ymax=max_,
        ymin=max(min_, 0)  # No metric falls bellow 0
    )

    if not filename:
        plt.show()
    else:
        save_fig(fig, filename)


def save_fig(fig: plt.Figure, filename: str):
    path = f"images/graphs/{EXPERIMENT_ID}/{filename}"

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    fig.savefig(fname=path)
    plt.close(fig)


def metric_per_quality(compression_format: str, metric: str, body_part: str = WILDCARD,
                       modality: str = WILDCARD, depth: str = WILDCARD,
                       spp: str = WILDCARD, bps: str = WILDCARD,
                       squeezed_data_fname: str = PROCEDURE_RESULTS_FILE + ".json",
                       raw_data_fname: str = PROCEDURE_RESULTS_FILE + ".csv",
                       save: bool = False):
    """Draws bar graph for metric results (mean + std error) per quality setting

    @param body_part:
    @param modality:
    @param metric:
    @param depth:
    @param spp:
    @param bps:
    @param compression_format:
    @param squeezed_data_fname:
    @param raw_data_fname:
    @param save:
    """

    modality = modality.upper()
    body_part = body_part.upper()
    metric = metric.lower()
    compression_format = compression_format.lower()

    with open(squeezed_data_fname) as f:
        data: dict = json.load(f)

    # Initialize x, y and err lists, respectively
    qualities, avg, std = [], [], []
    histogram = {}
    size = 0  # Cardinality of the data (how many images are evaluated)

    for key, value in data.items():
        if not key.endswith(compression_format):
            continue

        # histogram[quality] = metric.mean
        quality = key.split("-")[0]

        stats: dict[str, float] = get_stats(compression_format, bps=bps, depth=depth,
                                            metric=metric, modality=modality,
                                            body_part=body_part, spp=spp, quality=quality,
                                            raw_data_fname=raw_data_fname)

        quality = quality.replace("q", "d") if compression_format == "jxl" else quality
        histogram[quality] = stats["avg"]

        qualities.append(quality)
        avg.append(stats["avg"])
        std.append(stats["std"])

        size = stats["size"]

    if histogram == dict():
        print("No data found with the specified parameters!")
        exit(1)

    filename: str = f"{modality.lower()}_{body_part.lower()}_" \
                    f"{compression_format.lower()}_d{depth}_s{spp}_b{bps}_n{size}" if save else ""

    unit = f"({UNITS.get(metric)})" if UNITS.get(metric) is not None else ""
    draw_bars(qualities, avg, std, x_label="Quality values", y_label=f"{METRICS_DESCRIPTION[metric]} {unit}",
              title=f"'{modality}-{body_part}' images, '{compression_format}' format,"
                    f" depth='{depth}', spp='{spp}', bps='{bps}', #='{size}'",
              filename=filename)


def get_stats(compression_format: str, modality: str, depth: str, body_part: str,
              spp: str, bps: str, metric: str, quality: str = WILDCARD,
              raw_data_fname: str = f"{PROCEDURE_RESULTS_FILE}.csv") -> dict[str, float]:
    """Extract stats from the procedure_results*.json given the filters, allowing wildcard queries

    @param quality:
    @param compression_format: Format in question
    @param body_part: Body part filter
    @param modality: Modality of the medical image
    @param depth: Number of frames of the image
    @param spp: Samples per pixel filter
    @param bps: Bits per sample filter
    @param metric: Evaluation parameter
    @param raw_data_fname:
    @return: Dictionary containing statistics (min/max/avg/dev)
    """

    results = pd.read_csv(raw_data_fname)

    results = filter_data(body_part=body_part, bps=bps, compression_format=compression_format,
                          depth=depth, modality=modality, results=results, spp=spp, quality=quality)

    metric_column = results[metric].values

    return dict(
        min=min(metric_column), max=max(metric_column),
        avg=np.mean(metric_column), std=np.std(metric_column), size=metric_column.size
    )


def metric_per_metric(x_metric: str, y_metric: str, raw_data_fname: str,
                      body_part: str = WILDCARD,
                      modality: str = WILDCARD, depth: str = WILDCARD,
                      spp: str = WILDCARD, bps: str = WILDCARD,
                      compression_format: str = WILDCARD, save: bool = False):
    """Pair metrics with metrics and show relationship using a line graph

    @todo add quality parameter, quality=WILDCARD

    @param save: Whether to save the figure or not
    @param x_metric: Metric displayed in the x-axis
    @param y_metric: Metric displayed in the y-axis
    @param raw_data_fname: File name containing the raw data to be processed
    @param modality: Modality of the images to be studied
    @param body_part: Body part covered in the images
    @param depth: Number of frames of the image
    @param spp: Samples per pixel
    @param bps: Bits per sample
    @param compression_format: Compression format of the compression instances
    """

    x_metric, y_metric = x_metric.lower(), y_metric.lower()
    if modality != WILDCARD:
        modality = modality.upper()
    if body_part != WILDCARD:
        body_part = body_part.upper()
    compression_format = compression_format.lower()

    results = pd.read_csv(raw_data_fname)

    results = filter_data(body_part, bps, compression_format, depth, modality, results, spp)

    if results.empty:
        print("No data found with the specified attributes!")
        exit(1)

    x, y = [results[column].values for column in (x_metric, y_metric)]

    zipped = list(zip(x, y))
    zipped = sorted(zipped, key=lambda elem: elem[0])  # Sort by the x-axis

    x, y = list(zip(*zipped))

    filename: str = f"{modality.lower()}_{body_part.lower()}_{compression_format.lower()}" \
                    f"_d{depth}_s{spp}_b{bps}_n{results.shape[0]}" if save else f""
    chart_title = f"'{modality}-{body_part}' images, '{compression_format}' format," \
                  f" depth='{depth}', spp='{spp}', bps='{bps}', #='{results.shape[0]}'"

    draw_lines(x, y, x_label=METRICS_DESCRIPTION[x_metric], y_label=METRICS_DESCRIPTION[y_metric],
               title=chart_title, filename=filename)


def filter_data(body_part: str, bps: str, compression_format: str,
                depth: str, modality: str, results: pd.DataFrame, spp: str, quality=WILDCARD) -> pd.DataFrame:
    """

    @param body_part: Filter
    @param bps: Bits per sample - filter
    @param compression_format: Format - filter
    @param depth: Depth - filter
    @param modality: Filter
    @param results: Dataframe formatted data
    @param spp: Samples per pixel - filter
    @param quality: Quality setting - filter
    @return: Dataframe containing filtering data
    """
    lgt_expr_regex = re.compile(r"<|>\d+")
    if spp == WILDCARD:
        spp = r"\d+"
    elif lgt_expr_regex.fullmatch(spp) is not None:
        spp = util.number_lgt_regex(spp)
    if bps == WILDCARD:
        bps = r"\d+"
    elif lgt_expr_regex.fullmatch(bps) is not None:
        bps = util.number_lgt_regex(bps)
    if depth == WILDCARD:
        depth = r"\d+"
    elif lgt_expr_regex.fullmatch(depth) is not None:
        depth = util.number_lgt_regex(depth)
    if modality == WILDCARD:
        modality = WILDCARD_REGEX
    if body_part == WILDCARD:
        body_part = WILDCARD_REGEX
    if compression_format == WILDCARD:
        compression_format = WILDCARD_REGEX

    if quality == WILDCARD:
        quality = r"q\d+(.\d+)?"

    results = results.set_index("filename")
    results = results.filter(
        axis="index",
        regex=fr"{modality}_{body_part}_\w+_{spp}_{bps}_{depth}(_\d+)?(.apng)?_{quality}(-e\d)?.{compression_format}"
    )
    return results


def generate_charts():

    raw_data_filename = f"{PROCEDURE_RESULTS_FILE}_2.csv"
    squeezed_data_filename = f"{PROCEDURE_RESULTS_FILE}_2_bp.json"
    jpeg_raw_data_filename = f"{JPEG_EVAL_RESULTS_FILE}.csv"
    jpeg_squeezed_data_filename = f"{JPEG_EVAL_RESULTS_FILE}.json"

    filters: dict[str, dict[str, list[str]]] = dict(
        CT=dict(
            depth=["1"],
            body_part=["HEAD"],
            spp=["1"],
            bps=["12"],
        ),
        MG=dict(
            depth=["1", "57", "58", "59"],
            body_part=["BREAST"],
            spp=["1"],
            bps=["10"],
        ),
        SM=dict(
            depth=["1", "2", "6", "24", "96", "384"],
            body_part=["NA"],
            spp=["3"],
            bps=["8"],
        ),
    )

    ic_format = ImageCompressionFormat
    formats = [format_.name.lower() for format_ in (ic_format.JXL, ic_format.WEBP, ic_format.AVIF)]

    toggle_charts_save: bool = True

    for modality, mod_filters in filters.items():
        for depth, body_part, spp, bps, format_ in itertools.product(
                mod_filters["depth"], mod_filters["body_part"],
                mod_filters["spp"], mod_filters["bps"], formats):

            generate_chart(body_part=body_part, bps=bps, depth=depth,
                           jpeg_raw_data_filename=jpeg_raw_data_filename,
                           jpeg_squeezed_data_filename=jpeg_squeezed_data_filename, save=toggle_charts_save,
                           metric=METRIC, modality=modality, raw_data_filename=raw_data_filename, spp=spp,
                           squeezed_data_filename=squeezed_data_filename, y_metric=Y_METRIC, format_=format_)

    if toggle_charts_save:
        print("Chart files were saved!")


def generate_chart(body_part: str, bps: str, depth: str, jpeg_raw_data_filename: str,
                   jpeg_squeezed_data_filename: str, metric: str, modality: str,
                   raw_data_filename: str, spp: str, squeezed_data_filename: str,
                   y_metric: str, format_: str, save: bool = False):
    """

    @param format_: Image compression format to be evaluated
    @param body_part:
    @param bps:
    @param depth:
    @param metric:
    @param modality:
    @param spp:
    @param y_metric:
    @param raw_data_filename:
    @param squeezed_data_filename:
    @param jpeg_raw_data_filename:
    @param jpeg_squeezed_data_filename:
    @param save:
    @return:
    """

    match EVALUATE, EXPERIMENT:
        case GraphMode.IMAGE, Pipeline.MAIN:
            metric_per_image(modality=modality, metric=metric,
                             compression_format=format_, raw_data_fname=raw_data_filename, save=save)
        case GraphMode.QUALITY, Pipeline.MAIN:
            metric_per_quality(modality=modality, body_part=body_part, metric=metric, depth=depth, spp=spp, bps=bps,
                               compression_format=format_,
                               squeezed_data_fname=squeezed_data_filename,
                               raw_data_fname=raw_data_filename, save=save)
        case GraphMode.QUALITY, Pipeline.JPEG:
            metric_per_quality(modality=modality, body_part=body_part, depth=depth, metric=metric, spp=spp, bps=bps,
                               squeezed_data_fname=jpeg_squeezed_data_filename,
                               raw_data_fname=jpeg_raw_data_filename,
                               compression_format=ImageCompressionFormat.JPEG.name, save=save)
        case GraphMode.METRIC, Pipeline.MAIN:
            metric_per_metric(x_metric=metric, y_metric=y_metric, body_part=body_part,
                              modality=modality, depth=depth, spp=spp, bps=bps,
                              compression_format=format_,
                              raw_data_fname=raw_data_filename, save=save)
        case GraphMode.METRIC, Pipeline.JPEG:
            metric_per_metric(x_metric=metric, y_metric=y_metric, modality=modality, body_part=body_part,
                              depth=depth, compression_format="jpeg", spp="1", bps=bps, save=save,
                              raw_data_fname=f"{JPEG_EVAL_RESULTS_FILE}.csv")
        case _:
            print("Invalid settings!")
            exit(1)


METRIC = "ssim"
Y_METRIC = "cr"

# Enums
EVALUATE = GraphMode.METRIC
EXPERIMENT = Pipeline.MAIN
EXPERIMENT_ID: str = datetime.now().strftime(f"{EXPERIMENT.name}_{EVALUATE.name}_{METRIC.upper()}_%d-%h-%y_%Hh%M")


if __name__ == '__main__':
    generate_charts()
