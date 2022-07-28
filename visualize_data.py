"""
    Function names convention:
        y_per_x -> produces a 2D graph:
         y being a dependent variable (ssim, psnr, mse, ds, cs)
         x being a controlled/independent variable (effort, quality)
        fixed variables -> modality, compression format

"""
import json
import re
from enum import Enum

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

    plt.show()


def metric_per_quality(compression_format: str, metric: str, body_part: str = WILDCARD,
                       modality: str = WILDCARD, depth: str = WILDCARD,
                       spp: str = WILDCARD, bps: str = WILDCARD,
                       squeezed_data_fname: str = PROCEDURE_RESULTS_FILE + ".json",
                       raw_data_fname: str = PROCEDURE_RESULTS_FILE + ".csv"):
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

    if histogram == dict():
        print("No data found with the specified parameters!")
        exit(1)

    unit = f"({UNITS.get(metric)})" if UNITS.get(metric) is not None else ""
    draw_bars(qualities, avg, std, x_label="Quality values", y_label=f"{METRICS_DESCRIPTION[metric]} {unit}",
              title=f"{modality} images, {compression_format} format, depth={depth}, spp={spp}, bps={bps}")


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
        avg=np.mean(metric_column), std=np.std(metric_column)
    )


def metric_per_metric(x_metric: str, y_metric: str, raw_data_fname: str,
                      body_part: str = WILDCARD,
                      modality: str = WILDCARD, depth: str = WILDCARD,
                      spp: str = WILDCARD, bps: str = WILDCARD,
                      compression_format: str = WILDCARD):
    """Pair metrics with metrics and show relationship using a line graph

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

    chart_title = f"'{modality}-{body_part}' images, '{compression_format}' format," \
                  f" depth='{depth}', spp='{spp}', bps='{bps}'"

    results = pd.read_csv(raw_data_fname)

    results = filter_data(body_part, bps, compression_format, depth, modality, results, spp)

    if results.empty:
        print("No data found with the specified attributes!")
        exit(1)

    x, y = [results[column].values for column in (x_metric, y_metric)]

    zipped = list(zip(x, y))
    zipped = sorted(zipped, key=lambda elem: elem[0])  # Sort by the x-axis

    x, y = list(zip(*zipped))

    draw_lines(x, y, x_label=METRICS_DESCRIPTION[x_metric], y_label=METRICS_DESCRIPTION[y_metric],
               title=chart_title)


def filter_data(body_part, bps, compression_format, depth, modality, results, spp, quality=WILDCARD):
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


if __name__ == '__main__':

    # Aliases
    mode = GraphMode
    pip = Pipeline
    ic_format = ImageCompressionFormat

    EVALUATE = mode.QUALITY
    EXPERIMENT = pip.MAIN
    FORMAT = ic_format.JXL.name

    match EVALUATE, EXPERIMENT:
        case mode.IMAGE, pip.MAIN:
            metric_per_image(modality="CT", metric="ds", compression_format=FORMAT)  # for now, displays a line graph
        case mode.QUALITY, pip.MAIN:
            metric_per_quality(modality="CT", metric="ssim", depth="1", spp="*", bps=WILDCARD,
                               compression_format=FORMAT,
                               squeezed_data_fname=f"{PROCEDURE_RESULTS_FILE}_2_bp.json",
                               raw_data_fname=f"{PROCEDURE_RESULTS_FILE}_2.csv")
        case mode.QUALITY, pip.JPEG:
            metric_per_quality(modality="CT", depth="1", metric="ssim", spp="1",
                               squeezed_data_fname=f"{JPEG_EVAL_RESULTS_FILE}_1.json",
                               raw_data_fname=f"{JPEG_EVAL_RESULTS_FILE}.csv",
                               compression_format="jpeg")
        case mode.METRIC, pip.MAIN:
            metric_per_metric(x_metric="ssim", y_metric="ds", body_part="BREAST",
                              modality="MG", depth="1", spp="*", bps=WILDCARD,
                              compression_format=FORMAT,
                              raw_data_fname=f"{PROCEDURE_RESULTS_FILE}.csv")
        case mode.METRIC, pip.JPEG:
            metric_per_metric(x_metric="ssim", y_metric="cr", modality="CT", body_part="HEAD",
                              depth="1", compression_format="jpeg", spp="1", bps=WILDCARD,
                              raw_data_fname=f"{JPEG_EVAL_RESULTS_FILE}.csv")
        case _:
            print("Invalid settings!")
            exit(1)
