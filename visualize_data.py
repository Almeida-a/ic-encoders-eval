"""
    Function names convention:
        y_per_x -> produces a 2D graph:
         y being a dependent variable (ssim, psnr, mse, ds, cs)
         x being a controlled/independent variable (effort, quality)
        fixed variables -> modality, compression format

"""
import itertools
import os
import re
from datetime import datetime
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd

import util
from parameters import PROCEDURE_RESULTS_FILE, JPEG_EVAL_RESULTS_FILE, MINIMUM_AVIF_QUALITY, QUALITY_TOTAL_STEPS, \
    MAXIMUM_JXL_DISTANCE, MINIMUM_WEBP_QUALITY, MINIMUM_JPEG_QUALITY

TOGGLE_CHARTS_SAVE = True

WILDCARD_REGEX = r"[a-zA-Z0-9]+"

WILDCARD: str = "*"

MARGIN = .1  # ylim margin. e.g.: 10% margin

DARK_GREEN = "#007700"
DARK_PURPLE = "#4c0099"

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
                     raw_data_fname: str = PROCEDURE_RESULTS_FILE + ".csv"):
    """Given a data file, display statistics of kind metric = f(quality)

    Line graph portraying metric = f(quality), given the modality and format

    @todo low priority - Maybe use filter_data to narrow a bit the dataframe
    @param raw_data_fname: input data - file containing the procedure results
    @param modality: Used to filter the global data
    @param compression_format: Used to filter the global data
    @param metric: Evaluates a characteristic of a file compression (e.g.: ratio, similarity, speed).
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

    filename: str = f"{modality.lower()}_{compression_format.lower()}" if TOGGLE_CHARTS_SAVE else ""

    # Draw graph
    draw_lines(x=list(range(len(y))), y=y, y_label=metric,
               title=f"Modality: {modality}, Format: {compression_format}", filename=filename)


def draw_bars(keys: list, values: list[int | float | tuple], errs: list[int | float | tuple] = None,
              x_label: str = "", y_label: str = "", title: str = "", filename: str = "", **kwargs):
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
    keys, errs[0], errs[1], values[0], values[1] = sort_by_keys(keys, *errs, *values)

    is_multi_bar = type(values[0]) is tuple
    is_single_bar = type(values[0]) in {int, float}

    if is_multi_bar:

        bars_data = pd.DataFrame(
            dict(
                values1=values[0],
                values2=values[1],
                errors1=errs[0],
                errors2=errs[1],
            ), index=keys
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        ax2 = ax.twinx()

        width = 0.4
        bars_data["values1"].plot(kind='bar', ax=ax, color=DARK_GREEN,
                                  position=1, width=width, yerr=bars_data["errors1"])
        bars_data["values2"].plot(kind='bar', ax=ax2, color=DARK_PURPLE,
                                  position=0, width=width, yerr=bars_data["errors2"])

        ax.set_xlabel(x_label.upper())
        ax.set_ylabel(y_label.upper(), color=DARK_GREEN)
        ax2.set_ylabel(kwargs["y2_label"].upper(), color=DARK_PURPLE)

        # Calculate ylims
        ymax, ymin = calculate_ylims(values=bars_data["values1"].values, errs=bars_data["errors1"].values)
        ax.set_ylim(ymax=ymax, ymin=ymin)
        ymax, ymin = calculate_ylims(values=bars_data["values2"].values, errs=bars_data["errors2"].values)
        ax2.set_ylim(ymax=ymax, ymin=ymin)

    elif is_single_bar:
        fig = plt.figure()

        plt.bar(keys, values, yerr=errs, color=DARK_GREEN)

        plt.xlabel(x_label.upper())
        plt.ylabel(y_label.upper())

        max_, min_ = calculate_ylims(errs, values)
        plt.ylim(
            ymax=max_,
            ymin=max(min_, 0)  # No metric falls bellow 0
        )
    else:
        raise AssertionError(f"Invalid bar `values` variable type: '{type(values)}'.")
    plt.title(title.upper())

    if not filename:
        plt.show()
    else:
        save_fig(fig, filename)


def calculate_ylims(errs, values, allow_negatives: bool = False):
    min_: float = min(values[i] - errs[i] for i in range(len(values)))
    maxes: list[int | float] = [values[i] + errs[i] for i in range(len(values))]
    maxes.remove(float("inf")) if float("inf") in maxes else None
    max_: float = max(maxes)
    min_ -= (max_ - min_) * MARGIN
    max_ += (max_ - min_) * MARGIN

    if not allow_negatives:
        min_ = max(min_, 0)
        max_ = max(max_, 0)

    return max_, min_


def sort_by_keys(*args) -> list:
    """Sort all the lists by the first one

    @return:
    """
    zipped = zip(*args)
    zipped = list(sorted(zipped, key=lambda elem: float(elem[0]), reverse=False))
    return list(zip(*zipped))


def save_fig(fig: plt.Figure, filename: str):
    path = f"images/graphs/{EXPERIMENT_ID}/{filename}.png"

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    fig.savefig(fname=path)
    plt.close(fig)


def get_qualities(raw_data_fname: str, compression_format: str) -> list:
    """

    @param raw_data_fname:
    @param compression_format:
    @return:
    """

    df = pd.read_csv(raw_data_fname)

    value_matcher = re.compile(r"\d+")

    if compression_format == "jxl":
        matcher = re.compile(rf"\d+.\d+-e\d.{compression_format}")
        value_matcher = re.compile(r"\d+.\d+")
    elif compression_format in {"jpeg", "avif", "webp"}:
        matcher = re.compile(rf"\d+-e\d.{compression_format}")
    elif compression_format == "jpeg":
        matcher = re.compile(rf"\d+.{compression_format}")
    else:
        raise AssertionError(f"Invalid compression format: '{compression_format}'!")

    yielded_qualities: list = []

    for filename in df["filename"].values:
        extraction: list[str] = re.findall(matcher, filename)

        if not extraction:
            # empty - pattern not found
            continue

        quality: str = re.findall(value_matcher, extraction[0])[0]

        if quality not in yielded_qualities:
            yielded_qualities.append(quality)
            yield quality


def metric_per_quality(compression_format: str, body_part: str = WILDCARD,
                       modality: str = WILDCARD, depth: str = WILDCARD,
                       spp: str = WILDCARD, bps: str = WILDCARD,
                       raw_data_fname: str = PROCEDURE_RESULTS_FILE + ".csv"):
    """Draws bar graph for metric results (mean + std error) per quality setting

    @param body_part:
    @param modality:
    @param depth:
    @param spp:
    @param bps:
    @param compression_format:
    @param raw_data_fname:
    """

    modality = modality.upper()
    body_part = body_part.upper()
    compression_format = compression_format.lower()

    # Initialize x, y and err lists, respectively
    qualities, avg, std = [], [], []
    size = 0  # Cardinality of the data (how many images are evaluated)

    for quality in get_qualities(raw_data_fname, compression_format):

        stats_y1: dict[str, float] = get_stats(compression_format, bps=bps, depth=depth,
                                               metric=METRIC, modality=modality,
                                               body_part=body_part, spp=spp, quality=quality,
                                               raw_data_fname=raw_data_fname)
        stats_y2: dict[str, float] = get_stats(compression_format, bps=bps, depth=depth,
                                               metric=Y_METRIC, modality=modality,
                                               body_part=body_part, spp=spp, quality=quality,
                                               raw_data_fname=raw_data_fname)

        quality = quality.replace("q", "d") if compression_format == "jxl" else quality

        qualities.append(quality)
        avg.append((stats_y1["avg"], stats_y2["avg"]))
        std.append((stats_y1["std"], stats_y2["std"]))

        size = stats_y1["size"]

    # Format avg & std
    avg, std = (list(zip(*lst)) for lst in (avg, std))

    filename: str = f"{modality.lower()}_{body_part.lower()}_" \
                    f"d{depth}_s{spp}_b{bps}_n{size}_{compression_format.lower()}" if TOGGLE_CHARTS_SAVE else ""

    unit = f"({UNITS.get(METRIC)})" if UNITS.get(METRIC) is not None else ""
    unit2 = f"({UNITS.get(Y_METRIC)})" if UNITS.get(Y_METRIC) is not None else ""
    draw_bars(qualities, avg, std, x_label="Quality values",
              y_label=f"{METRICS_DESCRIPTION[METRIC]} {unit}", y2_label=f"{METRICS_DESCRIPTION[Y_METRIC]} {unit2}",
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
                      body_part: str = WILDCARD, quality: str = WILDCARD,
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
    @param quality: Quality setting filter
    """

    x_metric, y_metric = x_metric.lower(), y_metric.lower()
    if modality != WILDCARD:
        modality = modality.upper()
    if body_part != WILDCARD:
        body_part = body_part.upper()
    compression_format = compression_format.lower()

    results = pd.read_csv(raw_data_fname)

    results = filter_data(body_part, bps, compression_format, depth, modality, results, spp, quality=quality)

    if results.empty:
        print("No data found with the specified attributes!")
        exit(1)

    x, y = [results[column].values for column in (x_metric, y_metric)]

    zipped = list(zip(x, y))
    zipped = sorted(zipped, key=lambda elem: elem[0])  # Sort by the x-axis

    x, y = list(zip(*zipped))

    filename = f"{modality.lower()}_{body_part.lower()}_{compression_format.lower()}" \
               f"_d{depth}_s{spp}_b{bps}_n{results.shape[0]}_q{quality}" if TOGGLE_CHARTS_SAVE else f""
    chart_title = f"'{modality}-{body_part}' images, '{compression_format}' format,\n" \
                  f" depth='{depth}', spp='{spp}', bps='{bps}', q='{quality}', #='{results.shape[0]}'"

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
        quality = r"\d+(.\d+)?"

    results = results.set_index("filename")
    results = results.filter(
        axis="index",
        regex=fr"{modality}_{body_part}_\w+_{spp}_{bps}_{depth}(_\d+)?(.apng)?_q{quality}(-e\d)?.{compression_format}"
    )
    return results


def generate_charts():

    raw_data_filename = f"{PROCEDURE_RESULTS_FILE}_2.csv"
    jpeg_raw_data_filename = f"{JPEG_EVAL_RESULTS_FILE}.csv"

    img_type_filters: dict[str, dict[str, list[str]]] = dict(
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

    match EXPERIMENT:
        case Pipeline.MAIN:
            format_qualities = dict(
                jxl=list(np.linspace(0.0, MAXIMUM_JXL_DISTANCE, QUALITY_TOTAL_STEPS)),
                avif=list(np.linspace(MINIMUM_AVIF_QUALITY, 100, QUALITY_TOTAL_STEPS).astype(np.ubyte)),
                webp=list(np.linspace(MINIMUM_WEBP_QUALITY, 100, QUALITY_TOTAL_STEPS).astype(np.ubyte))
            )
        case Pipeline.JPEG:
            format_qualities = dict(
                jpeg=list(np.linspace(MINIMUM_JPEG_QUALITY, 100, QUALITY_TOTAL_STEPS))
            )
        case _:
            raise AssertionError("Invalid experiment")

    if EVALUATE == GraphMode.QUALITY:
        for key in format_qualities.keys():
            format_qualities[key] = [None]

    for (format_, qualities), (modality, mod_filters) in itertools.product(
            format_qualities.items(), img_type_filters.items()):
        for depth, body_part, spp, bps, quality in itertools.product(
                mod_filters["depth"], mod_filters["body_part"],
                mod_filters["spp"], mod_filters["bps"], qualities):
            generate_chart(body_part=body_part, bps=bps, depth=depth,
                           jpeg_raw_data_filename=jpeg_raw_data_filename, quality=quality,
                           metric=METRIC, modality=modality, raw_data_filename=raw_data_filename, spp=spp,
                           y_metric=Y_METRIC, format_=format_)

    if TOGGLE_CHARTS_SAVE:
        print("Chart files were saved!")


def generate_chart(body_part: str, bps: str, depth: str, jpeg_raw_data_filename: str,
                   metric: str, modality: str, quality: str,
                   raw_data_filename: str, spp: str, y_metric: str, format_: str):
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
    @param jpeg_raw_data_filename:
    @param quality:
    @return:
    """

    match EVALUATE, EXPERIMENT:
        case GraphMode.IMAGE, Pipeline.MAIN:
            metric_per_image(modality=modality, metric=metric,
                             compression_format=format_, raw_data_fname=raw_data_filename)
        case GraphMode.QUALITY, Pipeline.MAIN:
            metric_per_quality(modality=modality, body_part=body_part, depth=depth, spp=spp, bps=bps,
                               compression_format=format_,
                               raw_data_fname=raw_data_filename)
        case GraphMode.QUALITY, Pipeline.JPEG:
            metric_per_quality(modality=modality, body_part=body_part, depth=depth, spp=spp, bps=bps,
                               raw_data_fname=jpeg_raw_data_filename,
                               compression_format=ImageCompressionFormat.JPEG.name)
        case GraphMode.METRIC, Pipeline.MAIN:
            metric_per_metric(x_metric=metric, y_metric=y_metric, body_part=body_part,
                              modality=modality, depth=depth, spp=spp, bps=bps,
                              compression_format=format_, quality=quality,
                              raw_data_fname=raw_data_filename)
        case GraphMode.METRIC, Pipeline.JPEG:
            metric_per_metric(x_metric=metric, y_metric=y_metric, modality=modality, body_part=body_part,
                              depth=depth, compression_format="jpeg", spp="1", bps=bps, quality=quality,
                              raw_data_fname=f"{JPEG_EVAL_RESULTS_FILE}.csv")
        case _:
            print("Invalid settings!")
            exit(1)


# TODO automate the adjusting of these metrics (only when mode==MAIN)
METRIC = "ssim"
Y_METRIC = "cr"

# Enums
EVALUATE = GraphMode.QUALITY
EXPERIMENT = Pipeline.MAIN
EXPERIMENT_ID: str = datetime.now().strftime(f"DT_%d-%h-%y_%Hh%M_{EXPERIMENT.name}_{EVALUATE.name}_"
                                             f"{METRIC.upper()}_{Y_METRIC.upper()}")


if __name__ == '__main__':
    generate_charts()
