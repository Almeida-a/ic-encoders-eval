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


def metric_per_quality(compression_format: str, metric: str,
                       modality: str = WILDCARD, depth: str = WILDCARD,
                       spp: str = WILDCARD, bps: str = WILDCARD,
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
    histogram = {}

    for key, value in data.items():
        if not key.endswith(compression_format):
            continue

        stats: dict[str, float] = get_stats(value, bps=bps, depth=depth,
                                            metric=metric, modality=modality, spp=spp)

        # histogram[quality] = metric.mean
        quality = key.split("-")[0]
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


def get_stats(data: dict, modality: str, depth: str,
              spp: str, bps: str, metric: str) -> dict[str, float]:
    """Extract stats from the procedure_results*.json, allowing wildcard queries

    @param data: Main data to be queried
    @param modality: Modality of the medical image
    @param depth: Number of frames of the image
    @param spp: Samples per pixel
    @param bps: Bits per sample
    @param metric: Evaluation parameter
    @return: Dictionary containing statistics (min/max/avg/dev)
    """
    keys: dict[str, list | tuple] = {
        "modality": (*data.keys(),) if modality == WILDCARD else (modality,),
        "depth": []
    }

    for key_modality_ in keys["modality"]:
        keys["depth"].extend(
            data[key_modality_].keys() if depth == WILDCARD else [depth]
        )
    keys["depth"] = tuple(set(keys["depth"]))

    keys["spp"] = []
    for key_modality_ in keys["modality"]:
        for key_depth_ in keys["depth"]:
            keys["spp"].extend(
                data[key_modality_][key_depth_].keys()
                if spp == WILDCARD else [spp]
            )
    keys["spp"] = tuple(set(keys["spp"]))

    keys["bps"] = []
    for key_modality_ in keys["modality"]:
        for key_depth_ in keys["depth"]:
            for key_spp_ in keys["spp"]:
                keys["bps"].extend(
                    data[key_modality_][key_depth_][key_spp_].keys()
                    if bps == WILDCARD else [bps]
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


def metric_per_metric(x_metric: str, y_metric: str, raw_data_fname: str,
                      modality: str = WILDCARD, depth: str = WILDCARD,
                      spp: str = WILDCARD, bps: str = WILDCARD,
                      compression_format: str = WILDCARD):
    """Pair metrics with metrics and show relationship using a line graph

    @param x_metric: Metric displayed in the x-axis
    @param y_metric: Metric displayed in the y-axis
    @param raw_data_fname: File name containing the raw data to be processed
    @param modality: Modality of the images to be studied
    @param depth: Number of frames of the image
    @param spp: Samples per pixel
    @param bps: Bits per sample
    @param compression_format: Compression format of the compression instances
    """
    lgt_expr_regex = re.compile(r"<|>\d+")

    x_metric, y_metric = x_metric.lower(), y_metric.lower()
    if modality != WILDCARD:
        modality = modality.upper()
    compression_format = compression_format.lower()

    chart_title = f"'{modality}' images, '{compression_format}' format, depth='{depth}', spp='{spp}', bps='{bps}'"

    results = pd.read_csv(raw_data_fname)

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
    if compression_format == WILDCARD:
        compression_format = WILDCARD_REGEX

    results = results.set_index("filename")
    regex = fr"{modality}_\w+_\w+_{spp}_{bps}_{depth}(_\d+)?(.apng)?_q\d+(.\d+)?(-e\d)?.{compression_format}"

    results = results.filter(
        axis="index",
        regex=regex
    )

    if results.empty:
        print("No data found with the specified attributes!")
        exit(1)

    x, y = [results[column].values for column in (x_metric, y_metric)]

    zipped = list(zip(x, y))
    zipped = sorted(zipped, key=lambda elem: elem[0])  # Sort by the x-axis

    x, y = list(zip(*zipped))

    draw_lines(x, y, x_label=METRICS_DESCRIPTION[x_metric], y_label=METRICS_DESCRIPTION[y_metric],
               title=chart_title)


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

    EVALUATE = mode.METRIC
    EXPERIMENT = pip.MAIN
    FORMAT = ic_format.WEBP.name

    match EVALUATE, EXPERIMENT:
        case mode.IMAGE, pip.MAIN:
            metric_per_image(modality="CT", metric="ds", compression_format=FORMAT)  # for now, displays a line graph
        case mode.QUALITY, pip.MAIN:
            metric_per_quality(modality="SM", metric="ssim", depth="1", spp="*", bps=WILDCARD,
                               compression_format=FORMAT,
                               raw_data_fname=f"{PROCEDURE_RESULTS_FILE}.json")
        case mode.QUALITY, pip.JPEG:
            metric_per_quality(modality="CT", depth="1", metric="ssim", spp="1",
                               raw_data_fname=f"{JPEG_EVAL_RESULTS_FILE}_1.json",
                               compression_format="jpeg")
        case mode.METRIC, pip.MAIN:
            metric_per_metric(x_metric="ssim", y_metric="cr",
                              modality="SM", depth=">1", spp="*", bps=WILDCARD,
                              compression_format=FORMAT,
                              raw_data_fname=f"{PROCEDURE_RESULTS_FILE}.csv")
        case mode.METRIC, pip.JPEG:
            metric_per_metric(x_metric="ssim", y_metric="cr", modality="CT", depth="1",
                              compression_format="jpeg", spp="1", bps=WILDCARD,
                              raw_data_fname=f"{JPEG_EVAL_RESULTS_FILE}.csv")
        case _:
            print("Invalid settings!")
            exit(1)
