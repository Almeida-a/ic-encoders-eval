import re
from typing import List, Callable, Any
from typing import Optional

import matplotlib.pyplot as plt
import pandas
import pandas as pd

from parameters import PROCEDURE_RESULTS_FILE

"""
    Function names convention:
        y_per_x -> produces a 2D graph:
         y being a dependent variable (ssim, psnr, mse, ds, cs)
         x being a controlled/independent variable (effort, quality)
        fixed variables -> modality, compression format
        
"""


# TODO either produce a function for each case or implement a master
#  function (an intermediate solution can also be creating functions
#  for each case but they all call over a private master function.
#  This would maybe mitigate the perceived complexity for possible code readers)


def draw_graph(x: List[float], y: List[float],
               x_label: str = "", y_label: str = "", title: str = ""):
    """Draws a graph given a list of x and y values

    TODO add parameter to define type of graph (line, histogram, ...)

    :param x: Independent axis vector
    :param y: Dependent axis vector
    :param x_label:
    :param y_label:
    :param title:
    :return:
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


def metric_per_image(modality: str, metric: str,
                     raw_data_fname: str = PROCEDURE_RESULTS_FILE + ".csv",
                     compression_format: str = "jxl"):
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
    draw_graph(x=list(range(len(ssim_y))), y=ssim_y, y_label=metric, title=f"Modality: {modality},"
                                                                           f" Format: {compression_format}")


if __name__ == '__main__':
    metric_per_image(modality="CT", metric="cr")  # for now, displays a line graph
    # TODO use histogram - quality setting for each bar
