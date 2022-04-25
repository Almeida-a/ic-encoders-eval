import re
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
import pandas

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
    """

    :param x:
    :param y:
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


def ssim_per_quality(modality: str,
                     raw_data_fname: str = PROCEDURE_RESULTS_FILE+".csv",
                     compression_format: str = "jxl"):
    # Make sure the modality is uppercase
    modality = modality.upper()

    # Read file into dataframe
    df = pandas.read_csv(raw_data_fname)

    # Get the file names w/ the given modality, format and effort/speed value
    # Use regex
    filtered_fnames_x: List[Optional[str]] = list()
    for filename_x in df["filename"]:

        match = re.fullmatch(
            pattern=rf"{modality}_[a-zA-Z]+_(\d+_)?q\d+(.\d+)?-e\d.{compression_format}",
            string=filename_x
        )

        if match is not None:
            filtered_fnames_x.append(match.string)

    # Extract y list: ssim per fname in filtered_fnames
    ssim_y = list()
    for filename_x in filtered_fnames_x:
        ssim_y.append(float(df.loc[df["filename"] == filename_x]["ssim"]))

    # Draw graph
    draw_graph(x=list(range(len(ssim_y))), y=ssim_y)


if __name__ == '__main__':
    ssim_per_quality(modality="CT")
