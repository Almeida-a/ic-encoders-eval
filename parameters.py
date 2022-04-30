""" Procedure parameters

This script includes configuration values to the experiment including, but not limited to:
 * the dataset path
 * The lossless encoding format of the evaluation input images
"""

import metrics

LOSSLESS_EXTENSION: str = ".png"
PROCEDURE_RESULTS_FILE: str = "procedure_results"
JPEG_EVAL_RESULTS_FILE: str = "jpeg_eval_results"
DATASET_PATH: str = "images/dataset/"
# TODO low prio. implement this enforcement upon the code
# GRAYSCALE_MODE: bool = True

# Info flags
JXL_SUPPORTED_VERSIONS: tuple = ("0.6.1",)
CAVIF_RS_SUPPORTED_VERSIONS: tuple = ("1.3.4",)
AVIF_DECODE_SUPPORTED_VERSIONS: tuple = ("0.2.2",)
WEBP_SUPPORTED_VERSIONS: tuple = ("0.4.1",)

# Quality metrics dict
# Serves as a "database" holding the
#   image quality evaluating metrics and the respective functions reference
# TODO use this on the procedure instead of hard coding the metrics
qmetrics = dict(
    mse=metrics.mse, psnr=metrics.psnr, ssim=metrics.ssim
)
