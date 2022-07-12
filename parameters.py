"""Procedure parameters

This script includes configuration values to the experiment including, but not limited to:
 * the dataset path
 * The lossless encoding format of the evaluation input images
"""

import metrics

LOSSLESS_EXTENSION: str = ".png"

PROCEDURE_RESULTS_FILE: str = "procedure_results"
JPEG_EVAL_RESULTS_FILE: str = "jpeg_eval_results"
DATASET_PATH: str = "images/dataset/"
DATASET_COMPRESSED_PATH: str = "images/dataset_compressed/"

# Info flags
JXL_SUPPORTED_VERSIONS: tuple = ("0.6.1",)
CAVIF_RS_SUPPORTED_VERSIONS: tuple = ("1.3.4",)
AVIF_DECODE_SUPPORTED_VERSIONS: tuple = ("0.2.2",)
WEBP_SUPPORTED_VERSIONS: tuple = ("0.4.1",)

# FILENAME_KEYWORDS = id
MODALITY = 0
BODYPART = 1
COLORSPACE = 2
SAMPLES_PER_PIXEL = 3
BITS_PER_SAMPLE = 4
DEPTH = 5  # Number of frames in the image (> 1 if multi frame)

# Default settings per compression method
DEFAULTS = dict(
    jxl=dict(quality=1., effort=7),
    avif=dict(quality=80., speed=4),
    webp=dict(quality=75., effort=4)
)

# Quality metrics dict
# Serves as a "database" holding the
#   image quality evaluating metrics and the respective functions reference
# TODO (low prio.) use this on the procedure instead of hard coding the metrics
qmetrics = dict(
    mse=metrics.custom_mse, psnr=metrics.custom_psnr, ssim=metrics.ssim
)
