import PIL.Image
import cv2
import numpy as np
import pandas as pd
import rawpy as rawpy

import metrics

SHORT_MODE: bool = True
QUALITY_STEPS: int = 5

if __name__ == '__main__':
    # The purpose is to compress a set of uncompressed images in JPEG
    # Compression is performed using multiple quality configurations
    # For each quality configuration, compute the SSIM of the resulting image (versus the original)

    # Quality settings
    quality_parameters: np.ndarray = np.linspace(1, 100, QUALITY_STEPS)

    # How many files go through the evaluation
    file_count: int = 100
    if SHORT_MODE:
        file_count = 10

    # Compress using the above quality parameters
    # Save the compression ratio in a dataframe
    df = pd.DataFrame(data=dict(fname=[], original_size=[], compressed_size=[], CR=[], SSIM=[]))
    for file_i in range(file_count):
        file_path: str = "images/original/NEF/flat_" + str(file_i + 1).zfill(3) + ".NEF"
        file_name: str = ".".join(file_path.split("/")[-1].split(".")[:-1])
        for quality in quality_parameters:
            quality = int(quality)
            # Read input file
            # FIXME image seems too small. Should be way bigger than it is in img variable
            img = rawpy.imread(file_path)
            img = img.postprocess()
            # Compress input file
            comp_img = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            comp_img_bytes: bytes = np.array(comp_img[1]).tobytes()
            # Calculate compressed bitstream size
            cr = int(img.size / (len(comp_img_bytes) * 8))
            # TODO Calculate the SSIM between the images
            # Decode JPEG bitstream
            buffer: np.array = np.asarray(bytearray(comp_img_bytes), dtype=np.uint8)
            img_c = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            ssim = metrics.ssim(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
            )
            # Write to dataframe
            df = pd.concat([pd.DataFrame(dict(
                fname=[f"{file_name}_q{quality}"],
                original_size=[img.size],
                compressed_size=[len(comp_img_bytes) * 8],
                CR=[cr],
                SSIM=ssim
            )), df], ignore_index=True)
            # Report progress
            print(f"Progress:"
                  f"{round((file_i * quality_parameters.size + quality)/(file_count * quality_parameters.size), 2)}"
                  f"%", end="\r")
    df.to_csv("results.csv", index=False)
