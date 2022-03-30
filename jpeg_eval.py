import PIL.Image
import cv2
import numpy as np
import pandas as pd
import rawpy as rawpy

import metrics

QUALITY_STEPS: int = 5


def images() -> str:
    """
    Generator function
    :return: Sequence of image files path to be processed in experience
    """
    prefix: str = "images/original/NEF/flat_"

    for i in range(100):
        yield prefix + str(i+1).zfill(3) + ".NEF"


def main():
    # The purpose is to compress a set of uncompressed images in JPEG
    # Compression is performed using multiple quality configurations
    # For each quality configuration, compute the SSIM of the resulting image (versus the original)
    # Quality settings
    quality_parameters: np.ndarray = np.linspace(1, 100, QUALITY_STEPS)

    # Compress using the above quality parameters
    # Save the compression ratio in a dataframe
    df = pd.DataFrame(data=dict(fname=[], original_size=[], compressed_size=[], CR=[], SSIM=[]))
    for file_path in images():
        file_name: str = ".".join(file_path.split("/")[-1].split(".")[:-1])
        for quality in quality_parameters:
            quality = int(quality)

            # Read input file
            img = rawpy.imread(file_path)
            img = img.postprocess()

            # Compress input file
            comp_img = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            comp_img_bytes: bytes = np.array(comp_img[1]).tobytes()

            # Calculate compressed bitstream size
            cr = int(img.size / (len(comp_img_bytes) * 8))

            # Decode JPEG bitstream
            buffer: np.array = np.asarray(bytearray(comp_img_bytes), dtype=np.uint8)
            img_c = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

            # Calculate the SSIM between the images
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
    df.to_csv("results.csv", index=False)


if __name__ == '__main__':
    main()
