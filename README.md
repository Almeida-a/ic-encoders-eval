# Image Compression Evaluation Pipeline

This is a set of scripts that perform a quantitative evaluation
over the most recent image compression formats, such as Cloudinary's
JPEG XL, Google's WebP and AOMediaCodec's AVIF, for the scope of
medical imaging.

## Table of contents
1. [About](#about)
2. [Usage Guide](#usage-guide)
   1. [Setting up](#setting-up)
      1. [Hardware Requirements](#hardware-requirements)
      2. [Dependencies](#dependencies)
      3. [Dataset](#dataset)
   2. [Running](#running)
      1. [Main Pipeline](#main-pipeline)
      2. [JPEG Pipeline](#jpeg-pipeline)
3. [Feedback](#feedback)
4. [Licence](#license)

## About

### Codecs Evaluation Pipeline

This project is aimed to evaluate the most recent image compression formats.

## Usage Guide

### Setting up

#### Hardware Requirements
Currently, this software only works on *Linux* machines that have the
aforementioned codecs installed. See #set-up for how to set up
your machine.

#### Dependencies

Ensure you have `python3.10` installed, as well as the `pip` module.

Install the dicom toolkit software:
```shell
sudo apt install dcmtk
```
Install the required libraries to run the pipeline:
```shell
pip install -r requirements.txt
```

##### AVIF

First, install the [rust](https://rust-lang.org/tools/install)
toolkit with:
```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Secondly, install [cavif-rs](https://github.com/kornelski/cavif-rs)
```shell
cargo install cavif
```

##### JPEG XL
Install from source using:
```shell
sudo ./build_jxl.sh
```

##### WebP
Install through apt:
```shell
sudo apt install webp
```

#### Dataset

In order to run the program, you will need a set of dicom files to serve as your experiment
dataset upon which the results will be based on. Copy them to the folder `images/dataset_dicom`.

### Running

#### Main pipeline

To execute the main pipeline benchmark, run:
```shell
python3 dicom_parser.py  # pre-processing the dataset to (a)png image files
python3 procedure.py  # run encoders and captures metrics
```

Bear in mind that having a big dicom dataset can yield a long script execution time.

Two files `procedure_results.{json,csv}` are generated containing raw data with the benchmark results.


To generate the results in charts form, run:
```shell
python3 visualize_data.py
```
You can edit the script at the bottom in order to generate the results based on what you want to see.

#### JPEG Pipeline
To perform the benchmarking itself, run:
```shell
python3 jpeg_eval.py
```

Bear in mind that having a big dicom dataset can yield a long script execution time.

Two files `procedure_results.{json,csv}` are generated containing raw data with the benchmark results.

## Feedback
If you find a bug or want to send feedback, please feel free to open an issue or pull request!

## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/Almeida-a/ic-encoders-eval/blob/master/LICENSE) file for details.

