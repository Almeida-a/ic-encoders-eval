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
      3. [For x86_64 versions](#for-x86_64-versions)
      4. [Other instruction set versions](other-instruction-set-versions)
      5. [Dataset](#dataset)
   2. [Running](#running)

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

#### For x86_64 versions

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
Currently, I use a recent build from the [master](https://github.com/libjxl/libjxl) branch, not present on the releases.

Use a binary from version v0.7.0, build [ae95f45](https://github.com/libjxl/libjxl/commit/ae95f451e0d23a209fa22efac4771969a23dac99) (build from source).

Nextly, add `cjxl`/`djxl` binaries to your `PATH` variable.

##### WebP
Run the following commands
**if** you use bash:
```shell
curl https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.2.1-linux-x86-64.tar.gz --output ~/Downloads/libwebp-1.2.1-linux-x86-64.tar.gz
tar -C ~ -xzvf ~/Downloads/libwebp-1.2.1-linux-x86-64.tar.gz
echo 'export WEBP_HOME="$HOME/libwebp-1.2.1-linux-x86-64"' >> ~/.bashrc
echo 'export PATH="$WEBP_HOME/bin:$PATH"' >> ~/.bashrc
```
If you use zshell, switch ~/.bashrc with ~/.zshrc.

#### Other instruction set versions
If you have versions other than x86_64,
download the appropriate binaries for
[JPEG XL](https://github.com/libjxl/libjxl) (you need to build from source)
and [WebP](https://storage.googleapis.com/downloads.webmproject.org/releases/webp/index.html),
extract and append the binaries' paths to your PATH variable following the example shown at
the [x86_64](#for-x86_64-architectures) section.

#### Dataset

In order to run the program, you will need a set of dicom files to serve as your experiment
dataset upon which the results will be based on. Copy them to the folder `images/dataset_dicom`.

### Running

(Under Development)
