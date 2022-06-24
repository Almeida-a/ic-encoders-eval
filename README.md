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
      2. [For x86_64 architectures](#for-x86_64-architectures)
      3. [Other Architectures](#other-architectures)
      4. [Dataset](#dataset)
   2. [Running](#running)

## About

### Codecs Evaluation Pipeline

This project is aimed to evaluate the most recent image compression formats.


### Concepts

#### Image Compression

...

#### DICOM Standard

...

## Usage Guide

### Setting up

#### Hardware Requirements
Currently, this software only works on *Linux* machines that have the
aforementioned codecs installed. See #set-up for how to set up
your machine.

#### For x86_64 architectures

##### AVIF

First, install the [rust](https://rust-lang.org/tools/install)
toolkit with:
```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Secondly, install cavif-rs
```shell
cargo install cavif-rs
```

##### JPEG XL
Download the binaries for [x86_64-linux](https://github.com/libjxl/libjxl/releases/download/v0.6.1/jxl-linux-x86_64-static-v0.6.1.tar.gz).

Create a directory for the JPEG XL:
```shell
mkdir ~/jxl-linux-x86_64-static-v0.6.1
```

You will need to include the binaries into the PATH environment variable:

**If** you use bash:
```shell
tar -C ~/jxl-linux-x86_64-static-v0.6.1 -xzvf ~/Downloads/jxl-linux-x86_64-static-v0.6.1.tar.gz
echo 'export JXL="$HOME/jxl-linux-x86_64-static-v0.6.1"' >> ~/.bashrc
echo 'export PATH="$JXL/tools:$PATH"' >> ~/.bashrc
```
If you use zshell, switch ~/.bashrc with ~/.zshrc.

##### WebP
Download WebP for [x86_64-linux](https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.2.1-linux-x86-64.tar.gz)
. Then run the following commands
**if** you use bash:
```shell
tar -C ~/libwebp-1.2.1-linux-x86-64/ -xzvf ~/Downloads/libwebp-1.2.1-linux-x86-64.tar.gz
echo 'export WEBP_HOME="$HOME/libwebp-1.2.1-linux-x86-64"' >> ~/.bashrc
echo 'export PATH="$WEBP_HOME/bin:$PATH"' >> ~/.bashrc
```
If you use zshell, switch ~/.bashrc with ~/.zshrc.

#### Other architectures
If you have architectures other than x86_64,
download the appropriate binaries for
[JPEG XL](https://github.com/libjxl/libjxl/releases/tag/v0.6.1)
and [WebP](https://storage.googleapis.com/downloads.webmproject.org/releases/webp/index.html),
extract and append the binaries' paths to your PATH variable following the example shown at
the [x86_64](#for-x86_64-architectures) section.

#### Dataset

In order to run the program, you will need a set of dicom files to serve as your experiment
dataset upon which the results will be based on. Create the following directory in the
project root dir, and dump the dicom dataset there.
```shell
mkdir dataset_dicom
```
WARNINGS:
* The aforementioned directory's name is not arbitrary, you need to name it
exactly as instructed! The program will not find the dicom files otherwise.

* The program will not find any dicom files inside sub-folders of the dataset_dicom directory.

### Running

(Under Development)
