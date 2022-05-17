# Image Compression Evaluation Pipeline

This is a set of scripts that perform a quantitative evaluation
over the most recent image compression formats, such as Cloudinary's
JPEG XL, Google's WebP and AOMediaCodec's AVIF, for the scope of
medical imaging.

## Table of contents
1. [Setting up](#setting-up)
   1. [Requirements](#requirements)
   2. [For x86_64 architectures](#for-x86_64-architectures)
   3. [Other Architectures](#other-architectures)
2. [Running](#running)

## Setting up

### Requirements
Currently, this software only works on *Linux* machines that have the
aforementioned codecs installed. See #set-up for how to set up
your machine.

### For x86_64 architectures

#### AVIF

First, install the [rust](https://rust-lang.org/tools/install) toolkit:
```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Secondly, install cavif-rs
```shell
cargo install cavif-rs
```

#### JPEG XL
Download the binaries for [x86_64-linux](https://github.com/libjxl/libjxl/releases/download/v0.6.1/jxl-linux-x86_64-static-v0.6.1.tar.gz).

Create a directory for the JPEG XL:
```shell
mkdir ~/jxl-linux-x86_64-static-v0.6.1
```

You will need to include the binaries into the PATH environment variable:

**If** you use bash:
```shell
tar -C ~/jxl-linux-x86_64-static-v0.6.1 -xzvf ~/Downloads/jxl-linux-x86_64-static-v0.6.1.tar.gz
echo 'export JXL="~/jxl-linux-x86_64-static-v0.6.1/tools"' >> ~/.bashrc
echo 'export PATH="$JXL:$PATH"' >> ~/.bashrc
```
If you use zshell, switch ~/.bashrc with ~/.zshrc instead.

#### WebP
Download WebP for [x86_64-linux](https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.2.1-linux-x86-64.tar.gz)
.
```shell
tar -C ~/libwebp-1.2.1-linux-x86-64/ -xzvf ~/Downloads/libwebp-1.2.1-linux-x86-64.tar.gz
echo 'export WEBP="~/libwebp-1.2.1-linux-x86-64/bin/"' >> ~/.bashrc
echo 'export PATH="$WEBP:$PATH"' >> ~/.bashrc
```
If you use zshell, switch ~/.bashrc with ~/.zshrc instead.

### Other architectures
If you have architectures other than x86_64,
download the appropriate binaries for
[JPEG XL](https://github.com/libjxl/libjxl/releases/tag/v0.6.1)
and [WebP](https://storage.googleapis.com/downloads.webmproject.org/releases/webp/index.html),
extract and append the binaries' paths to your PATH variable.


## Running

(Under Development)