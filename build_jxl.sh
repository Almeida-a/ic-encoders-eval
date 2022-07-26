#!/bin/bash -ex
# Refer to https://github.com/varnav/makejxl/blob/main/build_cjxl.sh

if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

if [ "$(grep -c avx2 /proc/cpuinfo)" == 0 ]
  then echo "AVX2 support required"
  exit
fi
set -ex

apt-get update
apt-get -y install python3-pip git wget ca-certificates cmake pkg-config libbrotli-dev libgif-dev libjpeg-dev libopenexr-dev libpng-dev libwebp-dev clang

cd /tmp/
rm -rf libjxl
git clone https://github.com/libjxl/libjxl.git --recursive
cd libjxl
git checkout ae95f451e0d23a209fa22efac4771969a23dac99  # Locks the jxl version - delete to build from latest
rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF ..
cmake --build . -- -j"$(nproc)"

cp tools/cjxl /usr/local/bin/
cp tools/djxl /usr/local/bin/