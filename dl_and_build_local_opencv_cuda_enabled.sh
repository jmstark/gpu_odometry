#!/bin/sh

3RD_P_PATH="$(pwd)/third_party/include/"
REL_BUILD_PATH="/build/"
REL_INST_PATH="/install/"

DL_FILENAME="opencv-2.4.13.3"
DL_FILENAME_EXT="zip"
DL_PATH="https://github.com/opencv/opencv/archive/"


cd $3RD_P_PATH
wget "${DL_PATH}${DL_FILENAME}.${DL_FILENAME_EXT}"
unzip "${DL_FILENAME}.${DL_FILENAME_EXT}"
rm "${DL_FILENAME}.${DL_FILENAME_EXT}"
cd "${DL_FILENAME}"
ABS_INSTALL_PATH="$(pwd)/${REL_INST_PATH}"
mkdir $REL_BUILD_PATH
cd $REL_BUILD_PATH
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX="${ABS_INSTALL_PATH}"
make -j
make install
