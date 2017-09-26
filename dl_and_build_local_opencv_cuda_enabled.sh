#!/bin/sh

THIRD_P_PATH="$(pwd)/third_party/include/"
REL_BUILD_PATH="build/"
REL_INST_PATH="local-install/"

#https://github.com/opencv/opencv/archive/2.4.13.3.zip
DL_FILENAME="2.4.13.3"
DL_FILENAME_EXT="zip"
DL_PATH="https://github.com/opencv/opencv/archive/"
FOLDERNAME_UNPACKED="opencv-${DL_FILENAME}"

cd ${THIRD_P_PATH}
rm -f "${DL_FILENAME}.${DL_FILENAME_EXT}"
rm -rf "${FOLDERNAME_UNPACKED}"
wget "${DL_PATH}${DL_FILENAME}.${DL_FILENAME_EXT}"
unzip -q "${DL_FILENAME}.${DL_FILENAME_EXT}"
rm "${DL_FILENAME}.${DL_FILENAME_EXT}"
cd "${FOLDERNAME_UNPACKED}"
ABS_INSTALL_PATH="$(pwd)/${REL_INST_PATH}"
mkdir $REL_BUILD_PATH
cd $REL_BUILD_PATH
cmake -D CMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=OFF -D CMAKE_INSTALL_PREFIX="${ABS_INSTALL_PATH}" ..
make
make install
