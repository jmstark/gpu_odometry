#!/bin/sh

# we set this path to somewhere outside the project so that remote building doesn't transfer our whole
# OpenCV build directoy. That path must be supplied to CMakeLists.txt so it can find the libraries.
THIRD_P_PATH="${HOME}/cudalab-deps/"
REL_BUILD_PATH="build/"
REL_INST_PATH="local-install/"

#https://github.com/opencv/opencv/archive/2.4.13.3.zip
DL_FILENAME="2.4.13.3"
DL_FILENAME_EXT="zip"
DL_PATH="https://github.com/opencv/opencv/archive/"
FOLDERNAME_UNPACKED="opencv-${DL_FILENAME}"

mkdir -p ${THIRD_P_PATH}
cd ${THIRD_P_PATH}

ABS_INSTALL_PATH="$(pwd)/${FOLDERNAME_UNPACKED}/${REL_INST_PATH}"


if [ -d "${ABS_INSTALL_PATH}" ]; then 
	echo "Found local OpenCV install ${ABS_INSTALL_PATH}" 
	exit 0
else
	echo "Could not find local OpenCV install ${ABS_INSTALL_PATH}; commencing build."
fi

rm -f "${DL_FILENAME}.${DL_FILENAME_EXT}"
rm -rf "${FOLDERNAME_UNPACKED}"
wget "${DL_PATH}${DL_FILENAME}.${DL_FILENAME_EXT}"
unzip -q "${DL_FILENAME}.${DL_FILENAME_EXT}"
rm "${DL_FILENAME}.${DL_FILENAME_EXT}"
cd "${FOLDERNAME_UNPACKED}"
mkdir $REL_BUILD_PATH
cd $REL_BUILD_PATH
#cmake -D CMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=OFF -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0/ \
cmake -D CMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=ON -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0/ \
-D CMAKE_INSTALL_PREFIX="${ABS_INSTALL_PATH}" \
-DBUILD_opencv_core=ON -DBUILD_opencv_flann=ON -DBUILD_opencv_imgproc=ON -DBUILD_opencv_highgui=ON \
-DBUILD_opencv_features2d=ON -DBUILD_opencv_calib3d=ON -DBUILD_opencv_ml=ON -DBUILD_opencv_video=ON \
-DBUILD_opencv_legacy=ON -DBUILD_opencv_objdetect=ON -DBUILD_opencv_photo=ON -DBUILD_opencv_gpu=ON \
-DBUILD_opencv_ocl=OFF -DBUILD_opencv_nonfree=OFF -DBUILD_opencv_contrib=OFF -DBUILD_opencv_java=OFF \
-DBUILD_opencv_python=OFF -DBUILD_opencv_stitching=OFF -DBUILD_opencv_superres=OFF -DBUILD_opencv_ts=OFF \
-DBUILD_opencv_videostab=OFF \
..
make -j
make install
