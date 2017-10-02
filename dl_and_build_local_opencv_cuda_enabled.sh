#!/bin/sh

# we set this path to somewhere outside the project so that remote building doesn't transfer our whole
# OpenCV build directoy. That path must be supplied to CMakeLists.txt so it can find the libraries.
THIRD_P_PATH="${HOME}/cudalab-deps/"
REL_BUILD_PATH="build/"
REL_INST_PATH="local-install/"

#https://github.com/opencv/opencv/archive/3.3.0.zip
DL_FILENAME="3.3.0"
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

#rm -f "${DL_FILENAME}.${DL_FILENAME_EXT}"
#rm -rf "${FOLDERNAME_UNPACKED}"
#wget "${DL_PATH}${DL_FILENAME}.${DL_FILENAME_EXT}"
#unzip -q "${DL_FILENAME}.${DL_FILENAME_EXT}"
#rm "${DL_FILENAME}.${DL_FILENAME_EXT}"
cd "${FOLDERNAME_UNPACKED}"
mkdir $REL_BUILD_PATH
cd $REL_BUILD_PATH
cmake -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0/ \
-DCMAKE_INSTALL_PREFIX="${ABS_INSTALL_PATH}" \
-DWITH_CUDA=ON -DWITH_CUDALEGACY=ON -DBUILD_opencv_java=OFF \
-DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF \
-DBUILD_opencv_stitching=OFF -DBUILD_opencv_superres=OFF -DBUILD_opencv_ts=OFF \
-DBUILD_opencv_videostab=OFF \
-DWITH_VTK=ON -DWITH_OPENNI2=ON \
..
make -j2
make install


# -DOPENNI_PRIME_SENSOR_MODULE_BIN_DIR=/usr/prakt/s164/cudalab-deps/OpenNI2/Bin/x64-Release/OpenNI2/Drivers \
#-DOPENNI2_LIB_DIR=/home/josef/git/cudalab/third_party/OpenNI2/Bin/x64-Release/ \
#-DOPENNI2_INCLUDE_DIR=/home/josef/git/cudalab/third_party/OpenNI2/Include/ \
#-DOPENNI2_INSTALL_PATH=/home/josef/git/cudalab/third_party/OpenNI2/ \


#-DBUILD_opencv_core=ON -DBUILD_opencv_flann=ON -DBUILD_opencv_imgproc=ON -DBUILD_opencv_highgui=ON \
#-DBUILD_opencv_features2d=ON -DBUILD_opencv_calib3d=ON -DBUILD_opencv_ml=ON -DBUILD_opencv_video=ON \
#-DBUILD_opencv_legacy=ON -DBUILD_opencv_objdetect=ON -DBUILD_opencv_photo=ON -DBUILD_opencv_gpu=ON \