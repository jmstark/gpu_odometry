Robust odometry estimation with visualization, CUDA-acceleration and ASUS Xtion 0601 support
=============================================


Usage:
First:
$ cd build-release 

Dataset Mode:
$ ./dvo_gpu ../data/

Kinect Mode:
$ ./dvo_gpu

=============================================


Program compilation:

First, make sure that you have the required dependencies present and pointed to in the CMakeLists.txt file.
Then:

$ cd build-release
$ cmake ..
$ make


=============================================

Custom Dependencies:


NOTE:
For building the project in the TUM CUDA-lab, the required dependencies can be found pre-compiled in:
/usr/prakt/s164/cudalab-deps/opencv-3.3.0/local-install (it is world-readable)
It is compiled against the custom OpenNI2 (found in the same parent folder) needed for the ASUS Xtion 0601. You only need to adapt the path to OpenCV in CMakeLists.txt.



The project depends on OpenNI2 from the 'develop' branch of https://github.com/occipital/OpenNI2 and a custom OpenCV 3.3.
OpenCV 3.3 requires, apart from the standard modules, support for cuda, cudalegacy, vtk and OpenNI2 compiled in.
Executing ./dl_and_build_local_opencv_cuda_enabled.sh takes care of compiling OpenCV with the respective settings and installing it into ~/cudalab-deps/opencv-3.3.0/local-install.
Make sure to read and understand it before you run it. Do so at your own risk.

Before compiling OpenCV, you need to build OpenNI2 (Instructions for doing so on a TUM CUDA lab PC without root access):
Clone develop branch of https://github.com/occipital/OpenNI2/tree/develop
$ apt-get download libudev-dev
$ dpkg -x libudev-dev_229-4ubuntu19_amd64.deb libudev-install/
Copy libudev.dev into OpenNI2/ThirdParty/PSCommon/XnLib/Source/Linux
OpenNI2/ThirdParty/PSCommon/XnLib/Source/Linux/XnLinuxUSB.cpp change <> to "" around libudev.h: #include "libudev.h"
Copy libudev.so into OpenNI2/Bin/x64-Release/libudev.so
$ make
Go into folder Packaging
Comment out the whole python file ../Source/Documentation/Runme.py with ''' at end and beginning so that it does nothing (it doesnt work but isn't needed).
Back in Packaging folder, execute ReleaseVersion.py x64
Unpack the generated archive in Final into your desired install directory
Edit the file OpenNI-Linux-x64-2.3/Redist/OpenNI2/Drivers: Enable image registration
Comment out the root check and the udev copy in the install script
Execute the install script
Source the generated file: $ . OpenNIDevEnvironment
Then you can go ahead and execute the OpenCV build script.

