# Copyright (c) 2015 Robert Maier
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(DvoCpu C CXX)

# RelWithDebInfo, Release or Debug
SET(CMAKE_BUILD_TYPE RelWithDebInfo FORCE)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#==============================================================================
# Dependencies
#==============================================================================

# include sophus and Eigen
INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/third_party/include/ )

# Eigen
SET(EIGEN_ROOT ${PROJECT_SOURCE_DIR}/third_party/include/)
FIND_PACKAGE(Eigen REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})

# OpenCV
FIND_PACKAGE(OpenCV 2.4.8 COMPONENTS opencv_core opencv_highgui opencv_imgproc REQUIRED)

# CUDA
FIND_PACKAGE(CUDA REQUIRED)

#==============================================================================
# Application
#==============================================================================

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O1 -funroll-loops" )

# preprocessor define: source directory
ADD_DEFINITIONS(-DDVO_SOURCE_DIR=${PROJECT_SOURCE_DIR})

# applications
FILE(GLOB incs "src/*.hpp")
FILE(GLOB srcs "src/*.cpp")

CUDA_ADD_EXECUTABLE(dvo_cpu ${incs} ${srcs} )
TARGET_LINK_LIBRARIES(dvo_cpu ${OpenCV_LIBS})

