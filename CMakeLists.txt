# Copyright (c) 2015 Robert Maier
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(DvoGpu C CXX)

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
FIND_PACKAGE(OpenCV 2.4.8 COMPONENTS opencv_core opencv_highgui opencv_imgproc viz REQUIRED PATHS "/usr/local/")

# CUDA
FIND_PACKAGE(CUDA REQUIRED)
option(CUDA_USE_STATIC_CUDA_RUNTIME "Use the static version of the CUDA runtime library if available" FALSE)

#==============================================================================
# Application
#==============================================================================

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O1 -funroll-loops -D_FORCE_INLINES -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF" )
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --profile -lineinfo")

# preprocessor define: source directory
ADD_DEFINITIONS(-DDVO_SOURCE_DIR=${PROJECT_SOURCE_DIR})

# applications
FILE(GLOB incs "src/*.hpp")
FILE(GLOB srcs "src/*.cpp")
FILE(GLOB srcscu "src/*.cu")

CUDA_ADD_EXECUTABLE(dvo_gpu ${incs} ${srcs} ${srcscu})
TARGET_LINK_LIBRARIES(dvo_gpu ${OpenCV_LIBS} ${CUDA_CUBLAS_LIBRARIES})
