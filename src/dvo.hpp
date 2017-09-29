// Copyright 2016 Robert Maier, Technical University Munich
#ifndef DVO_H
#define DVO_H

#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>

#include "strided_range.hpp"
#include <thrust/inner_product.h>

#include <cuda_runtime.h>

#include "opencv2/gpu/gpu.hpp"
#include <cublas_v2.h>

#define NUM_STREAMS 8

extern cublasHandle_t handle;

typedef Eigen::Matrix<float, 6, 6> Mat6f;
typedef Eigen::Matrix<float, 6, 1> Vec6f;

class DVO
{
public:
    enum MinimizationAlgo
    {
        GaussNewton = 0,
        GradientDescent = 1,
        LevenbergMarquardt = 2
    };

    DVO();
    ~DVO();

    void init(int w, int h, const Eigen::Matrix3f &K);

    void buildPyramid(const cv::Mat &depth, const cv::Mat &gray, std::vector<cv::gpu::GpuMat> &depthPyramid, std::vector<cv::gpu::GpuMat> &grayPyramid);

    void align(const cv::Mat &depthRef, const cv::Mat &grayRef,
               const cv::Mat &depthCur, const cv::Mat &grayCur,
               Eigen::Matrix4f &pose);

    void align(const std::vector<cv::gpu::GpuMat> &depthRefPyramid, const std::vector<cv::gpu::GpuMat> &grayRefPyramid,
               const std::vector<cv::gpu::GpuMat> &depthCurPyramid, const std::vector<cv::gpu::GpuMat> &grayCurPyramid,
               Eigen::Matrix4f &pose);

private:
    cudaStream_t streams[NUM_STREAMS];

    cv::gpu::GpuMat downsampleGray(const cv::gpu::GpuMat &gray, int streamIdx);
    cv::gpu::GpuMat downsampleDepth(const cv::gpu::GpuMat &depth, int streamIdx);

    void convertSE3ToTf(const Vec6f &xi, Eigen::Matrix3f &rot, Eigen::Vector3f &t);
    void convertSE3ToTf(const Vec6f &xi, Eigen::Matrix4f &pose);
    void convertTfToSE3(const Eigen::Matrix3f &rot, const Eigen::Vector3f &t, Vec6f &xi);
    void convertTfToSE3(const Eigen::Matrix4f &pose, Vec6f &xi);

    cv::gpu::GpuMat convertToContGpuMat(const cv::Mat &m);

    void computeGradient(const cv::gpu::GpuMat &gray, cv::gpu::GpuMat &gradientx,cv::gpu::GpuMat &gradienty);
    float calculateError(float* residuals, int n);
    void calculateError(const cv::gpu::GpuMat &grayRef, const cv::gpu::GpuMat &depthRef,
                        const cv::gpu::GpuMat &grayCur, const cv::gpu::GpuMat &depthCur,
                        const Eigen::VectorXf &xi, const Eigen::Matrix3f &K,
                        float* residuals);


    void calculateMeanStdDev(float* residuals, float &mean, float &stdDev, int n);
    void computeAndApplyWeights(float* residuals, float* weights, int n);
    void applyWeights(const float* weights, float* residuals, int n);

    void deriveAnalytic(const cv::gpu::GpuMat &grayRef, const cv::gpu::GpuMat &depthRef,
                       const cv::gpu::GpuMat &grayCur, const cv::gpu::GpuMat &depthCur,
                       const cv::gpu::GpuMat &gradX_, const cv::gpu::GpuMat &gradY_,
                       const Eigen::VectorXf &xi, const Eigen::Matrix3f &K,
                       float* residuals, float* J);

    void compute_JtR(float* J, const float* residuals, Vec6f &b, int validRows);
    void compute_JtJ(const float* J, Mat6f &A, const float* weights, int validRows, bool useWeights);

    int numPyramidLevels_;
    std::vector<Eigen::Matrix3f> kPyramid_;
    std::vector<cv::Size> sizePyramid_;
    bool useWeights_;
    int numIterations_;

    std::vector<cv::gpu::GpuMat> gradX_;
    std::vector<cv::gpu::GpuMat> gradY_;
    std::vector<float*> d_J_;
    std::vector<float*> d_residuals_;
    std::vector<float*> d_weights_;

    MinimizationAlgo algo_;
};

#endif
