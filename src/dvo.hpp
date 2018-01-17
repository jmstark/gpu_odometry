// Copyright 2016 Robert Maier, Technical University Munich
#ifndef DVO_H
#define DVO_H

#ifndef WIN64
#define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>

#include "strided_range.hpp"
#include <thrust/inner_product.h>

#include <cuda_runtime.h>

#include <cublas_v2.h>

//Streams for overlapped GPU transfers/computations
#define NUM_STREAMS 8

extern cublasHandle_t handle;

typedef Eigen::Matrix<float, 6, 6> Mat6f;
typedef Eigen::Matrix<float, 6, 1> Vec6f;

class DVO
{
public:
	enum MinimizationAlgo
	{
		GaussNewton = 0, GradientDescent = 1, LevenbergMarquardt = 2
	};

	DVO();
	~DVO();

	void init(int w, int h, const Eigen::Matrix3f &K);

	void buildPyramid(const cv::Mat &depth, const cv::Mat &gray,
			std::vector<cv::cuda::GpuMat> &depthPyramid,
			std::vector<cv::cuda::GpuMat> &grayPyramid);

	void align(const cv::Mat &depthRef, const cv::Mat &grayRef,
			const cv::Mat &depthCur, const cv::Mat &grayCur,
			Eigen::Matrix4f &pose);

	void align(const std::vector<cv::cuda::GpuMat> &depthRefPyramid,
			const std::vector<cv::cuda::GpuMat> &grayRefPyramid,
			const std::vector<cv::cuda::GpuMat> &depthCurPyramid,
			const std::vector<cv::cuda::GpuMat> &grayCurPyramid,
			Eigen::Matrix4f &pose);

private:
	cudaStream_t streams[NUM_STREAMS];

	cv::cuda::GpuMat downsampleGray(const cv::cuda::GpuMat &gray,
			int streamIdx);
	cv::cuda::GpuMat downsampleDepth(const cv::cuda::GpuMat &depth,
			int streamIdx);

	void convertSE3ToTf(const Vec6f &xi, Eigen::Matrix3f &rot,
			Eigen::Vector3f &t);
	void convertSE3ToTf(const Vec6f &xi, Eigen::Matrix4f &pose);
	void convertTfToSE3(const Eigen::Matrix3f &rot, const Eigen::Vector3f &t,
			Vec6f &xi);
	void convertTfToSE3(const Eigen::Matrix4f &pose, Vec6f &xi);

	/**
	 * Convert conventional cv::Mat to cv::cuda::Mat for direct processing on GPU
	 */
	cv::cuda::GpuMat convertToContGpuMat(const cv::Mat &m);

	void computeGradient(const cv::cuda::GpuMat &gray,
			cv::cuda::GpuMat &gradientx, cv::cuda::GpuMat &gradienty);
	float calculateError(float* residuals, int n);
	void calculateError(const cv::cuda::GpuMat &grayRef,
			const cv::cuda::GpuMat &depthRef, const cv::cuda::GpuMat &grayCur,
			const cv::cuda::GpuMat &depthCur, const Eigen::VectorXf &xi,
			const Eigen::Matrix3f &K, float* residuals);

	void calculateMeanStdDev(float* residuals, float &mean, float &stdDev,
			int n);
	void computeWeights(float* residuals, float* weights, int n);

	void deriveAnalytic(const cv::cuda::GpuMat &grayRef,
			const cv::cuda::GpuMat &depthRef, const cv::cuda::GpuMat &grayCur,
			const cv::cuda::GpuMat &depthCur, const cv::cuda::GpuMat &gradX_,
			const cv::cuda::GpuMat &gradY_, const Eigen::VectorXf &xi,
			const Eigen::Matrix3f &K, float* d_residuals, bool useWeight,
			float* d_weights, float* d_J, Mat6f &A, Vec6f &b);

	int numPyramidLevels_;
	std::vector<Eigen::Matrix3f> kPyramid_;
	std::vector<cv::Size> sizePyramid_;
	bool useWeights_;
	int numIterations_;

	std::vector<cv::cuda::GpuMat> gradX_;
	std::vector<cv::cuda::GpuMat> gradY_;
	std::vector<float*> d_J_;
	std::vector<float*> d_residuals_;
	std::vector<float*> d_weights_;

	MinimizationAlgo algo_;
};

#endif
