// Copyright 2016 Robert Maier, Technical University Munich
#include "dvo.hpp"
#include "helper.h"

#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <stdio.h>
#include <ctime>

#include "helper.h"

#include <Eigen/Cholesky>
#include <sophus/se3.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>
#include <thrust/count.h>
#include <cublas_v2.h>

#include <math.h>
#include <thrust/execution_policy.h>

#define HUBER
//#define STUDENT_T
//#define CAUCHY

__constant__ float d_t[3];
__constant__ float d_K[9];
__constant__ float d_rotMat[9];

DVO::DVO() :
		numPyramidLevels_(5), useWeights_(true), numIterations_(500), algo_(
				GaussNewton)
{
	for (int i = 0; i < NUM_STREAMS; i++)
		cudaStreamCreate(&streams[i]);
}

DVO::~DVO()
{
	for (int i = 0; i < numPyramidLevels_; ++i)
	{
		cudaFree(d_J_[i]);
		CUDA_CHECK;
		cudaFree(d_residuals_[i]);
		CUDA_CHECK;
		cudaFree(d_weights_[i]);
		CUDA_CHECK;
	}
	for (int i = 0; i < NUM_STREAMS; i++)
		cudaStreamDestroy(streams[i]);
}

void DVO::init(int w, int h, const Eigen::Matrix3f &K)
{
	// pyramid level size
	int wDown = w;
	int hDown = h;
	int n = wDown * hDown;
	sizePyramid_.push_back(cv::Size(wDown, hDown));

	// gradients
	cv::cuda::GpuMat gradX = cv::cuda::createContinuous(h, w, CV_32FC1);
	gradX_.push_back(gradX);
	cv::cuda::GpuMat gradY = cv::cuda::createContinuous(h, w, CV_32FC1);
	gradY_.push_back(gradY);

	// Jacobian
	float* J;
	cudaMalloc(&J, sizeof(float) * n * 6);
	CUDA_CHECK;
	d_J_.push_back(J);
	// residuals
	float* d_residuals;
	cudaMalloc(&d_residuals, sizeof(float) * n);
	CUDA_CHECK;
	d_residuals_.push_back(d_residuals);
	// per-residual weights
	float* weights;
	cudaMalloc(&weights, sizeof(float) * n);
	CUDA_CHECK;
	d_weights_.push_back(weights);

	// camera matrix
	kPyramid_.push_back(K);

	for (int i = 1; i < numPyramidLevels_; ++i)
	{
		// pyramid level size
		wDown = wDown / 2;
		hDown = hDown / 2;
		int n = wDown * hDown;
		sizePyramid_.push_back(cv::Size(wDown, hDown));

		// gradients
		cv::cuda::GpuMat gradXdown = cv::cuda::createContinuous(hDown, wDown,
				CV_32FC1);
		gradX_.push_back(gradXdown);
		cv::cuda::GpuMat gradYdown = cv::cuda::createContinuous(hDown, wDown,
				CV_32FC1);
		gradY_.push_back(gradYdown);

		// Jacobian
		float* J;
		cudaMalloc(&J, sizeof(float) * n * 6);
		CUDA_CHECK;
		d_J_.push_back(J);
		// residuals
		float* d_residuals;
		cudaMalloc(&d_residuals, sizeof(float) * n);
		CUDA_CHECK;
		d_residuals_.push_back(d_residuals);
		// per-residual weights
		float* weights;
		cudaMalloc(&weights, sizeof(float) * n);
		CUDA_CHECK;
		d_weights_.push_back(weights);

		// downsample camera matrix
		Eigen::Matrix3f kDown = kPyramid_[i - 1];
		kDown(0, 2) += 0.5f;
		kDown(1, 2) += 0.5f;
		kDown.topLeftCorner(2, 3) = kDown.topLeftCorner(2, 3) * 0.5f;
		kDown(0, 2) -= 0.5f;
		kDown(1, 2) -= 0.5f;
		kPyramid_.push_back(kDown);
		//std::cout << "Camera matrix (level " << i << "): " << kDown << std::endl;
	}
}

void DVO::convertSE3ToTf(const Vec6f &xi, Eigen::Matrix3f &rot,
		Eigen::Vector3f &t)
{
	// rotation
	Sophus::SE3f se3 = Sophus::SE3f::exp(xi);
	Eigen::Matrix4f mat = se3.matrix();
	rot = mat.topLeftCorner(3, 3);
	t = mat.topRightCorner(3, 1);
}

void DVO::convertSE3ToTf(const Vec6f &xi, Eigen::Matrix4f &pose)
{
	Sophus::SE3f se3 = Sophus::SE3f::exp(xi);
	pose = se3.matrix();
}

void DVO::convertTfToSE3(const Eigen::Matrix3f &rot, const Eigen::Vector3f &t,
		Vec6f &xi)
{
	Sophus::SE3f se3(rot, t);
	xi = Sophus::SE3f::log(se3);
}

void DVO::convertTfToSE3(const Eigen::Matrix4f &pose, Vec6f &xi)
{
	Eigen::Matrix3f rot = pose.topLeftCorner(3, 3);
	Eigen::Vector3f t = pose.topRightCorner(3, 1);
	convertTfToSE3(rot, t, xi);
}

__global__ void downsampleGrayKernel(float* out, int w, int h, float* in)
{
	//Compute index
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int wDown = w / 2;
	int hDown = h / 2;
	//Do bounds check
	if (x < wDown && y < hDown && z < 1)
	{
		//downsample
		float sum = 0.0f;
		sum += in[2 * y * w + 2 * x] * 0.25f;
		sum += in[2 * y * w + 2 * x + 1] * 0.25f;
		sum += in[(2 * y + 1) * w + 2 * x] * 0.25f;
		sum += in[(2 * y + 1) * w + 2 * x + 1] * 0.25f;
		out[y * wDown + x] = sum;
	}
}

cv::cuda::GpuMat DVO::downsampleGray(const cv::cuda::GpuMat &gray,
		int streamIdx)
{
	float * d_in, *d_out;
	int w = gray.cols;
	int h = gray.rows;
	int wDown = w / 2;
	int hDown = h / 2;
	d_in = (float*) gray.data;

	cv::cuda::GpuMat grayDown = cv::cuda::createContinuous(hDown, wDown,
			gray.type());
	d_out = (float*) grayDown.data;

	dim3 block = dim3(64, 8, 1);
	dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y,
			1);
	downsampleGrayKernel<<<grid, block, 0, streams[streamIdx]>>>(d_out, w, h,
			d_in);

	return grayDown;
}

__global__ void downsampleDepthKernel(float* out, int w, int h, float* in)
{
	//Compute index
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int wDown = w / 2;
	int hDown = h / 2;
	//Do bounds check
	if (x < wDown && y < hDown && z < 1)
	{
		//Downsample
		float d0 = in[2 * y * w + 2 * x];
		float d1 = in[2 * y * w + 2 * x + 1];
		float d2 = in[(2 * y + 1) * w + 2 * x];
		float d3 = in[(2 * y + 1) * w + 2 * x + 1];

		int cnt = 0;
		float sum = 0.0f;
		if (d0 != 0.0f)
		{
			sum += 1.0f / d0;
			++cnt;
		}
		if (d1 != 0.0f)
		{
			sum += 1.0f / d1;
			++cnt;
		}
		if (d2 != 0.0f)
		{
			sum += 1.0f / d2;
			++cnt;
		}
		if (d3 != 0.0f)
		{
			sum += 1.0f / d3;
			++cnt;
		}

		if (cnt > 0)
		{
			float dInv = sum / float(cnt);
			if (dInv != 0.0f)
			{
				out[y * wDown + x] = 1.0f / dInv;
				return;
			}
		}
		//set otherwise uninitialized pixel if we did not enter the inner if-block
		out[y * wDown + x] = 0.0f;
	}
}

cv::cuda::GpuMat DVO::downsampleDepth(const cv::cuda::GpuMat &depth,
		int streamIdx)
{

	float * d_in, *d_out;
	int w = depth.cols;
	int h = depth.rows;
	int wDown = w / 2;
	int hDown = h / 2;
	d_in = (float*) depth.data;

	cv::cuda::GpuMat depthDown = cv::cuda::createContinuous(hDown, wDown,
			depth.type());
	d_out = (float*) depthDown.data;

	dim3 block = dim3(64, 8, 1);
	dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y,
			1);
	downsampleDepthKernel<<<grid, block, 0, streams[streamIdx]>>>(d_out, w, h,
			d_in);

	return depthDown;

}

__global__ void computeGradientKernel(float* outx, float *outy, const float* in,
		int w, int h)
{
	//Compute index
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	//Do bounds check
	if (x < w && y < h)
	{
		//if we are out of the specified range but still inside the frame, we need to set
		//the pixel anyway (analog to pre-initialization in the sequential code)
		outx[y * w + x] = 0.0f;
		outy[y * w + x] = 0.0f;

		float v0 = 0.0f;
		float v1 = 0.0f;

		// Along y direction
		if ((y - 1) >= 0 && (y + 1) < h)
		{
			v0 = in[(y - 1) * w + x];
			v1 = in[(y + 1) * w + x];
			outy[y * w + x] = 0.5f * (v1 - v0);
		}
		// Along x direction
		if ((x - 1) >= 0 && (x + 1) < w)
		{
			v0 = in[y * w + (x - 1)];
			v1 = in[y * w + (x + 1)];
			outx[y * w + x] = 0.5f * (v1 - v0);
		}
	}

}

void DVO::computeGradient(const cv::cuda::GpuMat &gray,
		cv::cuda::GpuMat &gradientx, cv::cuda::GpuMat &gradienty)
{

	// compute gradient manually using finite differences
	int w = gray.cols;
	int h = gray.rows;
	const float* d_ptrIn = (const float*) gray.data;
	gradientx.setTo(0);
	gradienty.setTo(0);
	float* d_ptrOutx = (float*) gradientx.data;
	float* d_ptrOuty = (float*) gradienty.data;
	dim3 block = dim3(64, 8, 1);
	dim3 grid = dim3((w + 1 + block.x) / block.x, (h + block.y) / block.y, 1);
	computeGradientKernel<<<grid, block>>>(d_ptrOutx, d_ptrOuty, d_ptrIn, w, h);
}

/**
 * Indicate if elements are valid, for aggregated counting of them
 */
struct is_nonzero: public thrust::unary_function<float, bool>
{
	__host__ __device__
	bool operator()(float x)
	{
		return x != 0.0f;
	}
};

/**
 * Square elements
 */
struct squareop: std::unary_function<float, float>
{
	__host__ __device__ float operator()(float data)
	{
		return data * data;
	}
};

/**
 * Error calculation using thrust GPU reduction
 * @param d_residuals
 * @param n number of elements
 * @return error
 */
float DVO::calculateError(float* d_residuals, int n)
{
	float error = 0.0f;

	thrust::device_ptr<float> dp_residuals = thrust::device_pointer_cast(
			d_residuals);

	//Count valid elements
	int numValid = thrust::count_if(dp_residuals, dp_residuals + n,
			is_nonzero());
	//square valid elements
	error = thrust::transform_reduce(dp_residuals, dp_residuals + n, squareop(),
			0.0f, thrust::plus<float>());

	//average
	if (numValid > 0)
		error = error / static_cast<float>(numValid);

	return error;
}

/**
 * Interpolate sub-pixels
 * @return interpolated value
 */
__host__ __device__ float d_interpolate(const float* ptrImgIntensity, float x,
		float y, int w, int h)
{
	float valCur = nan("");

#if 0
	// direct lookup, no interpolation
	int x0 = static_cast<int>(x + 0.5f);
	int y0 = static_cast<int>(y + 0.5f);
	if (x0 >= 0 && x0 < w && y0 >= 0 && y0 < h)
	valCur = ptrImgIntensity[y0*w + x0];
#else
	//bilinear interpolation
	int x0 = static_cast<int>(x);
	int y0 = static_cast<int>(y);
	int x1 = x0 + 1;
	int y1 = y0 + 1;

	float x1_weight = x - static_cast<float>(x0);
	float y1_weight = y - static_cast<float>(y0);
	float x0_weight = 1.0f - x1_weight;
	float y0_weight = 1.0f - y1_weight;

	if (x0 < 0 || x0 >= w)
		x0_weight = 0.0f;
	if (x1 < 0 || x1 >= w)
		x1_weight = 0.0f;
	if (y0 < 0 || y0 >= h)
		y0_weight = 0.0f;
	if (y1 < 0 || y1 >= h)
		y1_weight = 0.0f;
	float w00 = x0_weight * y0_weight;
	float w10 = x1_weight * y0_weight;
	float w01 = x0_weight * y1_weight;
	float w11 = x1_weight * y1_weight;

	float sumWeights = w00 + w10 + w01 + w11;
	float sum = 0.0f;
	if (w00 > 0.0f)
		sum += ptrImgIntensity[y0 * w + x0] * w00;
	if (w01 > 0.0f)
		sum += ptrImgIntensity[y1 * w + x0] * w01;
	if (w10 > 0.0f)
		sum += ptrImgIntensity[y0 * w + x1] * w10;
	if (w11 > 0.0f)
		sum += ptrImgIntensity[y1 * w + x1] * w11;

	if (sumWeights > 0.0f)
		valCur = sum / sumWeights;
#endif

	return valCur;
}

__global__ void g_residualKernel(const float* d_ptrGrayRef,
		const float* d_ptrDepthRef, const float* d_ptrGrayCur, float fx,
		float fy, float cx, float cy, int w, int h, float* d_residuals)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	// valid thread index
	if (x < w && y < h)
	{

		size_t idx = x + y * w;
		float residual = 0.0f;

		// backproject 2d pixel
		float dRef = d_ptrDepthRef[idx];

		// continue if valid depth data is available
		if (dRef > 0.0)
		{
			// to camera coordinates
			float x0 = (static_cast<float>(x) - cx) * 1.0f / fx;
			float y0 = (static_cast<float>(y) - cy) * 1.0f / fy;
			float homo = 1.0f;

			// apply known depth; to 3D coordinates
			x0 *= dRef;
			y0 *= dRef;
			float z0 = homo * dRef;

			// rotate and translate; Eigen uses column-major
			float x1 = d_rotMat[0] * x0 + d_rotMat[3] * y0 + d_rotMat[6] * z0
					+ d_t[0];
			float y1 = d_rotMat[1] * x0 + d_rotMat[4] * y0 + d_rotMat[7] * z0
					+ d_t[1];
			float z1 = d_rotMat[2] * x0 + d_rotMat[5] * y0 + d_rotMat[8] * z0
					+ d_t[2];

			if (z1 > 0.0f)
			{
				// project onto 2nd frame

				float x2 = (fx * x1 + cx * z1) / z1;
				float y2 = (fy * y1 + cy * z1) / z1;

				float valCur = d_interpolate(d_ptrGrayCur, x2, y2, w, h);
				if (!isnan(valCur))
				{
					float valRef = d_ptrGrayRef[idx];
					float valDiff = valRef - valCur;
					residual = valDiff;
				}
			}
		}
		d_residuals[idx] = residual;
	}
}

/**
 * Calculate residuals on GPU
 * @param d_residuals calculated output: residuals
 */
void DVO::calculateError(const cv::cuda::GpuMat &grayRef,
		const cv::cuda::GpuMat &depthRef, const cv::cuda::GpuMat &grayCur,
		const cv::cuda::GpuMat &depthCur, const Eigen::VectorXf &xi,
		const Eigen::Matrix3f &K, float* d_residuals)
{

	// create residual image
	int w = grayRef.cols;
	int h = grayRef.rows;

	// camera intrinsics
	float fx = K(0, 0);
	float fy = K(1, 1);
	float cx = K(0, 2);
	float cy = K(1, 2);

	// convert SE3 to rotation matrix and translation vector
	Eigen::Matrix3f rotMat;
	Eigen::Vector3f t;
	convertSE3ToTf(xi, rotMat, t);

	float* d_ptrGrayRef = (float*) grayRef.ptr();
	float* d_ptrDepthRef = (float*) depthRef.ptr();
	float* d_ptrGrayCur = (float*) grayCur.ptr();
	float* d_ptrDepthCur = (float*) depthCur.ptr();

	cudaMemcpyToSymbol(d_rotMat, rotMat.data(), 9);
	CUDA_CHECK;

	cudaMemcpyToSymbol(d_t, t.data(), 3);
	CUDA_CHECK;

	dim3 block = dim3(32, 8, 1);
	dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y,
			1);
	g_residualKernel<<<grid, block>>>(d_ptrGrayRef, d_ptrDepthRef, d_ptrGrayCur,
			fx, fy, cx, cy, w, h, d_residuals);
}

__global__ void computeHuberWeightsKernel(float* weights, float* residuals,
		int n, float k)
{
	//Compute index
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int i = x;
	//Do bounds check
	if (i < n && y < 1 && z < 1)
	{
		//compute robust Huber weights
		float w;
		if (std::abs(residuals[i]) <= k)
			w = 1.0f;
		else
			w = k / std::abs(residuals[i]);
		weights[i] = w;

	}
}

__global__ void computeStudentTWeightsKernel(float* weights, float* residuals,
		int n, int v, float sigma)
{
	//Compute index
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int i = x;
	//Do bounds check
	if (i < n && y < 1 && z < 1)
	{
		//compute robust Student T weights
		weights[i] = (1.0f * (v + 1))
				/ ((1.0f * v) + powf(residuals[i] / sigma, 2.0f));
	}
}

__global__ void computeCauchyWeightsKernel(float* weights, float* residuals,
		int n)
{
	//Compute index
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int i = x;
	//Do bounds check
	if (i < n && y < 1 && z < 1)
	{
		//compute robust Cauchy weights
		weights[i] = -2.0f / (1 + residuals[i] * residuals[i]);
	}
}

/**
 * shift elements by mean, and square them
 */
struct varianceshifteop: std::unary_function<float, float>
{
	varianceshifteop(float m) :
			mean(m)
	{ /* no-op */
	}

	const float mean;

	__device__ float operator()(float data) const
	{
		return (data - mean) * (data - mean);
	}
};

struct irls: std::unary_function<float, float>
{
	irls(float s_, float v_) :
			sd(s_), v(v_)
	{ /* no-op */
	}

	const float sd;
	const float v;

	__device__ float operator()(float data) const
	{
		return (data * data) * ((v + 1) / (v + powf(data / sd, 2)));
	}
};

void DVO::computeWeights(float* d_residuals, float* d_weights, int n)
{
#if 0
	// no weighting
	for (int i = 0; i < n; ++i)
	weights[i] = 1.0f;
#if 0
	// squared residuals
	for (int i = 0; i < n; ++i)
	residuals[i] = residuals[i] * residuals[i];
	return;
#endif
#endif

	float mean, stdDev;

	// wrap raw pointer with a device_ptr
	thrust::device_ptr<float> dp_residuals = thrust::device_pointer_cast(
			d_residuals);

	// sum elements and divide by the number of elements
	mean = thrust::reduce(dp_residuals, dp_residuals + n, 0.0f,
			thrust::plus<float>()) / n;

	// shift elements by mean, square, and add them
	float variance = thrust::transform_reduce(dp_residuals, dp_residuals + n,
			varianceshifteop(mean), 0.0f, thrust::plus<float>());

	// standard dev is just a sqrt away
	stdDev = std::sqrt(variance);

	// Parameters for huber distribution
	float k = 1.345f * stdDev;

	// Parameters for student t distribution
#ifdef STUDENT_T    
	int v = 5;
	float sd = stdDev;
	float oldSd;
	do
	{	oldSd = sd;
		sd = thrust::transform_reduce(
				dp_residuals,
				dp_residuals+n,
				irls(oldSd,v),
				0.0f,
				thrust::plus<float>());
	}while( std::abs(sd-oldSd) > 0.001f);
#endif 

	dim3 block = dim3(512, 1, 1);
	dim3 grid = dim3((n + block.x - 1) / block.x, 1, 1);
	//huber
#ifdef HUBER
	computeHuberWeightsKernel<<<grid, block>>>(d_weights, d_residuals, n, k);
#endif

	// student t 
#ifdef STUDENT_T
	computeStudentTWeightsKernel<<<grid,block>>>(d_weights, d_residuals, n, v,sd);
#endif

	//Cauchy
#ifdef CAUCHY
	computeCauchyWeightsKernel<<<grid,block>>>(d_weights, d_residuals, n);
#endif
}


/**
 * Manual rotation and translation of matrix
 */
__device__ void rotateAndTranslate(float *rot, float *t, float *v, float *res)
{
	for (int i = 0; i < 3; i++)
	{
		float sum = 0.f;
		for (int j = 0; j < 3; j++)
		{
			sum += rot[i + 3 * j] * v[j];
		}
		res[i] = sum + t[i];
	}

}

/**
 * GPU-accelerated gradient computation
 * @param d_Jr output gradient
 */
__global__ void computeAnalyticalGradient(float* d_ptrDepthRef, float *d_gradx,
		float *d_grady, int w, int h, float* d_residuals, bool useWeights,
		float* d_weights, float *d_Jr, float *d_Ai)
{

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < w && y < h)
	{
		size_t idx = x + (size_t) w * y;

		float fx = d_K[0];
		float fy = d_K[4];
		float cx = d_K[6];
		float cy = d_K[7];
		float fxInv = 1.0f / fx;
		float fyInv = 1.0f / fy;

		bool innerIfExecuted = false;

		// project 2d point back into 3d using its depth
		float dRef = d_ptrDepthRef[idx];
		if (dRef > 0.0f)
		{
			float x0 = (static_cast<float>(x) - cx) * fxInv;
			float y0 = (static_cast<float>(y) - cy) * fyInv;
			float scale = 1.0f;
			//scale = std::sqrt(x0*x0 + y0*y0 + 1.0);
			dRef = dRef * scale;
			x0 = x0 * dRef;
			y0 = y0 * dRef;

			// transform reference 3d point into current frame
			// reference 3d point
			// Eigen::Vector3f pt3Ref(x0, y0, dRef);
			float pt3Ref[3] =
			{ x0, y0, dRef };
			float pt3[3];

			rotateAndTranslate(d_rotMat, d_t, pt3Ref, pt3);

			if (pt3[2] > 0.0f)
			{

				float px = (fx * pt3[0] + cx * pt3[2]) / pt3[2];
				float py = (fy * pt3[1] + cy * pt3[2]) / pt3[2];

				// compute interpolated image gradient
				float dX = d_interpolate(d_gradx, px, py, w, h);
				float dY = d_interpolate(d_grady, px, py, w, h);

				if (!isnan(dX) && !isnan(dY))
				{
					innerIfExecuted = true;
					dX = fx * dX;
					dY = fy * dY;
					float pt3Zinv = 1.0f / pt3[2];

					float J[6];
					float weight = 1.0f;

					if (useWeights)
						weight = d_weights[idx];

					// shorter computation
					J[0] = (-1.0f * dX * pt3Zinv);
					J[1] = (-1.0f * dY * pt3Zinv);
					J[2] = ((dX * pt3[0] + dY * pt3[1]) * pt3Zinv * pt3Zinv);
					J[3] =
							((dX * pt3[0] * pt3[1]) * pt3Zinv * pt3Zinv
									+ dY
											* (1
													+ (pt3[1] * pt3Zinv)
															* (pt3[1] * pt3Zinv)));
					J[4] = (-dX
							* (1.0 + (pt3[0] * pt3Zinv) * (pt3[0] * pt3Zinv))
							- (dY * pt3[0] * pt3[1]) * pt3Zinv * pt3Zinv);
					J[5] = (-1.0f * (-dX * pt3[1] + dY * pt3[0]) * pt3Zinv);

					d_Jr[idx * 6 + 0] = J[0] * weight * d_residuals[idx];
					d_Jr[idx * 6 + 1] = J[1] * weight * d_residuals[idx];
					d_Jr[idx * 6 + 2] = J[2] * weight * d_residuals[idx];
					d_Jr[idx * 6 + 3] = J[3] * weight * d_residuals[idx];
					d_Jr[idx * 6 + 4] = J[4] * weight * d_residuals[idx];
					d_Jr[idx * 6 + 5] = J[5] * weight * d_residuals[idx];

					int i;
					int j;
					// from linear index to 2d matrix index
					for (int k = 0; k < 21; k++)
					{
						//line
						i = floor(
								(2.0f * 6 + 1
										- sqrtf(
												(2.0f * 6 + 1.0f)
														* (2.0f * 6 + 1.0f)
														- 8.0f * k)) / 2.0f);
						//column
						j = i + (k - 6 * i + i * (i - 1) / 2);

						d_Ai[idx * 21 + k] = J[i] * J[j] * weight;
					}
				}
			}
		}

		if (!innerIfExecuted)
		{
			for (int j = 0; j < 6; j++)
			{
				d_Jr[idx * 6 + j] = 0.0f;
			}
			for (int i = 0; i < 21; i++)
			{
				d_Ai[idx * 21 + i] = 0.0f;
			}
		}

	}
}

struct pixelA
{
	// A is 6x6 symmetric, therefore, 21 unique elements
	float a[21];
};

struct A_reduce: public thrust::binary_function<pixelA, pixelA, pixelA>
{
	__device__ pixelA operator()(pixelA Ai, pixelA Aj) const
	{
		struct pixelA res;
		for (int i = 0; i < 21; i++)
		{
			res.a[i] = Ai.a[i] + Aj.a[i];
		}

		return res;
	}

};

struct pixelb
{
	float b[6];
};

struct b_reduce: public thrust::binary_function<pixelb, pixelb, pixelb>
{
	__device__ pixelb operator()(pixelb bi, pixelb bj) const
	{
		struct pixelb res;
		for (int i = 0; i < 6; i++)
		{
			res.b[i] = bi.b[i] + bj.b[i];
		}

		return res;
	}

};

void DVO::deriveAnalytic(const cv::cuda::GpuMat &grayRef,
		const cv::cuda::GpuMat &depthRef, const cv::cuda::GpuMat &grayCur,
		const cv::cuda::GpuMat &depthCur, const cv::cuda::GpuMat &gradX,
		const cv::cuda::GpuMat &gradY, const Eigen::VectorXf &xi,
		const Eigen::Matrix3f &K, float* d_residuals, bool useWeights,
		float* d_weights, float* d_J, Mat6f &A, Vec6f &b)
{
	for (int i = 0; i < 6; i++)
	{
		b[i] = 0.0f;
		for (int j = 0; j < 6; j++)
		{
			A(i, j) = 0.0f;
		}
	}
	// reference input images
	int w = grayRef.cols;
	int h = grayRef.rows;
	int n = w * h;

	// camera intrinsics
	float fx = K(0, 0);
	float fy = K(1, 1);
	float cx = K(0, 2);
	float cy = K(1, 2);

	// convert SE3 to rotation matrix and translation vector
	Eigen::Matrix3f rotMat;
	Eigen::Vector3f t;
	convertSE3ToTf(xi, rotMat, t);

	float* d_ptrGrayRef = (float*) grayRef.ptr();
	float* d_ptrDepthRef = (float*) depthRef.ptr();
	float* d_ptrGrayCur = (float*) grayCur.ptr();
	float* d_ptrDepthCur = (float*) depthCur.ptr();

	// Allocating device memory
	float *d_gradx, *d_grady;

	d_gradx = (float*) gradX.data;
	d_grady = (float*) gradY.data;

	cudaMemcpyToSymbol(d_rotMat, rotMat.data(), 9 * sizeof(float));
	CUDA_CHECK;
	cudaMemcpyToSymbol(d_K, K.data(), 9 * sizeof(float));
	CUDA_CHECK;
	cudaMemcpyToSymbol(d_t, t.data(), 3 * sizeof(float));
	CUDA_CHECK;

	float *d_Ai;
	cudaMalloc(&d_Ai, sizeof(float) * 21 * n);

	dim3 block = dim3(32, 8, 1);
	dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y,
			1);

	g_residualKernel<<<grid, block>>>(d_ptrGrayRef, d_ptrDepthRef, d_ptrGrayCur,
			fx, fy, cx, cy, w, h, d_residuals);

	if (useWeights)
		computeWeights(d_residuals, d_weights, n); // W

	// d_J = Jt W r
	// d_Ai = A for every pixel
	computeAnalyticalGradient<<<grid, block>>>(d_ptrDepthRef, d_gradx, d_grady,
			w, h, d_residuals, useWeights, d_weights, d_J, d_Ai);

	thrust::device_ptr<struct pixelA> dp_As = thrust::device_pointer_cast(
			(struct pixelA *) d_Ai);
	thrust::device_ptr<struct pixelb> dp_bs = thrust::device_pointer_cast(
			(struct pixelb *) d_J);

	struct pixelA neutralA;
	for (int i = 0; i < 21; i++)
	{
		neutralA.a[i] = 0.0f;
	}

	struct pixelb neutralb;
	for (int i = 0; i < 6; i++)
	{
		neutralb.b[i] = 0.0f;
	}

	struct pixelA resA = thrust::reduce(dp_As, dp_As + n, neutralA, A_reduce());

	struct pixelb resb = thrust::reduce(dp_bs, dp_bs + n, neutralb, b_reduce());

	for (int i = 0; i < 6; i++)
	{
		b[i] = resb.b[i];
	}

	for (int k = 0; k < 21; k++)
	{
		int i = floor(
				(2.0f * 6 + 1
						- sqrtf(
								(2.0f * 6 + 1.0f) * (2.0f * 6 + 1.0f)
										- 8.0f * k)) / 2.0f);
		int j = i + (k - 6 * i + i * (i - 1) / 2);

		A(i, j) = resA.a[k];
		A(j, i) = resA.a[k];
	}

	cudaFree(d_Ai);
}

cv::cuda::GpuMat DVO::convertToContGpuMat(const cv::Mat &m)
{
	cv::cuda::GpuMat gpuM = cv::cuda::createContinuous(m.rows, m.cols,
			m.type());
	gpuM.upload(m);
	return gpuM;
}

/**
 * Builds pyramid of downsampled versions of an image
 */
void DVO::buildPyramid(const cv::Mat &depth, const cv::Mat &gray,
		std::vector<cv::cuda::GpuMat> &depthPyramid,
		std::vector<cv::cuda::GpuMat> &grayPyramid)
{
	grayPyramid.push_back(convertToContGpuMat(gray));
	depthPyramid.push_back(convertToContGpuMat(depth));

	for (int i = 1; i < numPyramidLevels_; ++i)
	{
		// downsample grayscale image
		cv::cuda::GpuMat grayDown = downsampleGray(grayPyramid[i - 1], 0);
		// downsample depth image
		cv::cuda::GpuMat depthDown = downsampleDepth(depthPyramid[i - 1], 1);
		cudaDeviceSynchronize();
		CUDA_CHECK;

		grayPyramid.push_back(grayDown);
		depthPyramid.push_back(depthDown);
	}
}

void DVO::align(const cv::Mat &depthRef, const cv::Mat &grayRef,
		const cv::Mat &depthCur, const cv::Mat &grayCur, Eigen::Matrix4f &pose)
{
	// downsampling

	std::vector<cv::cuda::GpuMat> grayRefGPUPyramid;
	std::vector<cv::cuda::GpuMat> depthRefGPUPyramid;

	buildPyramid(depthRef, grayRef, depthRefGPUPyramid, grayRefGPUPyramid);

	std::vector<cv::cuda::GpuMat> grayCurGPUPyramid;
	std::vector<cv::cuda::GpuMat> depthCurGPUPyramid;

	buildPyramid(depthCur, grayCur, depthCurGPUPyramid, grayCurGPUPyramid);

	align(depthRefGPUPyramid, grayRefGPUPyramid, depthCurGPUPyramid,
			grayCurGPUPyramid, pose);
}

/**
 * Function aligning current with previous image and computing relative pose
 * @param pose estimated pose (output)
 */
void DVO::align(const std::vector<cv::cuda::GpuMat> &depthRefGPUPyramid,
		const std::vector<cv::cuda::GpuMat> &grayRefGPUPyramid,
		const std::vector<cv::cuda::GpuMat> &depthCurGPUPyramid,
		const std::vector<cv::cuda::GpuMat> &grayCurGPUPyramid,
		Eigen::Matrix4f &pose)
{

	Vec6f xi;
	convertTfToSE3(pose, xi);

	Vec6f lastXi = Vec6f::Zero();

	int maxLevel = numPyramidLevels_ - 1;
	int minLevel = 1;
	float initGradDescStepSize = 1e-3f;
	float gradDescStepSize = initGradDescStepSize;

	Mat6f A;
	Vec6f b;

	Mat6f diagMatA = Mat6f::Identity();
	Vec6f delta;

	for (int lvl = maxLevel; lvl >= minLevel; --lvl)
	{
		float lambda = 0.1f;

		int w = sizePyramid_[lvl].width;
		int h = sizePyramid_[lvl].height;
		int n = w * h;

		cv::cuda::GpuMat grayRef = grayRefGPUPyramid[lvl];
		cv::cuda::GpuMat depthRef = depthRefGPUPyramid[lvl];
		cv::cuda::GpuMat grayCur = grayCurGPUPyramid[lvl];
		cv::cuda::GpuMat depthCur = depthCurGPUPyramid[lvl];
		Eigen::Matrix3f kLevel = kPyramid_[lvl];
		//std::cout << "level " << level << " (size " << depthRef.cols << "x" << depthRef.rows << ")" << std::endl;

		// compute gradient images
		computeGradient(grayCur, gradX_[lvl], gradY_[lvl]);
		//computeGradient(grayCur, gradY_[lvl], 1);

		float errorLast = std::numeric_limits<float>::max();
		for (int itr = 0; itr < numIterations_; ++itr)
		{
			// compute residuals and Jacobian
#if 0
			deriveNumeric(grayRef, depthRef, grayCur, depthCur, xi, kLevel, residuals_[lvl], J_[lvl]);
#else
			deriveAnalytic(grayRef, depthRef, grayCur, depthCur, gradX_[lvl],
					gradY_[lvl], xi, kLevel, d_residuals_[lvl], useWeights_,
					d_weights_[lvl], d_J_[lvl], A, b);
#endif

#if 0
			// compute and show error image
			cv::Mat errorImage;
			calculateErrorImage(residuals_[lvl], grayRef.cols, grayRef.rows, errorImage);
			std::stringstream ss;
			ss << "residuals_" << lvl << "_";
			ss << std::setw(2) << std::setfill('0') << itr << ".png";
			cv::imwrite(ss.str(), errorImage);
			cv::imshow("error", errorImage);
			cv::waitKey(0);
#endif

			// calculate error
			float error = calculateError(d_residuals_[lvl], n);

			// compute update
			//compute_JtR(d_J_[lvl], d_residuals_[lvl], b, n);
			if (algo_ == GradientDescent)
			{
				// Gradient Descent
				delta = -gradDescStepSize * b * (1.0f / b.norm());
			}
			else if (algo_ == GaussNewton)
			{
				// Gauss-Newton algorithm
				// solve using Cholesky LDLT decomposition
				delta = -(A.ldlt().solve(b));
			}
			else if (algo_ == LevenbergMarquardt)
			{
				// Levenberg-Marquardt algorithm
				diagMatA.diagonal() = lambda * A.diagonal();
				delta = -((A + diagMatA).ldlt().solve(b));
			}

			// apply update: left-multiplicative increment on SE3
			lastXi = xi;
			xi = Sophus::SE3f::log(
					Sophus::SE3f::exp(delta) * Sophus::SE3f::exp(xi));
#if 0
			std::cout << "delta = " << delta.transpose() << " size = " << delta.rows() << " x " << delta.cols() << std::endl;
			std::cout << "xi = " << xi.transpose() << std::endl;
#endif

			// compute error again
			error = calculateError(d_residuals_[lvl], n);

			if (algo_ == LevenbergMarquardt)
			{
				if (error >= errorLast)
				{
					lambda = lambda * 5.0f;
					xi = lastXi;

					if (lambda > 5.0f)
						break;
				}
				else
				{
					lambda = lambda / 1.5f;
				}
			}
			else if (algo_ == GaussNewton)
			{
				// break if no improvement (0.99 or 0.995)
				if (error / errorLast > 0.995f)
					break;
			}
			else if (algo_ == GradientDescent)
			{
				if (error >= errorLast)
				{
					gradDescStepSize = gradDescStepSize * 0.5f;
					if (gradDescStepSize <= initGradDescStepSize * 0.01f)
						gradDescStepSize = initGradDescStepSize * 0.01f;
					xi = lastXi;
				}
				else
				{
					gradDescStepSize = gradDescStepSize * 2.0f;
					if (gradDescStepSize >= initGradDescStepSize * 100.0f)
						gradDescStepSize = initGradDescStepSize * 100.0f;

					// break if no improvement (0.99 or 0.995)
					if (error / errorLast > 0.995f)
						break;
				}
			}

			errorLast = error;
		}
	}

	// store to output pose
	convertSE3ToTf(xi, pose);
}
