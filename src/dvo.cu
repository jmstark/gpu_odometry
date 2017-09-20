// Copyright 2016 Robert Maier, Technical University Munich
#include "dvo.hpp"
#include "helper.h"

#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <stdio.h>
#include <ctime>


#include <Eigen/Cholesky>
#include <sophus/se3.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>


#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
// cuda error checking
std::string prev_file = "";
int prev_line = 0;
void cuda_check(std::string file, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        std::cout << std::endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << std::endl;
        if (prev_line>0) std::cout << "Previous CUDA call:" << std::endl << prev_file << ", line " << prev_line << std::endl;
        exit(1);
    }
    prev_file = file;
    prev_line = line;
}


DVO::DVO() :
    numPyramidLevels_(5),
    useWeights_(true),
    numIterations_(500),
    algo_(GaussNewton)
{
}


DVO::~DVO()
{
    for (int i = 0; i < numPyramidLevels_; ++i)
    {
        delete[] J_[i];
        delete[] residuals_[i];
        delete[] weights_[i];
    }
}


void DVO::init(int w, int h, const Eigen::Matrix3f &K)
{
    // pyramid level size
    int wDown = w;
    int hDown = h;
    int n = wDown*hDown;
    sizePyramid_.push_back(cv::Size(wDown, hDown));

    // gradients
    cv::Mat gradX = cv::Mat::zeros(h, w, CV_32FC1);
    gradX_.push_back(gradX);
    cv::Mat gradY = cv::Mat::zeros(h, w, CV_32FC1);
    gradY_.push_back(gradY);

    // Jacobian
    float* J = new float[n*6];
    J_.push_back(J);
    // residuals
    float* residuals = new float[n];
    residuals_.push_back(residuals);
    // per-residual weights
    float* weights = new float[n];
    weights_.push_back(weights);

    // camera matrix
    kPyramid_.push_back(K);

    for (int i = 1; i < numPyramidLevels_; ++i)
    {
        // pyramid level size
        wDown = wDown / 2;
        hDown = hDown / 2;
        int n = wDown*hDown;
        sizePyramid_.push_back(cv::Size(wDown, hDown));

        // gradients
        cv::Mat gradXdown = cv::Mat::zeros(hDown, wDown, CV_32FC1);
        gradX_.push_back(gradXdown);
        cv::Mat gradYdown = cv::Mat::zeros(hDown, wDown, CV_32FC1);
        gradY_.push_back(gradYdown);

        // Jacobian
        float* J = new float[n*6];
        J_.push_back(J);
        // residuals
        float* residuals = new float[n];
        residuals_.push_back(residuals);
        // per-residual weights
        float* weights = new float[n];
        weights_.push_back(weights);

        // downsample camera matrix
        Eigen::Matrix3f kDown = kPyramid_[i-1];
        kDown(0, 2) += 0.5f;
        kDown(1, 2) += 0.5f;
        kDown.topLeftCorner(2, 3) = kDown.topLeftCorner(2, 3) * 0.5f;
        kDown(0, 2) -= 0.5f;
        kDown(1, 2) -= 0.5f;
        kPyramid_.push_back(kDown);
        //std::cout << "Camera matrix (level " << i << "): " << kDown << std::endl;
    }
}


void DVO::convertSE3ToTf(const Vec6f &xi, Eigen::Matrix3f &rot, Eigen::Vector3f &t)
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


void DVO::convertTfToSE3(const Eigen::Matrix3f &rot, const Eigen::Vector3f &t, Vec6f &xi)
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
    int wDown = w/2;
    int hDown = h/2;
	//Do bounds check
	if(x<wDown && y<hDown && z<1)
	{
        float sum = 0.0f;
        sum += in[2*y * w + 2*x] * 0.25f;
        sum += in[2*y * w + 2*x+1] * 0.25f;
        sum += in[(2*y+1) * w + 2*x] * 0.25f;
        sum += in[(2*y+1) * w + 2*x+1] * 0.25f;
        out[y*wDown + x] = sum;
	}
}

cv::Mat DVO::downsampleGray(const cv::Mat &gray)
{
	float * d_in, * d_out;
    int w = gray.cols;
    int h = gray.rows;
    int wDown = w/2;
    int hDown = h/2;
    cudaMalloc(&d_in,w*h*sizeof(float)); CUDA_CHECK;
    cudaMemcpy(d_in,gray.data,w*h*sizeof(float),cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMalloc(&d_out,wDown*hDown*sizeof(float)); CUDA_CHECK;

    dim3 block = dim3(64,8,1);
    dim3 grid = dim3((w+block.x-1) / block.x,
		(h+block.y - 1) / block.y,
		1);
    downsampleGrayKernel<<<grid,block>>>(d_out, w, h, d_in);
    cudaDeviceSynchronize(); CUDA_CHECK;

    cv::Mat grayDown = cv::Mat::zeros(hDown, wDown, gray.type());
    float* ptrOut = (float*)grayDown.data;

    cudaMemcpy(ptrOut,d_out,wDown*hDown*sizeof(float),cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaFree(d_out); CUDA_CHECK;
    cudaFree(d_in); CUDA_CHECK;
    return grayDown;

}


__global__ void downsampleDepthKernel(float* out, int w, int h, float* in)
{
	//Compute index
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
    int wDown = w/2;
    int hDown = h/2;
	//Do bounds check
	if(x<wDown && y<hDown && z<1)
	{
         float d0 = in[2*y * w + 2*x];
         float d1 = in[2*y * w + 2*x+1];
         float d2 = in[(2*y+1) * w + 2*x];
         float d3 = in[(2*y+1) * w + 2*x+1];

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
                 out[y*wDown + x] = 1.0f / dInv;
                 return;
             }
         }
         //set pixel if we did not enter the inner if-block
         out[y*wDown + x] = 0.0f;
	}
}


cv::Mat DVO::downsampleDepth(const cv::Mat &depth)
{

	float * d_in, * d_out;
    int w = depth.cols;
    int h = depth.rows;
    int wDown = w/2;
    int hDown = h/2;
    cudaMalloc(&d_in,w*h*sizeof(float)); CUDA_CHECK;
    cudaMemcpy(d_in,depth.data,w*h*sizeof(float),cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMalloc(&d_out,wDown*hDown*sizeof(float)); CUDA_CHECK;

    dim3 block = dim3(64,8,1);
    dim3 grid = dim3((w+block.x-1) / block.x,
		(h+block.y - 1) / block.y,
		1);
    downsampleDepthKernel<<<grid,block>>>(d_out, w, h, d_in);
    cudaDeviceSynchronize(); CUDA_CHECK;


    cv::Mat depthDown = cv::Mat::zeros(hDown, wDown, depth.type());
    float* ptrOut = (float*)depthDown.data;
    cudaMemcpy(ptrOut,d_out,wDown*hDown*sizeof(float),cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaFree(d_out); CUDA_CHECK;
    cudaFree(d_in); CUDA_CHECK;
    return depthDown;

}


void DVO::computeGradient(const cv::Mat &gray, cv::Mat &gradient, int direction)
{
    int dirX = 1;
    int dirY = 0;
    if (direction == 1)
    {
        dirX = 0;
        dirY = 1;
    }

    // compute gradient manually using finite differences
    int w = gray.cols;
    int h = gray.rows;
    const float* ptrIn = (const float*)gray.data;
    gradient.setTo(0);
    float* ptrOut = (float*)gradient.data;

    int yStart = dirY;
    int yEnd = h - dirY;
    int xStart = dirX;
    int xEnd = w - dirX;
    for (size_t y = yStart; y < yEnd; ++y)
    {
        for (size_t x = xStart; x < xEnd; ++x)
        {
            float v0;
            float v1;
            if (direction == 1)
            {
                // y-direction
                v0 = ptrIn[(y-1)*w + x];
                v1 = ptrIn[(y+1)*w + x];
            }
            else
            {
                // x-direction
                v0 = ptrIn[y*w + (x-1)];
                v1 = ptrIn[y*w + (x+1)];
            }
            ptrOut[y*w + x] = 0.5f * (v1 - v0);
        }
    }
}


float DVO::interpolate(const float* ptrImgIntensity, float x, float y, int w, int h)
{
    float valCur = std::numeric_limits<float>::quiet_NaN();

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
        sum += ptrImgIntensity[y0*w + x0] * w00;
    if (w01 > 0.0f)
        sum += ptrImgIntensity[y1*w + x0] * w01;
    if (w10 > 0.0f)
        sum += ptrImgIntensity[y0*w + x1] * w10;
    if (w11 > 0.0f)
        sum += ptrImgIntensity[y1*w + x1] * w11;

    if (sumWeights > 0.0f)
        valCur = sum / sumWeights;
#endif

    return valCur;
}


float DVO::calculateError(const float* residuals, int n)
{
    float error = 0.0f;
    int numValid = 0;
    for (int i = 0; i < n; ++i)
    {
        if (residuals[i] != 0.0f)
        {
            error += residuals[i] * residuals[i];
            ++numValid;
        }
    }
    if (numValid > 0)
        error = error / static_cast<float>(numValid);
    return error;
}


void DVO::calculateErrorImage(const float* residuals, int w, int h, cv::Mat &errorImage)
{
    cv::Mat imgResiduals = cv::Mat::zeros(h, w, CV_32FC1);
    float* ptrResiduals = (float*)imgResiduals.data;

    // fill residuals image
    for (size_t y = 0; y < h; ++y)
    {
        for (size_t x = 0; x < w; ++x)
        {
            size_t off = y*w + x;
            if (residuals[off] != 0.0f)
                ptrResiduals[off] = residuals[off];
        }
    }

    imgResiduals.convertTo(errorImage, CV_8SC1, 127.0);
}



__device__ float d_interpolate(const float* ptrImgIntensity, float x, float y, int w, int h)
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
        sum += ptrImgIntensity[y0*w + x0] * w00;
    if (w01 > 0.0f)
        sum += ptrImgIntensity[y1*w + x0] * w01;
    if (w10 > 0.0f)
        sum += ptrImgIntensity[y0*w + x1] * w10;
    if (w11 > 0.0f)
        sum += ptrImgIntensity[y1*w + x1] * w11;

    if (sumWeights > 0.0f)
        valCur = sum / sumWeights;
#endif

    return valCur;
}


texture<float,2,cudaReadModeElementType> texGrayCur;
__global__ void g_residualKernel(const float* d_ptrGrayRef,
                            const float* d_ptrDepthRef,
                            const float* d_ptrGrayCur,
                            const float* d_ptrRotation,
                            const float* d_ptrTranslation,
                            float fx, float fy, float cx, float cy, int w,int h,
                            float* d_residuals)
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    // valid thread index
    if(x < w && y < h) {

        size_t idx = x + y*w;
        float residual = 0.0f;

        // backproject 2d pixel
        float dRef = d_ptrDepthRef[idx];

        // continue if valid depth data is available
        if(dRef > 0.0) {
            // to camera coordinates
            float x0 = (static_cast<float>(x) - cx) * 1.0f/fx;
            float y0 = (static_cast<float>(y) - cy) * 1.0f/fy;
            float homo = 1.0f;

            // apply known depth; to 3D coordinates
            x0  *= dRef;
            y0  *= dRef;
            float z0 = homo * dRef;

            // rotate and translate; Eigen uses column-major
            float x1 = d_ptrRotation[0] * x0 + d_ptrRotation[3] * y0 +
                        d_ptrRotation[6] * z0 + d_ptrTranslation[0];
            float y1 = d_ptrRotation[1] * x0 + d_ptrRotation[4] * y0 +
                        d_ptrRotation[7] * z0 + d_ptrTranslation[1];
            float z1 = d_ptrRotation[2] * x0 + d_ptrRotation[5] * y0 +
                        d_ptrRotation[8] * z0 + d_ptrTranslation[2];

            if(z1 > 0.0f) {
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

                /*if(x2 >= 0 && x2 < w && y2 >= 0 && y2 < h) {
                    // interpolate
                    float valCur = tex2D(texGrayCur, x2, y2);
                    residual = d_ptrGrayRef[idx] - valCur;
                }*/
            }
        }
        d_residuals[idx] = residual;
    }
}


void DVO::calculateError(const cv::Mat &grayRef, const cv::Mat &depthRef,
                         const cv::Mat &grayCur, const cv::Mat &depthCur,
                         const Eigen::VectorXf &xi, const Eigen::Matrix3f &K,
                         float* residuals)
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

    /* ## for GpuMat use
    float* d_ptrGrayRef = (const float*)grayRef.data;
    float* d_ptrDepthRef = (const float*)depthRef.data;
    float* d_ptrGrayCur = (const float*)grayCur.data;
    float* d_ptrDepthCur = (const float*)depthCur.data;
    */

    float* d_ptrGrayRef;
    cudaMalloc(&d_ptrGrayRef, w*h*sizeof(float));
    cudaMemcpy(d_ptrGrayRef, (const float*)grayRef.data, w*h*sizeof(float), cudaMemcpyHostToDevice);

    float* d_ptrDepthRef;
    cudaMalloc(&d_ptrDepthRef, w*h*sizeof(float));
    cudaMemcpy(d_ptrDepthRef, (const float*)depthRef.data, w*h*sizeof(float), cudaMemcpyHostToDevice);

    float* d_ptrGrayCur;
    cudaMalloc(&d_ptrGrayCur, w*h*sizeof(float));
    cudaMemcpy(d_ptrGrayCur, (const float*)grayCur.data, w*h*sizeof(float), cudaMemcpyHostToDevice);

    /*texGrayCur.addressMode[0] = cudaAddressModeClamp;
    texGrayCur.addressMode[1] = cudaAddressModeClamp;
    texGrayCur.filterMode = cudaFilterModeLinear;
    texGrayCur.normalized = false;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    cudaBindTexture2D(NULL, &texGrayCur, d_ptrGrayCur, &desc, w, h, w * sizeof(d_ptrGrayCur[0]));
    */

    float* d_ptrRotation;
    cudaMalloc(&d_ptrRotation, 9*sizeof(float));
    cudaMemcpy(d_ptrRotation, rotMat.data(), 9*sizeof(float), cudaMemcpyHostToDevice);

    float* d_ptrTranslation;
    cudaMalloc(&d_ptrTranslation, 3*sizeof(float));
    cudaMemcpy(d_ptrTranslation, t.data(), 3*sizeof(float), cudaMemcpyHostToDevice);

    float* d_residuals;
    cudaMalloc(&d_residuals, w*h*sizeof(float));

    dim3 block = dim3(32,8,1);
    dim3 grid = dim3( (w + block.x -1) / block.x, (h+block.y -1) / block.y, 1);
    std::cout << "calling residual kernel" << std::endl;
    Timer ti;
    ti.start();
    g_residualKernel <<<grid,block>>> (d_ptrGrayRef, d_ptrDepthRef, d_ptrGrayCur, d_ptrRotation,
                                d_ptrTranslation, fx, fy, cx, cy, w, h, d_residuals);
    cudaDeviceSynchronize();
    ti.end();
    float tPassed = ti.get();
    std::cout << "took " << tPassed*1000 << " ms" << std::endl;


    cudaMemcpy(residuals, d_residuals, w*h*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_ptrGrayRef);
    cudaFree(d_ptrDepthRef);
    cudaFree(d_ptrRotation);
    cudaFree(d_ptrTranslation);
    cudaFree(d_residuals);
    //cudaUnbindTexture(texGrayCur);
    cudaFree(d_ptrGrayCur);
}



__global__ void computeHuberWeightsKernel(float* weights, float* residuals, int n, float k)
{
	//Compute index
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int i = x;
	//Do bounds check
	if(i<n && y < 1 && z < 1)
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



struct varianceshifteop
    : std::unary_function<float, float>
{
    varianceshifteop(float m)
        : mean(m)
    { /* no-op */ }

    const float mean;

    __device__ float operator()(float data) const
    {
    	return (data-mean)*(data-mean);
    }
};



void DVO::computeWeights(const float* residuals, float* weights, int n)
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


	float * d_weights, * d_residuals;
	cudaMalloc(&d_weights,n*sizeof(float)); CUDA_CHECK;
	cudaMemcpy(d_weights,weights,n*sizeof(float),cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMalloc(&d_residuals,n*sizeof(float)); CUDA_CHECK;
	cudaMemcpy(d_residuals,residuals,n*sizeof(float),cudaMemcpyHostToDevice); CUDA_CHECK;

    float mean, stdDev;

    // wrap raw pointer with a device_ptr
    thrust::device_ptr<float> dp_residuals = thrust::device_pointer_cast(d_residuals);

    // sum elements and divide by the number of elements
    mean = thrust::reduce(
        dp_residuals,
        dp_residuals+n,
        0.0f,
        thrust::plus<float>()) / n;

    // shift elements by mean, square, and add them
    float variance = thrust::transform_reduce(
    		dp_residuals,
    		dp_residuals+n,
            varianceshifteop(mean),
            0.0f,
            thrust::plus<float>());

    // standard dev is just a sqrt away
    stdDev = std::sqrt(variance);

    float k = 1.345f * stdDev;

    dim3 block = dim3(512,1,1);
    dim3 grid = dim3((n+block.x-1) / block.x,
		1,
		1);
    computeHuberWeightsKernel<<<grid,block>>>(d_weights, d_residuals, n, k);
    cudaDeviceSynchronize(); CUDA_CHECK;

	cudaMemcpy(weights,d_weights,n*sizeof(float),cudaMemcpyDeviceToHost); CUDA_CHECK;
	cudaFree(d_weights); CUDA_CHECK;
	cudaFree(d_residuals); CUDA_CHECK;
}

__global__ void applyWeightsKernel(const float* weights, float* residuals, int n)
{
	//Compute index
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int i = x;
	//Do bounds check
	if(i<n && y < 1 && z < 1)
	{
		residuals[i] = residuals[i] * weights[i];
	}
}


void DVO::applyWeights(const float* weights, float* residuals, int n)
{
	float * d_weights, * d_residuals;
	cudaMalloc(&d_weights,n*sizeof(float)); CUDA_CHECK;
	cudaMemcpy(d_weights,weights,n*sizeof(float),cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMalloc(&d_residuals,n*sizeof(float)); CUDA_CHECK;
	cudaMemcpy(d_residuals,residuals,n*sizeof(float),cudaMemcpyHostToDevice); CUDA_CHECK;

    dim3 block = dim3(512,1,1);
    dim3 grid = dim3((n+block.x-1) / block.x,
		1,
		1);
    applyWeightsKernel<<<grid,block>>>(d_weights, d_residuals, n);
    cudaDeviceSynchronize(); CUDA_CHECK;

	cudaMemcpy(residuals,d_residuals,n*sizeof(float),cudaMemcpyDeviceToHost); CUDA_CHECK;
	cudaFree(d_weights); CUDA_CHECK;
	cudaFree(d_residuals); CUDA_CHECK;

}


void DVO::deriveNumeric(const cv::Mat &grayRef, const cv::Mat &depthRef,
                                  const cv::Mat &grayCur, const cv::Mat &depthCur,
                                  const Eigen::VectorXf &xi, const Eigen::Matrix3f &K,
                                  float* residuals, float* J)
{
    float epsilon = 1e-6;
    float scale = 1.0f / epsilon;

    int w = grayRef.cols;
    int h = grayRef.rows;
    int n = w*h;

    // calculate per-pixel residuals
    calculateError(grayRef, depthRef, grayCur, depthCur, xi, K, residuals);

    // create and fill Jacobian column by column
    float* residualsInc = new float[n];
    for (int j = 0; j < 6; ++j)
    {
        Eigen::VectorXf unitVec = Eigen::VectorXf::Zero(6);
        unitVec[j] = epsilon;

        // left-multiplicative increment on SE3
        Eigen::VectorXf xiEps = Sophus::SE3f::log(Sophus::SE3f::exp(unitVec) * Sophus::SE3f::exp(xi));

        calculateError(grayRef, depthRef, grayCur, depthCur, xiEps, K, residualsInc);
        for (int i = 0; i < n; ++i)
            J[i*6 + j] = (residualsInc[i] - residuals[i]) * scale;
    }
    delete[] residualsInc;
}


void DVO::compute_JtR(const float* J, const float* residuals, Vec6f &b, int validRows)
{
    int n = 6;
    int m = validRows;

    // compute b = Jt*r
    for (int j = 0; j < n; ++j)
    {
        float val = 0.0f;
        for (int i = 0; i < m; ++i)
            val += J[i*6 + j] * residuals[i];
        b[j] = val;
    }
}


void DVO::compute_JtJ(const float* J, Mat6f &A, const float* weights, int validRows, bool useWeights)
{
    int n = 6;
    int m = validRows;

    // compute A = Jt*J
    for (int k = 0; k < n; ++k)
    {
        for (int j = 0; j < n; ++j)
        {
            float val = 0.0f;
            for (int i = 0; i < m; ++i)
            {
                float valSqr = J[i*6 + j] * J[i*6 + k];
                if (useWeights)
                    valSqr *= weights[i];
                val += valSqr;
            }
            A(k, j) = val;
        }
    }
}

__device__ float interpolate_gpu( const float* ptrImgIntensity, float x, float y, int w, int h)
{
	float valCur = 1.1f;

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
        sum += ptrImgIntensity[y0*w + x0] * w00;
    if (w01 > 0.0f)
        sum += ptrImgIntensity[y1*w + x0] * w01;
    if (w10 > 0.0f)
        sum += ptrImgIntensity[y0*w + x1] * w10;
    if (w11 > 0.0f)
        sum += ptrImgIntensity[y1*w + x1] * w11;

    if (sumWeights > 0.0f)
        valCur = sum / sumWeights;
#endif

    return valCur;
}

void matToInterleaved(float *a,const Eigen::Matrix3f &m)
{
	for(int i=0;i<3;i++)
	{
		for(int j=0;j<3;j++)
		{
			a[i+3*j] = m(i,j);
		}
	}
}

void vectorToInterleaved(float *a,const Eigen::Vector3f &v)
{
	for(int i=0;i<3;i++)
	{
		a[i] = v(i);
	}
}

__device__ void rotateAndTranslate(float *rot,float *t, float *v, float *res)
{
	for(int i = 0;i<3;i++)
	{
		float sum = 0.f;
		for(int j = 0;j<3;j++)
		{
			sum += rot[i+3*j]*v[j];
		}
		res[i] = sum + t[i];
	}

}

__device__ void multiply(float *mat,float *v,float *res)
{
	for(int i = 0;i<3;i++)
	{
		float sum = 0.f;
		for(int j = 0;j<3;j++)
		{
			sum += mat[i+3*j]*v[j];
		}
		res[i] = sum;
	}

}

__global__ void computeAnalyticalGradient(float *d_K,float* d_ptrDepthRef,float * d_rotMat, float* d_t,
		float *d_gradx,float *d_grady,int w, int h,float *d_J)
{

	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;

		
	if(x<w && y<h)
	{
		size_t idx = x + (size_t)w*y;

		float residualRowJ[6];

		// camera intrinsics
		//float fx = K(0, 0);
		float fx = d_K[0]; // Using d_K[i+3j] = k(i,j)
		//float fy = K(1, 1);
		float fy = d_K[4];
		//float cx = K(0, 2);
		float cx = d_K[6];
		//float cy = K(1, 2);
		float cy = d_K[7];
		
		float fxInv = 1.0f / fx;
		float fyInv = 1.0f / fy;

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
            float pt3Ref[3] = {x0,y0,dRef};
            float pt3[3];
           
            rotateAndTranslate(d_rotMat,d_t,pt3Ref,pt3);
            
            if (pt3[2] > 0.0f)
            {
            
                // project 3d point to 2d
                float pt2CurH[3];
                multiply(d_K,pt3,pt2CurH);
            	//Eigen::Vector3f pt2CurH = K * pt3;
               
                float ptZinv = 1.0f / pt2CurH[2];
                float px = pt2CurH[0] * ptZinv;
                float py = pt2CurH[1] * ptZinv;

                // compute interpolated image gradient
                float dX = interpolate_gpu(d_gradx, px, py, w, h);
                float dY = interpolate_gpu(d_grady, px, py, w, h);
               
                if (dX<=1 && dY<=1)
                {
                	//printf("dx = %f dy = %f \n",dX,dY);
                    dX = fx * dX;
                    dY = fy * dY;
                    float pt3Zinv = 1.0f / pt3[2];

                    // shorter computation
                    residualRowJ[0] = dX * pt3Zinv;
                    residualRowJ[1] = dY * pt3Zinv;
                    residualRowJ[2] = - (dX * pt3[0] + dY * pt3[1]) * pt3Zinv * pt3Zinv;
                    residualRowJ[3] = - (dX * pt3[0] * pt3[1]) * pt3Zinv * pt3Zinv - dY * (1 + (pt3[1] * pt3Zinv) * (pt3[1] * pt3Zinv));
                    residualRowJ[4] = + dX * (1.0 + (pt3[0] * pt3Zinv) * (pt3[0] * pt3Zinv)) + (dY * pt3[0] * pt3[1]) * pt3Zinv * pt3Zinv;
                    residualRowJ[5] = (- dX * pt3[1] + dY * pt3[0]) * pt3Zinv;
                }
            }
        }
		
        // set 1x6 Jacobian row for current residual
        // invert Jacobian according to kerl2012msc.pdf (necessary?)
        for (int j = 0; j < 6; ++j){
        	size_t jidx = idx*6 + j;
        	if(jidx < w*h*6) {
            	d_J[idx*6 + j] = - residualRowJ[j];
            	}
        }

	}


}




void DVO::deriveAnalytic(const cv::Mat &grayRef, const cv::Mat &depthRef,
                   const cv::Mat &grayCur, const cv::Mat &depthCur,
                   const cv::Mat &gradX, const cv::Mat &gradY,
                   const Eigen::VectorXf &xi, const Eigen::Matrix3f &K,
                   float* residuals, float* J)
{
    // reference input images
    int w = grayRef.cols;
    int h = grayRef.rows;
    int n = w*h;
    const float* ptrDepthRef = (const float*)depthRef.data;

    // camera intrinsics
    //float fx = K(0, 0);
    //float fy = K(1, 1);
   // float cx = K(0, 2);
    //float cy = K(1, 2);
    //float fxInv = 1.0f / fx;
    //float fyInv = 1.0f / fy;

    // convert SE3 to rotation matrix and translation vector
    Eigen::Matrix3f rotMat;
    Eigen::Vector3f t;
    convertSE3ToTf(xi, rotMat, t);

    // calculate per-pixel residuals
    calculateError(grayRef, depthRef, grayCur, depthCur, xi, K, residuals);

    // reference gradient images
    const float* ptrGradX = (const float*)gradX.data;
    const float* ptrGradY = (const float*)gradY.data;

    // Using multi threading
    dim3 block =  dim3(32,32,1);
    dim3 grid = dim3((w+block.x-1)/block.x,(h+block.y-1)/block.y,1);



    // Converting from Matrix and Vector to float
    float h_tr[3];
    float h_rot[9];
    float h_K[9];

    vectorToInterleaved(h_tr,t);
    matToInterleaved(h_rot,rotMat);
    matToInterleaved(h_K,K);

    // Allocating device memory
    float *d_ptrDepthRef,*d_J,*d_gradx,*d_grady,*d_t,*d_K,*d_rotMat;


    cudaMalloc(&d_J,6*n*sizeof(float));CUDA_CHECK;
    cudaMalloc(&d_ptrDepthRef,w*h*sizeof(float));CUDA_CHECK;
    cudaMalloc(&d_gradx,w*h*sizeof(float));CUDA_CHECK;
    cudaMalloc(&d_grady,w*h*sizeof(float));CUDA_CHECK;
    cudaMalloc(&d_rotMat,9*sizeof(float));CUDA_CHECK;
    cudaMalloc(&d_K,9*sizeof(float));CUDA_CHECK;
    cudaMalloc(&d_t,3*sizeof(float));CUDA_CHECK;


    cudaMemcpy(d_ptrDepthRef,ptrDepthRef,w*h*sizeof(float),cudaMemcpyHostToDevice);CUDA_CHECK;
    cudaMemcpy(d_gradx,ptrGradX,w*h*sizeof(float),cudaMemcpyHostToDevice);CUDA_CHECK;
    cudaMemcpy(d_grady,ptrGradY,w*h*sizeof(float),cudaMemcpyHostToDevice);CUDA_CHECK;
    cudaMemcpy(d_rotMat,h_rot,9*sizeof(float),cudaMemcpyHostToDevice);CUDA_CHECK;
    cudaMemcpy(d_K,h_K,9*sizeof(float),cudaMemcpyHostToDevice);CUDA_CHECK;
    cudaMemcpy(d_t,h_tr,3*sizeof(float),cudaMemcpyHostToDevice);CUDA_CHECK;

   
   
	

    computeAnalyticalGradient<<<grid,block>>>(d_K,d_ptrDepthRef,d_rotMat,d_t,d_gradx,d_grady,w,h,d_J);

    cudaMemcpy(J,d_J,6*n*sizeof(float),cudaMemcpyDeviceToHost);CUDA_CHECK;

    cudaFree(d_K);CUDA_CHECK;
    cudaFree(d_ptrDepthRef);CUDA_CHECK;
    cudaFree(d_rotMat);CUDA_CHECK;
    cudaFree(d_J);CUDA_CHECK;
    cudaFree(d_t);CUDA_CHECK;
    cudaFree(d_gradx);CUDA_CHECK;
    cudaFree(d_grady);CUDA_CHECK;

}


void DVO::buildPyramid(const cv::Mat &depth, const cv::Mat &gray, std::vector<cv::Mat> &depthPyramid, std::vector<cv::Mat> &grayPyramid)
{
    grayPyramid.push_back(gray);
    depthPyramid.push_back(depth);

    for (int i = 1; i < numPyramidLevels_; ++i)
    {
        // downsample grayscale image
        cv::Mat grayDown = downsampleGray(grayPyramid[i-1]);
        grayPyramid.push_back(grayDown);

        // downsample depth image
        cv::Mat depthDown = downsampleDepth(depthPyramid[i-1]);
        depthPyramid.push_back(depthDown);
    }
}


void DVO::align(const cv::Mat &depthRef, const cv::Mat &grayRef, const cv::Mat &depthCur, const cv::Mat &grayCur, Eigen::Matrix4f &pose)
{
    // downsampling
    std::vector<cv::Mat> grayRefPyramid;
    std::vector<cv::Mat> depthRefPyramid;
    buildPyramid(depthRef, grayRef, depthRefPyramid, grayRefPyramid);

    std::vector<cv::Mat> grayCurPyramid;
    std::vector<cv::Mat> depthCurPyramid;
    buildPyramid(depthCur, grayCur, depthCurPyramid, grayCurPyramid);

    align(depthRefPyramid, grayRefPyramid, depthCurPyramid, grayCurPyramid, pose);
}


void DVO::align(const std::vector<cv::Mat> &depthRefPyramid, const std::vector<cv::Mat> &grayRefPyramid,
                const std::vector<cv::Mat> &depthCurPyramid, const std::vector<cv::Mat> &grayCurPyramid,
                Eigen::Matrix4f &pose)
{
    Vec6f xi;
    convertTfToSE3(pose, xi);

    Vec6f lastXi = Vec6f::Zero();

    int maxLevel = numPyramidLevels_-1;
    int minLevel = 1;
    float initGradDescStepSize = 1e-3f;
    float gradDescStepSize = initGradDescStepSize;

    Mat6f A;
    Mat6f diagMatA = Mat6f::Identity();
    Vec6f delta;

    for (int lvl = maxLevel; lvl >= minLevel; --lvl)
    {
        float lambda = 0.1f;

        int w = sizePyramid_[lvl].width;
        int h = sizePyramid_[lvl].height;
        int n = w*h;

        cv::Mat grayRef = grayRefPyramid[lvl];
        cv::Mat depthRef = depthRefPyramid[lvl];
        cv::Mat grayCur = grayCurPyramid[lvl];
        cv::Mat depthCur = depthCurPyramid[lvl];
        Eigen::Matrix3f kLevel = kPyramid_[lvl];
        //std::cout << "level " << level << " (size " << depthRef.cols << "x" << depthRef.rows << ")" << std::endl;

        // compute gradient images
        computeGradient(grayCur, gradX_[lvl], 0);
        computeGradient(grayCur, gradY_[lvl], 1);

        float errorLast = std::numeric_limits<float>::max();
        for (int itr = 0; itr < numIterations_; ++itr)
        {
            // compute residuals and Jacobian
#if 0
            deriveNumeric(grayRef, depthRef, grayCur, depthCur, xi, kLevel, residuals_[lvl], J_[lvl]);
#else
            deriveAnalytic(grayRef, depthRef, grayCur, depthCur, gradX_[lvl], gradY_[lvl], xi, kLevel, residuals_[lvl], J_[lvl]);
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
            float error = calculateError(residuals_[lvl], n);

            if (useWeights_)
            {
                // compute robust weights
                computeWeights(residuals_[lvl], weights_[lvl], n);
                // apply robust weights
                applyWeights(weights_[lvl], residuals_[lvl], n);
            }

            // compute update
            Vec6f b;
            compute_JtR(J_[lvl], residuals_[lvl], b, n);

            if (algo_ == GradientDescent)
            {
                // Gradient Descent
                delta = -gradDescStepSize * b * (1.0f / b.norm());
            }
            else if (algo_ == GaussNewton)
            {
                // Gauss-Newton algorithm
                compute_JtJ(J_[lvl], A, weights_[lvl], n, useWeights_);
                // solve using Cholesky LDLT decomposition
                delta = -(A.ldlt().solve(b));
            }
            else if (algo_ == LevenbergMarquardt)
            {
                // Levenberg-Marquardt algorithm
                compute_JtJ(J_[lvl], A, weights_[lvl], n, useWeights_);
                diagMatA.diagonal() = lambda * A.diagonal();
                delta = -((A + diagMatA).ldlt().solve(b));
            }

            // apply update: left-multiplicative increment on SE3
            lastXi = xi;
            xi = Sophus::SE3f::log(Sophus::SE3f::exp(delta) * Sophus::SE3f::exp(xi));
#if 0
            std::cout << "delta = " << delta.transpose() << " size = " << delta.rows() << " x " << delta.cols() << std::endl;
            std::cout << "xi = " << xi.transpose() << std::endl;
#endif

            // compute error again
            error = calculateError(residuals_[lvl], n);

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
