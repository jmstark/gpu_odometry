#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iomanip>

#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include "dvo.hpp"
#include "tum_benchmark.hpp"

#define STR1(x)  #x
#define STR(x)  STR1(x)

cublasHandle_t handle;

int main(int argc, char *argv[])
{
    cv::VideoCapture* capture;
    bool useKinect = false;
    int maxFrames = -1;
    int numFrames;
    std::string assocFile;
    std::vector<std::string> filesColor;
    std::vector<std::string> filesDepth;
    std::vector<double> timestampsDepth;
    std::vector<double> timestampsColor;
    std::vector<double> timestamps;
    cv::Mat grayRef;
    cv::Mat depthRef;  
    int w, h;
    std::string dataFolder = std::string(STR(DVO_SOURCE_DIR)) + "/data/";
    cublasCreate(&handle);
    Eigen::Matrix3f K;

    // initialize intrinsic matrix: fr1
    K <<    517.3, 0.0, 318.6,
            0.0, 516.5, 255.3,
            0.0, 0.0, 1.0;
    //Code for initializing kinect stuff
    if(argc<2)
    {
        useKinect = true;
        // CAP_OPENNI2_ASUS driver does not support grabbing bgr-images (Image generator), while this one does
        capture = new cv::VideoCapture( cv::CAP_OPENNI2 );
       	if( !capture->isOpened() )
        {
            std::cout << "***** Can not open a capture object." << std::endl;
            return -1;
        }
        // Print some avalible device settings.
        if (capture->get(cv::CAP_OPENNI_DEPTH_GENERATOR_PRESENT))
        {
            std::cout << "\nDepth generator output mode:" << std::endl <<
                "FRAME_WIDTH      " << capture->get(cv::CAP_PROP_FRAME_WIDTH) << std::endl <<
                "FRAME_HEIGHT     " << capture->get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl <<
                "FRAME_MAX_DEPTH  " << capture->get(cv::CAP_PROP_OPENNI_FRAME_MAX_DEPTH) << " mm" << std::endl <<
                "FPS              " << capture->get(cv::CAP_PROP_FPS) << std::endl <<
                "REGISTRATION     " << capture->get(cv::CAP_PROP_OPENNI_REGISTRATION) << std::endl <<
                "IMG_GEN          " << capture->get(cv::CAP_OPENNI_IMAGE_GENERATOR_PRESENT) << std::endl;
        }
        else
        {
            std::cout << "\nDevice doesn't contain depth generator or it is not selected." << std::endl;
            return 1;
        }
        //get (first) reference images from camera
        capture->grab();
        capture->retrieve( depthRef, cv::CAP_OPENNI_DEPTH_MAP );
        capture->retrieve( grayRef, cv::CAP_OPENNI_GRAY_IMAGE );
        timestamps.push_back((double)cv::getTickCount()/cv::getTickFrequency());
    }
    //Code for initializing stuff for reading images from files
    else
    {
        dataFolder = std::string(argv[1]) + "/";
        // load file names
        assocFile = dataFolder + "rgbd_assoc.txt";
        if (!loadAssoc(assocFile, filesDepth, filesColor, timestampsDepth, timestampsColor))
        {
            std::cout << "Assoc file could not be loaded!" << std::endl;
            return 1;
        }
        numFrames = filesDepth.size();
        maxFrames = 100;
        timestamps.push_back(timestampsDepth[0]);
        grayRef = loadGray(dataFolder + filesColor[0]);
        depthRef = loadDepth(dataFolder + filesDepth[0]);
    }
    
    w = depthRef.cols;
    h = depthRef.rows;    
    Eigen::Matrix4f absPose = Eigen::Matrix4f::Identity();
    std::vector<Eigen::Matrix4f> poses;
    poses.push_back(absPose);

    DVO dvo;
    dvo.init(w, h, K);

    std::vector<cv::cuda::GpuMat> grayRefGPUPyramid;
    std::vector<cv::cuda::GpuMat> depthRefGPUPyramid;
    dvo.buildPyramid(depthRef, grayRef, depthRefGPUPyramid, grayRefGPUPyramid);

    // process frames
    double runtimeAvg = 0.0;
    int framesProcessed = 0;
    for (size_t i = 1; i < numFrames && (maxFrames < 0 || i < maxFrames); ++i)
    {
        std::cout << "aligning frames " << (i-1) << " and " << i  << std::endl;
        double timeDepth1;
        cv::Mat depthCur;
        cv::Mat grayCur;

        if(useKinect)
        {
            //get images from camera
            capture->grab();
            capture->retrieve( depthCur, cv::CAP_OPENNI_DEPTH_MAP );
            capture->retrieve( grayCur, cv::CAP_OPENNI_GRAY_IMAGE );  
            timeDepth1 = (double)cv::getTickCount()/cv::getTickFrequency();
        }
        else
        {
        // load input frame
        std::string fileColor1 = filesColor[i];
        std::string fileDepth1 = filesDepth[i];
        timeDepth1 = timestampsDepth[i];
        //std::cout << "File " << i << ": " << fileColor1 << ", " << fileDepth1 << std::endl;
        grayCur = loadGray(dataFolder + fileColor1);
        depthCur = loadDepth(dataFolder + fileDepth1);
        }
        // build pyramid
        std::vector<cv::cuda::GpuMat> grayCurGPUPyramid;
        std::vector<cv::cuda::GpuMat> depthCurGPUPyramid;
        dvo.buildPyramid(depthCur, grayCur, depthCurGPUPyramid, grayCurGPUPyramid);

        // frame alignment
        double tmr = (double)cv::getTickCount();

        Eigen::Matrix4f relPose = Eigen::Matrix4f::Identity();
        dvo.align(depthRefGPUPyramid, grayRefGPUPyramid, depthCurGPUPyramid, grayCurGPUPyramid, relPose);

        tmr = ((double)cv::getTickCount() - tmr)/cv::getTickFrequency();
        runtimeAvg += tmr;

        // concatenate poses
        absPose = absPose * relPose.inverse();
        poses.push_back(absPose);
        timestamps.push_back(timeDepth1);

        depthRefGPUPyramid = depthCurGPUPyramid;
        grayRefGPUPyramid = grayCurGPUPyramid;
        ++framesProcessed;
    }
    std::cout << "average runtime: " << (runtimeAvg / framesProcessed) * 1000.0 << " ms" << std::endl;

    // save poses
    savePoses(dataFolder + "traj.txt", poses, timestamps);

    // clean up
    cv::destroyAllWindows();
    cublasDestroy(handle);
    delete(capture);
    std::cout << "Direct Image Alignment finished." << std::endl;
    return 0;
}
