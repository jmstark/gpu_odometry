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
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz/viz3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/affine.hpp>

#include "dvo.hpp"
#include "tum_benchmark.hpp"


#define STR1(x)  #x
#define STR(x)  STR1(x)

cublasHandle_t handle;

int main(int argc, char *argv[])
{
    std::string dataFolder = std::string(STR(DVO_SOURCE_DIR)) + "/data/";

    cublasCreate(&handle);

    Eigen::Matrix3f K;
#if 1
    // initialize intrinsic matrix: fr1
    K <<    517.3, 0.0, 318.6,
            0.0, 516.5, 255.3,
            0.0, 0.0, 1.0;
    if(argc<2)
    {
      std::cerr<<"Usage: ./dvo <path to data folder>\n";
      return 1;
    }
    dataFolder = argv[1];
    //dataFolder = "/work/maierr/rgbd_data/rgbd_dataset_freiburg1_desk2/";
#else
    dataFolder = "/work/maierr/rgbd_data/rgbd_dataset_freiburg3_long_office_household/";
    // initialize intrinsic matrix: fr3
    K <<    535.4, 0.0, 320.1,
            0.0, 539.2, 247.6,
            0.0, 0.0, 1.0;
#endif
    //std::cout << "Camera matrix: " << K << std::endl;

    // load file names
    std::string assocFile = dataFolder + "rgbd_assoc.txt";
    std::cout << assocFile << std::endl;
    std::vector<std::string> filesColor;
    std::vector<std::string> filesDepth;
    std::vector<double> timestampsDepth;
    std::vector<double> timestampsColor;
    if (!loadAssoc(assocFile, filesDepth, filesColor, timestampsDepth, timestampsColor))
    {
        std::cout << "Assoc file could not be loaded!" << std::endl;
        return 1;
    }
    int numFrames = filesDepth.size();

    int maxFrames = -1;
    maxFrames = 100;

    Eigen::Matrix4f absPose = Eigen::Matrix4f::Identity();
    std::vector<Eigen::Matrix4f> poses;
    std::vector<double> timestamps;
    poses.push_back(absPose);
    timestamps.push_back(timestampsDepth[0]);

    cv::Mat grayRef = loadGray(dataFolder + filesColor[0]);
    cv::Mat depthRef = loadDepth(dataFolder + filesDepth[0]);
    int w = depthRef.cols;
    int h = depthRef.rows;

    DVO dvo;
    dvo.init(w, h, K);

    std::vector<cv::gpu::GpuMat> grayRefGPUPyramid;
    std::vector<cv::gpu::GpuMat> depthRefGPUPyramid;
    dvo.buildPyramid(depthRef, grayRef, depthRefGPUPyramid, grayRefGPUPyramid);

    // process frames
    double runtimeAvg = 0.0;
    int framesProcessed = 0;

    // code for visualization
    cv::viz::Viz3d mainWindow("Odometry");
    mainWindow.setWindowSize(cv::Size(1600,900));
    cv::Matx33f vizK(K(0,0),K(0,1),K(0,2),
                    K(1,0),K(1,1),K(1,2),
                    K(2,0),K(2,1),K(2,2));
    std::cout << vizK << std::endl;

    std::vector<cv::Affine3f> vizPoses;
    vizPoses.push_back(cv::Affine3f());


    for (size_t i = 1; i < numFrames && (maxFrames < 0 || i < maxFrames); ++i)
    {
        std::cout << "aligning frames " << (i-1) << " and " << i  << std::endl;

        // load input frame
        std::string fileColor1 = filesColor[i];
        std::string fileDepth1 = filesDepth[i];
        double timeDepth1 = timestampsDepth[i];
        //std::cout << "File " << i << ": " << fileColor1 << ", " << fileDepth1 << std::endl;
        cv::Mat grayCur = loadGray(dataFolder + fileColor1);
        cv::Mat depthCur = loadDepth(dataFolder + fileDepth1);
        // build pyramid
        std::vector<cv::gpu::GpuMat> grayCurGPUPyramid;
        std::vector<cv::gpu::GpuMat> depthCurGPUPyramid;
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

        float data[] = {
            absPose(0,0),absPose(0,1),absPose(0,2),absPose(0,3),
            absPose(1,0),absPose(1,1),absPose(1,2),absPose(1,3),
            absPose(2,0),absPose(2,1),absPose(2,2),absPose(2,3),
            absPose(3,0),absPose(3,1),absPose(3,2),absPose(3,3)
        };

        // matPose;
        //cv::eigen2cv(absPose, matPose);
        vizPoses.push_back(cv::Affine3f(data));

        /*mainWindow.showWidget("camera"+i,cv::viz::WCameraPosition(vizK,0.02),vizPoses[i]);

        cv::Vec3f start = vizPoses[i-1].translation();
        cv::Vec3f end = vizPoses[i].translation();

        mainWindow.showWidget("line"+i,cv::viz::WLine(cv::Point3f(start),cv::Point3f(end),cv::viz::Color::green()));
        */
        mainWindow.showWidget("cameras_line",cv::viz::WTrajectory(vizPoses, cv::viz::WTrajectory::PATH, 0.05, cv::viz::Color::green()));
        mainWindow.showWidget("cameras_frustums", cv::viz::WTrajectoryFrustums(vizPoses, vizK, 0.05, cv::viz::Color::red()));
        mainWindow.spinOnce(30);





    }
    std::cout << "average runtime: " << (runtimeAvg / framesProcessed) * 1000.0 << " ms" << std::endl;

    mainWindow.spin();
    // save poses
    savePoses(dataFolder + "traj.txt", poses, timestamps);

    // clean up
    cv::destroyAllWindows();
    cublasDestroy(handle);
    std::cout << "Direct Image Alignment finished." << std::endl;

    return 0;
}
