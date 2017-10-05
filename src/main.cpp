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
#include <opencv2/viz/viz3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/affine.hpp>

#include "dvo.hpp"
#include "tum_benchmark.hpp"


#define STR1(x)  #x
#define STR(x)  STR1(x)

#define SKIPPEDFRAMES 20

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

    cv::Mat colors(h,w,CV_8UC3);


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
	for(int i = 0; i < SKIPPEDFRAMES; i++)
            capture->grab();

        cv::Mat depthIn, grayIn;
        capture->retrieve( depthIn, cv::CAP_OPENNI_DEPTH_MAP );
      	capture->retrieve( grayIn, cv::CAP_OPENNI_GRAY_IMAGE );
        depthRef = convertDepth(depthIn);  
        grayRef = convertGray(grayIn);  
        cv::flip(grayRef,grayRef,1);
	cv::flip(depthRef,depthRef,1);
       	timestamps.push_back((double)cv::getTickCount()/cv::getTickFrequency());
	numFrames = 10000;
	maxFrames = 10000;
	
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
        maxFrames = 600;
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

    // code for visualization
    cv::viz::Viz3d mainWindow("Odometry");
    mainWindow.setWindowSize(cv::Size(1600,900));
    cv::Matx33f vizK(K(0,0),K(0,1),K(0,2),
                    K(1,0),K(1,1),K(1,2),
                    K(2,0),K(2,1),K(2,2));
    std::cout << vizK << std::endl;

    std::vector<cv::Affine3f> vizPoses;
    vizPoses.push_back(cv::Affine3f());
    mainWindow.setViewerPose(vizPoses[0]);


    for (size_t i = 1; i < numFrames && (maxFrames < 0 || i < maxFrames); ++i)
    {
        std::cout << "aligning frames " << (i-1) << " and " << i  << std::endl;
        double timeDepth1;
        cv::Mat depthCur;
        cv::Mat grayCur;

	cv::Mat depthIn, grayIn;

        if(useKinect)
        {
            //get images from camera
            capture->grab();
            capture->retrieve( depthIn, cv::CAP_OPENNI_DEPTH_MAP );
            capture->retrieve( grayIn, cv::CAP_OPENNI_GRAY_IMAGE );
	    depthCur = convertDepth(depthIn);  
	    grayCur = convertGray(grayIn);  
        cv::flip(grayCur,grayCur,1);
        cv::flip(depthCur,depthCur,1);

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
        grayCur.convertTo(colors, CV_8UC3, 255);


	cv::imshow( "depthCur", depthCur );
	cv::imshow( "grayCur", grayCur );		
	//cv::imshow( "depthRef", depthRef );
	//cv::imshow( "grayRef", grayRef );
	if( cv::waitKey(1) >= 0 )
        	break;
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
        mainWindow.showWidget("cameras_line",cv::viz::WTrajectory(vizPoses, cv::viz::WTrajectory::PATH, 0.1, cv::viz::Color::green()));
        mainWindow.showWidget("cameras_frustums", cv::viz::WTrajectoryFrustums(vizPoses, vizK, 0.1, cv::viz::Color::red()));

        //reconstruct current scene
        float fx = K(0,0);
        float fy = K(1,1);
        float cx = K(0,2);
        float cy = K(1,2);
        float fxInv = 1.0f / fx;
        float fyInv = 1.0f / fy;
        cv::Mat points(h,w,CV_32FC3);
        double dNaN = std::numeric_limits<float>::quiet_NaN();

        for(int x = 0; x < w; x++) {
	           for(int y = 0; y < h; y++) {
                   float depth = depthCur.at<float>(y,x);
			       if(depth > 0.0f) {
			           float x0 = (static_cast<float>(x) - cx) * fxInv;
        	                   float y0 = (static_cast<float>(y) - cy) * fyInv;
	                           float scale = 1.0f;

				   float x1 = depth * x0;
				   float y1 = depth * y0;
				   float z1 = depth * scale;
                                   points.at<cv::Vec3f>(y,x) = cv::Vec3f(x1,y1,z1);
                               } else {
                                    points.at<cv::Vec3f>(y,x) = cv::Vec3f(dNaN,dNaN,dNaN);
                                }
			            }
	   }
         cv::viz::WCloud cloudWidget(points, colors);
         mainWindow.showWidget("pointCloud", cloudWidget,vizPoses[i]);

         mainWindow.spinOnce(1);





    }
    std::cout << "average runtime: " << (runtimeAvg / framesProcessed) * 1000.0 << " ms" << std::endl;

    mainWindow.spin();
    // save poses
    savePoses(dataFolder + "traj.txt", poses, timestamps);

    // clean up
    cv::destroyAllWindows();
    cublasDestroy(handle);

    if(useKinect)
    	delete(capture);

    std::cout << "Direct Image Alignment finished." << std::endl;

    return 0;
}
