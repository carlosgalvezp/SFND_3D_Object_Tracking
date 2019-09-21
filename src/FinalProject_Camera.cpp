
/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.h"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"


/* MAIN PROGRAM */
std::vector<double> runExperiment(const DetectorType detectorType, const DescriptorType descriptorType)
{
    std::cout << "=======================================================" << std::endl;
    std::cout << "Experiment: " << detectorType << " + " << descriptorType << std::endl;
    std::cout << "=======================================================" << std::endl;

    /* INIT VARIABLES AND DATA STRUCTURES */
    // data location
    std::string dataPath = "../";

    // camera
    std::string imgBasePath = dataPath + "images/";
    std::string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    std::string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1;
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // object detection
    std::string yoloBasePath = dataPath + "dat/yolo/";
    std::string yoloClassesFile = yoloBasePath + "coco.names";
    std::string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    std::string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    std::string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    std::string lidarFileType = ".bin";

    // calibration data for camera and lidar
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector

    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;

    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;

    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    std::vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    // Camera-keypoint matching configuration
    MatcherType matcherType = MatcherType::BF;
    SelectorType selectorType = SelectorType::KNN;

    // Output data
    std::vector<double> all_ttc_camera;

    /* MAIN LOOP OVER ALL IMAGES */
    for (std::size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth)
    {
        /* LOAD IMAGE INTO BUFFER */
        // assemble filenames for current index
        std::ostringstream imgNumber;
        imgNumber << std::setfill('0') << std::setw(imgFillWidth) << imgStartIndex + imgIndex;
        std::string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file
        cv::Mat img = cv::imread(imgFullFilename);

        bVis = false;
        if (bVis)
        {
            cv::imshow("Input camera image", img);
            cv::waitKey(0);
        }
        bVis = false;

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = img;

        if (dataBuffer.size() < dataBufferSize)  // If the buffer is not yet full, simply push back
        {
            dataBuffer.push_back(frame);
        }
        else  // Otherwise shift data and place new frame in the back
        {
            // Shift contents in ring buffer
            for (std::size_t i = 1U; i < dataBuffer.size(); ++i)
            {
                dataBuffer[i - 1U] = dataBuffer[i];
            }

            // Add new image to the back
            dataBuffer.back() = frame;
        }

        std::cout << "#1 : LOAD IMAGE INTO BUFFER done" << std::endl;

        DataFrame& current_frame = dataBuffer.back();

        /* DETECT & CLASSIFY OBJECTS */

        float confThreshold = 0.2;
        float nmsThreshold = 0.4;

        bVis = false;
        detectObjects(current_frame.cameraImg, current_frame.boundingBoxes, confThreshold, nmsThreshold,
                      yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);
        bVis = false;

        std::cout << "#2 : DETECT & CLASSIFY OBJECTS done" << std::endl;

        /* CROP LIDAR POINTS */

        // load 3D Lidar points from file
        std::string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // remove Lidar points based on distance properties
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);

        current_frame.lidarPoints = lidarPoints;

        std::cout << "#3 : CROP LIDAR POINTS done" << std::endl;


        /* CLUSTER LIDAR POINT CLOUD */

        // associate Lidar points with camera-based ROI
        float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
        clusterLidarWithROI(current_frame.boundingBoxes, current_frame.lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

        // Visualize 3D objects
        bVis = false;
        if(bVis)
        {
            show3DObjects(current_frame.boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);
        }
        bVis = false;

        std::cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << std::endl;


        /* DETECT IMAGE KEYPOINTS */

        // convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor(current_frame.cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // extract 2D keypoints from current image
        std::vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        bVis = false;
        if (detectorType == DetectorType::SHITOMASI)
        {
            detKeypointsShiTomasi(keypoints, imgGray, bVis);
        }
        else if (detectorType == DetectorType::HARRIS)
        {
            detKeypointsHarris(keypoints, imgGray, bVis);
        }
        else
        {
            detKeypointsModern(keypoints, imgGray, detectorType, bVis);
        }
        bVis = false;

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType == DetectorType::SHITOMASI)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            std::cout << " NOTE: Keypoints have been limited!" << std::endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        current_frame.keypoints = keypoints;

        std::cout << "#5 : DETECT KEYPOINTS done" << std::endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        cv::Mat descriptors;
        descKeypoints(current_frame.keypoints, current_frame.cameraImg, descriptors, descriptorType);

        // push descriptors for current frame to end of data buffer
        current_frame.descriptors = descriptors;

        std::cout << "#6 : EXTRACT DESCRIPTORS done" << std::endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {
            DataFrame& previous_frame = *(dataBuffer.end() - 2);
            double ttcCamera = NAN;

            /* MATCH KEYPOINT DESCRIPTORS */
            std::vector<cv::DMatch> matches;

            matchDescriptors(previous_frame.keypoints, current_frame.keypoints,
                             previous_frame.descriptors, current_frame.descriptors,
                             matches, descriptorType, matcherType, selectorType);

            // store matches in current data frame
            current_frame.kptMatches = matches;

            bVis = false;
            if (bVis)
            {
                cv::Mat matchImg = current_frame.cameraImg.clone();
                cv::drawMatches(previous_frame.cameraImg, previous_frame.keypoints,
                                current_frame.cameraImg, current_frame.keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                std::vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                std::stringstream ss;
                ss << "[" << detectorType << ", " << descriptorType << "] "
                   << "Matching keypoints between two camera images";
                std::string windowName = ss.str();
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                std::cout << "Press key to continue to next image" << std::endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;

            std::cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << std::endl;

            /* TRACK 3D OBJECT BOUNDING BOXES */

            //// STUDENT ASSIGNMENT
            //// TASK FP.1 -> match list of 3D objects (std::vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
            std::map<int, int> bbBestMatches;
            matchBoundingBoxes(matches, previous_frame, current_frame, bbBestMatches); // associate bounding boxes between current and previous frame using keypoint matches
            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            current_frame.bbMatches = bbBestMatches;

            std::cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << std::endl;

            /* COMPUTE TTC ON OBJECT IN FRONT */

            // loop over all BB match pairs
            for (auto it1 = current_frame.bbMatches.begin(); it1 != current_frame.bbMatches.end(); ++it1)
            {
                // find bounding boxes associates with current match
                BoundingBox *prevBB, *currBB;
                for (auto it2 = current_frame.boundingBoxes.begin(); it2 != current_frame.boundingBoxes.end(); ++it2)
                {
                    if (it1->second == it2->boxID) // check wether current match partner corresponds to this BB
                    {
                        currBB = &(*it2);
                        break;
                    }
                }

                for (auto it2 = previous_frame.boundingBoxes.begin(); it2 != previous_frame.boundingBoxes.end(); ++it2)
                {
                    if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB
                    {
                        prevBB = &(*it2);
                        break;
                    }
                }

                // compute TTC for current match
                if( currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>0 ) // only compute TTC if we have Lidar points
                {
                    //// STUDENT ASSIGNMENT
                    //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                    double ttcLidar = computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate);
                    //// EOF STUDENT ASSIGNMENT

                    //// STUDENT ASSIGNMENT
                    //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                    //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
                    clusterKptMatchesWithROI(previous_frame.keypoints,
                                             current_frame.keypoints,
                                             current_frame.kptMatches,
                                             *currBB);
                    ttcCamera = computeTTCCamera(previous_frame.keypoints,
                                                 current_frame.keypoints,
                                                 currBB->kptMatches,
                                                 sensorFrameRate);
                    //// EOF STUDENT ASSIGNMENT

                    bVis = false;
                    if (bVis)
                    {
                        cv::Mat visImg = current_frame.cameraImg.clone();
                        showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
                        cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);

                        char str[200];
                        sprintf(str, "[%lu] TTC Lidar : %f s, TTC Camera : %f s", imgIndex, ttcLidar, ttcCamera);
                        putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));

                        std::string windowName = "Final Results : TTC";
                        cv::namedWindow(windowName, 4);
                        cv::imshow(windowName, visImg);
                        std::cout << "Press key to continue to next frame" << std::endl;
                        cv::waitKey(0);
                    }
                    bVis = false;

                } // eof TTC computation
            } // eof loop over all BB matches

            all_ttc_camera.push_back(ttcCamera);
        } // eof if data.size() > 1
    } // eof loop over all images

    return all_ttc_camera;
}

bool isValidExperiment(const DetectorType& detector_type, const DescriptorType& descriptor_type)
{
    // Cases documented not to work on UdacityHub
    bool output = true;

    if ((descriptor_type == DescriptorType::AKAZE) && (detector_type != DetectorType::AKAZE))
    {
        // AZAKE descriptor can only be used with KAZE or AKAZE keypoints
        output = false;
    }
    else if ((detector_type == DetectorType::SIFT) && (descriptor_type == DescriptorType::ORB))
    {
        // out-of-memory errors with this combination
        output = false;
    }

    return output;
}

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    // Define experiments
    const std::vector<DetectorType> detectors =
    {
        DetectorType::AKAZE,
        DetectorType::BRISK,
        DetectorType::FAST,
        DetectorType::HARRIS,
        DetectorType::ORB,
        DetectorType::SHITOMASI,
        DetectorType::SIFT,
    };

    const std::vector<DescriptorType> descriptors =
    {
        DescriptorType::BRISK,
        DescriptorType::BRIEF,
        DescriptorType::ORB,
        DescriptorType::FREAK,
        DescriptorType::AKAZE,
        DescriptorType::SIFT
    };

    std::vector<std::vector<double>> ttcs_camera;

    if (argc == 2 && std::string(argv[1]) == "best")
    {
        const DetectorType best_detector = DetectorType::FAST;
        const DescriptorType best_descriptor = DescriptorType::ORB;

        ttcs_camera.push_back(runExperiment(best_detector, best_descriptor));
    }
    else
    {
        // Run all combinations
        for (const DetectorType& detector_type : detectors)
        {
            for (const DescriptorType& descriptor_type : descriptors)
            {
                if (isValidExperiment(detector_type, descriptor_type))
                {
                    ttcs_camera.push_back(runExperiment(detector_type, descriptor_type));
                }
                else
                {
                    ttcs_camera.push_back(std::vector<double>(18, NAN));
                }
            }
        }
    }

    // Print data to copy-paste into spreadsheet
    for (const auto& ttc_data : ttcs_camera)
    {
        for (const double ttc_frame_i : ttc_data)
        {
            std::cout << ttc_frame_i << ", ";
        }
        std::cout << std::endl;
    }

    return 0;
}
