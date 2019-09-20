#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
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

enum class DetectorType
{
    SHITOMASI,
    HARRIS,
    FAST,
    BRISK,
    ORB,
    AKAZE,
    SIFT
};

enum class DescriptorType
{
    BRISK,
    BRIEF,
    ORB,
    FREAK,
    AKAZE,
    SIFT
};

enum class DescriptorFormat
{
    BINARY,
    HOG
};

enum class MatcherType
{
    BF,
    FLANN
};

enum class SelectorType
{
    NN,
    KNN
};

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints,
                        cv::Mat &img, bool bVis);

void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints,
                           cv::Mat &img,
                           bool bVis);

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints,
                        cv::Mat &img,
                        DetectorType detectorType,
                        bool bVis);

void descKeypoints(std::vector<cv::KeyPoint> &keypoints,
                   cv::Mat &img,
                   cv::Mat &descriptors,
                   DescriptorType descriptorType);

void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource,
                      std::vector<cv::KeyPoint> &kPtsRef,
                      cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches,
                      DescriptorType descriptorType,
                      MatcherType matcherType,
                      SelectorType selectorType);

std::ostream& operator<<(std::ostream& os, const DetectorType& x);
std::ostream& operator<<(std::ostream& os, const DescriptorType& x);
std::ostream& operator<<(std::ostream& os, const DescriptorFormat& x);
std::ostream& operator<<(std::ostream& os, const MatcherType& x);
std::ostream& operator<<(std::ostream& os, const SelectorType& x);

DescriptorFormat getDescriptorFormat(const DescriptorType& descriptor_type);

#endif /* matching2D_hpp */
