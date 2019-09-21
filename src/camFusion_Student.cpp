
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

namespace
{

double distance(const LidarPoint& a, const LidarPoint& b)
{
    return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

template<typename T>
double median(std::vector<T>& data)
{
    std::sort(data.begin(), data.end());
    T output{};

    const std::size_t size = data.size();

    if ((size % 2U) == 0U)
    {
        output = static_cast<T>(0.5) * (data[(size / 2U) - 1U] + data[size / 2U]);
    }
    else
    {
        output = data[size / 2U];
    }

    return output;
}

double findClosestLidarPointInLane(const std::vector<LidarPoint>& lidar_points)
{
    // Create a vector of x disances and sort them
    std::vector<double> x_distances;
    for (const LidarPoint& point : lidar_points)
    {
        x_distances.push_back(point.x);
    }

    std::sort(x_distances.begin(), x_distances.end());

    // Take the N closest points
    const std::size_t kNumberClosestPoints = 10U;
    std::vector<float> closest_x_distances(kNumberClosestPoints);
    std::copy(x_distances.begin(), x_distances.begin() + kNumberClosestPoints, closest_x_distances.begin());

    // Compute median to remove outliers
    return median(closest_x_distances);
}

template <typename T>
bool in(const T& x, const std::vector<T>& array)
{
    return (std::find(array.begin(), array.end(), x) != array.end());
}

struct BoxIdxPair
{
    int prev_box_idx;
    int curr_box_idx;
    int n_matched_keypoints;
};

bool operator==(const BoxIdxPair& a, const BoxIdxPair& b)
{
    return (a.prev_box_idx == b.prev_box_idx) &&
           (a.curr_box_idx == b.curr_box_idx);
}

}  // namespace


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        std::vector<std::vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (std::vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0;
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    std::string windowName = "3D Objects";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(const std::vector<cv::KeyPoint> &kptsPrev,  // train
                              const std::vector<cv::KeyPoint> &kptsCurr,  // query
                              const std::vector<cv::DMatch> &kptMatches,
                              BoundingBox &boundingBoxCurr)
{
    // First, compute median (robust) of distance between descriptors to filter out later
    std::vector<float> match_distances;
    for (const cv::DMatch& match : kptMatches)
    {
        match_distances.push_back(match.distance);
    }
    const float median_match_distance = median(match_distances);

    // Loop over matches and assign to bounding box if points are contained
    for (const cv::DMatch& match : kptMatches)
    {
        if (((match.distance > 0.5 * median_match_distance) ||
             (match.distance < 1.5 * median_match_distance)) &&
            boundingBoxCurr.roi.contains(kptsCurr[match.queryIdx].pt))
        {
            boundingBoxCurr.kptMatches.push_back(match);
        }
    }
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
double computeTTCCamera(const std::vector<cv::KeyPoint>& kptsPrev,
                        const std::vector<cv::KeyPoint>& kptsCurr,
                        const std::vector<cv::DMatch>& kptMatches,
                        const double frameRate)
{
    // compute distance ratios between all matched keypoints
    std::vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (const cv::DMatch& match1 : kptMatches)
    {
        // get current keypoint and its matched partner in the prev. frame
        const cv::KeyPoint& kpOuterCurr = kptsCurr[match1.trainIdx];
        const cv::KeyPoint& kpOuterPrev = kptsPrev[match1.queryIdx];

        for (const cv::DMatch& match2 : kptMatches)
        {
            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(match2.trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(match2.queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }
    }

    // only continue if list of distance ratios is not empty
    double TTC;
    if (distRatios.size() == 0)
    {
        TTC = NAN;
    }
    else
    {
        // compute camera-based TTC from distance ratios
        double medianDistRatio = median(distRatios);

        double dT = 1 / frameRate;
        TTC = -dT / (1 - medianDistRatio);
    }
    return TTC;
}


double computeTTCLidar(const std::vector<LidarPoint>& lidarPointsPrev,
                       const std::vector<LidarPoint>& lidarPointsCurr,
                       const double frameRate)
{
    // Find closest lidar point in front of the car, for both previous and current frame
    const double minXPrev = findClosestLidarPointInLane(lidarPointsPrev);
    const double minXCurr = findClosestLidarPointInLane(lidarPointsCurr);

    std::cout << "min x prev: " << minXPrev << ", min x curr: " << minXCurr << std::endl;

    // Compute and return TTC
    const double dT = 1.0 / frameRate;
    const double TTC = (minXCurr * dT) / (minXPrev - minXCurr);

    std::cout << "TTC LIDAR: " << TTC << std::endl;

    return TTC;
}


void matchBoundingBoxes(const std::vector<cv::DMatch>& matches,
                        const DataFrame& prevFrame,
                        const DataFrame& currFrame,
                        std::map<int, int>& bbBestMatches)
{
    // Prev frame = query
    // Curr frame = train
    // map<int, int> = map<prev, curr>
    // First, go through all matches and store the number of successfull matches
    std::vector<BoxIdxPair> box_matches;

    for (const cv::DMatch& match : matches)
    {
        for (const BoundingBox& prev_box : prevFrame.boundingBoxes)
        {
            for (const BoundingBox& curr_box : currFrame.boundingBoxes)
            {
                if (prev_box.roi.contains(prevFrame.keypoints[match.queryIdx].pt) &&
                    curr_box.roi.contains(currFrame.keypoints[match.trainIdx].pt))
                {
                    BoxIdxPair pair;
                    pair.prev_box_idx = prev_box.boxID;
                    pair.curr_box_idx = curr_box.boxID;

                    const auto it = std::find(box_matches.begin(), box_matches.end(), pair);
                    if (it == box_matches.end())
                    {
                        // This pair is new -> add to vector
                        pair.n_matched_keypoints = 1;
                        box_matches.push_back(pair);
                    }
                    else
                    {
                        // Existing pair -> increase number of keypoints
                        it->n_matched_keypoints++;
                    }
                }
            }
        }
    }

    // Sort matches keeping the ones with larger number of matches first
    std::sort(box_matches.begin(), box_matches.end(), [](const BoxIdxPair& a, const BoxIdxPair& b) {
        return a.n_matched_keypoints > b.n_matched_keypoints;
    });

    // Create final output
    std::vector<int> matched_prev_boxes;
    std::vector<int> matched_curr_boxes;

    for (const auto& box_match : box_matches)
    {
        const int idx_prev = box_match.prev_box_idx;
        const int idx_curr = box_match.curr_box_idx;

        if (!in(idx_prev, matched_prev_boxes) && !in(idx_curr, matched_curr_boxes))
        {
            bbBestMatches[idx_prev] = idx_curr;

            matched_prev_boxes.push_back(idx_prev);
            matched_curr_boxes.push_back(idx_curr);

            std::cout << "Matching prev box " << idx_prev << " with curr box " << idx_curr << std::endl;
        }
    }
}
