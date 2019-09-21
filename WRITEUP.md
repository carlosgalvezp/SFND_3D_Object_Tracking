Sensor Fusion Nanodegree - 3D Object Tracking Project
======================================================

In this document we summarize the work done for the 3D Object Tracking Project,
specifying how the different points in the rubric are fulfilled.

![](images/report/final_output.png)

FP.0 Final Report
--------------------
```
Provide a Writeup / README that includes all the rubric points and how you
addressed each one. You can submit your writeup as markdown or pdf.
```

This document is addressing the rubric points as requested.

FP.1 Match 3D Objects
---------------------
```
Implement the method "matchBoundingBoxes", which takes as input both the previous
and the current data frames and provides as output the ids of the matched regions
of interest (i.e. the boxID property). Matches must be the ones with the highest
number of keypoint correspondences.
```

The method `matchBoundingBoxes` is implemented in `camFusion_Student.cpp`.
A high-level description of the process is as follows:

0. Create a data structure `BoxIdxPair` that contains 3 elements:

   - The `boxID` of a bounding box in the previous frame.
   - The `boxID` of a bounding box in the current frame.
   - The number of keypoint matches between those two boxes.

1. Create a vector of `BoxIdxPair` (`box_matches` in the code) to store
   the number of valid matches for every previous-current box pair combination.

2. Go through all the input `kptMatches`. Go through all previous and current
   boxes. If a given combination of boxes contains the matched keypoints,
   increase a counter for that box pair. The result is that every combination
   of pairs of bounding boxes between previous and current frame have
   associated a number of keypoint matches.

2. Sort the vector `box_matches` according to the number of keypoint matches
   (the pairs with more keypoints go first).

3. Loop over the sorted vector and extract the pairs with higher number
   of correspondences, storing the output in `bbBestMatches`.
   Keep track of which boxes have already been assigned to ensure a 1:1 mapping.


FP.2 Compute Lidar-based TTC
---------------------------
```
Compute the time-to-collision in second for all matched 3D objects using only Lidar
measurements from the matched bounding boxes between current and previous frame.
```

This is implemented in the function `computeTTCLidar`, following the theory
given in the lectures:

```cpp
double computeTTCLidar(const std::vector<LidarPoint>& lidarPointsPrev,
                       const std::vector<LidarPoint>& lidarPointsCurr,
                       const double frameRate)
{
    // Find closest lidar point in front of the car, for both previous and current frame
    const double minXPrev = findClosestLidarPointInLane(lidarPointsPrev);
    const double minXCurr = findClosestLidarPointInLane(lidarPointsCurr);

    // Compute and return TTC
    const double dT = 1.0 / frameRate;
    const double TTC = (minXCurr * dT) / (minXPrev - minXCurr);

    return TTC;
}
```

We compute the closest lidar point in the ego lane, for both the previous
and current frame, and compute the TTC using that and the sensor frame rate.
Since the lidar points have already been filtered out to be in the ego
lane we don't need to care about that here.

However we do need to extract the closest point reliably, being robust against
sporious points that might end up between the ego lane and the actual
preceeding vehicle.

To do this, we implemented a helper function `findClosestLidarPointInLane`,
as follows:

```cpp
double findClosestLidarPointInLane(const std::vector<LidarPoint>& lidar_points)
{
    // Create a vector of x distances and sort them
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
```

The process is quite straightforward:

1. Create a vector to store the `x` component of all lidar points, which
   correspond to the distance to the preceeding vehicle.
2. Sort them to keep the closest points first.
3. Take the first `N` (in this case, `N = 10`) points from the sorted
   vector and store it (`closest_x_distances`). These points should contain
   mostly real points belonging to the rear of the preceeding vehicle,
   (which should have a similar value of distance),
   plus possibly a few outliers.
4. Finally, compute the estimate of distance to the preceeding vehicle
   by taking the `median` over `closest_x_distances`, which should be
   robust against possible outliers and return the true distance to the
   preceeding car.

Of course every lidar point (spurious or not) will have associated some
measurement error since the sensor is not perfect, but this is the best
estimate we can obtain with the data we have.

FP.3 Associate Keypoint Correspondences with Bounding Boxes
-----------------------------------------------------------
```
Prepare the TTC computation based on camera measurements by associating keypoint
correspondences to the bounding boxes which enclose them. All matches which satisfy
this condition must be added to a vector in the respective bounding box.
```

This is implemented in the `clusterKptMatchesWithROI` function. The high-level overview
is described below:

1. First, keep only the keypoint matches that have enclosed keypoints in the current frame.
2. For each of those selected matches, compute the Euclidean distance between the matched
   points. Most of these matches should have the same distance since they all should
   belong to a rigid object (the preceeding vehicle). Outliers (i.e. bad matches)
   will have distances way different than the other distances. Store all these distances
   in a vector (`keypoint_distances`).
3. Compute the median of this vector (`median_distance`) in order to get a robust estimate
   of what the distance between keypoints should be.
4. Finally, store in the output only those matches whose distance between keypoints
   is close enough to the median computed previously. We found empirically that keeping
   points that are `>0.5 * median` and `<2.0 * median` gives a good result, filtering out
   bad matches.


FP.4 Compute Camera-based TTC
-----------------------------
```
Compute the time-to-collision in second for all matched 3D objects using only
keypoint correspondences from the matched bounding boxes between current and
previous frame.
```
This is implemented in the `computeTTCCamera` function, following previous lectures.
The workflow is as follows:

1. Create a vector to store all the distance ratios between matches.
2. In a double for-loop iterating over the keypoint matches, compute the distances between
   all keypoints, for both the previous and current frame. For each combination,
   compute the `distanceRatio` as the ratio between the distance between keypoints
   in the current frame and the distance between keypoints in the previous frame.
3. Add all these distance ratios to a vector and compute the median of it, `medianDistRatio`, to
   obtain a robust estimate.
4. Finally compute the TTC with the formula given in the lectures:
   ```cpp
    double dT = 1.0 / frameRate;
    TTC = -dT / (1.0 - medianDistRatio);
    ```


FP.5 Performance Evaluation 1
-----------------------------
```
Find examples where the TTC estimate of the Lidar sensor does not seem plausible.
Describe your observations and provide a sound argumentation why you think this
happened.
```

To estimate the performance of TTC, we need an independent source of information
(ground truth). We don't have this available in the course, so our performance
evaluation will not be as reliable as it could be.

We estimate the "real" TTC manually by observing the lidar cloud in top-view.
The following data was collected:

```
Frame 0, t = 0.0: x_min = 7.97 m
Frame 1, t = 0.1: x_min = 7.91 m
Frame 2, t = 0.2: x_min = 7.85 m
Frame 3, t = 0.3: x_min = 7.79 m
```

Those distances were extracted from the top-view point cloud, were no significant
outliers were observed, so we should be able to trust these measurements.

We notice that distance to the preceeding vehicle is reduced by 0.06m every 0.1 seconds,
so the relative velocity between the ego and preceeding vehicle is 0.06 / 0.1 = 0.6 m/s.

Therefore, we would expect the following "real" estimated TTCs (assuming constant relative
velocity):
```
Frame 0, TTC = 13.28 m
Frame 1, TTC = 13.18 m
Frame 2, TTC = 13.08 m
Frame 3, TTC = 12.98 m
```

In our experiments, we observe that in general the TTC obtained from lidar is in line with
the previous "real" estimates, around 13 seconds TTC, decreasing over time as the ego vehicle
gets closer to the preceeding vehicle.

However we find a couple cases when the estimate is way off:

**Example 1**

*Observation*

In frame 6, we obtain a TTC from lidar of 7 seconds, which is too little:

![](images/report/lidar_7.png)

*Motivation*

Let's take a look at the lidar point clouds for frame 5 and frame 6:

Frame 5                               | Frame 6
--------------------------------------|------------------------------------------
![](images/report/lidar_points_5.png) | ![](images/report/lidar_points_6.png)

In Frame 5, `xmin = 7.64m` due to one outlier, but the implemented method
takes care of filtering that and instead the predicted measurement is
7.689 meters.

In Frame 6, `xmin = 7.58m`, but it's a large region of points having this
measurement, so it's not an outlier. If we look closely, we can see that the
lidar is now detecting a **different part of the car** that is a little bit
closer to the ego vehicle than before. Our code estimates now a distance
of 7.58 meters since there's no outliers.

The result is that the distance between ego and preceeding vehicle was
reduced by 0.1 meters (instead of the usual 0.06m) in 0.1 seconds, which
gives a higher relative speed of 1m/s and thus a smaller TTC.

**Example 2**

*Observation*

In frame 7, we obtain a TTC from lidar of 47 seconds, which is way too much:

![](images/report/lidar_47.png)

*Motivation*

Looking at the point clouds:

Frame 6                               | Frame 7
--------------------------------------|------------------------------------------
![](images/report/lidar_points_6.png) | ![](images/report/lidar_points_7.png)

The opposite problem happens now: in Frame 6 we detected a part of the car
that was closer to the ego, and now in Frame 7 we are back to detecting the
"usual" back of the car.

The algorithm computes the following measures of distance:

```
minXPrev: 7.5815, minXCurr: 7.5655
```

So now it's only a distance of 0.02 meters travelled in 0.1 seconds, which gives a relative
speed of 0.2 m/s (instead of 0.6 m/s) so we obtain a much larger TTC.

**Solution**
To solve these problems we should do tracking over more frames (instead of
the last 2 frames) to be more robust against what part of the preceeding
car is detected by the lidar.

FP.6 Performance Evaluation 2
-----------------------------
```
Run several detector / descriptor combinations and look at the differences in TTC
estimation. Find out which methods perform best and also include several examples
where camera-based TTC estimation is way off. As with Lidar, describe your
observations again and also look into potential reasons.
```
TODO
