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
TODO

FP.4 Compute Camera-based TTC
-----------------------------
```
Compute the time-to-collision in second for all matched 3D objects using only
keypoint correspondences from the matched bounding boxes between current and
previous frame.
```
TODO

FP.5 Performance Evaluation 1
-----------------------------
```
Find examples where the TTC estimate of the Lidar sensor does not seem plausible.
Describe your observations and provide a sound argumentation why you think this
happened.
```
TODO

FP.6 Performance Evaluation 2
-----------------------------
```
Run several detector / descriptor combinations and look at the differences in TTC
estimation. Find out which methods perform best and also include several examples
where camera-based TTC estimation is way off. As with Lidar, describe your
observations again and also look into potential reasons.
```
TODO


OBSERVATIONS
============
From top-view lidar, we observe the following distances to
the preceeding vehicle:

t = 0.0: 7.97 m
t = 0.1: 7.91 m
t = 0.2: 7.85 m
t = 0.3: 7.79 m

So the relative velocity to the preceeding vehicle is
approximately 0.06 m / 0.1s = 0.6 m/s

Therefore for t = 0.1 the real TTC would be:

7.91 / 0.6 = **13.18 seconds** (approximately)

And removing 0.1 seconds for the following frames assuming
constant velocity models.
