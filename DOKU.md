# Lane Detection Module

This Python code implements a basic lane detection algorithm using image prewitt gradients. It takes a grayscale image representing the road ahead and attempts to identify the left and right lane boundaries as sets of 2D points.

## How it Works

1.  **Grayscale Conversion:** The input color image is converted to grayscale.
2.  **Edge Detection:** The Prewitt edge detector is applied to find areas with strong intensity gradients (potential lane lines).
3.  **Thresholding:** The gradient magnitude image is thresholded to keep only the strongest edges. A fixed threshold of 50 is used.
4.  **Car and Stats Removal:** Pixels corresponding to the car's position and a lower status bar area are zeroed out to prevent detecting the car or UI elements as lanes.
5.  **Connected Components:** Remaining non-zero pixels are grouped into connected components (blobs).
6.  **Lane Identification:** The two connected components whose points are closest to the car position are assumed to be the left and right lanes.
7.  **Point Processing (Thinning & Interpolation):**
    * **Thinning:** For each detected lane component, the algorithm processes its points to reduce multiple detected edge points on the same horizontal line (same image row or `y` coordinate) to a single representative point. It iterates through the points and keeps approximately one point per row, with a check for horizontal separation (`abs(x - unique_y[y]) > 1`) to potentially include points that are horizontally distant within the same row. This helps in getting a clearer structure of the lane line.
    * **Interpolation:** After thinning, the remaining points for each lane are used to create a denser, smoother set of points. It sorts the thinned points by their vertical position (`y` coordinate, which corresponds to image row) and uses `np.interp` to generate 250 evenly spaced points between the minimum and maximum detected vertical positions. This effectively draws a smooth line through the detected points, providing a continuous representation of the lane boundary.
8.  **Coordinate Transformation:** The detected and interpolated points, which are in image coordinates (effectively row, column), are transformed using `align_to_wrapper` into a different (x, y) coordinate system to be used in the original image.
9.  **Return Points:** The function returns two arrays of points, one for the left lane and one for the right lane, in the target (x, y) coordinate system.

## Usage

Instantiate the `LaneDetection` class and call the `detect` method with your state image:

```python
from lane_detection import LaneDetection
import numpy as np

# state_image = ...

lane_detector = LaneDetection()
left_lane_points, right_lane_points = lane_detector.detect(state_image)

# left_lane_points and right_lane_points are now numpy arrays with shape (N, 2)
# in the target (x, y) coordinate system.
```

# Path Planning Module

This Python module implements a trajectory planner for autonomous driving. It receives two detected lane boundaries (left and right) and generates a smooth, optimized trajectory between them, as well as an estimate of the average road curvature.

## How it Works

1. **Midline Calculation:**  
   The method `build_waypoints()` computes the midpoint between the left and right lane boundaries to serve as the initial path centerline.

2. **Spline-Based Smoothing:**  
   The midpoints are fitted to a B-spline curve using `scipy.interpolate.splprep` for path smoothing. Separate spline parameters are used for the base and the optimized trajectory.

3. **Curvature Calculation:**  
   The first and second derivatives of the spline are used to calculate pointwise curvature. The median curvature is normalized and clipped to the range [–1, 1] based on a configurable threshold.

4. **Local Path Extraction:**  
   A local segment of the spline (typically ahead of the vehicle's fixed position) is extracted for further optimization. Normal vectors are computed along this path.

5. **Path Optimization via Curvature-Based Shifting:**  
   Each local path point is shifted sideways along its normal vector proportionally to the normalized curvature. This enables anticipatory cutting of inside curves.

6. **Output:**  
   The function `plan()` returns:
   - A smoothed and optimized trajectory as `(N, 2)` array,
   - A normalized curvature value for use in speed planning.

## Usage

```python
from path_planning import PathPlanning

planner = PathPlanning()
trajectory, curvature = planner.plan(left_lane_points, right_lane_points)
```

`left_lane_points` and `right_lane_points` are expected as `(N, 2)` NumPy arrays in world coordinates (x, y).

# Stanley Lateral Control Module

This Python code implements a lateral control system intended for trajectory following, based on a variation of the Stanley method. It calculates a steering command to keep a vehicle aligned with a given path.

## How it Works

The core logic resides in the `stanley` method, which is called by the main `control` function. It processes a given trajectory (a sequence of points) and the car's speed to compute a steering output.

Here's a breakdown of the key steps and features:

1.  **Trajectory Loading and Validation:**
    * Takes a `trajectory` (expected to be an `(N, 2)` NumPy array of points) and `speed` as input.
    * Includes error handling to check if the trajectory is valid. If the trajectory is `None`, empty, or in an incorrect format, the function returns the `last_steer` command calculated in the previous control step, preventing errors.
    * Note: The code includes `np.unique(trajectory, axis=0)` and `trajectory = trajectory[np.argsort(trajectory[:, 1])[::-1]]` which removes duplicate points and sorts the trajectory by the y-coordinate (highest to lowest).

2.  **Car Position and Lookahead Point Selection:**
    * The control uses a fixed car position (`self._car_position`, initialized to `[48, 64]`) for calculations. **Note:** This position is not updated by the `control` method itself in the provided code, which means the controller will always calculate errors relative to this fixed point unless `self._car_position` is updated externally.
    * It finds the point on the *processed and sorted* trajectory closest to this fixed car position.
    * Instead of targeting the closest point, it selects a `lookahead_index` which is `closest_index + 3` (clamped to the bounds of the trajectory array). This implements a **lookahead strategy**, aiming for a point several steps ahead on the path rather than the immediate next one. This helps in smoother tracking and anticipating curves.

3.  **Tangent Calculation:**
    * Calculates the tangent vector at the selected `lookahead_index` on the trajectory using `get_tangent_at_point`.

4.  **Error Calculation:**
    * **Heading Error:** Calculates the angle between the trajectory tangent vector at the lookahead point and a **fixed vector `np.array([0, 1])`**.
    * **Cross-Track Error:** Calculates the lateral distance from the fixed car position (`self._car_position`) to the trajectory line segment at the lookahead point. It does this by projecting the vector from `_car_position` to `next_point` onto the normal vector of the trajectory tangent.

5.  **Controller Activation/Deactivation:**
    * The controller **turns off** (returns `0.0` steering) under two specific conditions:
        * If the absolute `cross_error` is less than `0.4`. This indicates the car is considered sufficiently close to the path center line.
        * If the car `speed` is less than `1e-2`. This prevents unstable steering commands when the car is stationary or moving extremely slowly.
    * In both cases, if the speed is low, the `prev_head_error` is still updated.

6.  **Dynamic K2 Gain:**
    * The `K2` gain, which weights the cross-track error term in the Stanley formula, is made **dynamic**. An effective gain `K2_effective` is calculated using `K2 * (1 - np.exp(-abs(cross_error) / 3))`. This means the cross-track correction term's influence is smaller when the car is very close to the path and increases as the lateral error grows, providing stronger corrections when further off-path.

7.  **Steering Calculation (Stanley Formula):**
    * The steering command is calculated using the formula: `steer = arctan2(K2_effective * cross_error, speed + Ks + 1) + heading_error * K1`.
    * The first term is based on the cross-track error and speed (the cross-track error component of Stanley).
    * The second term is based on the heading error (the orientation error component of Stanley).

8.  **Steering Clipping:**
    * The final calculated `steer` value is **clipped** to the range `[-max_steer, max_steer]` using `np.clip`. `max_steer` is set to `1`. This limits the maximum steering angle applied.

9.  **State Update and Return:**
    * The calculated `steer` value is stored in `self.last_steer`. 
    * The final clipped `steer` value is returned.

## Key Methods

* `LateralControl()`: Constructor.
* `control(trajectory, speed)`: Main function to trigger the control calculation.
* `stanley(trajectory, speed)`: Contains the core control logic.
* `get_tangent_at_point(trajectory, index)`: Helper to find trajectory tangent.
* `angle_between_vectors(vec1, vec2)`: Helper to calculate signed angle.
* `angle_to_vec(angle)`: Helper to convert angle to vector.

# Longitudinal Control Module

This module implements a longitudinal controller for an autonomous vehicle. Its task is to regulate the vehicle's speed by computing appropriate acceleration or braking values based on the difference between the current speed and the target speed.

## How it Works

1. **Target Speed Prediction:**
   The `predict_target_speed()` function computes a target speed based on the current lane curvature and steering angle. Curves and strong steering reduce the target speed to ensure safe driving. Light curves and gentle steering have less effect. The combination of curvature and steering is weighted, scaled, and used to interpolate between a defined `max_speed` and `min_speed`.

2. **PID-Controlled Speed Adjustment:**
   The `control()` function calculates the difference (`error`) between the current and target speeds. It applies a PID controller (proportional, integral, derivative) to generate a control output.

   - The **P-term** reacts to the immediate difference between actual and target speeds.
   - The **I-term** accounts for accumulated past errors, helping to correct steady-state errors.
   - The **D-term** anticipates future errors based on how quickly the speed difference changes.

3. **Hybrid Control Strategy:**
   The PID output is not used exclusively. Instead:
   - The difference in speed (`delta_v`) is used to calculate a base braking or acceleration value.
   - The PID output is added as a correction to that base value to smooth transitions.
   - Output values are clipped to the range [0.0, 1.0].

4. **Braking and Acceleration Logic:**
   - If the vehicle is too fast (`delta_v > 0`), it brakes proportionally and adds any negative PID correction.
   - If too slow, it accelerates proportionally and adds positive PID correction.

## Example Usage

```python
controller = LongitudinalControl()
target_speed = controller.predict_target_speed(curvature, steering_angle)
acceleration, braking = controller.control(current_speed, target_speed)
```