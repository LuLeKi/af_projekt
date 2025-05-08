import numpy as np
from scipy.interpolate import splprep, splev 


class LateralControl:

    next_point = [0, 0] 
    trajectory = None
    prev_head_error = 0.0

    def __init__(self):
        self._car_position = np.array([48, 64])

    def get_tangent_at_point(self, trajectory: np.ndarray, index: int) -> float:
        if len(trajectory) < 2:
            return np.array([1.0, 0.0])

        if index <= 0:
            tangent = trajectory[1] - trajectory[0]
        elif index >= len(trajectory) - 1:
            tangent = trajectory[-1] - trajectory[-2]
        else:
            tangent = trajectory[index + 1] - trajectory[index]

        return tangent / np.linalg.norm(tangent)


    def angle_between_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        cross = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        dot = np.dot(vec1, vec2)
        angle = np.arctan2(cross, dot)
        return angle  

    def angle_to_vec(self, angle) -> np.ndarray:
        return np.array([np.cos(angle), np.sin(angle)])

    def stanley(self, car, trajectory: np.ndarray, speed: np.ndarray) -> float:
        K1 = 0.1
        K2 = 1.4
        Ks = 0.5

        trajectory = np.unique(trajectory, axis=0) 
        self.trajectory = trajectory

        
        # Separate into x and y
        x = trajectory[:, 0]
        y = trajectory[:, 1]

        # Create parametric spline representation
        tck, u = splprep([x, y], s=3, k=3)  # s is the smoothing factor, k is the degree of the spline

        # Evaluate the spline at more points for a smooth curve
        traj_linspace = np.linspace(0, 1, 120)
        x_fine, y_fine = splev(traj_linspace, tck) 
        center_traj_point = np.median(trajectory, axis=0)
        point_dist = np.linalg.norm(trajectory - center_traj_point, axis=1)
        scatter = np.mean(point_dist)
        if scatter > 12 and scatter < 26:
            trajectory = np.array([x_fine, y_fine]).T
            self.trajectory = trajectory
        print(f"scatter: {scatter}")
        
        # Original trajectory: shape (N, 2)
        dists = np.linalg.norm(trajectory - self._car_position, axis=1)
        # Vector from car to closest point
        closest_index = np.argmin(dists)
        #closest_point = trajectory[closest_index]
        lookahead_index =  min(closest_index + 15, len(trajectory) - 1) 
        next_point = trajectory[lookahead_index] 
        self.next_point = next_point

        trajectory_tangent_vec = self.get_tangent_at_point(trajectory, closest_index + 5)
        heading_error = self.angle_between_vectors(trajectory_tangent_vec, self.angle_to_vec(car.hull.angle)) 

        error_vec = next_point - self._car_position
        normal_vec = np.array([ trajectory_tangent_vec[1], -trajectory_tangent_vec[0]])
        cross_error = np.dot(error_vec, normal_vec) 

        if (speed < 1e-2): return 0.0

        alpha = 0.2  # smoothing factor
        heading_error = alpha + (1 - alpha) * heading_error
        cross_error = alpha  + (1 - alpha) * cross_error

        heading_derivative = abs(heading_error - self.prev_head_error) / 1e-2 
        print(f"heading_derivative: {heading_derivative}")
        steer = K1 * heading_error + np.arctan2((K2 * cross_error), (speed + 1e-2)) 
        steer -= Ks * heading_derivative
        self.prev_head_error = heading_error

        max_steer = 0.8
        ressteer = np.clip(steer, -max_steer, max_steer)

        # generate debug prints for all relevant variables
        print(f"car_position: {self._car_position}")
        print(f"next_point: {next_point}")
        print(f"trajectory_tangent_vec: {trajectory_tangent_vec}")
        print(f"heading_error: {heading_error}")
        print(f"cross_error: {cross_error}")
        print(f"steer: {steer}")
        print(f"ressteer: {ressteer}")
        print(f"max_steer: {max_steer}")
        print(f"traj len: {len(trajectory)}")
        return ressteer
    
    def control(self, car, trajectory: np.ndarray, speed: np.ndarray) -> float:
        return self.stanley(car, trajectory, speed)

