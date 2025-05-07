import numpy as np


class LateralControl:

    def __init__(self):
        self._car_position = np.array([48, 64])

    def get_tangent_angle_at_point(self, trajectory: np.ndarray, index: int) -> float:
        diff = np.diff(trajectory, axis=0)
        return diff[index]

    def angle_between_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return -np.arccos((np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))))

    def angle_to_vec(self, angle) -> np.ndarray:
        return np.array([np.cos(angle), np.sin(angle)])

    def stanley(self, car, trajectory: np.ndarray, speed: np.ndarray) -> float:
        K1 = 0.1
        K2 = 1
        print(trajectory)
        print("J", car.hull.angle)
        
        dists = np.linalg.norm(trajectory - self._car_position, axis=1)
        # Vector from car to closest point
        closest_index = np.argmin(dists)
        closest_point = trajectory[closest_index]

        trajectory_tangent_vec = self.get_tangent_angle_at_point(trajectory, closest_index)
        heading_error = self.angle_between_vectors(trajectory_tangent_vec, self.angle_to_vec(car.hull.angle)) 

        cross_error = np.abs(self._car_position - closest_point)[0]

        error_vec = closest_point - self._car_position
        # Normal vector to trajectory direction (rotate tangent by 90 degrees CCW)
        normal_vec = np.array([-trajectory_tangent_vec[1], trajectory_tangent_vec[0]])
        # Signed cross track error
        cross_error = np.dot(error_vec, normal_vec) 



        if (speed < 1e-5): return 0.0

        steer = K1 * heading_error + np.arctan2((K2 * cross_error), speed)

        ressteer = np.clip(steer, -1, 1)

        # generate debug prints for all relevant variables
        print(f"car_position: {self._car_position}")
        print(f"closest_point: {closest_point}")
        print(f"trajectory_tangent_vec: {trajectory_tangent_vec}")
        print(f"heading_error: {heading_error}")
        print(f"cross_error: {cross_error}")
        print(f"steer: {steer}")
        print(f"ressteer: {ressteer}")
        return ressteer
    
    def control(self, car, trajectory: np.ndarray, speed: np.ndarray) -> float:
        return self.stanley(car, trajectory, speed)

