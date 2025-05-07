import numpy as np


class LateralControl:

    def __init__(self):
        self._car_position = np.array([48, 64])

    def get_tangent_angle_at_point(self, trajectory: np.ndarray, index: int) -> float:
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
        angle = np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])
        # Normalize to [-pi, pi]
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def angle_to_vec(self, angle) -> np.ndarray:
        return np.array([np.cos(angle), np.sin(angle)])

    def stanley(self, car, trajectory: np.ndarray, speed: np.ndarray) -> float:
        K1 = 1
        K2 = 2
        Ks = 0.75

        trajectory = np.unique(trajectory, axis=0) 
        print(trajectory) 
        dists = np.linalg.norm(trajectory - self._car_position, axis=1)
        # Vector from car to closest point
        closest_index = np.argmin(dists)
        closest_point = trajectory[closest_index]

        trajectory_tangent_vec = self.get_tangent_angle_at_point(trajectory, closest_index)
        heading_error = self.angle_between_vectors(trajectory_tangent_vec, self.angle_to_vec(car.hull.angle)) 

        cross_error = np.abs(self._car_position - closest_point)[0]

        error_vec = closest_point - self._car_position
        normal_vec = np.array([ trajectory_tangent_vec[1], -trajectory_tangent_vec[0]])
        cross_error = np.dot(error_vec, normal_vec) 

        if (speed < 1e-5): return 0.0

        steer = K1 * heading_error + np.arctan2((K2 * cross_error), (Ks + speed))

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

