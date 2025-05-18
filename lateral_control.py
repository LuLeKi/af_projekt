import numpy as np
from scipy.interpolate import splprep, splev 
import matplotlib.pyplot as plt

PRINT_DEBUG = 0

class LateralControl:

    next_point = [0, 0] 
    trajectory = None
    prev_head_error = 0.0
    history = None
    tangent = None
    lookahead_index = 0
    last_steer = 0

    def __init__(self):
        self._car_position = np.array([48, 64])


    def get_tangent_at_point(self, trajectory: np.ndarray, index: int) -> float:
        trajectory = trajectory.argsort(axis=0)
        if len(trajectory) < 2:
            # Not enough points for a tangent, return a default (e.g., pointing forward along x-axis)
            return np.array([1.0, 0.0])

        # Ensure index is within bounds for tangent calculation
        # For points near the start or end, use forward or backward difference respectively.
        if index <= 0:
            # Tangent at the start is based on the first two points
            tangent = trajectory[min(1, len(trajectory)-1)] - trajectory[0]
        elif index >= len(trajectory) - 1:
            # Tangent at the end is based on the last two points
            tangent = trajectory[-1] - trajectory[max(0, len(trajectory)-2)]
        else:
            # Forward difference for other points
            tangent = trajectory[index + 1] - trajectory[index]
        
        norm = np.linalg.norm(tangent)
        if norm < 1e-6: # Avoid division by zero if points are coincident
            # If norm is too small, try to use a previous valid tangent or a default
            # For simplicity, returning a default here. A more robust solution might be needed.
            if index > 0: # Try backward difference if forward failed
                 tangent = trajectory[index] - trajectory[index-1]
                 norm = np.linalg.norm(tangent)
                 if norm < 1e-6:
                     return np.array([1.0, 0.0]) # Ultimate fallback
            else:
                 return np.array([1.0, 0.0]) # Ultimate fallback

        return tangent / norm 


    def angle_between_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        cross = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        dot = np.dot(vec1, vec2)
        angle = np.arctan2(cross, dot)
        return angle  

    def angle_to_vec(self, angle) -> np.ndarray:
        return np.array([np.cos(angle), np.sin(angle)])

    def stanley(self, trajectory: np.ndarray, speed: np.ndarray) -> float:
        K1 = 0.02
        K2 = 4.5 
        Ks = 0.2
        Kd = 0.2

        max_cross_error = 10
        max_steer = 1

        try:
            if isinstance(trajectory, tuple):
                trajectory = np.vstack(trajectory[0])
            else:
                trajectory = np.array(trajectory)
        except:
            return self.last_steer

        if len(trajectory) == 0:
            return self.last_steer
        if trajectory is None:
            return self.last_steer
        if trajectory.ndim != 2 or trajectory.shape[1] != 2:
            return self.last_steer

        trajectory = np.unique(trajectory, axis=0) 
        # sort trajectory by y from highest to lowest
        trajectory = trajectory[np.argsort(trajectory[:, 1])[::-1]] 
        self.trajectory = trajectory


        dists = np.linalg.norm(trajectory - self._car_position, axis=1)
        closest_index = np.argmin(dists)
        lookahead_index = min(closest_index + 3, max(len(trajectory) - 1, 0))
        self.lookahead_index = lookahead_index


        next_point = trajectory[lookahead_index]
        self.next_point = next_point


        trajectory_tangent_vec = self.get_tangent_at_point(trajectory, lookahead_index)
        self.tangent = trajectory_tangent_vec

        heading_error = self.angle_between_vectors(
            trajectory_tangent_vec,
            np.array([0, 1]),
        )

        error_vec = next_point - self._car_position
        normal_vec = np.array([-trajectory_tangent_vec[1], trajectory_tangent_vec[0]])
        cross_error = np.dot(error_vec, normal_vec)

        if abs(cross_error) < 0.4:
            return 0.0

        if speed < 1e-2:
            self.prev_head_error = heading_error
            return 0.0

        cross_error = np.clip(cross_error, -max_cross_error, max_cross_error)
        K2_effective = K2 * (1 - np.exp(-abs(cross_error) / 3))

        steer = np.arctan2(K2_effective * cross_error, speed + Ks + 1) + heading_error * K1

        delta_head_error = heading_error - self.prev_head_error
        self.prev_head_error = heading_error
        # Fügen Sie den Dämpfungsterm hinzu. Er wirkt der Änderung entgegen.
        damping_term = -1 * min(Kd * delta_head_error, 0.1)
        #steer += damping_term

        steer = np.clip(steer, -max_steer, max_steer)

        if (PRINT_DEBUG):
            # generate debug prints for all relevant variables
            print(f"car_position: {self._car_position}")
            print(f"next_point: {next_point}")
            print(f"trajectory_tangent_vec: {trajectory_tangent_vec}")
            print(f"k2_effective: {K2_effective}")
            print(f"heading_error: {heading_error}")
            print(f"cross_error: {cross_error}")
            print(f"steer: {steer}")
            print(f"max_steer: {max_steer}")
            print(f"traj len: {len(trajectory)}")
            print(f"heading damping: {delta_head_error}")

        self.last_steer = steer
        return steer 
    
    def control(self, trajectory: np.ndarray, speed: np.ndarray) -> float:
        return self.stanley(trajectory, speed)

