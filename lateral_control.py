import numpy as np

PRINT_DEBUG = 0

class LateralControl:
    """
    Implements the Stanley lateral control algorithm for trajectory following.
    """

    prev_head_error = 0.0
    history = None
    tangent = None
    last_steer = 0

    def __init__(self):
        self._car_position = np.array([48, 64])


    def get_tangent_at_point(self, trajectory: np.ndarray, index: int) -> float:
        """
        Calculates the tangent vector at a specific index along a trajectory. 

        Args:
            trajectory: (N, 2) array of ordered trajectory points.
            index: The index of the point at which to calculate the tangent.

        Returns:
            A unit 2D numpy array representing the tangent vector.
        """
        trajectory = trajectory.argsort(axis=0)
        if len(trajectory) < 2:
            return np.array([1.0, 0.0])

        # ensure index is within bounds for tangent calculation
        if index <= 0:
            tangent = trajectory[min(1, len(trajectory)-1)] - trajectory[0]
        elif index >= len(trajectory) - 1:
            tangent = trajectory[-1] - trajectory[max(0, len(trajectory)-2)]
        else:
            tangent = trajectory[index + 1] - trajectory[index]
        
        norm = np.linalg.norm(tangent)
        # avoid division by zero
        if norm < 1e-6: 
            # try backward difference if forward failed
            if index > 0: 
                 tangent = trajectory[index] - trajectory[index-1]
                 norm = np.linalg.norm(tangent)
                 if norm < 1e-6:
                     # fallback
                     return np.array([1.0, 0.0]) 
            else:
                 return np.array([1.0, 0.0]) 

        return tangent / norm 


    def angle_between_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculates the signed angle between two 2D vectors.

        Args:
            vec1: The first 2D vector.
            vec2: The second 2D vector.

        Returns:
            The angle in radians between vec1 and vec2, ranging from -pi to +pi.
        """
        cross = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        dot = np.dot(vec1, vec2)
        angle = np.arctan2(cross, dot)
        return angle  

    def angle_to_vec(self, angle) -> np.ndarray:
        """
        Converts an angle (in radians) to a unit 2D vector.

        Args:
            angle: The angle in radians.

        Returns:
            A unit 2D numpy array [cos(angle), sin(angle)].
        """
        return np.array([np.cos(angle), np.sin(angle)])

    def stanley(self, trajectory: np.ndarray, speed: np.ndarray) -> float:
        """
        Calculates the steering command using a variation of the Stanley method.
 
        Args:
            trajectory: (N, 2) array of trajectory points.
            speed: The car's current forward speed.

        Returns:
            The calculated steering command (float), typically in the range [-1, 1].
        """
        K1 = 0.02
        K2 = 4.5 
        Ks = 0.2

        max_cross_error = 10
        max_steer = 1

        # check trajectory for errors and use fallbacks
        try:
            if isinstance(trajectory, tuple):
                trajectory = np.vstack(trajectory[0])
            else:
                trajectory = np.array(trajectory)
        except:
            return self.last_steer

        # check traj len, use last steer if error
        if len(trajectory) == 0:
            return self.last_steer
        if trajectory is None:
            return self.last_steer
        if trajectory.ndim != 2 or trajectory.shape[1] != 2:
            return self.last_steer

        trajectory = np.unique(trajectory, axis=0) 
        # sort trajectory by y from highest to lowest
        trajectory = trajectory[np.argsort(trajectory[:, 1])[::-1]] 

        # find closest traj point to car
        dists = np.linalg.norm(trajectory - self._car_position, axis=1)
        closest_index = np.argmin(dists)
        lookahead_index = min(closest_index + 3, max(len(trajectory) - 1, 0))

        # select next point as aim for controller
        next_point = trajectory[lookahead_index]

        # get tangent at next point
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
        # dinamically change the effective K2
        K2_effective = K2 * (1 - np.exp(-abs(cross_error) / 3))

        # calc new steer with stanley formula
        steer = np.arctan2(K2_effective * cross_error, speed + Ks + 1) + heading_error * K1

        delta_head_error = heading_error - self.prev_head_error
        self.prev_head_error = heading_error

        # clip steer
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

