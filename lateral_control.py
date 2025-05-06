import numpy as np


class LateralControl:

    def __init__(self):
        self._car_position = np.array([48, 64])

    def control(self, trajectory: np.ndarray, speed: np.ndarray) -> float:
        print(np.subtract(self._car_position, trajectory))
        pass

