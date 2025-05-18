import numpy as np
from lane_detection import LaneDetection
from lateral_control import LateralControl
from longitudinal_control import LongitudinalControl
from path_planning import PathPlanning

class DummyHull:
    def __init__(self):
        self.angle = 0.0  

class Car:

    def __init__(self):
        self.hull = DummyHull()
        self._lane_detection = LaneDetection()
        self._path_planning = PathPlanning()
        self._lateral_control = LateralControl()
        self._longitudinal_control = LongitudinalControl()

    def next_action(self, observation: np.ndarray, info: dict[str, any]) -> list:
        """
        Bestimmt die nächste Aktion (Lenkung, Gas, Bremse) basierend auf Kamerabild und Fahrzeugdaten.
        """
        # 1. Spur erkennen
        left_lane_boundaries, right_lane_boundaries = self._lane_detection.detect(observation)

        # 2. Pfad planen und Krümmung berechnen
        trajectory, curvature = self._path_planning.plan(left_lane_boundaries, right_lane_boundaries)

        # 3. Querregelung (Stanley): Lenkwinkel berechnen
        steering_angle = self._lateral_control.control(trajectory, info["speed"])

        # 4. Zielgeschwindigkeit basierend auf Krümmung und Lenkwinkel
        target_speed = self._longitudinal_control.predict_target_speed(curvature, steering_angle)

        # 5. Längsregelung: Gas / Bremse berechnen
        acceleration, braking = self._longitudinal_control.control(info["speed"], target_speed)

        action = [steering_angle, acceleration, braking]

        return action
