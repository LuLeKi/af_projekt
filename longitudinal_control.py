import numpy as np
import time

class LongitudinalControl:
    def __init__(self):
        self.kp = 0.6
        self.ki = 0.1
        self.kd = 0.02

        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()
        self.last_throttle = 0.0

        self.brake_strength_multiplier = 1.6  # WICHTIG: war vorher nicht definiert

        self.max_speed = 80.0
        self.min_speed = 35.0

    def control(self, current_speed: float, curvature: np.ndarray, is_in_curve: bool) -> tuple:
        throttle = 0.0
        brake = 0.0

        # Zielgeschwindigkeit abhängig von Kurve
        curve_score = np.clip(np.max(np.abs(curvature)) * 25, 0.0, 1.0)
        base_target = self.max_speed - (self.max_speed - self.min_speed) * curve_score

        if not is_in_curve:
            base_target = max(base_target, 40.0)

        # PID-Regelung
        now = time.time()
        dt = max(now - self.last_time, 0.01)
        error = base_target - current_speed
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        pid = self.kp * error + self.ki * self.integral + self.kd * derivative

        if pid > 0:
            raw_throttle = np.clip(np.tanh(pid), 0.0, 1.0)

            if is_in_curve:
                max_curv = np.max(np.abs(curvature))
                curve_factor = np.clip(1.0 - max_curv * 50, 0.3, 1.0)
                max_throttle = curve_factor
            else:
                max_throttle = 1.0

            throttle = min(raw_throttle, max_throttle)
            throttle = min(throttle, self.last_throttle + 0.05)
        else:
            brake = np.clip(-pid * self.brake_strength_multiplier, 0.0, 1.0)

        self.last_throttle = throttle
        self.last_error = error
        self.last_time = now

        return throttle, brake
