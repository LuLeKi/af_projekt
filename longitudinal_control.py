import numpy as np
import time

class LongitudinalControl:
    def __init__(self,
                 max_speed=60,
                 min_speed=30,
                 brake_threshold=3.0,
                 low_speed_threshold=20,
                 low_speed_penalty_max=10,
                 acc_damping_steering_thresh=0.2,
                 acc_damping_curvature_thresh=0.2,
                 brake_boost_steering_thresh=0.5,
                 brake_boost_curvature_thresh=0.5,
                 max_brake_boost=0.5,
                 integral_limit=(-10, 10)):

        self.max_speed = max_speed
        self.min_speed = min_speed
        self.brake_threshold = brake_threshold
        self.low_speed_threshold = low_speed_threshold
        self.low_speed_penalty_max = low_speed_penalty_max

        self.acc_damping_steering_thresh = acc_damping_steering_thresh
        self.acc_damping_curvature_thresh = acc_damping_curvature_thresh
        self.brake_boost_steering_thresh = brake_boost_steering_thresh
        self.brake_boost_curvature_thresh = brake_boost_curvature_thresh
        self.max_brake_boost = max_brake_boost

        self.Kp_base = 0.05
        self.Ki_base = 0.005
        self.Kd_base = 0.01

        self.Kp = self.Kp_base
        self.Ki = self.Ki_base
        self.Kd = self.Kd_base

        self.error = 0.0
        self.integral = 0.0
        self.last_error = 0.0
        self.integral_limit = integral_limit
        self.last_target_speed = max_speed

        self.last_debug_time = 0

    def predict_target_speed(self, curvature: float, steering_angle: float, speed: float) -> float:
        curvature_factor = abs(curvature)
        steering_factor = min(abs(steering_angle), 1.0)

        # Gewichtung einstellbar
        combined_factor = 0.6 * curvature_factor + 0.4 * steering_factor
        combined_factor = np.clip(combined_factor**1.1, 0.0, 1.0)

        target_speed = self.max_speed - combined_factor * (self.max_speed - self.min_speed)

        if speed < self.low_speed_threshold:
            penalty_ratio = (self.low_speed_threshold - speed) / self.low_speed_threshold
            target_speed -= penalty_ratio * self.low_speed_penalty_max
            target_speed = max(target_speed, self.min_speed)

        self.Kp = self.Kp_base * (1 + combined_factor)
        self.Ki = self.Ki_base * (1 + 0.5 * combined_factor)
        self.Kd = self.Kd_base * (1 + 0.5 * combined_factor)

        self.debug_print(f"[TARGET] curv={curvature:.3f}, steer={steering_angle:.3f} → factor={combined_factor:.3f} → target={target_speed:.1f}")
        return target_speed

    def control(self, speed: float, target_speed: float, steering_angle: float, curvature: float) -> tuple[float, float]:
        error = target_speed - speed
        self.integral += error
        derivative = error - self.last_error
        self.last_error = error
        self.integral = np.clip(self.integral, *self.integral_limit)
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # Acceleration-Dämpfung
        steer_damping = np.clip(abs(steering_angle) / self.acc_damping_steering_thresh, 0.0, 1.0)
        curv_damping = np.clip(abs(curvature) / self.acc_damping_curvature_thresh, 0.0, 1.0)
        acc_penalty = 1.0 - 0.5 * (steer_damping + curv_damping)

        # Bremsen-Booster
        steer_boost = np.clip(abs(steering_angle) / self.brake_boost_steering_thresh, 0.0, 1.0)
        curv_boost = np.clip(abs(curvature) / self.brake_boost_curvature_thresh, 0.0, 1.0)
        brake_boost_factor = max(steer_boost, curv_boost) * self.max_brake_boost

        braking = 0
        acceleration = 0
        speed_drop = self.last_target_speed - target_speed
        self.last_target_speed = target_speed
        reason = "neutral"

        if speed > target_speed:
            delta = speed - target_speed
            if speed_drop > 2.0:
                braking = np.clip(speed_drop / 10, 0, 1)
                reason = f"predictive brake: Δtarget={speed_drop:.1f} → braking={braking:.2f}"
            elif delta > self.brake_threshold:
                braking = np.clip(delta / target_speed, 0, 1)
                reason = f"brake: delta={delta:.1f} > threshold={self.brake_threshold} → braking={braking:.2f}"
            else:
                reason = f"coast: delta={delta:.1f} → no brake"
        else:
            acceleration = np.clip(output, 0, 1)
            reason = f"accelerate: acc={acceleration:.2f}, penalty={acc_penalty:.2f}"

        acceleration *= acc_penalty
        braking *= 1.0 + brake_boost_factor

        self.debug_print(f"[LongCtrl] speed={speed:.1f}, target={target_speed:.1f}, steer={steering_angle:.2f}, curv={curvature:.2f} → {reason}")
        return acceleration, braking

    def debug_print(self, *args, **kwargs):
        now = time.time()
        if now - self.last_debug_time > 0.5:
            self.last_debug_time = now
            print(*args, **kwargs)
