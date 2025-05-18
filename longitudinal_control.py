import numpy as np
import time

class LongitudinalControl:
    def __init__(self):
        # === Zielgeschwindigkeiten ===
        self.max_speed = 60                    # maximale Geschwindigkeit (z. B. Geradeaus)
        self.min_speed = 25                    # minimale Geschwindigkeit (z. B. Kurve)

        # === Geschwindigkeitslogik ===
        self.brake_threshold = 0.9             # ab wie viel Überschuss gebremst wird
        self.low_speed_threshold = 25          # Mindestgeschwindigkeit, unter die nicht gebremst werden darf
        self.low_speed_penalty_max = 20         # zusätzliche Dämpfung bei sehr niedriger Geschwindigkeit

        # === Thresholds für Dämpfung / Boost ===
        self.acc_damping_steering_thresh = 0.01       # Lenkwinkel-Schwelle für Beschleunigungsdämpfung
        self.acc_damping_curvature_thresh = 0.01      # Krümmungs-Schwelle für Beschleunigungsdämpfung
        self.brake_boost_steering_thresh = 0.05       # Lenkwinkel-Schwelle für Bremsverstärkung
        self.brake_boost_curvature_thresh = 0.015     # Krümmungs-Schwelle für Bremsverstärkung
        self.max_brake_boost = 0.03                   # Maximaler Zusatzboost durch Boost-Logik

        # === Soft-/Hard-Brake-Grenzen ===
        self.soft_brake_curv_thresh = 0.005     # Ab dieser Krümmung leichte Bremse
        self.hard_brake_curv_thresh = 0.2     # Ab dieser Krümmung harte Bremse
        self.soft_brake_steer_thresh = 0.1    # Ab diesem Lenkwinkel leichte Bremse
        self.hard_brake_steer_thresh = 0.4    # Ab diesem Lenkwinkel harte Bremse
        self.soft_brake_boost = 0.3            # Verstärkung bei Soft-Brake
        self.hard_brake_boost = 0.8            # Verstärkung bei Hard-Brake

        # === PID-Regler ===
        self.Kp = 0.065
        self.Ki = 0.005
        self.Kd = 0.0175
        self.integral_limit = (10, 10)         # Begrenzung des Integralanteils
        self.error = 0.0
        self.integral = 0.0
        self.last_error = 0.0
        self.last_target_speed = self.max_speed

        # === Sonstiges ===
        self.last_debug_time = 0

    def predict_target_speed(self, curvature: float, steering_angle: float, speed: float) -> float:
        curvature_factor = abs(curvature)
        steering_factor = min(abs(steering_angle), 1.0)

        combined_factor = 0.4 * curvature_factor + 0.6 * steering_factor
        combined_factor = np.clip(combined_factor**1.1, 0.0, 1.0)

        target_speed = self.max_speed - combined_factor * (self.max_speed - self.min_speed)

        if speed < self.low_speed_threshold:
            penalty_ratio = (self.low_speed_threshold - speed) / self.low_speed_threshold
            target_speed -= penalty_ratio * self.low_speed_penalty_max
            target_speed = max(target_speed, self.min_speed)

        self.debug_print(f"[TARGET] curv={curvature:.3f}, steer={steering_angle:.3f} → factor={combined_factor:.3f} → target={target_speed:.1f}")
        return target_speed

    def control(self, speed: float, target_speed: float, steering_angle: float, curvature: float) -> tuple[float, float]:
        error = target_speed - speed
        self.integral += error
        derivative = error - self.last_error
        self.last_error = error
        self.integral = np.clip(self.integral, *self.integral_limit)
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        steer_damping = np.clip(abs(steering_angle) / self.acc_damping_steering_thresh, 0.0, 1.0)
        curv_damping = np.clip(abs(curvature) / self.acc_damping_curvature_thresh, 0.0, 1.0)
        acc_penalty = 1.0 - (0.7 * steer_damping + 0.3 * curv_damping)

        steer_boost = np.clip(abs(steering_angle) / self.brake_boost_steering_thresh, 0.0, 1.0)
        curv_boost = np.clip(abs(curvature) / self.brake_boost_curvature_thresh, 0.0, 1.0)
        brake_boost_factor = (0.3 * steer_boost + 0.7 * curv_boost) * self.max_brake_boost

        braking = 0
        acceleration = 0
        speed_drop = self.last_target_speed - target_speed
        self.last_target_speed = target_speed
        reason = "neutral"

        if speed > target_speed and speed > self.low_speed_threshold:
            delta = speed - target_speed
            if speed_drop > 2.0:
                braking = np.clip(speed_drop / 10, 0, 1)
                reason = f"predictive brake: Δtarget={speed_drop:.1f} → braking={braking:.2f}"
            elif delta > self.brake_threshold:
                braking = np.clip(delta / target_speed, 0, 1)
                reason = f"brake: delta={delta:.1f} > threshold={self.brake_threshold} → braking={braking:.2f}"
            else:
                reason = f"coast: delta={delta:.1f} → no brake"

            # Nicht unter low_speed_threshold bremsen
            if target_speed < self.low_speed_threshold:
                min_limit = self.low_speed_threshold + 0.5
                if speed < min_limit:
                    braking = 0
                    reason += f" → no brake (min {min_limit:.1f})"

        else:
            acceleration = np.clip(output, 0, 1)
            reason = f"accelerate: acc={acceleration:.2f}, penalty={acc_penalty:.2f}"

        acceleration *= acc_penalty

        # Zusätzliche Soft-/Hard-Brake-Logik
        soft_brake_triggered = (
            abs(curvature) > self.soft_brake_curv_thresh or
            abs(steering_angle) > self.soft_brake_steer_thresh
        )
        hard_brake_triggered = (
            abs(curvature) > self.hard_brake_curv_thresh or
            abs(steering_angle) > self.hard_brake_steer_thresh
        )

        if hard_brake_triggered:
            brake_boost = self.hard_brake_boost
            reason += " + hard_brake"
        elif soft_brake_triggered:
            brake_boost = self.soft_brake_boost
            reason += " + soft_brake"
        else:
            brake_boost = 0.0

        braking *= 1.0 + brake_boost + brake_boost_factor

        self.debug_print(f"[LongCtrl] speed={speed:.1f}, target={target_speed:.1f}, steer={steering_angle:.2f}, curv={curvature:.2f} → {reason}")
        return acceleration, braking

    def debug_print(self, *args, **kwargs):
        now = time.time()
        if now - self.last_debug_time > 0.5:
            self.last_debug_time = now
            print(*args, **kwargs)
