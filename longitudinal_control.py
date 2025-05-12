import numpy as np

class LongitudinalControl:
    def __init__(self):
            # PID Basiswerte
            self.Kp_base = 0.05
            self.Ki_base = 0.005
            self.Kd_base = 0.01

            # PID aktive Werte
            self.Kp = self.Kp_base
            self.Ki = self.Ki_base
            self.Kd = self.Kd_base

            # Fehlertracking
            self.error = 0.0
            self.integral = 0.0
            self.last_error = 0.0

            # === Parameter für Anpassung ===
            self.max_speed = 60
            self.min_speed = 30

            self.brake_threshold = 3.0  # Nur wenn wir >3 km/h über Zielspeed sind, bremsen


            self.steering_threshold = 0.5  # bei 0.5 beginnt max Dämpfung durch Lenkung
            self.low_speed_threshold = 20
            self.low_speed_penalty_max = 10

            self.integral_limit = (-10, 10)


    def predict_target_speed(self, curvature: float, steering_angle: float, speed: float) -> float:
        max_speed = self.max_speed
        min_speed = self.min_speed

        # Krümmungsfaktor (aus Path Planning)
        curvature_factor = min(abs(curvature), 1.0)

        # Steuerfaktor: ab 0.3 deutliches Lenken
        steering_factor = min(abs(steering_angle) / self.steering_threshold, 1.0)

        # Kombinierter Faktor (konservativ: max nehmen)
        combined_factor = max(curvature_factor, steering_factor)

        # Basis-Zielgeschwindigkeit
        target_speed = self.max_speed - combined_factor * (self.max_speed - self.min_speed)

        # Zusätzliche Dämpfung bei niedriger Geschwindigkeit
        if speed < self.low_speed_threshold:
            penalty_ratio = (self.low_speed_threshold - speed) / self.low_speed_threshold
            target_speed -= penalty_ratio * self.low_speed_penalty_max
            target_speed = max(target_speed, self.min_speed)

        # Dynamische PID-Anpassung
        self.Kp = self.Kp_base * (1 + combined_factor)
        self.Ki = self.Ki_base * (1 + 0.5 * combined_factor)
        self.Kd = self.Kd_base * (1 + 0.5 * combined_factor)

        return target_speed


    def control(self, speed: float, target_speed: float, steering_angle: float) -> tuple[float, float]:
            error = target_speed - speed
            self.integral += error
            derivative = error - self.last_error
            self.last_error = error

            self.integral = np.clip(self.integral, *self.integral_limit)

            output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

            steering_penalty = np.cos(abs(steering_angle) * np.pi / 2) ** 2

            if speed > target_speed:
                delta = speed - target_speed
                if delta > self.brake_threshold:
                    braking = np.clip(delta / target_speed, 0, 1)
                    acceleration = 0
                    braking /= steering_penalty
                else:
                    acceleration = 0  # rollen lassen
                    braking = 0
            else:
                acceleration = np.clip(output, 0, 1)
                braking = 0


            acceleration *= steering_penalty

            return acceleration, braking

