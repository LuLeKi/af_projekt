import numpy as np
import time
import math

class LongitudinalControl:
    """
    Führt die Längsregelung des Fahrzeugs durch.
    Bestimmt auf Basis der Zielgeschwindigkeit und Ist-Geschwindigkeit das Brems- und Gassignal.
    """

    def __init__(self):
        """Initialisiert Regelparameter, Geschwindigkeitsgrenzen und Zustände."""
        # Zielgeschwindigkeiten
        self.max_speed = 65  # km/h
        self.min_speed = 30  # km/h

        # Regelstärken für proportionale Reaktion (kein reiner PID-Output)
        self.acceleration_strength = 0.047  
        self.braking_strength = 0.046     

        # PID-Konstanten (optional genutzt)
        self.Kp = 0.08
        self.Ki = 0.026
        self.Kd = 0.049
        self.integral_limit = (10, 10)

        # Zustandsvariablen für PID-Regelung
        self.error = 0.0
        self.integral = 0.0
        self.last_error = 0.0
        self.last_target_speed = self.max_speed

        # Zeitsteuerung für Debug-Ausgabe
        self.last_debug_time = 0

    def predict_target_speed(self, curvature: float, steering_angle: float) -> float:
        """
        Berechnet die Zielgeschwindigkeit anhand der aktuellen Kurvenkrümmung und Lenkwinkel.

        Args:
            curvature (float): Streckenkrümmung (≥ 0).
            steering_angle (float): Lenkwinkel (in rad oder normiert).
            speed (float): Aktuelle Geschwindigkeit des Fahrzeugs.

        Returns:
            float: Zielgeschwindigkeit [km/h], mindestens min_speed.
        """
        curvature_factor = abs(curvature)
        steering_factor = abs(steering_angle)

        # Geringe Werte werden verstärkt (nicht zu stark bremsen auf Geraden)
        if curvature_factor < 0.022:
            curvature_factor *= 1.6
        if steering_factor < 0.055:
            steering_factor *= 2.0

        # Kombination mit Gewichtung: Kurve wirkt stärker als Lenkung
        combined_factor = 5.7 * (0.6 * curvature_factor + 0.4 * steering_factor)

        # Zieltempo aus max/min-Speed linear ableiten
        target_speed = self.max_speed - combined_factor * (self.max_speed - self.min_speed)
        target_speed = max(target_speed, self.min_speed)

        return target_speed

    def control(self, speed: float, target_speed: float) -> tuple[float, float]:
        """
        Führt die Längsregelung durch: bestimmt Brems- oder Beschleunigungssignal.

        Args:
            speed (float): Aktuelle Geschwindigkeit [km/h].
            target_speed (float): Vorgabe-Zielgeschwindigkeit [km/h].

        Returns:
            tuple[float, float]: (Beschleunigungswert, Bremswert), beide ∈ [0.0, 1.0]
        """
        error = target_speed - speed
        self.integral += error
        self.integral = np.clip(self.integral, *self.integral_limit)
        derivative = error - self.last_error
        self.last_error = error

        acceleration = 0.0
        braking = 0.0
        delta_v = speed - target_speed  # >0 → zu schnell

        if delta_v > 0:
            # Überschwindigkeit → proportional bremsen
            braking = np.clip(delta_v * self.braking_strength, 0, 1)
            reason = f"braking: Δv={delta_v:.2f} → {braking:.2f}"
        else:
            # Unterschwindigkeit → proportional beschleunigen
            acceleration = np.clip(-delta_v * self.acceleration_strength, 0, 1)
        return acceleration, braking
