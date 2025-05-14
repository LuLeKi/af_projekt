import numpy as np
from scipy.interpolate import splprep, splev

class PathPlanning:
    def __init__(self):
        """Initialisiert Parameter für Pfadplanung, Glättung und Krümmungsanalyse."""

        self.last_valid_trajectory = None
        self.last_valid_local_path = None
        self.last_valid_curvature = 0.0

        self.vehicle_position = np.array([48.0, 64.0])
        # Für Basisspur
        self.trajectory_scalar = 500
        self.trajecotry_smoothing = 5
        self.splinegrad = 2

        # Für optimierten Spur
        self.opti_trajectory_scalar = 1500
        self.opti_trajecotry_smoothing = 10
        self.opti_splinegrad = 2

        # Wann auf 100% Krümmung genormt wird
        self.curvature_clip_threshold = 0.04

        # Maximale Auslenkung bei 100% Krümmung
        self.max_cut_shift = 6

        # Spur länge
        self.path_length = 200

    def plan(self, left_lane: np.ndarray, right_lane: np.ndarray):
        """Plant eine optimierte Trajektorie zwischen linker und rechter Spurmarkierung."""
        # 1. berechne midpoints
        path_waypoints = self.build_waypoints(left_lane, right_lane)

        if path_waypoints is None or len(path_waypoints) < 3:
            print("[WARN] Leere oder zu kurze Wegpunktliste.")
            if (
                self.last_valid_trajectory is None or
                not isinstance(self.last_valid_trajectory, np.ndarray) or
                self.last_valid_trajectory.ndim != 2 or
                self.last_valid_trajectory.shape[1] != 2
            ):
                return np.empty((0, 2)), np.empty((0, 2)), 0.0
            return self.last_valid_trajectory, self.last_valid_local_path, self.last_valid_curvature


        # 2. erstelle und glätte funktion aus midpoints
        path_trajectory, spline_model, u_fine = self.build_trajectory(path_waypoints, False)
        # 3 . berechne krümmung an jedem mittelpunkt
        curvature = self.calculate_curvature(spline_model, u_fine)
        # 4. berechne mittlere krümmung von allen punkten und normiere von -1 bis 1
        normalized_curvature = self.normalize_curvature(curvature)
        # 5. berechne normalenvektoren für Richtung an allen Punkten
        normals = self.calculate_normals(path_trajectory)
        # 6. setze anfangspunkt von pfad auf fahrzeug koordinaten
        local_path, local_normals = self.extract_local_path(path_trajectory, normals)
        # 7. optimiere pfad an krümmung um kurven besser zu schneiden
        optimized_local_waypoints = self.optimize_path(local_path, local_normals, normalized_curvature)
        # 8. erstelle und glätte funktion aus optimierten midpoints
        optimized_local_path = self.build_trajectory(optimized_local_waypoints, True)


        # 9. letzte gültige Werte speichern
        self.last_valid_trajectory = optimized_local_path
        self.last_valid_local_path = local_path
        self.last_valid_curvature = normalized_curvature

        # Rückgabe
        return optimized_local_path, local_path, normalized_curvature  

    def build_waypoints(self, left_lane, right_lane):
        """Berechnet mittlere Spurpunkte zwischen linker und rechter Spurbegrenzung."""
        min_len = min(len(left_lane), len(right_lane))
        return (left_lane[:min_len] + right_lane[:min_len]) / 2.0

    def build_trajectory(self, waypoints, optimized=False):
        """Interpoliert eine glatte Trajektorie aus gegebenen Wegpunkten mithilfe von Splines.
            Wählt Parameter für Glättung abhängig vom "optimized" Parameter"""
        if len(waypoints) >= 3:
            if optimized:
                smoothing = self.opti_trajecotry_smoothing
                splinegrad = self.opti_splinegrad
                scalar = self.opti_trajectory_scalar
            else:
                smoothing = self.trajecotry_smoothing
                splinegrad = self.splinegrad
                scalar = self.trajectory_scalar

            spline_model = splprep([waypoints[:, 0], waypoints[:, 1]], s=smoothing, k=splinegrad)[0]
            u_fine = np.linspace(0, 1, scalar)
            x_spline, y_spline = splev(u_fine, spline_model)

            if optimized:
                return np.vstack((x_spline, y_spline)).T
            else:
                return np.vstack((x_spline, y_spline)).T, spline_model, u_fine
        return waypoints, None, None
    
    def calculate_curvature(self, spline_model, u_fine):
        """Berechnet die Krümmung der Trajektorie auf Basis des Spline-Modells."""
        if spline_model is None or u_fine is None:
            return np.array([])  # ← richtiges Format, kein float!
        dx, dy = splev(u_fine, spline_model, der=1)
        ddx, ddy = splev(u_fine, spline_model, der=2)
        curvature = (dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
        return curvature

    def normalize_curvature(self, curvature):
        """Normiert die mittlere Krümmung auf einen Bereich von -1 bis 1."""
        if not isinstance(curvature, np.ndarray) or curvature.size == 0:
            return 0.0
        median_curv = np.median(curvature)
        threshold = self.curvature_clip_threshold
        return np.clip(median_curv / threshold, -1.0, 1.0)

    def calculate_normals(self, waypoints):
        """Berechnet Normalenvektoren entlang des Pfades."""
        normals = []
        for i in range(1, len(waypoints)):
            direction = waypoints[i] - waypoints[i-1]
            norm = np.linalg.norm(direction)
            normals.append(np.array([-direction[1], direction[0]]) / norm if norm else np.array([0.0, 0.0]))
        normals.append(normals[-1] if normals else np.array([0.0, 0.0]))
        return np.array(normals)
    
    def find_closest_index(self, waypoints, position):
        """Findet den Index des Wegpunkts, der der aktuellen Fahrzeugposition am nächsten ist."""
        distances = np.linalg.norm(waypoints - position, axis=1)
        return int(np.argmin(distances))

    def extract_local_path(self, waypoints, normals):
        """Extrahiert einen lokalen Teilpfad ab der Fahrzeugposition inklusive zugehöriger Normalen."""
        length = self.path_length
        closest_idx = self.find_closest_index(waypoints, self.vehicle_position)
        return waypoints[closest_idx:closest_idx+length], normals[closest_idx:closest_idx+length]
    
    def optimize_path(self, local_path, local_normals, normalized_curvature):
        """Passt den lokalen Pfad basierend auf Krümmung und Normalen an um Kurven besser zu schneiden."""
        adjusted = local_path.copy()
        shift = normalized_curvature * self.max_cut_shift
        for i in range(len(adjusted)):
            adjusted[i] += local_normals[i] * shift
        return np.array(adjusted)
