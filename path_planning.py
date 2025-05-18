import numpy as np
from scipy.interpolate import splprep, splev

class PathPlanning:
    """
    Klasse zur Pfadplanung zwischen Spurmarkierungen.
    Erzeugt glatte Trajektorien, berechnet Krümmung und passt den Pfad in Kurven an.
    """

    def __init__(self):
        """Initialisiert Planungsparameter und Konfigurationswerte."""
        self.vehicle_position = np.array([48.0, 64.0])

        # Basisspur-Parameter
        self.trajectory_scalar = 500
        self.trajectory_smoothing = 5
        self.spline_grad = 2

        # Optimierte Spur-Parameter
        self.opti_trajectory_scalar = 1200
        self.opti_trajectory_smoothing = 9.4
        self.opti_spline_grad = 2

        # Krümmungsnormierung
        self.curvature_clip_threshold = 0.045
        self.max_cut_shift = 5.8

        # Pfadlänge in Punkten
        self.path_length = 200

        # Letzte gültige Rückgaben
        self.last_valid_trajectory = None
        self.last_valid_local_path = None
        self.last_valid_curvature = 0.0

    def plan(self, left_lane: np.ndarray, right_lane: np.ndarray):
        """
        Plant eine optimierte Trajektorie zwischen linker und rechter Spurmarkierung.
        Gibt auch den unbearbeiteten lokalen Pfad und die normierte mittlere Krümmung zurück.
        """
        path_waypoints = self.build_waypoints(left_lane, right_lane)

        if path_waypoints is None or len(path_waypoints) < 3:
            if (self.last_valid_trajectory is None or
                not isinstance(self.last_valid_trajectory, np.ndarray) or
                self.last_valid_trajectory.ndim != 2 or
                self.last_valid_trajectory.shape[1] != 2):
                return np.empty((0, 2)), np.empty((0, 2)), 0.0
            return self.last_valid_trajectory, self.last_valid_local_path, self.last_valid_curvature

        # Glatte Basis-Trajektorie
        path_trajectory, spline_model, u_fine = self.build_trajectory(path_waypoints, optimized=False)
        curvature = self.calculate_curvature(spline_model, u_fine)
        normalized_curvature = self.normalize_curvature(curvature)
        normals = self.calculate_normals(path_trajectory)
        local_path, local_normals = self.extract_local_path(path_trajectory, normals)
        optimized_waypoints = self.optimize_path(local_path, local_normals, normalized_curvature)
        optimized_local_path = self.build_trajectory(optimized_waypoints, optimized=True)

        # Gültige Daten speichern
        self.last_valid_trajectory = optimized_local_path
        self.last_valid_local_path = local_path
        self.last_valid_curvature = normalized_curvature

        return optimized_local_path, normalized_curvature

    def build_waypoints(self, left_lane: np.ndarray, right_lane: np.ndarray) -> np.ndarray:
        """Berechnet Mittelspurpunkte zwischen linker und rechter Spur."""
        min_len = min(len(left_lane), len(right_lane))
        return (left_lane[:min_len] + right_lane[:min_len]) / 2.0

    def build_trajectory(self, waypoints: np.ndarray, optimized: bool):
        """
        Erzeugt glatte Trajektorie über gegebenen Wegpunkten mittels B-Spline.
        Bei `optimized=True` wird nur die Spline-Kurve zurückgegeben.
        """
        if optimized:
            smoothing = self.opti_trajectory_smoothing
            spline_grad = self.opti_spline_grad
            scalar = self.opti_trajectory_scalar
        else:
            smoothing = self.trajectory_smoothing
            spline_grad = self.spline_grad
            scalar = self.trajectory_scalar

        if (isinstance(waypoints, np.ndarray) and waypoints.ndim == 2 and
            waypoints.shape[1] == 2 and len(waypoints) >= spline_grad + 1 and
            not np.isnan(waypoints).any() and not np.isinf(waypoints).any()):
            try:
                spline_model = splprep([waypoints[:, 0], waypoints[:, 1]], s=smoothing, k=spline_grad)[0]
                u_fine = np.linspace(0, 1, scalar)
                x_spline, y_spline = splev(u_fine, spline_model)
                spline_path = np.vstack((x_spline, y_spline)).T
                return spline_path if optimized else (spline_path, spline_model, u_fine)
            except Exception:
                return waypoints, None, None

        return waypoints, None, None

    def calculate_curvature(self, spline_model, u_fine):
        """Berechnet Punktweise die Krümmung einer Spline-Kurve."""
        if spline_model is None or u_fine is None:
            return np.array([])
        dx, dy = splev(u_fine, spline_model, der=1)
        ddx, ddy = splev(u_fine, spline_model, der=2)
        curvature = (dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
        return curvature

    def normalize_curvature(self, curvature: np.ndarray) -> float:
        """
        Normiert die mittlere Krümmung auf [-1, 1] bezogen auf Schwellwert.
        """
        if not isinstance(curvature, np.ndarray) or curvature.size == 0:
            return 0.0
        median_curv = np.median(curvature)
        return np.clip(median_curv / self.curvature_clip_threshold, -1.0, 1.0)

    def calculate_normals(self, waypoints: np.ndarray) -> np.ndarray:
        """Berechnet Normalenvektoren (senkrechte Richtungen) entlang einer Linie aus Punkten."""
        normals = []
        for i in range(1, len(waypoints)):
            direction = waypoints[i] - waypoints[i - 1]
            norm = np.linalg.norm(direction)
            normal = np.array([-direction[1], direction[0]]) / norm if norm else np.array([0.0, 0.0])
            normals.append(normal)
        normals.append(normals[-1] if normals else np.array([0.0, 0.0]))
        return np.array(normals)

    def find_closest_index(self, waypoints: np.ndarray, position: np.ndarray) -> int:
        """Gibt Index des Wegpunkts zurück, der der Fahrzeugposition am nächsten ist."""
        distances = np.linalg.norm(waypoints - position, axis=1)
        return int(np.argmin(distances))

    def extract_local_path(self, waypoints: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extrahiert lokalen Pfadabschnitt um Fahrzeugposition inkl. Normalenvektoren."""
        start_idx = self.find_closest_index(waypoints, self.vehicle_position)
        end_idx = start_idx + self.path_length
        return waypoints[start_idx:end_idx], normals[start_idx:end_idx]

    def optimize_path(self, local_path: np.ndarray, local_normals: np.ndarray, normalized_curvature: float) -> np.ndarray:
        """
        Verschiebt die Punkte des lokalen Pfads entlang ihrer Normalen,
        um in Kurven den Pfad enger zu schneiden.
        """
        shift = normalized_curvature * self.max_cut_shift
        return local_path + local_normals * shift
