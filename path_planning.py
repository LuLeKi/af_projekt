import numpy as np
import time
from scipy.interpolate import splprep, splev

class PathPlanning:
    """
    PathPlanning: Pfadplanung auf Basis von Fahrspurerkennung.
    Behandelt Spurverluste, Streckenende und dynamische Pfadanpassung.
    """

    def __init__(self):
        """Initialisiert alle Statusvariablen."""
        self._last_debug_time = time.time()
        self.vehicle_position = np.array([48.0, 64.0])
        self.last_valid_waypoints = None
        self.last_valid_curvature = 0.0
        self.invalid_counter = 0
        self.track_finished = False

    def plan(self, left_lane: np.ndarray, right_lane: np.ndarray):
        """
        Plant den nächsten Fahrzeugpfad basierend auf Spurgrenzen.

        Args:
            left_lane (np.ndarray): Punkte der linken Spurbegrenzung.
            right_lane (np.ndarray): Punkte der rechten Spurbegrenzung.

        Returns:
            Tuple[np.ndarray, float]: Anpassungspfad und normierte Krümmung.
        """
        if self.track_finished:
            return self.generate_emergency_stop_path()

        if not self.lanes_are_valid(left_lane, right_lane):
            return self.handle_invalid_data()

        waypoints, tck, u_fine = self.build_waypoints(left_lane, right_lane)

        if waypoints is None or len(waypoints) == 0:
            return self.handle_invalid_data()

        curvature = self.calculate_curvature(tck, u_fine)
        normalized_curvature = self.normalize_curvature(curvature)
        normals = self.calculate_normals(waypoints)

        local_path, local_normals = self.extract_local_path(waypoints, normals)

        if len(local_path) < 3:
            return self.handle_invalid_data()

        adjusted_path = self.adjust_and_smooth_path(local_path, local_normals, normalized_curvature)

        self.update_valid_state(adjusted_path, normalized_curvature)
        self.print_debug_info(left_lane, right_lane, waypoints, curvature, normalized_curvature, local_path, adjusted_path)

        return adjusted_path, normalized_curvature

    def lanes_are_valid(self, left_lane, right_lane):
        """Prüft ob die Fahrspurgrenzen plausibel und synchron sind."""
        if len(left_lane) == 0 or len(right_lane) == 0:
            return False
        num_points = min(len(left_lane), len(right_lane))
        distances = np.linalg.norm(left_lane[:num_points] - right_lane[:num_points], axis=1)
        return not np.any(distances < 5.0) and not np.any(distances > 100.0)

    def build_waypoints(self, left_lane, right_lane):
        """
        Erzeugt Wegpunkte basierend auf Mittellinien-Interpolation.

        Returns:
            Tuple[np.ndarray, optional]: Wegpunkte und Spline-Parameter falls verfügbar.
        """
        midpoints = (left_lane[:len(right_lane)] + right_lane[:len(left_lane)]) / 2.0
        if len(midpoints) >= 3:
            tck, u = splprep([midpoints[:, 0], midpoints[:, 1]], s=5.0, k=2)
            u_fine = np.linspace(0, 1, 500)
            x, y = splev(u_fine, tck)
            return np.vstack((x, y)).T, tck, u_fine
        return midpoints, None, None

    def handle_invalid_data(self):
        """
        Handhabt ungültige oder fehlerhafte Fahrspurinformationen.
        Gibt Dummy-Pfad oder zuletzt gültigen Pfad zurück.
        """
        self.invalid_counter += 1
        if self.invalid_counter < 50:
            return self.generate_dummy_path()
        if self.invalid_counter > 200 and not self.track_finished:
            print("\n→ Zu lange keine Spur gefunden! Strecke wird als beendet angesehen.\n")
            self.track_finished = True
            return self.generate_emergency_stop_path()
        return (self.last_valid_waypoints, self.last_valid_curvature) if self.last_valid_waypoints is not None else self.generate_dummy_path()

    def generate_emergency_stop_path(self):
        """Erstellt einen Notfallpfad zum sofortigen Anhalten."""
        stop_path = np.array([[self.vehicle_position[0], self.vehicle_position[1]]])
        return stop_path, 0.0

    def generate_dummy_path(self):
        """Erstellt einen kurzen Notfallpfad leicht nach vorne."""
        dummy_path = np.array([[self.vehicle_position[0], self.vehicle_position[1] + 5.0]])
        return dummy_path, 0.0

    def update_valid_state(self, path, curvature):
        """Aktualisiert den letzten gültigen Pfad und setzt Fehlerzähler zurück."""
        self.last_valid_waypoints = path.copy()
        self.last_valid_curvature = curvature
        self.invalid_counter = 0

    def calculate_curvature(self, tck, u_fine):
        """
        Berechnet die mittlere absolute Krümmung eines Splines.

        Args:
            tck: Spline-Parameter.
            u_fine: Verfeinerte Stützstellen.

        Returns:
            float: Mittlere Krümmung.
        """
        if tck is None or u_fine is None:
            return 0.0
        dx, dy = splev(u_fine, tck, der=1)
        ddx, ddy = splev(u_fine, tck, der=2)
        curvature = (dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
        mean_curvature = np.mean(np.abs(curvature)) if curvature.size > 0 else 0.0
        return 0.0 if mean_curvature > 1.0 else mean_curvature

    def normalize_curvature(self, curvature, min_radius=10.0, max_radius=1000.0):
        """
        Normalisiert die Krümmung auf einen Wert zwischen 0 und 1.
        
        Args:
            curvature (float): Echte Krümmung.

        Returns:
            float: Normierte Krümmung.
        """
        if curvature == 0:
            return 0.0
        radius = np.clip(1.0 / abs(curvature), min_radius, max_radius)
        return (max_radius - radius) / (max_radius - min_radius)

    def calculate_normals(self, waypoints):
        """
        Berechnet Normalenvektoren entlang der Mittellinie.
        
        Args:
            waypoints (np.ndarray): Punkte entlang der Spurmittellinie.

        Returns:
            np.ndarray: Array von Normalenvektoren.
        """
        normals = []
        for i in range(1, len(waypoints)):
            direction = waypoints[i] - waypoints[i-1]
            norm = np.linalg.norm(direction)
            normals.append(np.array([-direction[1], direction[0]]) / norm if norm else np.array([0.0, 0.0]))
        normals.append(normals[-1] if normals else np.array([0.0, 0.0]))
        return np.array(normals)

    def extract_local_path(self, waypoints, normals, length=50):
        """
        Extrahiert lokalen Pfadabschnitt basierend auf Fahrzeugposition.

        Args:
            waypoints (np.ndarray): Alle verfügbaren Wegpunkte.
            normals (np.ndarray): Berechnete Normalen.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Lokale Wegpunkte und zugehörige Normalen.
        """
        closest_idx = self.find_closest_index(waypoints, self.vehicle_position)
        return waypoints[closest_idx:closest_idx+length], normals[closest_idx:closest_idx+length]

    def find_closest_index(self, waypoints, position):
        """Findet Index des nächsten Wegpunkts zur aktuellen Fahrzeugposition."""
        distances = np.linalg.norm(waypoints - position, axis=1)
        return np.argmin(distances)

    def adjust_and_smooth_path(self, local_path, local_normals, normalized_curvature):
        """
        Verschiebt den lokalen Pfad dynamisch entlang der Normalen und glättet ihn.

        Args:
            local_path (np.ndarray): Lokaler unbearbeiteter Pfad.
            local_normals (np.ndarray): Normalenvektoren entlang des Pfads.
            normalized_curvature (float): Normierte Krümmung.

        Returns:
            np.ndarray: Angepasster und geglätteter Pfad.
        """
        adjusted = local_path.copy()
        shift = normalized_curvature * 5.0
        for i in range(len(adjusted)):
            adjusted[i] += local_normals[i] * shift
        if len(adjusted) < 3:
            return adjusted
        tck, u = splprep([adjusted[:, 0], adjusted[:, 1]], s=1.0, k=2)
        u_fine = np.linspace(0, 1, len(adjusted))
        x, y = splev(u_fine, tck)
        return np.vstack((x, y)).T

    def print_debug_info(self, left, right, midpoints, curvature, normalized_curvature, local_path, adjusted_path):
        """Gibt aktuelle Pfad- und Krümmungsinformationen aus (nur jede Sekunde einmal)."""
        now = time.time()
        if now - self._last_debug_time >= 1.0:
            print(f"L: {left[0]} | R: {right[0]} | M: {midpoints[0]}")
            print(f"Krümmung: {curvature:.5f} | Normierte Krümmung: {normalized_curvature:.2f}")
            if len(local_path) > 0:
                print(f"Lokaler Pfad: {local_path[0]} | Angepasst: {adjusted_path[0]}")
            print("-" * 60)
            self._last_debug_time = now
