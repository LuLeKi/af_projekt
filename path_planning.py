import numpy as np
import time
from scipy.interpolate import splprep, splev

class PathPlanning:
    def __init__(self):
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

        self._last_debug_time = time.time()

    def plan(self, left_lane: np.ndarray, right_lane: np.ndarray):

        path_waypoints = self.build_waypoints(left_lane, right_lane)
        path_trajectory, spline_model, u_fine = self.build_trajectory(path_waypoints, False)

        curvature = self.calculate_curvature(spline_model, u_fine)
        normalized_curvature = self.normalize_curvature(curvature)
        normals = self.calculate_normals(path_trajectory)

        local_path, local_normals = self.extract_local_path(path_trajectory, normals)

        optimized_local_waypoints = self.optimize_path(local_path, local_normals, normalized_curvature)

        optimized_local_path = self.build_trajectory(optimized_local_waypoints, True)
        if self.should_debug():
            self.debug(local_path, optimized_local_waypoints, local_normals, curvature, normalized_curvature)


        # export unoptimierten local path für debug auf test_path_planning
        # return optimized_local_path, local_path, normalized_curvature
        return optimized_local_path, normalized_curvature

    def build_waypoints(self, left_lane, right_lane):
        min_len = min(len(left_lane), len(right_lane))
        return (left_lane[:min_len] + right_lane[:min_len]) / 2.0

    def build_trajectory(self, waypoints, optimized=False):
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
        if spline_model is None or u_fine is None:
            return 0.0
        dx, dy = splev(u_fine, spline_model, der=1)
        ddx, ddy = splev(u_fine, spline_model, der=2)
        curvature = (dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
        return curvature

    def normalize_curvature(self, curvature):
        if curvature is None or len(curvature) == 0:
            return 0.0
        median_curv = np.median(curvature)
        threshold = self.curvature_clip_threshold
        return np.clip(median_curv / threshold, -1.0, 1.0)

    def calculate_normals(self, waypoints):
        normals = []
        for i in range(1, len(waypoints)):
            direction = waypoints[i] - waypoints[i-1]
            norm = np.linalg.norm(direction)
            normals.append(np.array([-direction[1], direction[0]]) / norm if norm else np.array([0.0, 0.0]))
        normals.append(normals[-1] if normals else np.array([0.0, 0.0]))
        return np.array(normals)

    def optimize_path(self, local_path, local_normals, normalized_curvature):
        adjusted = local_path.copy()
        shift = normalized_curvature * self.max_cut_shift
        for i in range(len(adjusted)):
            adjusted[i] += local_normals[i] * shift
        return np.array(adjusted)
    
    def find_closest_index(self, waypoints, position):
        distances = np.linalg.norm(waypoints - position, axis=1)
        return int(np.argmin(distances))

    def extract_local_path(self, waypoints, normals):
        length = self.path_length
        closest_idx = self.find_closest_index(waypoints, self.vehicle_position)
        return waypoints[closest_idx:closest_idx+length], normals[closest_idx:closest_idx+length]

    def should_debug(self, interval=1):
        now = time.time()
        if now - self._last_debug_time >= interval:
            self._last_debug_time = now
            return True
        return False
   
    def debug(self, trajectory_path, adjusted, normals, raw_curvature, normalized_curvature_scalar):
        idx = self.find_closest_index(trajectory_path, self.vehicle_position)
        print("\n================== [DEBUG: PATH OPTIMIZATION] ==================")
        print(f"🚗 Fahrzeugposition      : {self.vehicle_position}")
        print(f"🧮 Parameter:")
        print(f"   max_cut_shift         = {self.max_cut_shift}")
        print(f"   curvature_clip_thres  = {self.curvature_clip_threshold}")
        print(f"   splinegrad            = {self.splinegrad} (original) / {self.opti_splinegrad} (optimiert)")
        print(f"   smoothing             = {self.trajecotry_smoothing} / {self.opti_trajecotry_smoothing}")
        print(f"   scalar                = {self.trajectory_scalar} / {self.opti_trajectory_scalar}")
        print(f"📈 Normierte Krümmung    : {normalized_curvature_scalar:+.3f} ➞ Shift: {normalized_curvature_scalar * self.max_cut_shift:+.3f}")
        print("----------------------------------------------------------------")

        for j in range(idx, min(idx + 5, len(adjusted))):
            raw = raw_curvature[j] if isinstance(raw_curvature, np.ndarray) else raw_curvature
            shift_j = normalized_curvature_scalar * self.max_cut_shift
            dist_before = np.linalg.norm(self.vehicle_position - trajectory_path[j])
            dist_after  = np.linalg.norm(self.vehicle_position - adjusted[j])
            offset_vec  = adjusted[j] - trajectory_path[j]
            offset_len  = np.linalg.norm(offset_vec)

            print(f"[{j}]")
            print(f"  raw_curv         = {raw:+.5f}")
            print(f"  normal           = {normals[j]}")
            print(f"  offset (delta)   = {offset_vec}, len = {offset_len:.3f}")
            print(f"  dist before/after= {dist_before:.2f} / {dist_after:.2f}")
            if abs(shift_j) > 0.8 * self.max_cut_shift:
                print("  ⚠️  Achtung: Nahe max. Verschiebung!")
            if abs(normalized_curvature_scalar) > 0.9:
                print("  ⚠️  Achtung: hohe Norm-Krümmung!")
            print("----------------------------------------------------------------")
