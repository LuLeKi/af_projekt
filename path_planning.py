import numpy as np
import time
from scipy.interpolate import splprep, splev

class PathPlanning:
    def __init__(self):
        self.vehicle_position = np.array([48.0, 64.0])

        # Für rohe Trajektorie – Basislinie, darf „sanfter“ sein
        self.trajectory_scalar = 1500
        self.trajecotry_smoothing = 15
        self.splinegrad = 2

        # Für optimierten Pfad – darf feiner auf Kurven reagieren
        self.opti_trajectory_scalar = 1500
        self.opti_trajecotry_smoothing = 10 # weniger glätten
        self.opti_splinegrad = 2            # kein Overshoot

        self.curvature_clip_threshold = 0.02
        self.max_cut_shift = 8

        self.distance_threshold = 100

        self._last_debug_time = time.time()

    def plan(self, left_lane: np.ndarray, right_lane: np.ndarray):
        path_waypoints = self.build_waypoints(left_lane, right_lane)
        path_trajectory, spline_model, u_fine = self.build_trajectory(path_waypoints, False)

        curvature = self.calculate_curvature(spline_model, u_fine)
        normalized_curvature = self.normalize_curvature(curvature)
        normals = self.calculate_normals(spline_model, u_fine)

        optimized_path_waypoints = self.optimize_path(path_trajectory, normals, curvature, normalized_curvature)
        optimized_path_trajectory = self.build_trajectory(optimized_path_waypoints, True)

        optimized_local_path = self.extract_local_path(optimized_path_trajectory)
        local_path = self.extract_local_path(path_trajectory)
        return optimized_local_path, local_path, curvature

    def build_waypoints(self, left_lane, right_lane):
        min_len = min(len(left_lane), len(right_lane))
        return (left_lane[:min_len] + right_lane[:min_len]) / 2.0

    def build_trajectory(self, waypoints, optimized=False):
        if optimized:
            smoothing = self.opti_trajecotry_smoothing
            splinegrad = self.opti_splinegrad
            scalar = self.opti_trajectory_scalar
        else:
            smoothing = self.trajecotry_smoothing
            splinegrad = self.splinegrad
            scalar = self.trajectory_scalar

        spline_model = splprep([waypoints[:, 0], waypoints[:, 1]],
                            s=smoothing, k=splinegrad)[0]
        u_fine = np.linspace(0, 1, scalar)
        x_spline, y_spline = splev(u_fine, spline_model)

        if optimized:
            return np.vstack((x_spline, y_spline)).T
        else:
            return np.vstack((x_spline, y_spline)).T, spline_model, u_fine

    
    
    def calculate_curvature(self, spline_model, u_fine):
        dx, dy = splev(u_fine, spline_model, der=1)
        ddx, ddy = splev(u_fine, spline_model, der=2)
        numerator = dx * ddy - dy * ddx
        denominator = (dx**2 + dy**2) ** 1.5
        with np.errstate(divide='ignore', invalid='ignore'):
            curvature = np.where(denominator != 0, numerator / denominator, 0.0)
        return curvature
    
    def normalize_curvature(self, curvature: np.ndarray) -> np.ndarray:
        return np.clip(curvature / self.curvature_clip_threshold, -1.0, 1.0)


    def calculate_normals(self, spline_model, u_values: np.ndarray) -> np.ndarray:
        # 1. Tangenten berechnen (1. Ableitung der Spline-Funktion)
        dx, dy = splev(u_values, spline_model, der=1)
        tangents = np.stack((dx, dy), axis=-1)  # Shape: (N, 2)

        # 2. Normieren der Tangenten
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents_normalized = np.divide(tangents, norms, out=np.zeros_like(tangents), where=norms != 0)

        # 3. Normalen durch 90°-Rotation der Tangenten (rechtsdrehend)
        # [x, y] → [-y, x]
        normals = np.stack((-tangents_normalized[:, 1], tangents_normalized[:, 0]), axis=-1)

        return normals


    def optimize_path(self, trajectory_path, normals, curvature, normalized_curvature):

        adjusted = trajectory_path.copy()
        for i in range(len(adjusted)):
            shift = normalized_curvature[i] * self.max_cut_shift  
            adjusted[i] += normals[i] * shift

            if self.should_debug():
                self.debug(trajectory_path, adjusted, normals, curvature, normalized_curvature)


        return adjusted


    def should_debug(self, interval=1):
        now = time.time()
        if now - self._last_debug_time >= interval:
            self._last_debug_time = now
            return True
        return False
   
    def debug(self, trajectory_path, adjusted, normals, raw_curvature, normalized_curvature):
        idx = self.find_closest_index(trajectory_path, self.vehicle_position)
        print("\n================== [FULL DEBUG: PATH OPTIMIZATION] ==================")
        print(f"Fahrzeugposition: {self.vehicle_position}")
        print(f"Parameter: max_cut_shift={self.max_cut_shift}, "
            f"curvature_clip_threshold={self.curvature_clip_threshold}, "
            f"splinegrad={self.splinegrad}/{self.opti_splinegrad}, "
            f"smoothing={self.trajecotry_smoothing}/{self.opti_trajecotry_smoothing}, "
            f"scalar={self.trajectory_scalar}/{self.opti_trajectory_scalar}")
        print("---------------------------------------------------------------------")

        for j in range(idx, min(idx + 5, len(adjusted))):
            raw = raw_curvature[j]
            norm = normalized_curvature[j]
            shift_j = norm * self.max_cut_shift
            tangent = adjusted[j] - trajectory_path[j]
            tangent_len = np.linalg.norm(tangent)
            normal_len = np.linalg.norm(normals[j])
            dist_to_vehicle = np.linalg.norm(trajectory_path[j] - self.vehicle_position)

            print(f"[{j}]")
            print(f"  raw_curv      = {raw:+.5f}")
            print(f"  norm_curv     = {norm:+.2f}")
            print(f"  shift         = {shift_j:+.3f} (max = {self.max_cut_shift})")
            print(f"  old_pos       = {trajectory_path[j]}")
            print(f"  new_pos       = {adjusted[j]}")
            print(f"  normal        = {normals[j]}, |n| = {normal_len:.3f}")
            print(f"  shift_vec     = {tangent}, |shift_vec| = {tangent_len:.3f}")
            print(f"  dist_to_vehicle = {dist_to_vehicle:.2f}")
            if abs(norm) > 5:
                print("  ⚠️  Achtung: sehr starke normierte Krümmung!")
            if abs(shift_j) > 0.8 * self.max_cut_shift:
                print("  ⚠️  Achtung: fast maximaler Shift erreicht!")
            print("---------------------------------------------------------------------")


        
    def find_closest_index(self, waypoints, position):
        distances = np.linalg.norm(waypoints - position, axis=1)
        return int(np.argmin(distances))

    def extract_local_path(self, trajectory_points):
        closest_idx = self.find_closest_index(trajectory_points, self.vehicle_position)
        local_path = []
        total_distance = 0.0
        for i in range(closest_idx, len(trajectory_points) - 1):
            current_point = trajectory_points[i]
            next_point = trajectory_points[i + 1]
            segment_length = np.linalg.norm(next_point - current_point)
            total_distance += segment_length
            local_path.append(current_point)
            if total_distance >= self.distance_threshold:
                break
        return np.array(local_path)
