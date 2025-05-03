import argparse
import cv2
import gymnasium as gym
import numpy as np
from env_wrapper import CarRacingEnvWrapper
from input_controller import InputController
from path_planning import PathPlanning
from lane_detection import LaneDetection


def run(env, input_controller: InputController):
    path_planning = PathPlanning()
    lane_detection = LaneDetection()

    seed = int(np.random.randint(0, int(1e6)))
    state_image, info = env.reset(seed=seed)
    total_reward = 0.0

    while not input_controller.quit:
        # Spurerkennung
        left_lane, right_lane = lane_detection.detect(state_image)

        # Absicherung: Nur planen, wenn genug Punkte da sind
        if len(left_lane) < 5 or len(right_lane) < 5:
            print("[WARN] Zu wenige Spurpunkte erkannt, Frame übersprungen.")
            input_controller.update()
            state_image, r, done, trunc, info = env.step([0.0, 0.0, 0.0])
            total_reward += r
            continue

        try:
            optimized_waypoints, original_waypoints, curvature = path_planning.plan(left_lane, right_lane)
        except Exception as e:
            print(f"[ERROR] Fehler bei Pfadplanung: {e}")
            input_controller.update()
            state_image, r, done, trunc, info = env.step([0.0, 0.0, 0.0])
            total_reward += r
            continue

        cv_image = np.asarray(state_image, dtype=np.uint8)

        # Optional: ungültige Punkte filtern
        optimized_waypoints = [p for p in optimized_waypoints if isinstance(p, (list, np.ndarray)) and len(p) == 2]
        original_waypoints = [p for p in original_waypoints if isinstance(p, (list, np.ndarray)) and len(p) == 2]

        # Optimierter Pfad: grün
        for point in np.array(optimized_waypoints, dtype=np.int32):
            if 0 < point[0] < 96 and 0 < point[1] < 84:
                cv_image[int(point[1]), int(point[0])] = [0, 255, 0]  # Grün

        # Originaler Pfad: lila
        for point in np.array(original_waypoints, dtype=np.int32):
            if 0 < point[0] < 96 and 0 < point[1] < 84:
                cv_image[int(point[1]), int(point[0])] = [255, 0, 255]  # Lila

        # Linke Spur: rot
        for point in left_lane:
            if 0 < point[0] < 96 and 0 < point[1] < 84:
                cv_image[int(point[1]), int(point[0])] = [255, 0, 0]

        # Rechte Spur: blau
        for point in right_lane:
            if 0 < point[0] < 96 and 0 < point[1] < 84:
                cv_image[int(point[1]), int(point[0])] = [0, 0, 255]

        # Anzeige vorbereiten
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        cv_image = cv2.resize(cv_image, (cv_image.shape[1] * 6, cv_image.shape[0] * 6))
        cv2.imshow("Car Racing - Path Planning (Lane Detection)", cv_image)
        cv2.waitKey(1)

        # Step the environment
        input_controller.update()
        a = [
            input_controller.steer,
            input_controller.accelerate,
            input_controller.brake,
        ]
        state_image, r, done, trunc, info = env.step(a)
        total_reward += r

        # Reset bei Done oder Skip
        if done or input_controller.skip:
            print(f"seed: {seed:06d}     reward: {total_reward:06.2F}")
            input_controller.skip = False
            seed = int(np.random.randint(0, int(1e6)))
            state_image, info = env.reset(seed=seed)
            total_reward = 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_display", action="store_true", default=False)
    args = parser.parse_args()

    render_mode = "rgb_array" if args.no_display else "human"
    env = CarRacingEnvWrapper(
        gym.make("CarRacing-v3", render_mode=render_mode, domain_randomize=False)
    )
    input_controller = InputController()

    run(env, input_controller)
    env.reset()


if __name__ == "__main__":
    main()
