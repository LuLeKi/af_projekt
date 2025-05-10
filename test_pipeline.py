import argparse

import cv2
import gymnasium as gym
import numpy as np
from env_wrapper import CarRacingEnvWrapper
from input_controller import InputController
from lane_detection import LaneDetection
from lateral_control import LateralControl
from longitudinal_control import LongitudinalControl
from matplotlib import pyplot as plt
from path_planning import PathPlanning


def run(env, input_controller: InputController):
    lane_detection = LaneDetection()
    path_planning = PathPlanning()
    lateral_control = LateralControl()
    longitudinal_control = LongitudinalControl()

    seed = int(np.random.randint(0, int(1e6)))
    state_image, info = env.reset(seed=seed)
    total_reward = 0.0

    fig = plt.figure()
    plt.ion()
    plt.show()
    speed_history = []
    target_speed_history = []

    while not input_controller.quit:
        left_lane_boundaries, right_lane_boundaries = lane_detection.detect(state_image)
        trajectory, t, curvature = path_planning.plan(
            left_lane_boundaries, right_lane_boundaries
        )
        steering_angle = lateral_control.control(env.unwrapped.car, t, info["speed"])
        target_speed = longitudinal_control.predict_target_speed(curvature)
        target_speed = 15 
        #acceleration, braking = longitudinal_control.control(
        #    info["speed"], target_speed, steering_angle
        #)
        acceleration = 0.05
        braking = 0
        speed_history.append(info["speed"])
        target_speed_history.append(target_speed)

        cv_image = np.asarray(state_image, dtype=np.uint8)
        for index, point in enumerate( lateral_control.trajectory):
            if 0 < point[0] < 96 and 0 < point[1] < 96:
                cv_image[int(point[1]), int(point[0])] = [255, 255, 255] if index != lateral_control.lookahead_index else [0, 255, 0]
        for point in left_lane_boundaries:
        #for point in info["left_lane_boundary"]:
            if 0 < point[0] < 96 and 0 < point[1] < 84:
                cv_image[int(point[1]), int(point[0])] = [255, 0, 0]
        for point in right_lane_boundaries:
        #for point in info["right_lane_boundary"]:
            if 0 < point[0] < 96 and 0 < point[1] < 84:
                cv_image[int(point[1]), int(point[0])] = [0, 0, 255]


        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        cv_image = cv2.resize(cv_image, (cv_image.shape[1] * 6, cv_image.shape[0] * 6))
        cv2.imshow("Car Racing - Pipeline", cv_image)
        cv2.waitKey(1)

        # Longitudinal control plot
        plt.gcf().clear()
        plt.plot(speed_history, c="green")
        plt.plot(target_speed_history)
        try:
            fig.canvas.flush_events()
        except:
            pass

        # Step the environment
        a = [steering_angle, acceleration if info["speed"] < target_speed else 0, braking]
        state_image, r, done, trunc, info = env.step(a)
        total_reward += r

        # Reset environment if the run is skipped
        input_controller.update()
        if done or input_controller.skip:
            print(f"seed: {seed:06d}     reward: {total_reward:06.2F}")

            input_controller.skip = False
            seed = int(np.random.randint(0, int(1e6)))
            state_image, info = env.reset(seed=seed)
            total_reward = 0.0

            speed_history = []
            target_speed_history = []


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
