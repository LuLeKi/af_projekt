###
###     This Test Files shows the unoptimized path as a yellow line and the optimized as a green line
###         It uses the left and right_lane_boundarys either out of ENV WRAPPER or LANE DETECTION

import argparse

import cv2
import gymnasium as gym
import numpy as np

from env_wrapper import CarRacingEnvWrapper
from lane_detection import LaneDetection


from input_controller import InputController
from path_planning import PathPlanning


lane_detection_lanes = False     # True → Lane Detection nutzen, False → env-wrapper



def run(env, input_controller: InputController):
    path_planning = PathPlanning()
    lane_detection = LaneDetection()

    seed = int(np.random.randint(0, int(1e6)))
    state_image, info = env.reset(seed=seed)
    total_reward = 0.0

    while not input_controller.quit:
        if lane_detection_lanes:
            left_lane, right_lane = lane_detection.detect(state_image)
        else:
            left_lane, right_lane = info["left_lane_boundary"], info["right_lane_boundary"]

        opti_way_points, way_points, curvature = path_planning.plan(left_lane, right_lane)

        cv_image = np.asarray(state_image, dtype=np.uint8)
        way_points = np.array(way_points, dtype=np.int32)
        #for opti waypoints
        opti_way_points = np.array(opti_way_points, dtype=np.int32)

        for point in opti_way_points:
            if 0 < point[0] < 96 and 0 < point[1] < 84:
                cv_image[int(point[1]), int(point[0])] = [0, 255, 0]
        #show unoptimized way_points
        for point in way_points:
            if 0 < point[0] < 96 and 0 < point[1] < 84:
                cv_image[int(point[1]), int(point[0])] = [255, 255, 0]

        for point in left_lane:
            if 0 < point[0] < 96 and 0 < point[1] < 84:
                cv_image[int(point[1]), int(point[0])] = [255, 0, 0]
        for point in right_lane:
            if 0 < point[0] < 96 and 0 < point[1] < 84:
                cv_image[int(point[1]), int(point[0])] = [0, 0, 255]

        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        cv_image = cv2.resize(cv_image, (cv_image.shape[1] * 6, cv_image.shape[0] * 6))
        cv2.imshow("Car Racing - Path Planning", cv_image)
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

        # Reset environment if the run is skipped
        input_controller.update()
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
