import argparse
import gymnasium as gym
import numpy as np
import cv2  
import atexit

from input_controller import InputController
from matplotlib import pyplot as plt

from env_wrapper import CarRacingEnvWrapper
from longitudinal_control import LongitudinalControl
from path_planning import PathPlanning           
from lane_detection import LaneDetection         
from lateral_control import LateralControl      

# Flags für Auswahl der Spurquelle und Lenkregelung
lane_detection_lanes = True     # True → Lane Detection nutzen, False → env-wrapper
lateral_control_steering = True # True → Regler, False → manuelle Steuerung (wasd)
enable_plotting = False 
enable_view_path = True 

if enable_plotting:
    fig = plt.figure()
    plt.ion()
    plt.show()


def run(env, input_controller: InputController):
    # Instanziierung der Module
    longitudinal_control = LongitudinalControl()
    path_planning = PathPlanning()
    lane_detection = LaneDetection() 

    if lateral_control_steering:
        lateral_control = LateralControl()

    # Reset-Phase (nach Instanzen, vor Loop)
    seed = int(np.random.randint(0, int(1e6)))
    state_image, info = env.reset(seed=seed)
    total_reward = 0.0

    speed_history = []
    target_speed_history = []
    plot_counter = 0

    while not input_controller.quit:
        # Spurquelle wählen (Lane Detection oder Env-Werte)
        if lane_detection_lanes:
            left_lane, right_lane = lane_detection.detect(state_image)
        else:
            left_lane, right_lane = info["left_lane_boundary"], info["right_lane_boundary"]

        # Trajektorie + Krümmung berechnen
        trajectory, _, curvature = path_planning.plan(left_lane, right_lane)

        # Lenkwinkel berechnen (manuell oder per Regler)
        if lateral_control_steering:
            steering = lateral_control.control(env.unwrapped.car, trajectory, info["speed"])
        else:
            steering = input_controller.steer
            

        # Zielgeschwindigkeit & Reglersteuerung
        target_speed = longitudinal_control.predict_target_speed(curvature, info["speed"], steering)
        acceleration, braking = longitudinal_control.control(info["speed"], target_speed, steering)

        speed_history.append(info["speed"])
        target_speed_history.append(target_speed)
        if enable_view_path:
            visualize_planning_view(state_image, left_lane, right_lane, trajectory)

  

        # Plot-Update
        if enable_plotting:
            plot_counter += 1
            if plot_counter % 4 == 0:  # Alle ~4 Frames ≈ 15 FPS bei 60Hz
                plt.gcf().clear()
                plt.plot(speed_history[-200:], c="green", label="Ist-Speed")  # Nur letzte 200 Punkte
                plt.plot(target_speed_history[-200:], label="Ziel-Speed")
                plt.legend()
                try:
                    fig.canvas.flush_events()
                except:
                    pass

        # Nächster Schritt in der Simulation
        input_controller.update()
        a = [steering, acceleration, braking]  #steering jetzt aus Bedingung, nicht input_controller.steer direkt
        state_image, r, done, trunc, info = env.step(a)
        total_reward += r

        # Run ggf. neu starten
        input_controller.update()
        if done or input_controller.skip:
            print(f"seed: {seed:06d}     reward: {total_reward:06.2F}")
            input_controller.skip = False
            seed = int(np.random.randint(0, int(1e6)))
            state_image, info = env.reset(seed=seed)
            total_reward = 0.0
            speed_history = []
            target_speed_history = []


def visualize_planning_view(state_image, left_lane, right_lane, trajectory):
    cv_image = np.asarray(state_image, dtype=np.uint8).copy()

    # Konvertiere alle Punktlisten einmalig zu np.int32
    if (isinstance(trajectory, tuple)): 
        trajectory = np.array(trajectory[0], dtype=np.int32)
    else:
        trajectory = np.array(trajectory, dtype=np.int32)
    left_lane = np.array(left_lane, dtype=np.int32)
    right_lane = np.array(right_lane, dtype=np.int32)

    for point in trajectory:
        if 0 < point[0] < 96 and 0 < point[1] < 84:
            cv_image[point[1], point[0]] = [0, 255, 0]  # Grün: optimierte Spur

    for point in left_lane:
        if 0 < point[0] < 96 and 0 < point[1] < 84:
            cv_image[point[1], point[0]] = [255, 0, 0]  # Rot: linke Spur

    for point in right_lane:
        if 0 < point[0] < 96 and 0 < point[1] < 84:
            cv_image[point[1], point[0]] = [0, 0, 255]  # Blau: rechte Spur

    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    cv_image = cv2.resize(cv_image, (cv_image.shape[1] * 6, cv_image.shape[0] * 6))
    cv2.imshow("Planning View", cv_image)
    cv2.waitKey(1)



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
    if enable_view_path:
        atexit.register(cv2.destroyAllWindows)


if __name__ == "__main__":
    main()
