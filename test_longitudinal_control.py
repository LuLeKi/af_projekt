import argparse

import gymnasium as gym
import numpy as np
from env_wrapper import CarRacingEnvWrapper
from input_controller import InputController
from longitudinal_control import LongitudinalControl
from matplotlib import pyplot as plt

fig = plt.figure()
plt.ion()
plt.show()


def run(env, input_controller: InputController):
    longitudinal_control = LongitudinalControl()

    seed = int(np.random.randint(0, int(1e6)))
    state_image, info = env.reset(seed=seed)
    total_reward = 0.0

    speed_history = []
    target_speed_history = []

    # <<< Gleich am Anfang Krümmung aus der Zukunft bestimmen
    if "trajectory" in info:
        info["curvature"] = estimate_future_curvature(info["trajectory"], future_idx=20, window_size=10)
    else:
        info["curvature"] = 0.0

    while not input_controller.quit:
        # Step 1: Steuerung berechnen
        target_speed = longitudinal_control.predict_target_speed(info["curvature"])
        acceleration, braking = longitudinal_control.control(info["speed"], target_speed)

        speed_history.append(info["speed"])
        target_speed_history.append(target_speed)
        print(f"Current Speed: {info['speed']:.2f} km/h | Target Speed: {target_speed:.2f} km/h | Curvature: {info['curvature']:.4f}")

        # Plotting
        plt.gcf().clear()
        plt.plot(speed_history, c="green", label="Speed")
        plt.plot(target_speed_history, c="blue", label="Target Speed")
        plt.legend()
        try:
            fig.canvas.flush_events()
        except:
            pass

        # Step 2: Eingabe holen und Aktion ausführen
        input_controller.update()
        a = [input_controller.steer, acceleration, braking]
        state_image, r, done, trunc, info = env.step(a)
        total_reward += r

        # <<< Nach dem neuen Schritt: Zukunfts-Krümmung neu auswerten
        if "trajectory" in info:
            info["curvature"] = estimate_future_curvature(info["trajectory"], future_idx=20, window_size=10)
        else:
            info["curvature"] = 0.0

        # Step 3: Reset wenn fertig
        input_controller.update()
        if done or input_controller.skip:
            print(f"seed: {seed:06d}     reward: {total_reward:06.2F}")
            input_controller.skip = False
            seed = int(np.random.randint(0, int(1e6)))
            state_image, info = env.reset(seed=seed)
            total_reward = 0.0
            speed_history = []
            target_speed_history = []

            if "trajectory" in info:
                info["curvature"] = estimate_future_curvature(info["trajectory"], future_idx=20, window_size=10)
            else:
                info["curvature"] = 0.0


def estimate_future_curvature(trajectory: np.ndarray, future_idx: int = 20, window_size: int = 10) -> float:
    """
    Schätzt die mittlere Krümmung in der Zukunft (nicht am aktuellen Fahrzeug).
    """
    if trajectory is None or len(trajectory) < 3:
        return 0.0

    future_idx = min(future_idx, len(trajectory) - 2)
    end_idx = min(future_idx + window_size, len(trajectory))

    section = trajectory[future_idx:end_idx]

    if len(section) < 3:
        return 0.0

    dx = np.gradient(section[:, 0])
    dy = np.gradient(section[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    mean_curvature = np.mean(np.abs(curvature))

    return np.clip(mean_curvature, 0.0, 1.0)





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
