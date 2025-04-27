import argparse
import gymnasium as gym
from env_wrapper import CarRacingEnvWrapper
from input_controller import InputController
import numpy as np

def run(env, input_controller: InputController):
    seed = int(np.random.randint(0, int(1e6)))
    state_image, info = env.reset(seed=seed)

    while not input_controller.quit:
        # Steuerung updaten
        input_controller.update()

        # Action: [Steering, Gas, Brake]
        action = [input_controller.steer, input_controller.accelerate, input_controller.brake]

        state_image, reward, done, trunc, info = env.step(action)

        # Reset wenn Auto kaputt oder Runde zu Ende
        if done or input_controller.skip:
            input_controller.skip = False
            seed = int(np.random.randint(0, int(1e6)))
            state_image, info = env.reset(seed=seed)

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
