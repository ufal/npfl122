#!/usr/bin/env python3
import numpy as np

import car_racing_evaluator

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=None, type=int, help="Training episodes.")
    parser.add_argument("--frame_skip", default=1, type=int, help="Repeat actions for given number of frames.")
    parser.add_argument("--frame_history", default=1, type=int, help="Number of past frames to stack together.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")

    parser.add_argument("--alpha", default=None, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=None, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=None, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=None, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=None, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = car_racing_evaluator.environment()

    # TODO: Implement a variation to Deep Q Network algorithm.

    # Perform a training episode
    state, done = env.reset(), False
    while not done:
        if args.render_each and (env.episode + 1) % args.render_each == 0:
            env.render()

        action = [0, 1, 0]
        next_state, reward, done, _ = env.step(action)
