#!/usr/bin/env python3
import numpy as np

import cart_pole_evaluator

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=None, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")

    parser.add_argument("--epsilon", default=None, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=None, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=None, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_evaluator.environment()

    # TODO: Implement Monte-Carlo RL algorithm.
    #
    # The overall structure of the code follows.

    while training:
        # Perform a training episode
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            next_state, reward, done, _ = env.step(action)

    # Perform last 100 evaluation episodes
