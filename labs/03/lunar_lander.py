#!/usr/bin/env python3
import argparse

import gym
import numpy as np

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=None, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=None, type=float, help="Exploration factor.")
parser.add_argument("--gamma", default=None, type=float, help="Discounting factor.")

def main(env, args):
    # Fix random seed
    np.random.seed(args.seed)

    # TODO: Implement a suitable RL algorithm.

    training = True
    while training:
        # To generate expert trajectory, you can use
        state, trajectory = env.expert_trajectory()

        # TODO: Perform a training episode
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            state, reward, done, _ = env.step(action)

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # TODO: Choose (greedy) action
            action = None
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(wrappers.DiscreteLunarLanderWrapper(gym.make("LunarLander-v2")), args.seed)

    main(env, args)
