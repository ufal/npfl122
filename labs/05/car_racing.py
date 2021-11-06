#!/usr/bin/env python3
import argparse
import collections
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

import car_racing_environment
import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--frame_skip", default=1, type=int, help="Frame skip.")

def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    if args.recodex:
        # TODO: Perform evaluation of a trained model.

        while True:
            state, done = env.reset(start_evaluation=True), False
            while not done:
                # TODO: Choose an action
                action = None
                state, reward, done, _ = env.step(action)

    else:
        # TODO: Perform training

        # If you want to create N multiprocessing parallel environments, use
        #   vector_env = gym.vector.AsyncVectorEnv([lambda: gym.make("CarRacingSoftFS{}-v0".format(args.frame_skip))] * N)
        #   vector_env.seed(args.seed) # The individual environments will get incremental seeds

        pass


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CarRacingSoftFS{}-v0".format(args.frame_skip)), args.seed, evaluate_for=15, report_each=1)

    main(env, args)
