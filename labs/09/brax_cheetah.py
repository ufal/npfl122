#!/usr/bin/env python3
import argparse
import collections
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--env", default="halfcheetah", type=str, help="Environment.")

def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    def evaluate_episode(start_evaluation:bool = False) -> float:
        rewards, state, done = 0, env.reset(start_evaluation), False
        while not done:
            if args.render_each and (env.episode + 1) % args.render_each == 0:
                # Store the current state in the visualization buffer.
                env.render("html")

            # TODO: Choose an action and perform a step
            action = None
            state, reward, done, _ = env.step(action)
            rewards += reward

        if args.render_each and env.episode % args.render_each == 0:
            # Produce an HTML visualization using all the stored states.
            env.render("html", path="{}{}.html".format(args.env, env.episode))
        return rewards

    # Evaluation in ReCodEx
    if args.recodex:
        while True:
            evaluate_episode(start_evaluation=True)

    # TODO: Perform training.
    #
    # Note that the SAC had issues with exploding gradients (the model started
    # to predict NaNs after several updates); the problem went away after
    # passing `clipnorm=10` to the `tf.optimizers.Adam`. Note that the
    # value `10` is my first try and definitely not an optimal value.
    #
    # Vectorized Brax environment can be created using
    #   venv = wrappers.BraxWrapper(args.env, workers=args.threads)
    raise NotImplementedError()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(wrappers.BraxWrapper(args.env), args.seed)

    main(env, args)
