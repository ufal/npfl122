#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import car_racing_evaluator

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=None, type=int, help="Training episodes.")
    parser.add_argument("--frame_skip", default=1, type=int, help="Repeat actions for given number of frames.")
    parser.add_argument("--frame_history", default=1, type=int, help="Number of past frames to stack together.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

    parser.add_argument("--alpha", default=None, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=None, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=None, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=None, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=None, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = car_racing_evaluator.environment(args.frame_skip)

    # TODO: Implement a variation to Deep Q Network algorithm.
    #
    # Example: How to perform an episode with "always gas" agent.
    state, done = env.reset(), False
    while not done:
        if args.render_each and (env.episode + 1) % args.render_each == 0:
            env.render()

        action = [0, 1, 0]
        next_state, reward, done, _ = env.step(action)

    # After training (or loading the model), you should run the evaluation:
    while True:
        state, done = env.reset(True), False
        while not done:
            # Choose greedy action
            state, reward, done, _ = env.step(action)
