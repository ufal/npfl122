#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import cart_pole_pixels_evaluator

class Network:
    def __init__(self, env, args):
        # TODO: Create a suitable network.

        # Warning: If you plan to use Keras `.train_on_batch` and/or `.predict_on_batch`
        # methods, pass `experimental_run_tf_function=False` to compile. There is
        # a bug in TF 2.0 which causes the `*_on_batch` methods not to use `tf.function`.

        # Otherwise, if you are training manually, using `tf.function` is a good idea
        # to get good performance.
        raise NotImplementedError()

    def train(self, states, actions, returns):
        states, actions, returns = np.array(states, np.float32), np.array(actions, np.int32), np.array(returns, np.float32)
        # TODO
        raise NotImplementedError()

    def predict(self, states):
        states = np.array(states, np.float32)
        # TODO
        raise NotImplementedError()


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=None, type=int, help="Number of episodes to train on.")
    parser.add_argument("--episodes", default=None, type=int, help="Training episodes.")
    parser.add_argument("--gamma", default=None, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layers", default=None, type=int, help="Number of hidden layers.")
    parser.add_argument("--hidden_layer_size", default=None, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=None, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = cart_pole_pixels_evaluator.environment()

    # Construct the network
    network = Network(env, args)

    # TODO: Training

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            # TODO: Compute action `probabilities` using `network.predict` and current `state`

            # Choose greedy action this time
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)
