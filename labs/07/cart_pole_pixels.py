#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import cart_pole_pixels_evaluator

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, state_shape, num_actions):
        with self.session.graph.as_default():
            self.states = tf.placeholder(tf.float32, [None] + state_shape)
            self.actions = tf.placeholder(tf.int32, [None])
            self.returns = tf.placeholder(tf.float32, [None])

            # TODO: Add network running inference.
            #
            # For generality, we assume the result is in `self.predictions`.
            #
            # Only this part of the network will be saved, in order not to save
            # optimizer variables (e.g., estimates of the gradient moments).

            # Saver for the inference network
            self.saver = tf.train.Saver()

            # TODO: Training using operation `self.training`.

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    def predict(self, states):
        return self.session.run(self.predictions, {self.states: states})

    def train(self, states, actions, returns):
        self.session.run(self.training, {self.states: states, self.actions: actions, self.returns: returns })

    def save(self, path):
        self.saver.save(self.session, path, write_meta_graph=False, write_state=False)

    def load(self, path):
        self.saver.restore(self.session, path)

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, type=str, help="Checkpoint path.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_pixels_evaluator.environment()

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, env.state_shape, env.actions)

    # Load the checkpoint if required
    if args.checkpoint:
        # Try extract it from embedded_data
        try:
            import embedded_data
            embedded_data.extract()
        except:
            pass
        network.load(args.checkpoint)

        # TODO: Evaluation

    else:
        # TODO: Training
        while not evaluating:
            pass

        # Save the trained model
        network.save("cart_pole_pixels/model")
