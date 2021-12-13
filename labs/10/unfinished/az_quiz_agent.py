#!/usr/bin/env python3
import argparse
import collections
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

import az_quiz

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.3, type=float, help="MCTS root Dirichlet alpha")
parser.add_argument("--batch_size", default=512, type=int, help="Number of game positions to train on.")
parser.add_argument("--num_simulations", default=100, type=int, help="Number of simulations in one MCTS.")
parser.add_argument("--evaluate_each", default=None, type=int, help="Evaluate each number of iterations.")
parser.add_argument("--train_for", default=None, type=int, help="Update steps in every iteration.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--sampling_moves", default=8, type=int, help="Sampling moves.")
parser.add_argument("--sim_games", default=None, type=int, help="Simulated games to generate in every iteration.")
parser.add_argument("--window_length", default=100000, type=int, help="Replay buffer max length.")
args = parser.parse_args()

class Agent:
    # TODO: Define an agent network. A possible architecture known to work consits of
    # - 5 convolutional layers with 3x3 kernel and 15-20 filters,
    # - a policy head, which first uses 3x3 convolution to reduce the number of channels
    #   to 2, flattens the representation, and finally uses a dense layer with softmax
    #   activation to produce the policy,
    # - a value head, which again uses 3x3 convolution to reduce the number of channels
    #   to 2, flattens, and produces expected return using an output dense layer with
    #   tanh activation.
    pass
