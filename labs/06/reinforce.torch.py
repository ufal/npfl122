#!/usr/bin/env python3
import argparse
import os

import gym
import numpy as np
import torch
import torch.nn as nn

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
parser.add_argument("--episodes", default=None, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=None, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=None, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=None, type=float, help="Learning rate.")


class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # Use GPU if available.
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # TODO: Create a suitable model. The predict method assumes
        # it is stored as `self._model` and that it returns logits.
        self._model = nn.Sequential(
            ...
        ).to(self._device)

        # TODO: Define an optimizer. Using `torch.optim.Adam` optimizer with
        # the given `args.learning_rate` is a good default.
        self._optimizer = ...

        # TODO: Define the loss (using a suitable `nn.?Loss` class).
        self._loss = ...

        # PyTorch uses Kaiming (=He) uniform initializer for weights and uniform for biases.
        # Tensorflow uses Glorot (=Xavier) uniform for weights and zeros for biases.
        # In some experiments, the TensorFlow initialization works slightly better for RL,
        # so you can uncomment the following lines to employ it.
        #   def init_weights_as_tensorflow(m):
        #       if isinstance(m, nn.Linear):
        #           nn.init.xavier_uniform_(m.weight)
        #           nn.init.zeros_(m.bias)
        #   self._model.apply(init_weights_as_tensorflow)

    # TODO: Define a training method.
    #
    # The `wrappers.typed_np_function` automatically converts input arguments
    # to NumPy arrays of given type, and converts the result to a NumPy array.
    @wrappers.typed_np_function(np.float32, np.int64, np.float32)
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        raise NotImplementedError()

    @wrappers.typed_np_function(np.float32)
    def predict(self, states: np.ndarray) -> np.ndarray:
        states = torch.from_numpy(states).to(self._device)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(states)
            policy = nn.functional.softmax(logits, dim=-1)
            return policy.cpu().numpy()


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and number of threads
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset()[0], False
            while not done:
                # TODO: Choose `action` according to probabilities
                # distribution (see `np.random.choice`), which you
                # can compute using `network.predict` and current `state`.
                action = ...

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # TODO: Compute returns by summing rewards (with discounting)

            # TODO: Add states, actions and returns to the training batch

        # TODO: Train using the generated batch.

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO: Choose a greedy action
            action = ...
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CartPole-v1"), args.seed, args.render_each)

    main(env, args)
