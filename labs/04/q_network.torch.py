#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections

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
parser.add_argument("--epsilon", default=None, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=None, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=None, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=None, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=None, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=None, type=float, help="Learning rate.")
parser.add_argument("--target_update_freq", default=None, type=int, help="Target update frequency.")


class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # Use GPU if available.
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # TODO: Create a suitable model and store it as `self._model`.
        self._model = nn.Sequential(
            ...
        ).to(self._device)

        # TODO: Define an optimizer (probably from `torch.optim`).
        self._optimizer = ...

        # TODO: Define the loss (most likely some `nn.?Loss`).
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

    # Define a training method. Generally you have two possibilities
    # - pass new q_values of all actions for a given state; all but one are the same as before
    # - pass only one new q_value for a given state, and include the index of the action to which
    #   the new q_value belongs
    # The code below implements the first option, but you can change it if you want.
    #
    # The `wrappers.typed_np_function` automatically converts input arguments
    # to NumPy arrays of given type, and converts the result to a NumPy array.
    @wrappers.typed_np_function(np.float32, np.float32)
    def train(self, states: np.ndarray, q_values: np.ndarray) -> None:
        states = torch.from_numpy(states).to(self._device)
        q_values = torch.from_numpy(q_values).to(self._device)

        self._model.train()
        self._optimizer.zero_grad()
        predictions = self._model(states)
        loss = self._loss(predictions, q_values)
        loss.backward()
        self._optimizer.step()

    @wrappers.typed_np_function(np.float32)
    def predict(self, states: np.ndarray) -> np.ndarray:
        states = torch.from_numpy(states).to(self._device)
        self._model.eval()
        with torch.no_grad():
            return self._model(states).cpu().numpy()

    # If you want to use target network, the following method copies weights from
    # a given Network to the current one.
    def copy_weights_from(self, other: Network) -> None:
        params = dict(self._model.named_parameters())
        params_other = dict(other._model.named_parameters())
        with torch.no_grad():
            for name, value in params_other.items():
                params[name].data.copy_(value.data)


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and number of threads
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque()
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    epsilon = args.epsilon
    training = True
    while training:
        # Perform episode
        state, done = env.reset()[0], False
        while not done:
            # TODO: Choose an action.
            # You can compute the q_values of a given state by
            #   q_values = network.predict([state])[0]
            action = ...

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Append state, action, reward, done and next_state to replay_buffer
            replay_buffer.append(Transition(state, action, reward, done, next_state))

            # TODO: If the replay_buffer is large enough, perform a training batch
            # from `args.batch_size` uniformly randomly chosen transitions.
            #
            # After you choose `states` and suitable targets, you can train the network as
            #   network.train(states, ...)

            state = next_state

        if args.epsilon_final_at:
            epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO: Choose (greedy) action
            action = ...
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CartPole-v1"), args.seed, args.render_each)

    main(env, args)
