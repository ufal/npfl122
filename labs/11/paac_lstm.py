#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # Report only TF errors by default

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
parser.add_argument("--entropy_regularization", default=None, type=float, help="Entropy regularization weight.")
parser.add_argument("--env", default="LunarLander-v2", type=str, help="Environment.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=None, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=None, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--lstm_size", default=None, type=int, help="LSTM dimensionality.")
parser.add_argument("--n", default=5, type=int, help="LSTM unroll steps.")
parser.add_argument("--transform_epsilon", default=None, type=int, help="Reward transform epsilon.")
parser.add_argument("--workers", default=None, type=int, help="Number of parallel workers.")

class Network:
    def __init__(self, env, args):
        # TODO: Similarly to reinforce with baseline, define two components:
        # - actor, which predicts distribution over the actions
        # - critic, which predicts the value function
        #
        # The actor should consist of
        # - a fully connected ReLU layer with `args.hidden_layer_size`
        # - a LSTM cell with `args.lstm_size` units
        # - a fully connected layer with `args.action_space.n` units and
        #   softmax activation.
        #
        # The critic is just the usual network with a single hidden layer
        # of size `args.hidden_layer_size`.
        #
        # Use the usual losses and Adam optimizer with given `args.learning_rate`.
        raise NotImplementedError()

    # The `wrappers.typed_np_function` automatically converts input arguments
    # to NumPy arrays of given type, and converts the result to a NumPy array.
    @wrappers.typed_np_function(np.float32, np.float32, np.int32, np.float32, np.float32)
    @tf.function
    def train(self, memories, states, actions, rewards, dones):
        # TODO: Perform training. Notably:
        # - compute n-step returns using the given information. I use the transformed
        #   rewards, but feel free to choose not to.
        # - train the critic using the n-step returns
        # - train the agent using policy gradient theorem. Be careful to reset memories
        #   on episode ends.
        raise NotImplementedError()

    @wrappers.typed_np_function(np.int32)
    def zero_memory(self, batch_size):
        # TODO: Return initial zero memory for a given batch size.
        raise NotImplementedError()

    @wrappers.typed_np_function(np.float32, np.float32)
    @tf.function
    def predict_actions(self, states, memory):
        # TODO: Given the states and memories (LSTM states),
        # return predicted action probabilities and new memories.
        raise NotImplementedError()

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_values(self, states):
        # TODO: Return estimates of value function.
        raise NotImplementedError()

def main(env, args):
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    def evaluate_episode(start_evaluation=False):
        rewards, state, memory, done = 0, env.reset(start_evaluation), network.zero_memory(1), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # Predict the action using the greedy policy
            [action_probs], memory = network.predict_actions([state], memory)
            action = np.argmax(action_probs)
            state, reward, done, _ = env.step(action)
            rewards += reward
        return rewards

    # Create the vectorized environment
    vector_env = gym.vector.AsyncVectorEnv([lambda: gym.make(env.spec.id)] * args.workers)

    zero_memories = network.zero_memory(args.workers)
    states, memories = vector_env.reset(), zero_memories

    training = True
    while training:
        # Training
        for _ in range(args.evaluate_each):
            batch_memories, batch_states, batch_actions, batch_rewards, batch_dones = memories, [], [], [], []
            for _ in range(args.n):
                # Choose actions using network.predict_actions
                action_probs, memories = network.predict_actions(states, memories)
                actions = [np.random.choice(env.action_space.n, p=probs) for probs in action_probs]

                # Perform steps in the vectorized environment
                next_states, rewards, dones, _ = vector_env.step(actions)

                # Append the information to the trajectory
                batch_states.append(states)
                batch_actions.append(actions)
                batch_rewards.append(rewards)
                batch_dones.append(dones)

                states = next_states
                memories = np.where(np.expand_dims(dones, -1), zero_memories, memories)

            batch_states.append(states)
            network.train(batch_memories, batch_states, batch_actions, batch_rewards, batch_dones)

        # Periodic evaluation
        for _ in range(args.evaluate_for):
            evaluate_episode()

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make(args.env), args.seed)

    main(env, args)
