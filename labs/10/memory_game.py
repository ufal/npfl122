#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import memory_game_evaluator

class Network:
    def __init__(self, env, args):
        self.args = args
        self.env = env

        # Define the agent inputs: a memory, and a state.
        memory = tf.keras.layers.Input(shape=[args.memory_cells, args.memory_cell_size], dtype=tf.float32)
        state = tf.keras.layers.Input(shape=len(env.states), dtype=tf.int32)

        # Encode the input state, which is a (card, observation) pair,
        # by representing each element as one-hot and concatenating them, resulting
        # in a vector of length sum(env.states).
        encoded_input = tf.keras.layers.Concatenate()([tf.one_hot(state[:, i], env.states[i]) for i in range(len(env.states))])

        # TODO: Generate a read key for memory read from the encoded input, by using
        # a ReLU hidden layer of size `args.hidden_layer` followed by a dense layer
        # with `args.memory_cell_size` units and tanh activation (to keep the memory
        # content in limited range).

        # TODO: Read the memory using the generated read key. Notably, compute cosine
        # similarity of the key and every memory row, apply softmax to generate
        # a weight distribution over the rows, and finally take a weighted average of
        # the memory rows.

        # TODO: Using concatenated encoded input and the read value, use a ReLU hidden
        # layer of size `args.hidden_layer` followed by a dense layer with
        # `env.actions` units and softmax activation to produce a policy.

        assert sum(env.states) == args.memory_cell_size
        # TODO: Perform memory write. For faster convergence, append directly
        # the `encoded_input` to the memory, i.e., add it as a first memory row, and drop
        # the last memory row to keep memory size constant.

        # Create the agent
        self._agent = tf.keras.Model(inputs=[memory, state], outputs=[updated_memory, policy])

        # TODO: Prepare an optimizer and a loss

    def zero_memory(self):
        # TODO: Return an empty memory. It should be a TF tensor
        # with shape (self.args.memory_cells, self.args.memory_cell_size).
        raise NotImplementedError()

    @tf.function
    def _train(self, states, targets):
        # TODO: Given a batch of sequences of `states` (each being a (card, symbol) pair),
        # train the network to predict the required `targets`.
        #
        # Specifically, start with a batch of empty memories, and run the agent
        # sequentially as many times as necessary, using `targets` as gold labels.
        #
        # Note that the sequences can be of different length, so you have to
        # provide a way of padding the episodes.
        raise NotImplementedError()

    def train(self, episodes):
        # TODO: Given a list of episodes, prepare the arguments
        # of the self._train method, and execute it.
        raise NotImplementedError()

    @tf.function
    def _predict(self, memory, state):
        return self._agent([memory, state])

    def predict(self, memory, state):
        memory, state = tf.convert_to_tensor(memory), tf.convert_to_tensor(state)
        return self._predict(memory, state)


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=None, type=int, help="Number of episodes to train on.")
    parser.add_argument("--cards", default=4, type=int, help="Number of cards in the memory game.")
    parser.add_argument("--hidden_layer", default=None, type=int, help="Hidden layer size; default 8*`cards`")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--memory_cells", default=None, type=int, help="Number of memory cells; default 2*`cards`")
    parser.add_argument("--memory_cell_size", default=None, type=int, help="Memory cell size; default 3/2*`cards`")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    if args.hidden_layer is None:
        args.hidden_layer = 8 * args.cards
    if args.memory_cells is None:
        args.memory_cells = 2 * args.cards
    if args.memory_cell_size is None:
        args.memory_cell_size = 3 * args.cards // 2

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = memory_game_evaluator.environment(args.cards)

    # Construct the network
    network = Network(env, args)

    def evaluate_episode(evaluating=False):
        rewards = 0
        state, memory, done = env.reset(evaluating), network.zero_memory(), False
        while not done:
            # TODO: Find out which action to use
            action = None

            state, reward, done, _ = env.step(action)
            rewards += reward
        return rewards

    # Training
    training = True
    while training:
        # Generate required number of episodes
        episodes = []
        for _ in range(args.batch_size):
            episodes.append(env.expert_episode())

        # Train the network
        network.train(episodes)

        # TODO: Maybe evaluate the current performance, using
        # `evaluate_episode()` method returning the achieved return,
        # and setting `training=False` when the performance is high enough.


    # Final evaluation
    while True:
        evaluate_episode(evaluating=True)
