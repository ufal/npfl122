#!/usr/bin/env python3
import itertools
import sys

import numpy as np
import tensorflow as tf

import memory_game_evaluator

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, num_observations, num_actions):
        with self.session.graph.as_default():
            memory_size = num_observations + num_actions

            self.steps = tf.placeholder(tf.int32, [None])
            self.states = tf.placeholder(tf.float32, [None, 2 * args.rnn_dim + args.memory_cells * memory_size])
            self.observations = tf.placeholder(tf.int32, [None, None])
            self.actions = tf.placeholder(tf.int32, [None, None])
            self.returns = tf.placeholder(tf.float32, [None, None])

            # One-hot encoding of observations and actions
            inputs = tf.concat([tf.one_hot(self.observations, num_observations), tf.one_hot(self.actions[:, :-1], num_actions)], axis=2)

            # Decode state
            lstm_states = tf.nn.rnn_cell.LSTMStateTuple(*tf.split(self.states[:, :2 * args.rnn_dim], 2, axis=1))
            if args.memory_cells:
                memory = tf.reshape(self.states[:, 2 * args.rnn_dim:], [-1, args.memory_cells, memory_size])

                # Run RNN with the external memory, using `lstm_states` as
                # initial states and `memory` as initial memory.

                lstm_cell = tf.nn.rnn_cell.LSTMCell(args.rnn_dim)
                def rnn_step(states, inputs):
                    lstm_states, read_values, memory = states

                    # TODO: Run LSTM, processing concatenation of `inputs` and `read_values` as inputs
                    # and `lstm_states` as states, generating `outputs` and `lstm_states`.

                    # TODO: Read from memory, using `args.read_keys` number
                    # of read keys. For each read key
                    # - take `memory_size` unused elements of the beginning of `outputs`
                    # - compute cosine similarity of the read key and all memory locations
                    # - process the similarities by a softmax
                    # - produce `read_value` by reading the memory according to the softmaxed weights
                    # Finally, concatenate all `read_value`s into a `read_values`.

                    # TODO: Write to `memory`, by storing `inputs` in the first memory location and
                    # shifting the other (therefore, dropping the last one).

                    return lstm_states, read_values, memory

                # Run tf.scan which iteratively applies `rnn_step`.
                lstm_states, read_values, memory = tf.scan(
                    rnn_step, tf.transpose(inputs, [1, 0, 2]),
                    initializer=(lstm_states, tf.zeros([tf.shape(inputs)[0], args.read_keys * memory_size]), memory))

                read_values = tf.transpose(read_values, [1, 0, 2])

                # Generate outputs, ignoring the elements already used by read keys
                outputs = tf.transpose(lstm_states.h[:, :, args.read_keys * memory_size:], [1, 0, 2])

                # Construct indices to the last step in every sequence and
                # use them to find corresponding `lstm_states` and `memory`.
                ends = tf.stack([tf.range(tf.shape(inputs)[0]), self.steps - 1], axis=1)
                lstm_states = tf.nn.rnn_cell.LSTMStateTuple(tf.gather_nd(lstm_states.c, ends), tf.gather_nd(lstm_states.h, ends))
                memory = tf.gather_nd(memory, ends)

                # Compute hidden layer by concatenating inputs, outputs and read_values.
                hidden_layer = tf.concat([outputs, inputs, read_values], axis=2)
                # The predicted states are the LSTM states and the memory.
                self.predicted_states = tf.concat(list(lstm_states) + [tf.reshape(memory, [-1, args.memory_cells * memory_size])], axis=1)
            else:
                # TODO: Plain RNN without memory, using lstm_states as initial states.
                # The goal is to generate `hidden_layer` with per-step hidden layer values
                # and `self.predicted_states` with ending LSTM states per batch instance.
                pass

            # Predict action probabilities
            logits = tf.layers.dense(tf.layers.dense(hidden_layer, args.rnn_dim, activation=tf.nn.relu), num_actions)
            self.probabilities = tf.nn.softmax(logits)

            # Saver
            self.saver = tf.train.Saver()

            # Training
            weights = tf.sequence_mask(self.steps, dtype=tf.float32)
            loss = tf.losses.sparse_softmax_cross_entropy(self.actions[:, 1:], logits, weights=weights * self.returns)
            loss += -args.entropy_regularization * tf.reduce_mean(tf.distributions.Categorical(logits=logits).entropy() * weights)
            self.training = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=tf.train.create_global_step(), name="training")

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    def zero_state(self):
        return np.zeros([self.states.shape[1].value])

    def predict(self, state, observation, action):
        [state], [[probabilities]] = self.session.run([self.predicted_states, self.probabilities],
                                                      {self.steps: [1], self.states: [state],
                                                       self.observations: [[observation]], self.actions: [[action, action]]})
        return state, probabilities

    def train(self, steps, states, observations, actions, returns):
        self.session.run(self.training, {self.steps: steps, self.states: states, self.observations: observations,
                                         self.actions: actions, self.returns: returns})


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=5, type=int, help="Number of episodes to train on.")
    parser.add_argument("--baseline_decay", default=0.99, type=float, help="Baseline exponential decay.")
    parser.add_argument("--cards", default=None, type=int, help="Number of cards in the memory game.")
    parser.add_argument("--entropy_regularization", default=0.1, type=float, help="Entropy regularization weight.")
    parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of episodes.")
    parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate for number of episodes.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--memory_cells", default=0, type=int, help="Memory cells.")
    parser.add_argument("--read_keys", default=1, type=int, help="Number of read keys.")
    parser.add_argument("--rnn_dim", default=64, type=int, help="RNN dimensionality.")
    parser.add_argument("--learning_rate", default=0.005, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create the environment
    env = memory_game_evaluator.environment(args.cards)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, env.states, env.actions)

    def evaluate_episode(evaluating=False):
        rewards = 0
        observation, state, action, done = env.reset(evaluating), network.zero_state(), 0, False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            state, probabilities = network.predict(state, observation, action)
            action = np.argmax(probabilities)
            observation, reward, done, _ = env.step(action)
            rewards += reward
        return rewards

    baseline, episodes = np.zeros([1], dtype=np.float32), 0
    while True:
        # Training
        batch_observations, batch_actions, batch_returns = [], [], []
        for episode in range(args.evaluate_each):
            # Perform episode
            observations, actions, rewards = [], [0], []
            observation, state, action, done = env.reset(), network.zero_state(), 0, False
            while not done:
                state, probabilities = network.predict(state, observation, action)
                action = np.random.choice(np.arange(len(probabilities)), p=probabilities)

                next_observation, reward, done, _ = env.step(action)

                observations.append(observation)
                actions.append(action)
                rewards.append(reward)

                observation = next_observation

            # Accumulate rewards
            for i in reversed(range(len(rewards) - 1)):
                rewards[i] += args.gamma * rewards[i + 1]

            # Update baseline with exponential averaging
            for i, reward in enumerate(rewards):
                if i >= len(baseline): baseline = np.pad(baseline, (0, 1), "constant")
                baseline[i] = args.baseline_decay * baseline[i] + (1 - args.baseline_decay) * reward

            # Append episode to a batch
            batch_observations.append(observations)
            batch_actions.append(actions)
            batch_returns.append(rewards)

            # Train if enough data has been generated
            if len(batch_observations) % args.batch_size == 0 or episode + 1 == args.evaluate_each:
                network.train(list(map(len, batch_observations)), [network.zero_state()] * len(batch_observations),
                              np.array(list(itertools.zip_longest(*batch_observations, fillvalue=0))).T,
                              np.array(list(itertools.zip_longest(*batch_actions, fillvalue=0))).T,
                              np.array(list(itertools.zip_longest(*batch_returns, fillvalue=0))).T - baseline[:max(map(len, batch_observations))])
                batch_observations, batch_actions, batch_returns = [], [], []

        # Periodic evaluation
        returns = []
        for _ in range(args.evaluate_for):
            returns.append(evaluate_episode())
        print("Evaluation of {} episodes: {}".format(args.evaluate_for, np.mean(returns)), file=sys.stderr, flush=True)
