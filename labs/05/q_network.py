#!/usr/bin/env python3
import collections

import numpy as np
import tensorflow as tf

import cart_pole_evaluator

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
            self.q_values = tf.placeholder(tf.float32, [None, num_actions])

            # Compute the q_values
            hidden = self.states
            for _ in range(args.hidden_layers):
                hidden = tf.layers.dense(hidden, args.hidden_layer_size, activation=tf.nn.relu)
            self.predicted_values = tf.layers.dense(hidden, num_actions)

            # Training
            loss = tf.losses.mean_squared_error(self.q_values, self.predicted_values)
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step, name="training")

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    def predict(self, states):
        return self.session.run(self.predicted_values, {self.states: states})

    def train(self, states, q_values):
        self.session.run(self.training, {self.states: states, self.q_values: q_values})

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--episodes", default=1000, type=int, help="Episodes for epsilon decay.")
    parser.add_argument("--epsilon", default=0.3, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layers", default=2, type=int, help="Number of hidden layers.")
    parser.add_argument("--hidden_layer_size", default=20, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, env.state_shape, env.actions)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque()
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    evaluating = False
    epsilon = args.epsilon
    while True:
        # Perform episode
        state, done = env.reset(evaluating), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # TODO: compute action using epsilon-greedy policy. You can compute
            # the q_values of a given state using
            #   q_values = network.predict([state])[0]

            next_state, reward, done, _ = env.step(action)

            # Append state, action, reward, done and next_state to replay_buffer
            replay_buffer.append(Transition(state, action, reward, done, next_state))

            # TODO: If the replay_buffer is large enough, preform a training batch
            # of size `args.batch_size` (you can train on every `args.batch_size`-th step,
            # but in this simple task also on each step).
            #
            # After you choose `states` and their target `q_values`, you train the network as
            #   network.train(states, q_values)

            state = next_state

        # TODO: Decide if we want to start evaluating

        if not evaluating:
            if args.epsilon_final:
                epsilon = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
