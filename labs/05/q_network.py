#!/usr/bin/env python3
import collections

import numpy as np
import tensorflow as tf

import cart_pole_evaluator

class Network:
    def __init__(self, env, args):
        # TODO: Create a suitable network

        # Warning: If you plan to use Keras `.train_on_batch` and/or `.predict_on_batch`
        # methods, pass `experimental_run_tf_function=False` to compile. There is
        # a bug in TF 2.0 which causes the `*_on_batch` methods not to use `tf.function`.

        # Otherwise, if you are training manually, using `tf.function` is a good idea
        # to get good performance.

    # Define a training method. Generally you have two possibilities
    # - pass new q_values of all actions for a given state; all but one are the same as before
    # - pass only one new q_value for a given state, including the index of the action to which
    #   the new q_value belongs
    def train(self, states, ...):
        # TODO
        pass

    def predict(self, states):
        # TODO
        pass


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--episodes", default=1000, type=int, help="Episodes for epsilon decay.")
    parser.add_argument("--epsilon", default=0.3, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layers", default=1, type=int, help="Number of hidden layers.")
    parser.add_argument("--hidden_layer_size", default=20, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False)

    # Construct the network
    network = Network(env, args)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque()
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    evaluating = False
    epsilon = args.epsilon
    while training:
        # Perform episode
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # TODO: compute action using epsilon-greedy policy. You can compute
            # the q_values of a given state using
            #   q_values = network.predict(np.array([state], np.float32))[0]

            next_state, reward, done, _ = env.step(action)

            # Append state, action, reward, done and next_state to replay_buffer
            replay_buffer.append(Transition(state, action, reward, done, next_state))

            # TODO: If the replay_buffer is large enough, preform a training batch
            # of `args.batch_size` uniformly randomly chosen transitions.
            #
            # After you choose `states` and suitable targets, you can train the network as
            #   network.train(states, ...)

            state = next_state

        if args.epsilon_final:
            epsilon = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            action = np.argmax(network.predict(np.array([state], np.float32))[0])
            state, reward, done, _ = env.step(action)
