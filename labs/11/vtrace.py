#!/usr/bin/env python3
import collections
import sys

import numpy as np
import tensorflow as tf

import gym_evaluator

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, state_shape, num_actions):
        with self.session.graph.as_default():
            self.steps = tf.placeholder(tf.int32, [None])
            self.states = tf.placeholder(tf.float32, [None, args.n + 1] + state_shape)
            self.actions = tf.placeholder(tf.int32, [None, args.n])
            self.action_probabilities = tf.placeholder(tf.float32, [None, args.n])
            self.rewards = tf.placeholder(tf.float32, [None, args.n])

            # Compute the action logits
            hidden_layer = tf.layers.dense(self.states[:, :args.n], args.hidden_layer, activation=tf.nn.relu)
            logits = tf.layers.dense(hidden_layer, num_actions)
            self.probabilities = tf.nn.softmax(logits)

            hidden_layer = tf.layers.dense(self.states, args.hidden_layer, activation=tf.nn.relu)
            values = tf.layers.dense(hidden_layer, 1)[:, :, 0]
            values *= tf.to_float(tf.sequence_mask(self.steps, maxlen=args.n + 1))

            # Training
            loss = self.vtrace(args, self.actions, self.action_probabilities, self.rewards, logits, values)
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step, name="training")

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    def vtrace(self, args, actions, actions_probabilities, rewards, actor_logits, critic_values):
        """Compute loss for V-trace algorithm.

        Arguments:
          args:
              command line arguments
          actions: [batch_size, n]
              chosen actions
          action_probabilities: [batch_size, n]
              probability of the chosen actions under behaviour policy;
              guaranteed to be 1 for actions after episode termination
          rewards: [batch_size, n]
              observed rewards;
              guaranteed to be 0 for rewards after episode termination
          actor_logits: [batch_size, n, num_actions]
              logits of actions under current (target) policy
          critic_values: [batch_size, n+1]
              critic estimation of values of encountered states;
              guaranteed to be 0 for states after episode termination
        """

        # TODO: Compute target policy probability of given actions
        # into `actor_action_probabilities`, i.e., symbolically
        #   actor_action_probabilities = softmax(actor_logits)[:, :, actions[:, :]]

        rhos, cs = [], []
        # TODO: Compute clipped rho-s and c-s, as a Python list with args.n
        # elements, each being a tf.Tensor (values for a whole batch).
        # The value rhos[i] and cs[i] should be importance sampling ratio for actions[:, i]
        # clipped by args.clip_rho and args.clip_c, respectively.

        vs = [None] * (args.n + 1)
        # TODO: Compute vs from the last one to the first one.
        # The last vs[args.n] is just critic_values[:, args.n]
        # The others can be computed recursively as
        #   vs[t] = critic_values[:, t] + delta_t V + gamma * cs[t] * (vs[t+1] - critic_values[:, t+1])

        # TODO: Define and return loss, which consists of
        # - usuall actor loss, with weights being tf.stop_gradient of
        #    - the importance sampling factor (rhos[0])
        #    - estimated q_value (computed as rewards + gamma * vs[1])
        #      minus the baseline of critic_values
        #  - negative mean of args.entropy_regularization times
        #      the entropy of actor logits (actor_logits[:, 0])
        #  - mean square error of tf.stop_gradient(vs[0]) and
        #      critic values
        return loss

    def predict_probabilities(self, state):
        return self.session.run(self.probabilities, {self.states: np.tile(state, (1, self.states.shape[1].value, 1))})[0, 0]

    def train(self, steps, states, actions, action_probabilities, rewards):
        self.session.run(self.training, {self.steps: steps, self.states: states, self.actions: actions,
                                         self.action_probabilities: action_probabilities, self.rewards: rewards})

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Number of transitions to train on.")
    parser.add_argument("--clip_c", default=1., type=float, help="Clip value for c in V-trace.")
    parser.add_argument("--clip_rho", default=1., type=float, help="Clip value for rho in V-trace.")
    parser.add_argument("--entropy_regularization", default=0.1, type=float, help="Entropy regularization weight.")
    parser.add_argument("--env", default="CartPole-v1", type=str, help="Environment.")
    parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of episodes.")
    parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate for number of episodes.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=30, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--n", default=4, type=int, help="Number of steps to use in V-trace.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Perform ReCodEx evaluation.")
    parser.add_argument("--replay_buffer_maxlen", default=None, type=int, help="Replay buffer maxlen.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--target", default=475., type=float, help="Target return.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create the environment
    env = gym_evaluator.GymEnvironment(args.env)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, env.state_shape, env.actions)

    # ReCodEx evaluation
    if args.recodex:
        import vtrace_recodex
        vtrace_recodex.evaluate(network)

    # Replay memory
    replay_buffer = collections.deque(maxlen=args.replay_buffer_maxlen)
    Transition = collections.namedtuple("Transition", ["state", "action", "action_probability", "reward", "done"])

    def evaluate_episode(evaluating=False):
        rewards = 0
        state, done = env.reset(evaluating), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            probabilities = network.predict_probabilities(state)
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)
            rewards += reward
        return rewards

    while True:
        # Training
        for _ in range(args.evaluate_each):
            state, done = env.reset(), False
            while not done:
                probabilities = network.predict_probabilities(state)
                action = np.random.choice(np.arange(len(probabilities)), p=probabilities)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.append(Transition(state, action, probabilities[action], reward, done))
                state = next_state

                if len(replay_buffer) > args.n * args.batch_size:
                    steps = np.zeros((args.batch_size), dtype=np.int32)
                    states = np.zeros([args.batch_size, args.n + 1] + env.state_shape, dtype=np.float32)
                    actions = np.zeros((args.batch_size, args.n), dtype=np.int32)
                    action_probabilities = np.ones((args.batch_size, args.n), dtype=np.float32)
                    rewards = np.zeros((args.batch_size, args.n), dtype=np.float32)

                    batch = np.random.choice(len(replay_buffer) - args.n, size=args.batch_size, replace=False)
                    for i in range(args.batch_size):
                        for j in range(args.n):
                            item = replay_buffer[batch[i] + j]
                            states[i, j] = item.state
                            actions[i, j] = item.action
                            action_probabilities[i, j] = item.action_probability
                            rewards[i, j] = item.reward
                            if item.done:
                                steps[i] = j + 1
                                break
                        else:
                            steps[i] = args.n + 1
                            states[i, args.n] = replay_buffer[batch[i] + args.n].state

                    network.train(steps, states, actions, action_probabilities, rewards)

        # Periodic evaluation
        returns = []
        for _ in range(args.evaluate_for):
            returns.append(evaluate_episode())
        print("Evaluation of {} episodes: {}".format(args.evaluate_for, np.mean(returns)), file=sys.stderr)

        if np.mean(returns) >= args.target:
            print("Reached mean average return of {}, running final evaluation.".format(args.target))
            while True:
                evaluate_episode(True)
