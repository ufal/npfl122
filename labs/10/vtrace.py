#!/usr/bin/env python3
import collections

import numpy as np
import tensorflow as tf

import gym_evaluator

class Network:
    def __init__(self, env, args):
        # Store the arguments regularization
        self.args = args

        # TODO: Create the actor. The input should be a batch of _sequences_ of
        # states (so the input shape is `[None] + env.state_shape`), each state
        # processed independently by the same network with a dense layer of
        # args.hidden_layer units with ReLU activation, followed by an softmax
        # layer with `env.actions` units.
        #
        # We use sequences of states on the input, because we want to predict
        # probabilities of up to `args.n` following states.
        #
        # We train the actor using sparse categorical crossentropy loss
        # and Adam optimizer with args.learning_rate.

        # TODO: Create the critic. The input should be again a batch of _sequences_
        # of states, each processed independently by a network with a dense layer of
        # args.hidden_layer units with ReLU activation, followed by a dense layer
        # with 1 output and no activation.
        #
        # We train the critic using MSE loss and Adam optimizer with args.learning_rate.

    # Do not change the method signature, as this method is used for testing in ReCodEx.
    @staticmethod
    def vtrace(args, actions, action_probabilities, rewards, actor_probabilities, critic_values):
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
          actor_probabilities: [batch_size, n, num_actions]
              probabilities of actions under current (target) policy
          critic_values: [batch_size, n+1]
              critic estimation of values of encountered states;
              guaranteed to be 0 for states after episode termination
        """

        # TODO: Compute target policy probability of given actions
        # into `actor_action_probabilities`, i.e., symbolically
        #   actor_action_probabilities = actor_probabilities[:, :, actions[:, :]]

        rhos, cs = [], []
        # TODO: Compute clipped rho-s and c-s, as a Python list with
        # args.n elements, each a tensor (values for a whole batch).
        # The value rhos[i] and cs[i] should be importance sampling
        # ratio for actions[:, i] clipped by `args.clip_rho` and
        # `args.clip_c`, respectively.

        vs = [None] * (args.n + 1)
        # TODO: Compute vs from the last one to the first one.
        # The `vs[args.n]` is just `critic_values[:, args.n]`
        # The others can be computed recursively as
        #   vs[t] = critic_values[:, t] + delta_t V + gamma * cs[t] * (vs[t+1] - critic_values[:, t+1])

        # TODO: Return a pair with following elements:
        # - coefficient for actor loss, i.e., a product of the importance
        #   sampling factor (rhos[0]) and the estimated q_value
        #   (rewards + gamma * vs[1]) minus the baseline of critic_values
        # - target for the critic, i.e., vs[0]

    @tf.function
    def train(self, steps, states, actions, action_probabilities, rewards):
        # TODO: Run the actor on first `args.n` states and the critic on `args.n+1` states

        # TODO: Only first `steps` of `states` are valid (so `steps` might be `args.n+1`
        # if all `states` are non-terminal), so the critic predictions for the
        # states after the `steps` ones must be set to zero.

        # TODO: Run the `vtrace` method, with the last two arguments being the actor
        # and critic predictions, obtaining `actor_weights` and `critic_targets`.

        # TODO: Train the actor, using the first state of every batch instance, with
        # - sparse categorical crossentropy loss, weighted by `actor_weights`
        # - plus entropy regularization with weights self.args.entropy_regularization.
        #   Entropy of a given categorical distribution `d` is
        #     tf.reduce_sum(-d * tf.math.log(d), axis=-1)

        # TODO: Train the critic using the first state of every batch instance,
        # utilizing MSE loss with `critic_targets` as gold values.

    @tf.function
    def _predict_actions(self, states):
        return self._actor(states)

    def predict_actions(self, states):
        states = np.expand_dims(np.array(states, np.float32), axis=1)
        return self._predict_actions(states).numpy()[:, 0]


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=None, type=int, help="Number of transitions to train on.")
    parser.add_argument("--clip_c", default=1., type=float, help="Clip value for c in V-trace.")
    parser.add_argument("--clip_rho", default=1., type=float, help="Clip value for rho in V-trace.")
    parser.add_argument("--entropy_regularization", default=0.1, type=float, help="Entropy regularization weight.")
    parser.add_argument("--env", default="CartPole-v1", type=str, help="Environment.")
    parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of episodes.")
    parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate for number of episodes.")
    parser.add_argument("--gamma", default=None, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=None, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=None, type=float, help="Learning rate.")
    parser.add_argument("--n", default=None, type=int, help="Number of steps to use in V-trace.")
    parser.add_argument("--replay_buffer_maxlen", default=None, type=int, help="Replay buffer maxlen.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--target_return", default=495, type=float, help="Target return.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = gym_evaluator.GymEnvironment(args.env)

    # Construct the network
    network = Network(env, args)

    # Replay memory
    replay_buffer = collections.deque(maxlen=args.replay_buffer_maxlen)
    Transition = collections.namedtuple("Transition", ["state", "action", "action_probability", "reward", "done"])

    def evaluate_episode(evaluating=False):
        rewards = 0
        state, done = env.reset(evaluating), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            probabilities = network.predict_actions([state])[0]
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)
            rewards += reward
        return rewards

    while True:
        # Training
        for _ in range(args.evaluate_each):
            state, done = env.reset(), False
            while not done:
                probabilities = network.predict_actions([state])[0]
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

                    # TODO: Prepare a batch.
                    #
                    # Each batch instance is a sequence of `args.n+1` consecutive `states` and
                    # `args.n` consecutive `actions`, `action_probabilities` and `rewards`.
                    # The `steps` indicate how many `states` in range [1,2,...,args.n+1] are valid.
                    #
                    # To generate a batch, sample `args.batch_size` indices from replay_buffer
                    # (ignoring the last `args.n` ones to avoid overflow). Then fill for every
                    # sampled index the consecutive states, actions, action_probabilities and
                    # rewards -- if `done` is not set, all of them are filled and `steps` is
                    # set to `args.n+1`. If `done` is set, only a subset of states, actions,
                    # action_probabilities and rewards are set, and `steps` is set to the
                    # number of valid states (<`args.n+1`).
                    network.train(steps, states, actions, action_probabilities, rewards)

        # Periodic evaluation
        returns = []
        for _ in range(args.evaluate_for):
            returns.append(evaluate_episode())
        print("Evaluation of {} episodes: {}".format(args.evaluate_for, np.mean(returns)))

        if np.mean(returns) >= args.target_return:
            print("Reached mean average return of {}, running final evaluation.".format(np.mean(returns)))
            while True:
                evaluate_episode(True)
