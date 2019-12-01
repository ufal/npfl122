#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import continuous_mountain_car_evaluator

# This class is a bare version of tfp.distributions.Normal
class Normal:
    def __init__(self, loc, scale):
        self.loc = tf.convert_to_tensor(loc, dtype=tf.float32)
        self.scale = tf.convert_to_tensor(scale, dtype=tf.float32)

    def log_prob(self, x):
        log_unnormalized = -0.5 * tf.math.squared_difference(x / self.scale, self.loc / self.scale)
        log_normalization = 0.5 * np.log(2. * np.pi) + tf.math.log(self.scale)
        return log_unnormalized - log_normalization

    def entropy(self):
        log_normalization = 0.5 * np.log(2. * np.pi) + tf.math.log(self.scale)
        entropy = 0.5 + log_normalization
        return entropy * tf.ones_like(self.loc)

    def sample_n(self, n, seed=None):
        shape = tf.concat([[n], tf.broadcast_dynamic_shape(tf.shape(self.loc), tf.shape(self.scale))], axis=0)
        sampled = tf.random.normal(shape=shape, mean=0., stddev=1., dtype=tf.float32, seed=seed)
        return sampled * self.scale + self.loc

class Network:
    def __init__(self, env, args):
        assert len(env.action_shape) == 1
        action_components = env.action_shape[0]

        self.entropy_regularization = args.entropy_regularization

        # TODO: Create `_model`, which: processes `states`. Because `states` are
        # vectors of tile indices, you need to convert them to one-hot-like
        # encoding. I.e., for batch example i, state should be a vector of
        # length `weights` with `tiles` ones on indices `states[i,
        # 0..`tiles`-1] and the rest being zeros.
        #
        # The model computes `mus` and `sds`, each of shape [batch_size, action_components].
        # Compute each independently using `states` as input, adding a fully connected
        # layer with args.hidden_layer units and ReLU activation. Then:
        # - For `mus` add a fully connected layer with `actions` outputs.
        #   To avoid `mus` moving from the required [-1,1] range, you can apply
        #   `tf.tanh` activation.
        # - For `sds` add a fully connected layer with `actions` outputs
        #   and `tf.nn.softplus` activation.
        # The model also computes `values`, starting with `states` and
        # - add a fully connected layer of size args.hidden_layer and ReLU activation
        # - add a fully connected layer with 1 output and no activation

        self._model = tf.keras.Model(inputs=inputs, outputs=[mus, sds, values])
        self._optimizer = tf.optimizers.Adam(args.learning_rate)

    @tf.function
    def _train(self, states, actions, returns):
        # TODO: Run the model on given states and compute
        # `sds`, `mus` and `values`. Then create `action_distribution` using
        # `Normal` distribution class and computed `mus` and `sds`.
        #
        # TODO: Compute `loss` as a sum of three losses:
        # - negative log probability of the `actions` in the `action_distribution`
        #   (using `log_prob` method). You need to sum the log probabilities
        #   of subactions for a single batch example (using `tf.reduce_sum` with `axis=1`).
        #   Then weight the resulting vector by `(returns - tf.stop_gradient(values))`
        #   and compute its mean.
        # - negative value of the distribution entropy (use `entropy` method of
        #   the `action_distribution`) weighted by `args.entropy_regularization`.
        # - mean square error of the `returns` and `values`

    def train(self, states, actions, returns):
        states, actions, returns = np.array(states, np.int32), np.array(actions, np.float32), np.array(returns, np.float32)
        self._train(states, actions, returns)

    @tf.function
    def _predict(self, states):
        return self._model(states, training=False)

    def predict_actions(self, states):
        states = np.array(states, np.int32)
        mus, sds, _ = self._predict(states)
        return mus.numpy(), sds.numpy()

    def predict_values(self, states):
        states = np.array(states, np.int32)
        _, _, values = self._predict(states)
        return values.numpy()[:, 0]

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--entropy_regularization", default=0.1, type=float, help="Entropy regularization weight.")
    parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of batches.")
    parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate for number of batches.")
    parser.add_argument("--gamma", default=None, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=None, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=None, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--tiles", default=8, type=int, help="Tiles to use.")
    parser.add_argument("--workers", default=1, type=int, help="Number of parallel workers.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = continuous_mountain_car_evaluator.environment(tiles=args.tiles)
    action_lows, action_highs = env.action_ranges

    # Construct the network
    network = Network(env, args)

    # Initialize parallel workers by env.parallel_init
    states = env.parallel_init(args.workers)
    while True:
        # Training
        for _ in range(args.evaluate_each):
            # TODO: Choose actions using network.predict_actions.
            # using np.random.normal to sample action and np.clip
            # to clip it using action_lows and action_highs,

            # TODO: Perform steps by env.parallel_step

            # TODO: Compute return estimates by
            # - extracting next_states from steps
            # - computing value function approximation in next_states
            # - estimating returns by reward + (0 if done else args.gamma * next_state_value)

            # TODO: Train network using current states, chosen actions and estimated returns

        # Periodic evaluation
        returns = []
        for _ in range(args.evaluate_for):
            returns.append(0)
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                action = network.predict_actions([state])[0][0]
                state, reward, done, _ = env.step(action)
                returns[-1] += reward
        print("Evaluation of {} episodes: {}".format(args.evaluate_for, np.mean(returns)))

    # On the end perform final evaluations with `env.reset(True)`
