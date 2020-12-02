#!/usr/bin/env python3
import argparse

import gym
import numpy as np

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate alpha.")
parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration epsilon factor.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor gamma.")
parser.add_argument("--lambda", default=None, type=float, help="Trace factor lambda, if any.")
parser.add_argument("--n", default=1, type=int, help="Use n-step method.")
parser.add_argument("--off_policy", default=False, action="store_true", help="Off-policy (less exploratory target)")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--vtrace_clip", default=None, type=float, help="V-Trace clip rho and c, if any.")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("Taxi-v3"), seed=args.seed, report_each=100)

    # Extract a deterministic MDP into three NumPy arrays
    # - R[state][action] is the reward
    # - D[state][action] is the True/False value indicating end of episode
    # - N[state][action] is the next state
    N, R, D = np.split(
        np.array([[env.P[s][a][0][1:] for a in range(env.action_space.n)] for s in range(env.observation_space.n)]),
        3, axis=-1)
    N, R, D = N.squeeze(-1), R.squeeze(-1), D.squeeze(-1)

    # Create a random seed generator
    generator = np.random.RandomState(args.seed)

    # As `lambda` is a Python keyword we cannot access the atributte directly with args.lambda.
    lambda_ = getattr(args, 'lambda')
    V = np.zeros(env.observation_space.n)

    for _ in range(args.episodes):
        state, done = env.reset(), False

        # Generate episode and update V using the given TD method
        while not done:
            best_action = np.argmax(R[state] + (1 - D[state]) * args.gamma * V[N[state]])
            action = best_action if generator.uniform() >= args.epsilon else env.action_space.sample()
            action_prob = args.epsilon / env.action_space.n + (1 - args.epsilon) * (action == best_action)

            next_state, reward, done, _ = env.step(action)

            if args.off_policy:
                target_policy = np.eye(env.action_space.n)[np.argmax(R + (1 - D) * args.gamma * V[N], axis=-1)]
                target_epsilon = args.epsilon / 3 if args.off_policy else args.epsilon
                target_policy = (1 - target_epsilon) * target_policy + target_epsilon / env.action_space.n * np.ones_like(target_policy)

            # TODO: Perform the update to the state value function `V`, using
            # a TD update with the following parameters:
            # - `args.n`: use `args.n`-step method
            # - `args.off_policy`:
            #    - if False, the epsilon-greedy behaviour policy is also the target policy
            #    - if True, the target policy is an epsilon/3-greedy policy
            # - if `args.lambda` is not None, use eligibility traces
            # - if `args.vtrace_clip` is not None, clip the importance sample ratios with it
            #
            # Perform the updates as soon as you can -- whenever you have all the information
            # to update `V[state]`, do it. For each `action` use its corresponding
            # `action_prob` at the time of taking the `action` as the behaviour policy action
            # probability, and current `target_policy` as the target policy (everywhere
            # in the update).
            #
            # Do not forget that when `done` is True, bootstrapping on the
            # `next_state` is not used.
            #
            # Also note that when the episode ends and `args.n` > 1, there will
            # be several states that also need to be updated. Perform the updates
            # in the order in which you encountered the states in the trajectory
            # and during these updates, use the `target_policy` computed above
            # (do not modify it during these post-episode updates).

            state = next_state

    return V

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
