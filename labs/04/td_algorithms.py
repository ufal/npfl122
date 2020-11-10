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
parser.add_argument("--mode", default="sarsa", type=str, help="Mode (sarsa/expected_sarsa/tree_backup).")
parser.add_argument("--n", default=1, type=int, help="Use n-step method.")
parser.add_argument("--off_policy", default=False, action="store_true", help="Off-policy; use greedy as target")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("Taxi-v3"), seed=args.seed, report_each=100)

    # Fix random seed and create a generator
    generator = np.random.RandomState(args.seed)

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    for _ in range(args.episodes):
        next_state, done = env.reset(), False

        # Generate episode and update Q using the given TD method
        next_action = np.argmax(Q[next_state]) if generator.uniform() >= args.epsilon else env.action_space.sample()
        next_action_prob = args.epsilon / env.action_space.n + (1 - args.epsilon) * (next_action == np.argmax(Q[next_state]))
        while not done:
            action, action_prob, state = next_action, next_action_prob, next_state
            next_state, reward, done, _ = env.step(action)
            if not done:
                next_action = np.argmax(Q[next_state]) if generator.uniform() >= args.epsilon else env.action_space.sample()
                next_action_prob = args.epsilon / env.action_space.n + (1 - args.epsilon) * (next_action == np.argmax(Q[next_state]))

            target_policy = np.eye(env.action_space.n)[np.argmax(Q, axis=1)]
            if not args.off_policy:
                target_policy = (1 - args.epsilon) * target_policy + args.epsilon / env.action_space.n * np.ones_like(target_policy)

            # TODO: Perform the update to the state-action value function `Q`, using
            # a TD update with the following parameters:
            # - `args.n`: use `args.n`-step method
            # - `args.off_policy`:
            #    - if False, the epsilon-greedy behaviour policy is also the target policy
            #    - if True, the target policy is the greedy policy
            #      - for SARSA (with any `args.n`) and expected SARSA (with `args.n` > 1),
            #        importance sampling must be used
            # - `args.mode`: this argument can have the following values:
            #   - "sarsa": regular SARSA algorithm
            #   - "expected_sarsa": expected SARSA algorithm
            #   - "tree_backup": tree backup algorithm
            #
            # Perform the updates as soon as you can -- whenever you have all the information
            # to update `Q[state, action]`, do it. For each `action` use its corresponding
            # `action_prob` at the time of taking the `action` as the behaviour policy action
            # probability, and current `target_policy` as the target policy (everywhere
            # in the update).
            #
            # Do not forget that when `done` is True, bootstrapping on the
            # `next_state` is not used.
            #
            # Also note that when the episode ends and `args.n` > 1, there will
            # be several state-action pairs that also need to be updated. Perform
            # the updates in the order in which you encountered the state-action
            # pairs and during these updates, use the `target_policy` computed
            # above (do not modify it during these post-episode updates).

    return Q

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
