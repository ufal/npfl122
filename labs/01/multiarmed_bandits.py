#!/usr/bin/env python3
import numpy as np

class MultiArmedBandits():
    def __init__(self, bandits, episode_length):
        self._bandits = []
        for _ in range(bandits):
            self._bandits.append(np.random.normal(0., 1.))
        self._done = True
        self._episode_length = episode_length
        #print("Initialized {}-armed bandit, maximum average reward is {}".format(bandits, np.max(self._bandits)))

    def reset(self):
        self._done = False
        self._trials = 0
        return None

    def step(self, action):
        if self._done:
            raise ValueError("Cannot step in MultiArmedBandits when there is no running episode")
        self._trials += 1
        self._done = self._trials == self._episode_length
        reward = np.random.normal(self._bandits[action], 1.)
        return None, reward, self._done, {}


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandits", default=10, type=int, help="Number of bandits.")
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    parser.add_argument("--episode_length", default=1000, type=int, help="Number of trials per episode.")

    parser.add_argument("--mode", default="greedy", type=str, help="Mode to use -- greedy, ucb and gradient.")
    parser.add_argument("--alpha", default=0, type=float, help="Learning rate to use (if applicable).")
    parser.add_argument("--c", default=1., type=float, help="Confidence level in ucb.")
    parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor (if applicable).")
    parser.add_argument("--initial", default=0, type=float, help="Initial value function levels.")
    args = parser.parse_args()

    env = MultiArmedBandits(args.bandits, args.episode_length)

    average_rewards = []
    for episode in range(args.episodes):
        env.reset()

        # TODO: Initialize required values (depending on mode).

        average_rewards.append(0)
        done = False
        while not done:
            # TODO: Action selection according to mode
            if args.mode == "greedy":
                action =
            elif args.mode == "ucb":
                action =
            elif args.mode == "gradient":
                action =

            _, reward, done, _ = env.step(action)
            average_rewards[-1] += reward / args.episode_length

            # TODO: Update parameters

    # Print out final score as mean and variance of all obtained rewards.
    print("Final score: {}, variance: {}".format(np.mean(average_rewards), np.var(average_rewards)))
