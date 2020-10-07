#!/usr/bin/env python3
import sys

import gym
import numpy as np

class EvaluationWrapper(gym.Wrapper):
    def __init__(self, env, seed=None, evaluate_for=100):
        super().__init__(env)
        self._evaluate_for = evaluate_for

        self.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

        self._episode_running = False
        self._episode_returns = []
        self._evaluating_from = None

    @property
    def episode(self):
        return len(self._episode_returns)

    def reset(self, start_evaluation=False):
        if self._evaluating_from is not None and self._episode_running:
            raise RuntimeError("Cannot reset a running episode after `start_evaluation=True`")

        if start_evaluation and self._evaluating_from is None:
            self._evaluating_from = self.episode

        self._episode_running = True
        self._episode_return = 0
        return super().reset()

    def step(self, action):
        if not self._episode_running:
            raise RuntimeError("Cannot run `step` on environments without an active episode, run `reset` first")

        observation, reward, done, info = super().step(action)

        self._episode_return += reward
        if done:
            self._episode_running = False
            self._episode_returns.append(self._episode_return)

            if self.episode % 10 == 0:
                print("Episode {}, mean {}-episode return {:.2f} +-{:.2f}".format(
                    self.episode, self._evaluate_for, np.mean(self._episode_returns[-self._evaluate_for:]),
                    np.std(self._episode_returns[-self._evaluate_for:])), file=sys.stderr)
            if self._evaluating_from is not None and self.episode >= self._evaluating_from + self._evaluate_for:
                print("The mean {}-episode return after evaluation {:.2f} +-{:.2f}".format(
                    self._evaluate_for, np.mean(self._episode_returns[-self._evaluate_for:]),
                    np.std(self._episode_returns[-self._evaluate_for:]), file=sys.stderr))
                self.close()
                sys.exit(0)

        return observation, reward, done, info

class DiscretizationWrapper(gym.ObservationWrapper):
    def __init__(self, env, separators):
        super().__init__(env)
        self._separators = separators

        states = 1
        for separator in separators:
            states *= 1 + len(separator)
        self.observation_space = gym.spaces.Discrete(states)

    def observation(self, observations):
        state = 0
        for observation, separator in zip(observations, self._separators):
            state *= 1 + len(separator)
            state += np.digitize(observation, separator)
        return state

class DiscreteCartPoleWrapper(DiscretizationWrapper):
    def __init__(self, env, bins=8):
        super().__init__(env, [
            np.linspace(-2.4, 2.4, num=bins + 1)[1:-1], # cart position
            np.linspace(-3, 3, num=bins + 1)[1:-1],     # pole angle
            np.linspace(-0.5, 0.5, num=bins + 1)[1:-1], # cart velocity
            np.linspace(-2, 2, num=bins + 1)[1:-1],     # pole angle velocity
        ])
