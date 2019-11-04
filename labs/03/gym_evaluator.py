#!/usr/bin/env python3
import math
import sys

import gym
import numpy as np

class GymEnvironment:
    def __init__(self, env, separators=None, tiles=None):
        self._env = gym.make(env)
        self._env.seed(42)

        self._separators = separators
        self._tiles = tiles
        if self._separators is not None:
            self._first_tile_states, self._rest_tiles_states = 1, 1
            for separator in separators:
                self._first_tile_states *= 1 + len(separator)
                self._rest_tiles_states *= 2 + len(separator)
            if tiles:
                self._separator_offsets, self._separator_tops = [], []
                for separator in separators:
                    self._separator_offsets.append(0 if len(separator) <= 1 else (separator[1] - separator[0]) / tiles)
                    self._separator_tops.append(math.inf if len(separator) <= 1 else separator[-1] + (separator[1] - separator[0]))

        self._evaluating_from = None
        self._episode_return = 0
        self._episode_returns = []
        self._episode_ended = True

    def _maybe_discretize(self, observation):
        if self._separators is not None:
            state = 0
            for i in range(len(self._separators)):
                state *= 1 + len(self._separators[i])
                state += np.digitize(observation[i], self._separators[i])
            if self._tiles:
                states = [state]
                for t in range(1, self._tiles):
                    state = 0
                    for i in range(len(self._separators)):
                        state *= 2 + len(self._separators[i])
                        value = observation[i] + ((t * (2 * i + 1)) % self._tiles) * self._separator_offsets[i]
                        if value > self._separator_tops[i]:
                            state += 1 + len(self._separators[i])
                        else:
                            state += np.digitize(value, self._separators[i])
                    states.append(self._first_tile_states + (t - 1) * self._rest_tiles_states + state)
                observation = states
            else:
                observation = state

        return observation

    @property
    def states(self):
        if hasattr(self._env.observation_space, "n"):
            return self._env.observation_space.n
        else:
            if self._separators is not None:
                states = self._first_tile_states
                if self._tiles:
                    states += (self._tiles - 1) * self._rest_tiles_states
                return states
            raise RuntimeError("Continuous environments have infinitely many states")

    @property
    def weights(self):
        if self._separators is not None and self._tiles:
            return self._first_tile_states + (self._tiles - 1) * self._rest_tiles_states
        raise RuntimeError("Only environments with tile encoding have weights")

    @property
    def state_shape(self):
        if self._separators is not None:
            return [] if not self._tiles else [self._tiles]
        else:
            return list(self._env.observation_space.shape)

    @property
    def actions(self):
        if hasattr(self._env.action_space, "n"):
            return self._env.action_space.n
        else:
            raise RuntimeError("The environment has continuous action space, cannot return number of actions")

    @property
    def action_shape(self):
        if hasattr(self._env.action_space, "shape"):
            return list(self._env.action_space.shape)
        else:
            return []

    @property
    def action_ranges(self):
        if not hasattr(self._env.action_space, "shape"):
            raise RuntimeError("The environment does not have continuous actions, cannot return action ranges")
        if hasattr(self._env.action_space, "low") and hasattr(self._env.action_space, "high"):
            return list(self._env.action_space.low), list(self._env.action_space.high)
        else:
            raise RuntimeError("The environment has no action ranges defined")

    @property
    def episode(self):
        return len(self._episode_returns)

    def reset(self, start_evaluate=False):
        if start_evaluate and self._evaluating_from is None:
            self._evaluating_from = self.episode

        self._episode_ended = False
        return self._maybe_discretize(self._env.reset())

    def step(self, action):
        if self._episode_ended:
            raise RuntimeError("Cannot run `step` on environments without an active episode, run `reset` first")

        observation, reward, done, info = self._env.step(action)

        self._episode_return += reward
        if done:
            self._episode_ended = True
            self._episode_returns.append(self._episode_return)

            if self.episode % 10 == 0:
                print("Episode {}, mean 100-episode return {}".format(
                    self.episode, np.mean(self._episode_returns[-100:])), file=sys.stderr)
            if self._evaluating_from is not None and self.episode >= self._evaluating_from + 100:
                print("The mean 100-episode return after evaluation {}".format(np.mean(self._episode_returns[-100:])))
                sys.exit(0)

            self._episode_return = 0

        return self._maybe_discretize(observation), reward, done, info

    def render(self):
        self._env.render()
