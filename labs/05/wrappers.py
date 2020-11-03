#!/usr/bin/env python3
import sys

import gym
import numpy as np

class EvaluationWrapper(gym.Wrapper):
    def __init__(self, env, seed=None, evaluate_for=100, report_each=10):
        super().__init__(env)
        self._evaluate_for = evaluate_for
        self._report_each = report_each

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

            if self.episode % self._report_each == 0:
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
    def __init__(self, env, separators, tiles=None):
        super().__init__(env)
        self._separators = separators
        self._tiles = tiles

        if tiles is None:
            states = 1
            for separator in separators:
                states *= 1 + len(separator)
            self.observation_space = gym.spaces.Discrete(states)
        else:
            self._first_tile_states, self._rest_tiles_states = 1, 1
            for separator in separators:
                self._first_tile_states *= 1 + len(separator)
                self._rest_tiles_states *= 2 + len(separator)
            self.observation_space = gym.spaces.MultiDiscrete([
                self._first_tile_states + i * self._rest_tiles_states for i in range(tiles)])

            self._separator_offsets, self._separator_tops = [], []
            for separator in separators:
                self._separator_offsets.append(0 if len(separator) <= 1 else (separator[1] - separator[0]) / tiles)
                self._separator_tops.append(math.inf if len(separator) <= 1 else separator[-1] + (separator[1] - separator[0]))


    def observation(self, observations):
        state = 0
        for observation, separator in zip(observations, self._separators):
            state *= 1 + len(separator)
            state += np.digitize(observation, separator)
        if self._tiles is None:
            return state
        else:
            states = [state]
            for t in range(1, self._tiles):
                state = 0
                for i in range(len(self._separators)):
                    state *= 2 + len(self._separators[i])
                    value = observations[i] + ((t * (2 * i + 1)) % self._tiles) * self._separator_offsets[i]
                    if value > self._separator_tops[i]:
                        state += 1 + len(self._separators[i])
                    else:
                        state += np.digitize(value, self._separators[i])
                states.append(self._first_tile_states + (t - 1) * self._rest_tiles_states + state)
            return states


class DiscreteCartPoleWrapper(DiscretizationWrapper):
    def __init__(self, env, bins=8):
        super().__init__(env, [
            np.linspace(-2.4, 2.4, num=bins + 1)[1:-1], # cart position
            np.linspace(-3, 3, num=bins + 1)[1:-1],     # pole angle
            np.linspace(-0.5, 0.5, num=bins + 1)[1:-1], # cart velocity
            np.linspace(-2, 2, num=bins + 1)[1:-1],     # pole angle velocity
        ])


class DiscreteMountainCarWrapper(DiscretizationWrapper):
    def __init__(self, env, bins=None, tiles=None):
        if bins is None:
            bins = 24 if tiles is None or tiles <= 1 else 12 if tiles <= 3 else 8
        super().__init__(env, [
            np.linspace(-1.2, 0.6, num=bins + 1)[1:-1],   # car position
            np.linspace(-0.07, 0.07, num=bins + 1)[1:-1], # car velocity
        ], tiles)


class DiscreteLunarLanderWrapper(DiscretizationWrapper):
    def __init__(self, env):
        super().__init__(env, [
            np.linspace(-.4,   .4, num=5 + 1)[1:-1],   # x
            np.linspace(-.075,1.35,num=6 + 1)[1:-1],   # y
            np.linspace(-.5,   .5, num=5 + 1)[1:-1],   # vel x
            np.linspace(-.8,   .8, num=7 + 1)[1:-1],   # vel y
            np.linspace(-.2,   .2, num=3 + 1)[1:-1],   # rot
            np.linspace(-.2,   .2, num=5 + 1)[1:-1],   # ang vel
            [.5], #lc
            [.5], #rc
        ])

        self._expert = gym.make("LunarLander-v2")
        self._expert.seed(42)

    def expert_trajectory(self):
        state, trajectory, done = self._expert.reset(), [], False
        initial_state = self.observation(state)
        while not done:
            action = gym.envs.box2d.lunar_lander.heuristic(self._expert, state)
            state, reward, done, _ = self._expert.step(action)
            trajectory.append((action, reward, self.observation(state)))
        return initial_state, trajectory


gym.envs.register(
    id="MountainCar1000-v0",
    entry_point="gym.envs.classic_control:MountainCarEnv",
    max_episode_steps=1000,
    reward_threshold=-110.0,
)
