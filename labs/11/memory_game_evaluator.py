#!/usr/bin/env python3
import gym
import gym.envs.classic_control
import numpy as np

class MemoryGame(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, cards):
        assert cards % 2 == 0

        self._cards = cards
        self.observation_space = gym.spaces.Discrete(cards // 2)
        self.action_space = gym.spaces.Discrete(cards)

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self._symbols = self.np_random.permutation(np.repeat(np.arange(self._cards // 2), 2))
        self._removed = np.zeros((self._cards,), dtype=np.bool)
        self._last_action = 0

        return self._symbols[0]

    def _step(self, action):
        assert action >= 0 and action < self._cards

        reward = -1
        if self._symbols[self._last_action] == self._symbols[action] \
                and self._last_action != action \
                and not self._removed[action]:
            reward = +2
            self._removed[self._last_action] = True
            self._removed[action] = True
        self._last_action = action

        return self._symbols[action], reward, self._removed.all(), {}

    def _render(self, mode='human', close=False):
        if close: return

        formatted = ["Memory game:"]
        for i in range(self._cards):
            formatted.append(str(self._symbols[i]) if not self._removed[i] else "X")
        print(" ".join(formatted))

memory_games = set()

import gym_evaluator
def environment(cards):
    env_name = "MemoryGame{}-v0".format(cards)
    if env_name not in memory_games:
        gym.envs.register(id=env_name,
                          entry_point=lambda: MemoryGame(cards),
                          max_episode_steps=2 * cards,
                          reward_threshold=0)
        memory_games.add(env_name)
    return gym_evaluator.GymEnvironment(env_name)
