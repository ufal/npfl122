#!/usr/bin/env python3
import gym
import numpy as np

class MemoryGame(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, cards: int):
        assert cards > 0 and cards % 2 == 0

        self._cards = cards
        self._expert = None

        self.observation_space = gym.spaces.MultiDiscrete([cards, cards // 2])
        self.action_space = gym.spaces.Discrete(cards + 1)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._symbols = self.np_random.permutation(np.repeat(np.arange(self._cards // 2), 2))
        self._removed = bytearray(self._cards)
        self._used = bytearray(self._cards)
        self._unused_action = 0
        self._last_action = 0

        return self.step(0)[0]

    def step(self, action: int):
        assert action >= 0 and action <= self._cards
        if action == self._cards:
            action = self._unused_action
            self._unused_action += self._unused_action + 1 < self._cards

        self._used[action] = True
        while self._unused_action + 1 < self._cards and self._used[self._unused_action]:
            self._unused_action += 1

        reward = -1
        if self._symbols[self._last_action] == self._symbols[action] \
                and self._last_action != action \
                and not self._removed[action]:
            reward = +2
            self._removed[self._last_action] = True
            self._removed[action] = True
        self._last_action = action

        return (action, self._symbols[action]), reward, all(self._removed), {}

    def render(self, mode='human'):
        formatted = ["Memory game:"]
        for i in range(self._cards):
            formatted.append(str(self._symbols[i]) if not self._removed[i] else "X")
        formatted.append("Last action: {}".format(self._last_action))
        print(" ".join(formatted))

    def expert_episode(self) -> list[tuple[int, int]]:
        if self._expert is None:
            self._expert = MemoryGame(self._cards)
            self._expert.seed(42)

        state = self._expert.reset()
        episode, seen, done = [], {}, False
        while not done:
            last_action, observation = state
            if observation in seen:
                action = seen.pop(observation)
                if action == last_action - 1:
                    action = self._cards
            else:
                seen[observation] = last_action
                action = self._cards

            episode.append((state, action))
            state, _, done, _ = self._expert.step(action)
        episode.append((state, None))
        return episode


_memory_games = set()
def make(cards: int) -> gym.Env:
    env_name = "MemoryGame{}-v0".format(cards)
    if env_name not in _memory_games:
        gym.envs.register(id=env_name,
                          entry_point=lambda: MemoryGame(cards),
                          max_episode_steps=2 * cards,
                          reward_threshold=0)
        _memory_games.add(env_name)
    return gym.make(env_name)
