#!/usr/bin/env python3
import gym
import numpy as np


class MemoryGame(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, cards: int, render_mode=None):
        assert cards > 0 and cards % 2 == 0

        self._cards = cards
        self._expert = None

        self.observation_space = gym.spaces.MultiDiscrete([cards, cards // 2])
        self.action_space = gym.spaces.Discrete(cards + 1)
        self.render_mode = render_mode

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._symbols = self.np_random.permutation(np.repeat(np.arange(self._cards // 2), 2))
        self._removed = bytearray(self._cards)
        self._used = bytearray(self._cards)
        self._unused_action = 0
        self._last_action = 0

        return self.step(0)[0], {}

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

        if self.render_mode == "human":
            self.render()
        return np.array([action, self._symbols[action]]), reward, all(self._removed), False, {}

    def render(self, mode='human'):
        formatted = ["Memory game:"]
        for i in range(self._cards):
            formatted.append(str(self._symbols[i]) if not self._removed[i] else "X")
        formatted.append("Last action: {}".format(self._last_action))
        print(" ".join(formatted))

    def expert_episode(self) -> list[tuple[int, int]]:
        if self._expert is None:
            self._expert = make_memory_game(self._cards)
            self._expert.reset(seed=42)

        state = self._expert.reset()[0]
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
            state, _, terminated, truncated, _ = self._expert.step(action)
            done = terminated or truncated
        episode.append((state, None))
        return episode


def make_memory_game(cards: int):
    return gym.wrappers.TimeLimit(MemoryGame(cards), max_episode_steps=2 * cards)


gym.envs.register(
    id="MemoryGame-v0",
    entry_point=make_memory_game,
    reward_threshold=0,
)
