#!/usr/bin/env python3
import gym
import numpy as np

class MultiCollect(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 25}

    def __init__(self, agents: int):
        assert agents > 0
        self._agents = agents

        self.observation_space = gym.spaces.Box(np.full(6 * agents, -10, dtype=np.float32), np.full(6 * agents, 10, dtype=np.float32))
        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(5)] * agents)
        self._viewer = None

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._agents_pos = self.np_random.uniform(-10, 10, size=[self._agents, 2])
        self._agents_vel = self.np_random.randint(-4, 4 + 1, size=[self._agents, 2])
        self._centers = self.np_random.uniform(-10, 10, size=[self._agents, 2])
        self._centers_hit = np.zeros(self._agents, dtype=np.int32)

        return self.step(np.zeros(self._agents, dtype=np.int32))[0]

    def step(self, action: list[int]):
        action = np.asarray(action)
        assert len(action) == self._agents
        assert np.all(action >= 0) and np.all(action < 5)

        # Update speeds
        self._agents_vel[:, 0] += (action == 1)
        self._agents_vel[:, 0] -= (action == 2)
        self._agents_vel[:, 1] += (action == 3)
        self._agents_vel[:, 1] -= (action == 4)
        self._agents_vel = np.clip(self._agents_vel, -4, 4)

        # Update pos
        self._agents_pos += self._agents_vel / 8

        # Compute reward and update the hit information
        rewards = np.zeros(self._agents)
        distances = np.linalg.norm(self._centers[:, np.newaxis] - self._agents_pos[np.newaxis, :], axis=-1)
        for i in range(self._agents):
            a = np.argmin(distances[i])
            closest = distances[i][a]
            if closest < 1:
                rewards[a] += 1
                self._centers_hit[i] += 1
                if self._centers_hit[i] >= 10:
                    rewards[a] += 50
                    self._centers_hit[i] = 0
                    self._centers[i] = self.np_random.uniform(-10, 10, size=2)
            else:
                rewards[a] += 1 - (closest - 1) / 10
                self._centers_hit[i] = 0

        distances = np.linalg.norm(self._agents_pos[:, np.newaxis] - self._agents_pos[np.newaxis, :], axis=-1)
        closest = np.min(distances + np.eye(self._agents), axis=1)
        self._agents_hit = closest < 1
        rewards -= self._agents_hit

        state = np.concatenate([self._centers.ravel(), self._agents_pos.ravel(), self._agents_vel.ravel() / 8])
        return state, np.mean(rewards), False, {"agent_rewards": rewards}

    def render(self, mode="human"):
        W = 600
        PIXELS = 30 # 600 - (10 + 10)

        if self._viewer is None:
            from gym.envs.classic_control import rendering

            self._viewer = rendering.Viewer(W, W)

            self._viewer_centers = []
            for _ in range(self._agents):
                circle = rendering.make_circle(radius=PIXELS, res=30, filled=True)
                transform = rendering.Transform()
                circle.add_attr(transform)
                self._viewer.add_geom(circle)
                self._viewer_centers.append((circle, transform))

            self._viewer_agents_pos = []
            for _ in range(self._agents):
                circle = rendering.make_circle(radius=0.5 * PIXELS, res=30, filled=True)
                transform = rendering.Transform()
                circle.add_attr(transform)
                self._viewer.add_geom(circle)
                self._viewer_agents_pos.append((circle, transform))

            self._viewer_agents_vel = []
            for i in range(self._agents):
                self._viewer_agents_vel.append([[0, 0], [0, 0]])
                line = rendering.make_polyline(self._viewer_agents_vel[-1])
                line.add_attr(self._viewer_agents_pos[i][1])
                self._viewer.add_geom(line)

        if not hasattr(self, "_centers"):
            return None

        for (circle, transform), center, hit in zip(self._viewer_centers, self._centers, self._centers_hit):
            circle.set_color(0, 1.0 if hit else 0.7, 0)
            transform.set_translation(center[0] * PIXELS + W / 2, center[1] * PIXELS + W / 2)
        for (circle, transform), pos, hit in zip(self._viewer_agents_pos, self._agents_pos, self._agents_hit):
            circle.set_color(1.0 * hit, 0, 0.7 * (1 - hit))
            transform.set_translation(pos[0] * PIXELS + W / 2, pos[1] * PIXELS + W / 2)
        for line, vel in zip(self._viewer_agents_vel, self._agents_vel):
            line[1] = vel * PIXELS

        return self._viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self._viewer:
            self._viewer.close()
            self._viewer = None

class SingleCollect(MultiCollect):
    def __init__(self):
        super().__init__(1)
        self.action_space = self.action_space[0]

    def step(self, action):
        action = np.asarray(action)
        if not action.shape: action = np.expand_dims(action, axis=0)
        return super().step(action)


gym.envs.register(id="SingleCollect-v0", entry_point=SingleCollect, max_episode_steps=250, reward_threshold=0)
gym.envs.register(id="MultiCollect1-v0", entry_point=lambda: MultiCollect(1), max_episode_steps=250, reward_threshold=0)
gym.envs.register(id="MultiCollect2-v0", entry_point=lambda: MultiCollect(2), max_episode_steps=250, reward_threshold=0)
gym.envs.register(id="MultiCollect3-v0", entry_point=lambda: MultiCollect(3), max_episode_steps=250, reward_threshold=0)
gym.envs.register(id="MultiCollect4-v0", entry_point=lambda: MultiCollect(4), max_episode_steps=250, reward_threshold=0)
gym.envs.register(id="MultiCollect5-v0", entry_point=lambda: MultiCollect(5), max_episode_steps=250, reward_threshold=0)


# Allow running the environment and controlling it with arrows
if __name__=="__main__":
    import pyglet
    import time

    action, done = 0, False
    def key_press(k, mod):
        global action, done
        if k==0xff0d: done = True
        if k==pyglet.window.key.LEFT:  action = 2
        if k==pyglet.window.key.RIGHT: action = 1
        if k==pyglet.window.key.UP:  action = 3
        if k==pyglet.window.key.DOWN: action = 4

    env = gym.make("SingleCollect-v0")
    env.render()
    env.env._viewer.window.on_key_press = key_press

    while True:
        env.reset()
        rewards = []
        done = False
        while not done:
            _, reward, done, _ = env.step(action)
            action = 0
            env.render()
            rewards.append(reward)
            if len(rewards) % 25 == 0:
                print("Rewards for last 25 timesteps: {}".format(np.sum(rewards[-25:])))
            time.sleep(0.04)
        print("Episode ended with a return of {}".format(np.sum(rewards)))

    env.close()
