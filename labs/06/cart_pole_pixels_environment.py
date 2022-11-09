#!/usr/bin/env python3
import gym
import gym.envs.classic_control
import numpy as np

import pygame


class CartPolePixels(gym.envs.classic_control.CartPoleEnv):
    H, W = 80, 80
    SCALE = 8

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self._images = 3
        self._image = np.zeros([self.H, self.W, self._images], dtype=np.uint8)
        self._screen = None

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=[self.H, self.W, self._images], dtype=np.uint8)

    def reset(self, seed=None, options=None):
        render_mode = self.render_mode
        self.render_mode = None
        state, info = super().reset(seed=seed, options=options)
        self.render_mode = render_mode

        for step in range(self._images):
            observation = self._draw(state)

        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):
        render_mode = self.render_mode
        self.render_mode = None
        state, reward, termination, truncation, info = super().step(action)
        self.render_mode = render_mode

        observation = self._draw(state)

        if self.render_mode == "human":
            self.render()
        return observation, reward, termination, truncation, info

    def render(self):
        assert self.render_mode in self.metadata["render_modes"]

        if self.render_mode == "rgb_array":
            return np.copy(self._image)

        if self._screen is None:
            pygame.init()
            pygame.display.init()
            self._screen = pygame.display.set_mode((self.W * self.SCALE, self.H * self.SCALE))
            self._screen_surface = pygame.Surface((self.W * self.SCALE, self.H * self.SCALE))
            self._clock = pygame.time.Clock()

        pygame.pixelcopy.array_to_surface(
            self._screen_surface, self._image.repeat(8, axis=0).repeat(8, axis=1).transpose([1, 0, 2]))

        pygame.event.pump()
        self._screen.blit(self._screen_surface, (0, 0))
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])
        pygame.event.pump()

    def close(self):
        if self._screen is not None:
            self._screen = None
            pygame.display.quit()
            pygame.quit()

    def _draw(self, state):
        for i in range(self._images - 1):
            self._image[:, :, i] = self._image[:, :, i + 1]

        cart = int(40 + state[0] / 3 * 40)
        cart_poly = [(70, cart - 10), (80, cart - 10), (80, cart + 10), (70, cart + 10)]

        pole_x = int(40 + (state[0] + np.sin(state[2]) * 4.2) / 3 * 40)
        pole_y = int(70 - np.cos(state[2]) * 5.2 / 3 * 40)
        pole_poly = [(pole_y, pole_x - 2), (70, cart - 2), (70, cart + 2), (pole_y, pole_x + 2)]

        self._image[:, :, self._images - 1] = 0
        self._fill_polygon(cart_poly, self._image[:, :, self._images - 1], 128)
        self._fill_polygon(pole_poly, self._image[:, :, self._images - 1], 255)
        return np.copy(self._image)

    # Taken from https://github.com/luispedro/mahotas/blob/master/mahotas/polygon.py
    def _fill_polygon(self, polygon, canvas, color=1):
        '''
        fill_polygon([(y0,x0), (y1,x1),...], canvas, color=1)
        Draw a filled polygon in canvas
        Parameters
        ----------
        polygon : list of pairs
            a list of (y,x) points
        canvas : ndarray
            where to draw, will be modified in place
        color : integer, optional
            which colour to use (default: 1)
        '''
        # algorithm adapted from: http://www.alienryderflex.com/polygon_fill/
        if not len(polygon):
            return
        min_y = min(y for y, x in polygon)
        max_y = max(y for y, x in polygon)
        polygon = [(float(y), float(x)) for y, x in polygon]
        if max_y < canvas.shape[0]:
            max_y += 1
        for y in range(min_y, max_y):
            nodes = []
            j = -1
            for i, p in enumerate(polygon):
                pj = polygon[j]
                if p[0] < y and pj[0] >= y or pj[0] < y and p[0] >= y:
                    dy = pj[0] - p[0]
                    if dy:
                        nodes.append((p[1] + (y - p[0]) / (pj[0] - p[0]) * (pj[1] - p[1])))
                    elif p[0] == y:
                        nodes.append(p[1])
                j = i
            nodes.sort()
            for n, nn in zip(nodes[::2], nodes[1::2]):
                nn += 1
                canvas[y, int(n):int(nn)] = color


#################################
# Environment for NPFL122 class #
#################################

gym.envs.register(
    id="CartPolePixels-v1",
    entry_point=CartPolePixels,
    max_episode_steps=500,
    reward_threshold=475,
)

# Allow running the environment and controlling it with arrows
if __name__ == "__main__":
    env = CartPolePixels(render_mode="human")
    env.metadata["render_fps"] = 10

    quit = False
    while not quit:
        env.reset()
        steps, action, restart = 0, 0, False
        while True:
            # Handle input
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = 0
                    if event.key == pygame.K_RIGHT:
                        action = 1
                    if event.key == pygame.K_RETURN:
                        restart = True
                    if event.key == pygame.K_ESCAPE:
                        quit = True
                if event.type == pygame.QUIT:
                    quit = True

            # Perform the step
            _, _, terminated, truncated, _ = env.step(action)

            steps += 1
            if terminated or truncated or restart or quit:
                break
        print("Episode ended after {} steps".format(steps))

    env.close()
