#!/usr/bin/env python3
import gym
import numpy as np

import gym_evaluator

def environment(discrete=True):
    if discrete:
        separators = [
            np.linspace(-.4,   .4, num=5 + 1)[1:-1],   # x
            np.linspace(-.075,1.35,num=6 + 1)[1:-1],   # y
            np.linspace(-.5,   .5, num=5 + 1)[1:-1],   # vel x
            np.linspace(-.8,   .8, num=7 + 1)[1:-1],   # vel y
            np.linspace(-.2,   .2, num=3 + 1)[1:-1],   # rot
            np.linspace(-.2,   .2, num=5 + 1)[1:-1],   # ang vel
            [.5], #lc
            [.5], #rc
        ]
        evaluator = gym_evaluator.GymEnvironment("LunarLander-v2", separators=separators)
    else:
        evaluator = gym_evaluator.GymEnvironment("LunarLander-v2")

    evaluator._expert = gym.make("LunarLander-v2")
    evaluator._expert.seed(42)
    evaluator._expert.continuous = not discrete
    def expert_trajectory():
        state, trajectory, done = evaluator._expert.reset(), [], False
        initial_state = evaluator._maybe_discretize(state)
        while not done:
            action = gym.envs.box2d.lunar_lander.heuristic(evaluator._expert, state)
            state, reward, done, _ = evaluator._expert.step(action)
            trajectory.append((action, reward, evaluator._maybe_discretize(state)))
        return initial_state, trajectory
    evaluator.expert_trajectory = expert_trajectory

    return evaluator
