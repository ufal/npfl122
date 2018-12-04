#!/usr/bin/env python3
import gym
import numpy as np

import gym_evaluator

def environment(discrete=True, tiles=None):
    if discrete:
        bins = 24 if tiles is None or tiles <= 1 else 12 if tiles <= 3 else 8
        separators = [
            np.linspace(-1.2, 0.6, num=bins + 1)[1:-1],   # car position
            np.linspace(-0.07, 0.07, num=bins + 1)[1:-1], # car velocity
        ]
        return gym_evaluator.GymEnvironment("MountainCarContinuous-v0", separators=separators, tiles=tiles)

    return gym_evaluator.GymEnvironment("MountainCarContinuous-v0")
