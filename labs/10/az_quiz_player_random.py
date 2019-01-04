#!/usr/bin/env python3
import numpy as np

import az_quiz

class Player:
    def play(self, az_quiz):
        action = None
        while action is None or not az_quiz.valid(action):
            action = np.random.randint(az_quiz.actions)

        return action

