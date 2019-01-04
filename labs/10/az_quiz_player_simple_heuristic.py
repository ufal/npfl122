#!/usr/bin/env python3
import numpy as np

import az_quiz

class Player:
    CENTER = 12
    ANCHORS = [4, 16, 19]

    def play(self, az_quiz):
        if az_quiz.valid(self.CENTER): return self.CENTER

        any_anchor = any(map(az_quiz.valid, self.ANCHORS))

        action = None
        while action is None or not az_quiz.valid(action):
            if any_anchor:
                action = np.random.choice(self.ANCHORS)
            else:
                action = np.random.randint(az_quiz.actions)

        return action
