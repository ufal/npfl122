#!/usr/bin/env python3
import numpy as np

import az_quiz

class Player:
    def play(self, az_quiz):
        action = None
        while action is None or not az_quiz.valid(action):
            action = np.random.randint(az_quiz.actions)

        return action

if __name__ == "__main__":
    import az_quiz_evaluator_recodex
    az_quiz_evaluator_recodex.evaluate(Player())
