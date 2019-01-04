#!/usr/bin/env python3
import numpy as np

import az_quiz

class Player:
    def play(self, az_quiz):
        az_quiz.render()

        action = None
        while action is None or not az_quiz.valid(action):
            try:
                action = int(input("Action for player {}: ".format(az_quiz.to_play)))
            except ValueError:
                pass

        return action
