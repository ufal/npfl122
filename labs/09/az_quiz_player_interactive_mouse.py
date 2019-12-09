#!/usr/bin/env python3
import numpy as np

import az_quiz

class Player:
    def play(self, az_quiz):
        print("Choose action for player {}: ".format(az_quiz.to_play), end="", flush=True)
        action = az_quiz.mouse_input()
        print("action {}".format(action), flush=True)

        return action
