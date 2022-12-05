#!/usr/bin/env python3
import argparse

import numpy as np

import az_quiz

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.


class Player:
    CENTER = 12
    ANCHORS = [4, 16, 19]

    def play(self, az_quiz):
        if az_quiz.valid(self.CENTER):
            return self.CENTER

        any_anchor = any(map(az_quiz.valid, self.ANCHORS))

        action = None
        while action is None or not az_quiz.valid(action):
            if any_anchor:
                action = np.random.choice(self.ANCHORS)
            else:
                action = np.random.randint(az_quiz.actions)

        return action


def main(args):
    return Player()
