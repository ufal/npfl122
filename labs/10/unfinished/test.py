#!/usr/bin/env python3
import numpy as np

import az_quiz
import az_quiz_cpp

def evaluate(x):
    print(x.dtype, x.shape, x)
    return np.ones(1, np.float32), np.ones([1, 28], np.float32)

game = az_quiz.AZQuiz(False)
game.move(13)
game.move(25)
game.move(7)
az_quiz_cpp.mcts(game._board, evaluate, 100, 0., 0.)
