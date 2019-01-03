#!/usr/bin/env python3
import numpy as np

class AZQuiz:
    actions = 28
    N = 7
    C = 4

    def __init__(self, randomized):
        self._board = np.zeros([self.N, self.N], dtype=np.uint8)
        self._randomized = randomized
        self._to_play = 0
        self._winner = None
        self._viewer = None

    def clone(self):
        clone = AZQuiz(self._randomized)
        clone._board[:, :] = self._board
        clone._to_play = self._to_play
        clone._winner = self._winner
        return clone

    @property
    def board(self):
        board = np.zeros([self.N, self.N, self.C], dtype=np.bool)
        for j in range(self.N):
            for i in range(j + 1):
                board[j, i] = self._REPRESENTATION[self._board[j, i]]
        return board

    @property
    def to_play(self):
        return self._to_play if self._winner is None else None

    @property
    def winner(self):
        return self._winner

    def swap_players(self):
        self._board = self._SWAP_PLAYERS[self._board]
        self._to_play = 1 - self._to_play
        self._winner = 1 - self._winner if self._winner is not None else None

    def valid(self, action):
        return self._winner is None and action >= 0 and action < self.actions \
            and self._board[self._ACTION_Y[action], self._ACTION_X[action]] < 2

    def move(self, action):
        return self._move(action, np.random.uniform() if self._randomized else 0)

    def all_moves(self, action):
        success = self.clone()
        success._move(action, 0.)
        if not self._randomized:
            return [(1.0, success)]

        failure = self.clone()
        failure._move(action, 1.)
        if self._board[self._ACTION_Y[action], self._ACTION_X[action]] == 0:
            success_probability = self._INITIAL_QUESTION_PROB
        else:
            success_probability = self._ADDITIONAL_QUESTION_PROB
        return [(success_probability, success), (1. - success_probability, failure)]

    def _move(self, action, random_value):
        if not self.valid(action):
            raise ValueError("An invalid action to AZQuiz.move")

        if self._board[self._ACTION_Y[action], self._ACTION_X[action]] == 0:
            if random_value <= self._INITIAL_QUESTION_PROB:
                self._board[self._ACTION_Y[action], self._ACTION_X[action]] = 2 + self._to_play
            else:
                self._board[self._ACTION_Y[action], self._ACTION_X[action]] = 1
        else:
            if random_value > self._ADDITIONAL_QUESTION_PROB:
                self._to_play = 1 - self._to_play
            self._board[self._ACTION_Y[action], self._ACTION_X[action]] = 2 + self._to_play
        self._to_play = 1 - self._to_play

        edges, visited = np.zeros(2, dtype=np.bool), np.zeros([self.N, self.N], dtype=np.bool)
        for j in range(self.N):
            edges[:] = False
            field = self._board[j, 0]
            if field >= 2:
                self._traverse(j, 0, field, edges, visited)
                if edges.all():
                    self._winner = field - 2

    def _traverse(self, j, i, field, edges, visited):
        if visited[j, i]: return
        visited[j, i] = True

        if j == i: edges[0] = True
        if j == self.N - 1: edges[1] = True
        if j - 1 >= 0:
            if i - 1 >= 0 and self._board[j - 1, i - 1] == field: self._traverse(j - 1, i - 1, field, edges, visited)
            if self._board[j - 1, i] == field: self._traverse(j - 1, i, field, edges, visited)
        if i - 1 >= 0 and self._board[j, i - 1] == field: self._traverse(j, i - 1, field, edges, visited)
        if i + 1 < self.N and self._board[j, i + 1] == field: self._traverse(j, i + 1, field, edges, visited)
        if j + 1 < self.N:
            if self._board[j + 1, i] == field: self._traverse(j + 1, i, field, edges, visited)
            if i + 1 < self.N and self._board[j + 1, i + 1] == field: self._traverse(j + 1, i + 1, field, edges, visited)

    def render(self):
        A = 40
        I = 35
        W = A * 13
        COLORS = np.array([[210, 210, 210], [58, 58, 58], [58, 147, 192], [254, 147, 17]], dtype=np.uint8)

        image = np.zeros([W, W, 3], dtype=np.uint8)
        for j in range(self.N):
            for i in range(j + 1):
                x = int((i - j/2) * 2 * 0.866 * A + W/2)
                y = int((j - (self.N - 1)/2) * 1.5 * A + W/2)
                for yo in range(-I, I + 1):
                    xo = int(min(2 * 0.866 * (I - abs(yo)), 0.866 * I))
                    image[y + yo, x - xo:x + xo + 1] = COLORS[self._board[j, i]]

        if self._viewer is None:
            from gym.envs.classic_control import rendering
            self._viewer = rendering.SimpleImageViewer()
        self._viewer.imshow(image)

    _INITIAL_QUESTION_PROB = 0.8
    _ADDITIONAL_QUESTION_PROB = 0.7

    _ACTION_Y = np.array([0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,6], dtype=np.int8)
    _ACTION_X = np.array([0,0,1,0,1,2,0,1,2,3,0,1,2,3,4,0,1,2,3,4,5,0,1,2,3,4,5,6], dtype=np.int8)
    _REPRESENTATION = np.array([[0,0,0,1], [0,0,1,1], [1,0,0,1], [0,1,0,1]], dtype=np.bool)
    _SWAP_PLAYERS = np.array([0, 1, 3, 2])

if __name__ == "__main__":
    quiz = AZQuiz(True)
    while quiz.winner is None:
        quiz.render()

        action = None
        while action is None or not quiz.valid(action):
            try:
                action = int(input("Action for player {}: ".format(quiz.to_play)))
            except KeyboardInterrupt:
                raise
            except:
                pass
        quiz.move(action)
    print("Congratulation player {}".format(quiz.winner))
