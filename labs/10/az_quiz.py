#!/usr/bin/env python3
import numpy as np

class AZQuiz:
    actions = 28
    N = 7
    C = 3

    def __init__(self):
        self._board = np.zeros([self.N, self.N, self.C], dtype=np.bool)
        self._to_play = 0
        self._winner = None
        self._viewer = None

    def clone(self):
        clone = AZQuiz()
        clone._board[:, :, :] = self._board
        clone._to_play = self._to_play
        clone._winner = self._winner
        return clone

    @property
    def board(self):
        return np.copy(self._board)

    @property
    def to_play(self):
        return self._to_play if not self._winner else None

    @property
    def winner(self):
        return self._winner

    def swap_players(self):
        self._board[:, :, [0, 1]] = self._board[:, :, [1, 0]]
        self._to_play = 1 - self._to_play
        self._winner = 1 - self._winner if self._winner is not None else None

    def valid(self, action):
        return self._winner is None and action >= 0 and action < self.actions \
            and not self._board[self._ACTION_Y[action], self._ACTION_X[action], 0:2].any()

    def move(self, action):
        if not self.valid(action):
            raise ValueError("An invalid action to AZQuiz.move")

        field = self._board[self._ACTION_Y[action], self._ACTION_X[action]]
        if not field[2]:
            if np.random.uniform() <= self._INITIAL_QUESTION_PROB:
                field[self._to_play] = 1
            else:
                field[2] = 1
        else:
            if np.random.uniform() <= self._ADDITIONAL_QUESTION_PROB:
                field[self._to_play] = 1
            else:
                field[1 - self._to_play] = 1
                self._to_play = 1 - self._to_play
            field[2] = 0
        self._to_play = 1 - self._to_play

        edges, visited = np.zeros(2, dtype=np.bool), np.zeros([self.N, self.N], dtype=np.bool)
        for j in range(self.N):
            edges[:] = False
            player = 0 if self._board[j, 0, 0] else 1 if self._board[j, 0, 1] else None
            if player is not None:
                self._traverse(j, 0, player, edges, visited)
                if edges.all():
                    self._winner = player

    def _traverse(self, j, i, player, edges, visited):
        if visited[j, i]: return
        visited[j, i] = True

        if j == i: edges[0] = True
        if j == self.N - 1: edges[1] = True
        if j - 1 >= 0:
            if i - 1 >= 0 and self._board[j - 1, i - 1, player]: self._traverse(j - 1, i - 1, player, edges, visited)
            if self._board[j - 1, i, player]: self._traverse(j - 1, i, player, edges, visited)
        if i - 1 >= 0 and self._board[j, i - 1, player]: self._traverse(j, i - 1, player, edges, visited)
        if i + 1 < self.N and self._board[j, i + 1, player]: self._traverse(j, i + 1, player, edges, visited)
        if j + 1 < self.N:
            if self._board[j + 1, i, player]: self._traverse(j + 1, i, player, edges, visited)
            if i + 1 < self.N and self._board[j + 1, i + 1, player]: self._traverse(j + 1, i + 1, player, edges, visited)

    def render(self):
        A = 40
        I = 35
        W = A * 13
        COLORS = np.array([[210, 210, 210], [58, 58, 58], [254, 147, 17], [0, 0, 0], [58, 147, 192]], dtype=np.uint8)

        image = np.zeros([W, W, 3], dtype=np.uint8)
        for j in range(self.N):
            for i in range(j + 1):
                x = int((i - j/2) * 2 * 0.866 * A + W/2)
                y = int((j - (self.N - 1)/2) * 1.5 * A + W/2)
                for yo in range(-I, I + 1):
                    xo = int(min(2 * 0.866 * (I - abs(yo)), 0.866 * I))
                    image[y + yo, x - xo:x + xo + 1] = COLORS[np.polyval(self._board[j, i], 2)]

        if self._viewer is None:
            from gym.envs.classic_control import rendering
            self._viewer = rendering.SimpleImageViewer()
        self._viewer.imshow(image)

    _INITIAL_QUESTION_PROB = 0.8
    _ADDITIONAL_QUESTION_PROB = 0.7

    _ACTION_Y = np.array([0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,6], dtype=np.int8)
    _ACTION_X = np.array([0,0,1,0,1,2,0,1,2,3,0,1,2,3,4,0,1,2,3,4,5,0,1,2,3,4,5,6], dtype=np.int8)

if __name__ == "__main__":
    quiz = AZQuiz()
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
