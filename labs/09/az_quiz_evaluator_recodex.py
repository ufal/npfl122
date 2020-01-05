#!/usr/bin/env python3

# This module will be available in ReCodEx.

import az_quiz
import az_quiz_player_simple_heuristic

class Player:
    def play(self, az_quiz):
        raise NotImplementedError()

def evaluate(player, games=50, randomized=False):
    # Will be implemented slightly differently in ReCodEx

    players = [player, az_quiz_player_simple_heuristic.Player()]
    wins = [0, 0]
    for i in range(games):
        for to_start in range(2):
            game = az_quiz.AZQuiz(randomized)
            while game.winner is None:
                game.move(players[to_start ^ game.to_play].play(game.clone()))
            if game.winner == to_start:
                wins[to_start] += 1

        print("First player win rate after {} games: {:.2f}% ({:.2f}% and {:.2f}% when starting and not starting)".format(
            2 * i + 2, 100 * (wins[0] + wins[1]) / (2 * i + 2), 100 * wins[0] / (i + 1), 100 * wins[1] / (i + 1)))
