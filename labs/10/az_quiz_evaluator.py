#!/usr/bin/env python3
import sys
import time

import az_quiz

class Player:
    def play(self, az_quiz):
        raise NotImplementedError()

def evaluate(players, games, randomized, render):
    wins = [0, 0]
    for i in range(games):
        for to_start in range(2):
            game = az_quiz.AZQuiz(randomized)
            to_play = to_start
            try:
                while game.winner is None:
                    game.move(players[to_play].play(game.clone()))
                    to_play = 1 - to_play
                    if render:
                        game.render()
                        time.sleep(0.3)
            except ValueError:
                pass
            if to_play == 1:
                wins[to_start] += 1
            if render:
                time.sleep(1.0)

        print("First player win rate: {:.2f}% ({:.2f}% and {:.2f}% when starting and not starting)".format(
            100 * (wins[0] + wins[1]) / (2 * i + 2), 100 * wins[0] / (i + 1), 100 * wins[1] / (i + 1)), file=sys.stderr)


if __name__ == "__main__":
    import argparse
    import importlib

    parser = argparse.ArgumentParser()
    parser.add_argument("player_1", type=str, help="First player module")
    parser.add_argument("player_2", type=str, help="Second player module")
    parser.add_argument("--games", default=100, type=int, help="Number of alternating games to evaluate")
    parser.add_argument("--randomized", default=False, action="store_true", help="Is answering allowed to fail and generate random results")
    parser.add_argument("--render", default=False, action="store_true", help="Should the games be rendered")
    args = parser.parse_args()

    if args.player_1.endswith(".py"): args.player_1 = args.player_1[:-3]
    if args.player_2.endswith(".py"): args.player_2 = args.player_2[:-3]

    evaluate(
        [importlib.import_module(args.player_1).Player(),
         importlib.import_module(args.player_2).Player()],
        games=args.games,
        randomized=args.randomized,
        render=args.render,
    )
