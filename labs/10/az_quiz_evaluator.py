#!/usr/bin/env python3
import argparse
import importlib
import sys
import time

import az_quiz

def load_player(player):
    if player.endswith(".py"):
        player = player[:-3]

    module = importlib.import_module(player)
    args = module.parser.parse_args([])
    args.recodex = True
    return module.main(args)

def evaluate(players, games, randomized, render):
    wins = [0, 0]
    for i in range(games):
        for to_start in range(2):
            game = az_quiz.AZQuiz(randomized)
            try:
                while game.winner is None:
                    game.move(players[to_start ^ game.to_play].play(game.clone()))
                    if render:
                        game.render()
                        time.sleep(0.3)
            except ValueError:
                pass
            if game.winner == to_start:
                wins[to_start] += 1
            if render:
                time.sleep(1.0)

        print("First player win rate after {} games: {:.2f}% ({:.2f}% and {:.2f}% when starting and not starting)".format(
            2 * i + 2, 100 * (wins[0] + wins[1]) / (2 * i + 2), 100 * wins[0] / (i + 1), 100 * wins[1] / (i + 1)), file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("player_1", type=str, help="First player module")
    parser.add_argument("player_2", type=str, help="Second player module")
    parser.add_argument("--games", default=50, type=int, help="Number of alternating games to evaluate")
    parser.add_argument("--randomized", default=False, action="store_true", help="Is answering allowed to fail and generate random results")
    parser.add_argument("--render", default=False, action="store_true", help="Should the games be rendered")
    args = parser.parse_args()

    evaluate(
        [load_player(args.player_1), load_player(args.player_2)],
        games=args.games,
        randomized=args.randomized,
        render=args.render,
    )
