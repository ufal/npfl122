### Assignment: memory_game
#### Date: Deadline: Feb 17, 23:59
#### Points: **5** bonus only

In this bonus-only exercise we explore a partially observable environment.
Consider a one-player variant of a memory game (pexeso), where a player repeatedly
flip cards. If the player flips two cards with the same symbol in succession,
the cards are removed and the player recieves a reward of +2. Otherwise the
player recieves a reward of -1. An episode ends when all cards are removed.
For a given even $N$, there are $N$ actions – the card indices, and $N/2$
observations, which encode card symbol. Every episode can be ended using
at most $3N/2$ actions. The environment is provided by the
[memory_game_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/11/memory_game_evaluator.py)
module.

Your goal is to solve the environment using an agent utilizing a LSTM cell.
The reinforce algorithm with baseline seems an appropriate choice.

ReCodEx evaluates your solution on environments with 4, 6, 8 and 16 cards
(utilizing the `--cards` argument). For each card number, 1000 episodes are
simulated and your solution gets 1 point (2 points for 16 cards) if the average
return is positive.

A template [memory_game.py](https://github.com/ufal/npfl122/tree/master/labs/11/memory_game.py)
is available. Depending on `memory_cells` argument, it employs either a vanilla
LSTM or LSTM with external memory. Note that I was able to train it only for 4, 6,
and 8 cards.
