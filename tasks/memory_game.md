### Assignment: memory_game
#### Date: Deadline: Jan 27, 23:59
#### Points: **5** bonus only

In this bonus-only exercise we explore a partially observable environment.
Consider a one-player variant of a memory game (pexeso), where a player repeatedly
flip cards. If the player flips two cards with the same symbol in succession,
the cards are removed and the player recieves a reward of +2. Otherwise the
player recieves a reward of -1. An episode ends when all cards are removed.

For a given even $N$, there are $N$ actions – the card indices, and $N/2$
observations, which encode card symbol. Every episode can be ended using
at most $3N/2$ actions.

Your goal is to solve the environment using an agent utilizing a LSTM cell.
The reinforce algorithm with baseline seems an appropriate choice.

The templates will be added shortly.
