### Assignment: memory_game_rl
#### Date: Deadline: Feb 13, 23:59
#### Points: 5 points; any non-zero number counts as solved for passing the exam with grade 1 by solving all the assignments

This is a continuation of the `memory_game` assignment.

In this task, your goal is to solve the memory game environment
using reinforcement learning. That is, you must not use the
`env.expert_episode` method during training. You can start with the
[memory_game_rl.py](https://github.com/ufal/npfl122/tree/master/labs/12/memory_game_rl.py)
template, which extends the `memory_game` template by generating
training episodes suitable for some reinforcement learning algorithm.

ReCodEx evaluates your solution on environments with 4, 6 and 8 cards (utilizing
the `--cards` argument). For each card number, your solution gets 2 points
(1 point for 4 cards) if the average return is nonnegative. You can train the agent
directly in ReCodEx (the time limit is 15 minutes), or submit a pre-trained one.
