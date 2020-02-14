### Assignment: memory_game_rl
#### Date: Deadline: Mar 15, 23:59
#### Points: 5 points

This is a continuation of the `memory_game` assignment.

In this task, your goal is to solve the memory game environment
using reinforcement learning. That is, you must not use the
`env.expert_episode` method during training.

ReCodEx evaluates your solution on environments with 4, 6 and 8 cards (utilizing
the `--cards` argument). For each card number, your solution gets 2 points
(1 point for 4 cards) if the average return is nonnegative. You can train the agent
directly in ReCodEx, or submit a pre-trained one.

There is no specific template for this assignment, reuse the
[memory_game.py](https://github.com/ufal/npfl122/tree/master/labs/10/memory_game.py)
for the previous assignment.

**Note that you must not submit `gym_environment.py` nor
`memory_game_evaluator.py` to ReCodEx.**
