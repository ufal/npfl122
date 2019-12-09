### Assignment: walker
#### Date: Deadline: Jan 05, 23:59
#### Points: 8 points + 10 bonus

In this exercise exploring continuous robot control,
try solving the [BipedalWalker-v2 environment](https://gym.openai.com/envs/BipedalWalker-v2)
environment from the [OpenAI Gym](https://gym.openai.com/).

To create the evaluator, use
[gym_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/08/gym_evaluator.py)`.GymEvaluator("BipedalWalker-v2")`.
The environment is continuous, states and actions are described at
[OpenAI Gym Wiki](https://github.com/openai/gym/wiki/BipedalWalker-v2).

In ReCodEx, you are expected to submit an already trained model,
which is evaluated on 100 episodes with a total time
limit of 10 minutes. If your average return is at least 100, you obtain
8 points. The task is also a _competition_ and at most 10 points will be awarded
according to relative ordering of your solution performances.

You can start with the [ddpg.py](https://github.com/ufal/npfl122/tree/master/labs/08/ddpg.py)
template, only set `args.env` to `BipedalWalker-v2`.

**Note that you must not submit `gym_evaluator.py` to ReCodEx.**
