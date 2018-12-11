### Assignment: walker
#### Date: Deadline: Jan 06, 23:59
#### Points: **10** bonus only

In this bonus-only exercise exploring continuous robot control,
try solving the [BipedalWalker-v2 environment](https://gym.openai.com/envs/BipedalWalker-v2)
environment from the [OpenAI Gym](https://gym.openai.com/).

To create the evaluator, use
[gym_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/02/gym_evaluator.py)`.GymEvaluator("BipedalWalker-v2")`.
The environment is continuous, states and actions are described at
[OpenAI Gym Wiki](https://github.com/openai/gym/wiki/BipedalWalker-v2).

The task is a _competition_ and at most 10 points will be awarded according to
relative ordering of your solution performances. In ReCodEx, your solution
will be evaluated on 100 different tracks with a total time limit of 10 minutes.
If your average return is at least 0, ReCodEx shows the solution as correct.

You can start with the [ddpg.py](https://github.com/ufal/npfl122/tree/master/labs/09/ddpg.py)
template, only set `args.env` to `BipedalWalker-v2`.
