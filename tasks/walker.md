### Assignment: walker
#### Date: Deadline: Dec 08, 23:59
#### Points: 8 points + 10 bonus

In this exercise exploring continuous robot control,
try solving the [BipedalWalker-v3 environment](https://gym.openai.com/envs/BipedalWalker-v2)
environment from the [OpenAI Gym](https://gym.openai.com/).
The environment is continuous, states and actions are described at
[OpenAI Gym Wiki](https://github.com/openai/gym/wiki/BipedalWalker-v2).

In ReCodEx, you are expected to submit an already trained model,
which is evaluated with two seeds, each for 100 episodes with a time
limit of 10 minutes. If your average return is at least 100, you obtain
8 points. The task is also a _competition_ and at most 10 points will be awarded
according to relative ordering of your solution performances.

You can start with the [walker.py](https://github.com/ufal/npfl122/tree/master/labs/08/walker.py)
template, but should probably reuse a lot of code from
[ddpg.py](https://github.com/ufal/npfl122/tree/master/labs/07/ddpg.py).
