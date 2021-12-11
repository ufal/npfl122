### Assignment: walker
#### Date: Deadline: ~~Dec 05~~ Dec 12, 23:59
#### Points: 5 points

In this exercise exploring continuous robot control,
try solving the [BipedalWalker-v3 environment](https://gym.openai.com/envs/BipedalWalker-v2)
environment from the [OpenAI Gym](https://gym.openai.com/).
The environment is continuous, states and actions are described at
[OpenAI Gym Wiki](https://github.com/openai/gym/wiki/BipedalWalker-v2).

**Note that the penalty of `-100` on crash makes the training considerably slower.
Even if all of DDPG, TD3 and SAC can be trained with original rewards, overriding
the reward at the end of episode to `0` speeds up training considerably.**

In ReCodEx, you are expected to submit an already trained model,
which is evaluated with two seeds, each for 100 episodes with a time
limit of 10 minutes. If your average return is at least 200, you obtain
5 points.

The [walker.py](https://github.com/ufal/npfl122/tree/master/labs/08/walker.py)
template contains the skeleton for implementing the SAC agent, but you can
also solve the assignment with DDPG/TD3.
