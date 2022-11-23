### Assignment: walker
#### Date: Deadline: Dec 05, 7:59 a.m.
#### Points: 5 points

In this exercise we explore continuous robot control
by solving the continuous [BipedalWalker-v3 environment](https://www.gymlibrary.dev/environments/box2d/bipedal_walker/)
from the [Gym library](https://www.gymlibrary.dev/).

**Note that the penalty of `-100` on crash makes the training considerably slower.
Even if all of DDPG, TD3 and SAC can be trained with original rewards, overriding
the reward at the end of episode to `0` speeds up training considerably.**

In ReCodEx, you are expected to submit an already trained model,
which is evaluated with two seeds, each for 100 episodes with a time
limit of 10 minutes. If your average return is at least 200, you obtain
5 points.

The [walker.py](https://github.com/ufal/npfl122/tree/master/labs/08/walker.py)
template contains the skeleton for implementing the SAC agent, but you can
also solve the assignment with DDPG/TD3. The PyTorch template
[walker.torch.py](https://github.com/ufal/npfl122/tree/master/labs/08/walker.torch.py)
is also available.
