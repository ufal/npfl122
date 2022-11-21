### Assignment: walker_hardcore
#### Date: Deadline: Dec 05, 7:59 a.m.
#### Points: 6 points + 8 bonus

As an extension of the `walker` assignment, solve the
_hardcore_ version of the [BipedalWalker-v3 environment](https://www.gymlibrary.dev/environments/box2d/bipedal_walker/)
from the [Gym library](https://www.gymlibrary.dev/).

**Note that the penalty of `-100` on crash can discourage or even stop training,
so overriding the reward at the end of episode to `0` (or descresing it
substantially) makes the training considerably easier (I have not surpassed
return `0` with neither TD3 nor SAC with the original `-100` penalty).**

In ReCodEx, you are expected to submit an already trained model,
which is evaluated with three seeds, each for 100 episodes with a time
limit of 10 minutes. If your average return is at least 100, you obtain
6 points. The task is also a [_competition_](https://ufal.mff.cuni.cz/courses/npfl122/2223-winter#competitions),
and at most 8 points will be awarded according to relative ordering of your
solutions.

The [walker_hardcore.py](https://github.com/ufal/npfl122/tree/master/labs/08/walker_hardcore.py)
template shows a basic structure of evaluaton in ReCodEx, but
you most likely want to start either with [ddpg.py](https://github.com/ufal/npfl122/tree/master/labs/07/ddpg.py).
or with [walker.py](https://github.com/ufal/npfl122/tree/master/labs/08/walker.py)
and just change the `env` argument to `BipedalWalkerHardcore-v3`.
