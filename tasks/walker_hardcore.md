### Assignment: walker_hardcore
#### Date: Deadline: ~~Dec 05~~ Dec 12, 23:59
#### Points: 6 points + 8 bonus

As an extension of the `walker` assignment, solve the
[BipedalWalkerHardcore-v3 environment](https://gym.openai.com/envs/BipedalWalkerHardcore-v2)
environment from the [OpenAI Gym](https://gym.openai.com/).

**Note that the penalty of `-100` on crash can discourage or even stop training,
so decreasing it substantially or even setting it to `0` makes the training easier
(I could not surpass return 0 with neither TD3 nor SAC with the `-100` penalty).**

In ReCodEx, you are expected to submit an already trained model,
which is evaluated with three seeds, each for 100 episodes with a time
limit of 10 minutes. If your average return is at least 100, you obtain
6 points. The task is also a [_competition_](https://ufal.mff.cuni.cz/courses/npfl122/2122-winter#competitions)
and at most 8 points will be awarded according to relative ordering of your
solution performances.

The [walker_hardcore.py](https://github.com/ufal/npfl122/tree/master/labs/08/walker_hardcore.py)
template shows a basic structure of evaluaton in ReCodEx, but
you most likely want to start either with [ddpg.py](https://github.com/ufal/npfl122/tree/master/labs/07/ddpg.py).
or with [walker.py](https://github.com/ufal/npfl122/tree/master/labs/08/walker.py)
and just change the `env` argument to `BipedalWalkerHardcore-v3`.
