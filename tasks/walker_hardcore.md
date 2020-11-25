### Assignment: walker_hardcore
#### Date: Deadline: Dec 08, 23:59
#### Points: 10 bonus

As an extesnion of the `walker` assignment, try solving the
[BipedalWalkerHardcore-v3 environment](https://gym.openai.com/envs/BipedalWalkerHardcore-v2)
environment from the [OpenAI Gym](https://gym.openai.com/).

The task is a _competition only_ and at most 10 points will be awarded according to
relative ordering of your solution performances. In ReCodEx, your solution
will be evaluated with two seeds, each for 100 episodes with a time limit of 10 minutes.
If your average return is at least 0, ReCodEx shows the solution as correct.

You can start with the [ddpg.py](https://github.com/ufal/npfl122/tree/master/labs/07/ddpg.py)
template, only set `args.env` to `BipedalWalkerHardcore-v3`.
