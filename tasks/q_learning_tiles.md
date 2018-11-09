### Assignment: q_learning_tiles
#### Date: Deadline: Nov 18, 23:59
#### Points: **compulsory**

Improve the `q_learning` task performance on the
[MountainCar-v0 environment](https://gym.openai.com/envs/MountainCar-v0)
environment using tile coding. Your goal is to reach an average reward of -110
during 100 evaluation episodes.

Use the updated [mountain_car_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/03/mountain_car_evaluator.py)
module (depending on updated [gym_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/02/gym_evaluator.py)
to interact with the discretized environment. The environment
methods and properties are described in the `monte_carlo` assignment, with the
following change:
- The `state` returned by the `env.step` method is a _list_ containing tile
  indices of the current state. Each of these indices are smaller than
  `env.states` (and therefore `env.states` returns in fact a number of
  tiles, or weights in the parametrization of the `Q` function approximation).

  The default number of offset tiles is `args.tiles=8`, but you can use any
  number you want (but the assignment is solvable with 8).

You can start with the [q_learning_tiles.py](https://github.com/ufal/npfl122/tree/master/labs/04/q_learning_tiles.py)
template, which parses several useful parameters, creates the environment
and illustrates the overall usage. Implementing Q-learning is enough to pass
the assignment, even if both N-step Sarsa and Tree Backup converge a little
faster.

During evaluation in ReCodEx, three different random seeds will be employed, and
you will get a point for each setting where you reach the required reward.
The time limit for each test is 5 minutes.
