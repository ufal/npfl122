### Assignment: q_learning_tiles
#### Date: Deadline: Nov 18, 23:59
#### Points: **compulsory**

Improve the `q_learning` task performance on the
[MountainCar-v0 environment](https://gym.openai.com/envs/MountainCar-v0)
environment using linear function approximation with tile coding.
Your goal is to reach an average reward of -110 during 100 evaluation episodes.

Use the updated [mountain_car_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/03/mountain_car_evaluator.py)
module (depending on updated [gym_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/02/gym_evaluator.py))
to interact with the discretized environment. The environment
methods and properties are described in the `monte_carlo` assignment, with the
following changes:
- The `env.weights` method return the number of weights of the linear function
  approximation.
- The `state` returned by the `env.step` method is a _list_ containing weight
  indices of the current state (i.e., the feature vector of the state consists
  of zeros and ones, and only the indices of the ones are returned). The
  (action-)value function for a state is therefore approximated as a sum of the
  weights whose indices are returned by `env.step`.

  The default number of tiles in tile encoding (i.e., the size of the list with
  weight indices) is `args.tiles=8`, but you can use any number you want (but
  the assignment is solvable with 8).

You can start with the [q_learning_tiles.py](https://github.com/ufal/npfl122/tree/master/labs/04/q_learning_tiles.py)
template, which parses several useful parameters, creates the environment
and illustrates the overall usage. Implementing Q-learning is enough to pass
the assignment, even if both N-step Sarsa and Tree Backup converge a little
faster.

During evaluation in ReCodEx, three different random seeds will be employed, and
you will get a point for each setting where you reach the required reward.
The time limit for each test is 5 minutes.
