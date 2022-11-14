### Assignment: paac_continuous
#### Date: Deadline: Nov 28, 7:59 a.m.
#### Points: 5 points

Solve the [MountainCarContinuous-v0 environment](https://www.gymlibrary.dev/environments/classic_control/mountain_car_continuous/)
using parallel actor-critic algorithm with continuous actions.
When actions are continuous, `env.action_space` is the same `Box` space
as `env.observation_space`, offering:
- `env.action_space.shape`, which specifies the shape of actions (you can assume
  actions are always a 1D vector),
- `env.action_space.low` and `env.action_space.high`, which specify the ranges
  of the corresponding actions.

Your goal is to reach an average return of 90 during 100 evaluation episodes.

Start with the [paac_continuous.py](https://github.com/ufal/npfl122/tree/master/labs/07/paac_continuous.py)
template, which provides a simple network implementation in TensorFlow. Feel
free to use PyTorch or JAX instead, if you like.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 10 minutes.
