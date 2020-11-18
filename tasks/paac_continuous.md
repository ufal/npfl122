### Assignment: paac_continuous
#### Date: Deadline: Dec 01, 23:59
#### Points: 6 points

Solve the [MountainCarContinuous-v0 environment](https://gym.openai.com/envs/MountainCarContinuous-v0/)
environment using parallel actor-critic algorithm with continuous actions.
When actions are continuous, `env.action_space` is the same `Box` space
as `env.observation_space`, offering:
- `env.action_space.shape`, which specifies the shape of actions (you can assume
  actions are always a 1D vector),
- `env.action_space.low` and `env.action_space.high`, which specify the ranges
  of the corresponding actions.

Your goal is to reach an average return of 90 during 100 evaluation episodes.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 10 minutes.

For the time being, the template is not yet ready. You can look at the
[last year's template](https://github.com/ufal/npfl122/tree/past-1920/labs/08/paac_continuous.py),
but note that it will not work in ReCodEx.
