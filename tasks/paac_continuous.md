### Assignment: paac_continuous
#### Date: Deadline: Dec 15, 23:59
#### Points: 6 points

Using the [paac_continuous.py](https://github.com/ufal/npfl122/tree/master/labs/08/paac_continuous.py)
template, solve the [MountainCarContinuous-v0 environment](https://gym.openai.com/envs/MountainCarContinuous-v0/)
environment using parallel actor-critic algorithm with continuous actions.

The `gym_environment` now provides two additional methods:
- `action_shape`: returns required shape of continuous action. You can
  assume the actions are always an one-dimensional vector.
- `action_ranges`: returns a pair of vectors `low`, `high`. These denote
  valid ranges for the actions, so `low[i]`$≤$`action[i]`$≤$`high[i]`.

Your goal is to reach an average return of 90 during 100 evaluation episodes.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 10 minutes.
