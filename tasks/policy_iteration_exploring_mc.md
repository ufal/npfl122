### Assignment: policy_iteration_exploring_mc
#### Date: Deadline: Oct 24, 23:59
#### Points: 2 points

Starting with [policy_iteration_exploring_mc.py](https://github.com/ufal/npfl122/tree/master/labs/02/policy_iteration_exploring_mc.py),
extend the `policy_iteration` assignment to perform policy evaluation
by using Monte Carlo estimation with exploring starts.

The estimation can now be performed model-free (without the access to the full
MDP dynamics), therefore, the `GridWorld.step` returns a randomly sampled
result instead of a full distribution.

