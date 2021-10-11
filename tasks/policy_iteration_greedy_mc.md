### Assignment: policy_iteration_greedy_mc
#### Date: Deadline: Oct 24, 23:59
#### Points: 2 points

Starting with [policy_iteration_greedy_mc.py](https://github.com/ufal/npfl122/tree/master/labs/02/policy_iteration_greedy_mc.py),
extend the `policy_iteration_exploring_mc` assignment to perform policy
evaluation by using $ε$-greedy Monte Carlo estimation.

For the sake of replicability, use the provided
`GridWorld.epsilon_greedy(epsilon, greedy_action)` method, which returns
a random action with probability of `epsilon` and otherwise returns the
given `greedy_action`.

