### Assignment: importance_sampling
#### Date: Deadline: Nov 10, 23:59
#### Points: 4 points

Using the [FrozenLake-v0 environment](https://gym.openai.com/envs/FrozenLake-v0)
environment, implement Monte Carlo weighted importance sampling to estimate
state value function $V$ of target policy, which uniformly chooses either action
1 (down) or action 2 (right), utilizing behaviour policy, which uniformly
chooses among all four actions.

Start with the [importance_sampling.py](https://github.com/ufal/npfl122/tree/master/labs/03/importance_sampling.py)
template, which creates the environment and generates episodes according to
behaviour policy.

For $1000$ episodes, the output of your program should be the following:
```
 0.00  0.00  0.00  0.00
 0.00  0.00  0.00  0.00
 0.00  0.00  0.21  0.00
 0.00  0.00  0.45  0.00
```
