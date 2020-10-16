### Assignment: policy_iteration_greedy_mc
#### Date: Deadline: Oct 27, 23:59
#### Points: 2 points
#### Examples: policy_iteration_greedy_mc_examples

Starting with [policy_iteration_greedy_mc.py](https://github.com/ufal/npfl122/tree/master/labs/02/policy_iteration_greedy_mc.py),
extend the `policy_iteration_exploring_mc` assignment to perform policy
evaluation by using $ε$-greedy Monte Carlo estimation.

For the sake of replicability, use the provided
`GridWorld.epsilon_greedy(epsilon, greedy_action)` method, which returns
a random action with probability of `epsilon` and otherwise returns the
given `greedy_action`.

#### Examples Start: policy_iteration_greedy_mc_examples
Note that your results may sometimes be slightly different (for example because of varying floating point arithmetic on your CPU).
- `python3 policy_iteration_greedy_mc.py --gamma=0.95 --seed=42 --steps=1`
```
    0.00↑    0.00↑    0.00↑    0.00↑
    0.00↑             0.00→    0.00→
    0.00↑    0.00↑    0.00→    0.00→
```
- `python3 policy_iteration_greedy_mc.py --gamma=0.95 --seed=42 --steps=10`
```
   -0.90↓   -0.12←    0.00←  -13.60↑
    0.62→             0.51←    0.00←
    0.57←    1.79→   -7.55←    7.78↓
```
- `python3 policy_iteration_greedy_mc.py --gamma=0.95 --seed=42 --steps=50`
```
   -0.20↓   -0.02←    0.00←   -5.94↑
    0.12→             0.14←   -0.27←
    0.11←    0.00↓   -1.40←    5.55↓
```
- `python3 policy_iteration_greedy_mc.py --gamma=0.95 --seed=42 --steps=100`
```
   -0.10↓   -0.01←    0.00←   -4.48↑
    0.06→            -0.82←   -0.75←
    0.01←    0.00↓   -1.37←    3.40↓
```
- `python3 policy_iteration_greedy_mc.py --gamma=0.95 --seed=42 --steps=200`
```
   -0.05↓   -0.01←    0.00←   -3.42↑
    0.03→            -0.39←   -1.82←
    0.01←    0.03↓   -0.99←    3.66↓
```
- `python3 policy_iteration_greedy_mc.py --gamma=0.95 --seed=42 --steps=500`
```
   -0.05↓   -0.00←   -0.29←   -2.90↑
    0.01→            -0.61←   -1.28←
   -0.02←    0.01↓   -0.50←    2.94↓
```
#### Examples End:
