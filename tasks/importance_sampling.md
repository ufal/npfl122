### Assignment: importance_sampling
#### Date: Deadline: Oct 31, 7:59 a.m.
#### Points: 2 points
#### Tests: importance_sampling_tests

Using the [FrozenLake-v1 environment](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/),
implement Monte Carlo weighted importance sampling to estimate
state value function $V$ of target policy, which uniformly chooses either action
1 (down) or action 2 (right), utilizing behaviour policy, which uniformly
chooses among all four actions.

Start with the [importance_sampling.py](https://github.com/ufal/npfl122/tree/master/labs/03/importance_sampling.py)
template, which creates the environment and generates episodes according to
behaviour policy.

#### Tests Start: importance_sampling_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 importance_sampling.py --episodes=500`
```
 0.00  0.00  0.09  0.24
 0.00  0.00  0.30  0.00
 0.00  0.11  0.32  0.00
 0.00  0.25  0.33  0.00
```
- `python3 importance_sampling.py --episodes=5000`
```
 0.03  0.00  0.01  0.03
 0.04  0.00  0.09  0.00
 0.10  0.24  0.23  0.00
 0.00  0.44  0.49  0.00
```
- `python3 importance_sampling.py --episodes=50000`
```
 0.03  0.02  0.05  0.01
 0.13  0.00  0.07  0.00
 0.21  0.33  0.36  0.00
 0.00  0.35  0.76  0.00
```
#### Tests End:
