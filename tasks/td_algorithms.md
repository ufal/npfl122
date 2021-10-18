### Assignment: td_algorithms
#### Date: Deadline: Nov 07, 23:59
#### Points: 4 points
#### Examples: td_algorithms_examples
#### Tests: td_algorithms_tests

Starting with the [td_algorithms.py](https://github.com/ufal/npfl122/tree/master/labs/03/td_algorithms.py)
template, implement all of the following $n$-step TD methods variants:
- SARSA, expected SARSA and Tree backup;
- either on-policy (with $ε$-greedy behaviour policy) or off-policy
  (with the same behaviour policy, but greedy target policy).

#### Examples Start: td_algorithms_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 td_algorithms.py --mode=sarsa --n=1`
```
Episode 200, mean 100-episode return -223.71 +-93.59
Episode 400, mean 100-episode return -139.87 +-94.98
Episode 600, mean 100-episode return -79.92 +-78.13
Episode 800, mean 100-episode return -45.47 +-61.87
Episode 1000, mean 100-episode return -23.00 +-37.69
```
- `python3 td_algorithms.py --mode=sarsa --n=1 --off_policy`
```
Episode 200, mean 100-episode return -226.79 +-86.95
Episode 400, mean 100-episode return -121.31 +-98.15
Episode 600, mean 100-episode return -75.73 +-82.94
Episode 800, mean 100-episode return -32.20 +-54.12
Episode 1000, mean 100-episode return -12.77 +-31.50
```
- `python3 td_algorithms.py --mode=sarsa --n=4`
```
Episode 200, mean 100-episode return -250.47 +-162.91
Episode 400, mean 100-episode return -84.50 +-136.47
Episode 600, mean 100-episode return -13.44 +-46.17
Episode 800, mean 100-episode return -2.66 +-14.03
Episode 1000, mean 100-episode return -1.50 +-12.57
```
- `python3 td_algorithms.py --mode=sarsa --n=4 --off_policy`
```
Episode 200, mean 100-episode return -355.38 +-114.57
Episode 400, mean 100-episode return -195.13 +-184.38
Episode 600, mean 100-episode return -35.85 +-89.91
Episode 800, mean 100-episode return -2.21 +-13.34
Episode 1000, mean 100-episode return -0.94 +-12.37
```
- `python3 td_algorithms.py --mode=expected_sarsa --n=1`
```
Episode 200, mean 100-episode return -236.48 +-88.08
Episode 400, mean 100-episode return -130.54 +-97.12
Episode 600, mean 100-episode return -80.92 +-75.09
Episode 800, mean 100-episode return -45.32 +-58.35
Episode 1000, mean 100-episode return -18.23 +-33.90
```
- `python3 td_algorithms.py --mode=expected_sarsa --n=1 --off_policy`
```
Episode 200, mean 100-episode return -211.49 +-96.62
Episode 400, mean 100-episode return -133.54 +-94.77
Episode 600, mean 100-episode return -61.63 +-71.41
Episode 800, mean 100-episode return -27.20 +-40.28
Episode 1000, mean 100-episode return -12.80 +-25.67
```
- `python3 td_algorithms.py --mode=expected_sarsa --n=4`
```
Episode 200, mean 100-episode return -208.38 +-160.04
Episode 400, mean 100-episode return -60.27 +-117.63
Episode 600, mean 100-episode return -4.22 +-18.40
Episode 800, mean 100-episode return -1.46 +-11.79
Episode 1000, mean 100-episode return -1.50 +-17.14
```
- `python3 td_algorithms.py --mode=expected_sarsa --n=4 --off_policy`
```
Episode 200, mean 100-episode return -360.77 +-127.22
Episode 400, mean 100-episode return -232.19 +-196.49
Episode 600, mean 100-episode return -71.42 +-142.10
Episode 800, mean 100-episode return -7.37 +-26.92
Episode 1000, mean 100-episode return -5.71 +-27.27
```
- `python3 td_algorithms.py --mode=tree_backup --n=1`
```
Episode 200, mean 100-episode return -236.48 +-88.08
Episode 400, mean 100-episode return -130.54 +-97.12
Episode 600, mean 100-episode return -80.92 +-75.09
Episode 800, mean 100-episode return -45.32 +-58.35
Episode 1000, mean 100-episode return -18.23 +-33.90
```
- `python3 td_algorithms.py --mode=tree_backup --n=1 --off_policy`
```
Episode 200, mean 100-episode return -211.49 +-96.62
Episode 400, mean 100-episode return -133.54 +-94.77
Episode 600, mean 100-episode return -61.63 +-71.41
Episode 800, mean 100-episode return -27.20 +-40.28
Episode 1000, mean 100-episode return -12.80 +-25.67
```
- `python3 td_algorithms.py --mode=tree_backup --n=4`
```
Episode 200, mean 100-episode return -240.56 +-121.74
Episode 400, mean 100-episode return -67.22 +-120.22
Episode 600, mean 100-episode return -7.58 +-42.29
Episode 800, mean 100-episode return -7.12 +-41.08
Episode 1000, mean 100-episode return 0.61 +-8.93
```
- `python3 td_algorithms.py --mode=tree_backup --n=4 --off_policy`
```
Episode 200, mean 100-episode return -302.59 +-137.42
Episode 400, mean 100-episode return -99.90 +-149.18
Episode 600, mean 100-episode return -9.82 +-36.69
Episode 800, mean 100-episode return -0.14 +-9.99
Episode 1000, mean 100-episode return 0.11 +-9.68
```
#### Examples End:
#### Tests Start: td_algorithms_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 td_algorithms.py --episodes=10 --mode=sarsa --n=1`
```
Episode 10, mean 100-episode return -650.00 +-103.40
```
- `python3 td_algorithms.py --episodes=10 --mode=sarsa --n=1 --off_policy`
```
Episode 10, mean 100-episode return -575.50 +-161.10
```
- `python3 td_algorithms.py --episodes=10 --mode=sarsa --n=4`
```
Episode 10, mean 100-episode return -736.10 +-63.01
```
- `python3 td_algorithms.py --episodes=10 --mode=sarsa --n=4 --off_policy`
```
Episode 10, mean 100-episode return -602.20 +-174.02
```
- `python3 td_algorithms.py --episodes=10 --mode=expected_sarsa --n=1`
```
Episode 10, mean 100-episode return -673.90 +-80.11
```
- `python3 td_algorithms.py --episodes=10 --mode=expected_sarsa --n=1 --off_policy`
```
Episode 10, mean 100-episode return -632.00 +-109.27
```
- `python3 td_algorithms.py --episodes=10 --mode=expected_sarsa --n=4`
```
Episode 10, mean 100-episode return -737.00 +-76.37
```
- `python3 td_algorithms.py --episodes=10 --mode=expected_sarsa --n=4 --off_policy`
```
Episode 10, mean 100-episode return -560.90 +-147.25
```
- `python3 td_algorithms.py --episodes=10 --mode=tree_backup --n=1`
```
Episode 10, mean 100-episode return -673.90 +-80.11
```
- `python3 td_algorithms.py --episodes=10 --mode=tree_backup --n=1 --off_policy`
```
Episode 10, mean 100-episode return -632.00 +-109.27
```
- `python3 td_algorithms.py --episodes=10 --mode=tree_backup --n=4`
```
Episode 10, mean 100-episode return -708.50 +-169.01
```
- `python3 td_algorithms.py --episodes=10 --mode=tree_backup --n=4 --off_policy`
```
Episode 10, mean 100-episode return -695.00 +-97.85
```
#### Tests End:
