### Assignment: td_algorithms
#### Date: Deadline: ~~Nov 10, 23:59~~ Nov 17, 23:59
#### Points: 5 points
#### Examples: td_algorithms_examples

Starting with the [td_algorithms.py](https://github.com/ufal/npfl122/tree/master/labs/04/td_algorithms.py)
template, implement all of the following $n$-step TD methods variants:
- SARSA, expected SARSA and Tree backup;
- either on-policy (with $ε$-greedy behaviour policy) or off-policy
  (with the same behaviour policy, but greedy target policy).

This assignment is new, so if you think you have implemented everything
correctly and it does not pass, do not hesitate to write me.

#### Examples Start: td_algorithms_examples
Note that your results may sometimes be slightly different (for example because of varying floating point arithmetic on your CPU).
- `python3 td_algorithms.py --mode=sarsa --n=1`
```
Episode 100, mean 100-episode return -359.90 +-180.01
Episode 200, mean 100-episode return -229.65 +-95.70
Episode 300, mean 100-episode return -188.63 +-110.47
Episode 400, mean 100-episode return -142.89 +-103.14
Episode 500, mean 100-episode return -119.60 +-95.95
Episode 600, mean 100-episode return -75.24 +-69.79
Episode 700, mean 100-episode return -61.72 +-66.71
Episode 800, mean 100-episode return -47.05 +-53.93
Episode 900, mean 100-episode return -27.54 +-47.37
Episode 1000, mean 100-episode return -20.95 +-30.10
```
- `python3 td_algorithms.py --mode=sarsa --n=1 --off_policy`
```
Episode 100, mean 100-episode return -373.39 +-178.89
Episode 200, mean 100-episode return -222.31 +-97.32
Episode 300, mean 100-episode return -188.14 +-98.75
Episode 400, mean 100-episode return -132.91 +-94.43
Episode 500, mean 100-episode return -101.87 +-98.58
Episode 600, mean 100-episode return -60.76 +-68.50
Episode 700, mean 100-episode return -51.04 +-59.31
Episode 800, mean 100-episode return -30.87 +-47.22
Episode 900, mean 100-episode return -22.93 +-37.66
Episode 1000, mean 100-episode return -15.19 +-33.73
```
- `python3 td_algorithms.py --mode=sarsa --n=4`
```
Episode 100, mean 100-episode return -531.91 +-254.05
Episode 200, mean 100-episode return -227.53 +-140.60
Episode 300, mean 100-episode return -160.83 +-168.68
Episode 400, mean 100-episode return -57.30 +-105.12
Episode 500, mean 100-episode return -16.90 +-57.42
Episode 600, mean 100-episode return -5.51 +-23.77
Episode 700, mean 100-episode return -7.75 +-26.05
Episode 800, mean 100-episode return -2.56 +-13.42
Episode 900, mean 100-episode return -0.14 +-12.36
Episode 1000, mean 100-episode return -0.82 +-11.08
```
- `python3 td_algorithms.py --mode=sarsa --n=4 --off_policy`
```
Episode 100, mean 100-episode return -553.47 +-180.17
Episode 200, mean 100-episode return -375.44 +-119.84
Episode 300, mean 100-episode return -365.94 +-138.19
Episode 400, mean 100-episode return -304.91 +-142.79
Episode 500, mean 100-episode return -153.13 +-170.85
Episode 600, mean 100-episode return -58.07 +-125.97
Episode 700, mean 100-episode return -13.95 +-45.24
Episode 800, mean 100-episode return -10.65 +-50.26
Episode 900, mean 100-episode return -3.50 +-16.14
Episode 1000, mean 100-episode return -2.70 +-13.37
```
- `python3 td_algorithms.py --mode=expected_sarsa --n=1`
```
Episode 100, mean 100-episode return -361.82 +-191.18
Episode 200, mean 100-episode return -223.74 +-89.97
Episode 300, mean 100-episode return -185.13 +-105.88
Episode 400, mean 100-episode return -133.16 +-92.85
Episode 500, mean 100-episode return -96.37 +-93.79
Episode 600, mean 100-episode return -63.07 +-71.77
Episode 700, mean 100-episode return -58.80 +-74.54
Episode 800, mean 100-episode return -40.60 +-51.91
Episode 900, mean 100-episode return -34.29 +-49.64
Episode 1000, mean 100-episode return -17.55 +-33.08
```
- `python3 td_algorithms.py --mode=expected_sarsa --n=1 --off_policy`
```
Episode 100, mean 100-episode return -361.52 +-183.75
Episode 200, mean 100-episode return -225.36 +-100.96
Episode 300, mean 100-episode return -187.48 +-97.69
Episode 400, mean 100-episode return -134.88 +-96.69
Episode 500, mean 100-episode return -100.00 +-80.72
Episode 600, mean 100-episode return -66.45 +-67.23
Episode 700, mean 100-episode return -41.29 +-48.97
Episode 800, mean 100-episode return -33.64 +-44.79
Episode 900, mean 100-episode return -19.59 +-32.56
Episode 1000, mean 100-episode return -16.06 +-30.43
```
- `python3 td_algorithms.py --mode=expected_sarsa --n=4`
```
Episode 100, mean 100-episode return -547.51 +-209.08
Episode 200, mean 100-episode return -264.92 +-127.12
Episode 300, mean 100-episode return -171.70 +-175.02
Episode 400, mean 100-episode return -74.00 +-144.16
Episode 500, mean 100-episode return -36.78 +-79.85
Episode 600, mean 100-episode return -5.32 +-23.55
Episode 700, mean 100-episode return -2.52 +-16.14
Episode 800, mean 100-episode return 1.14 +-11.30
Episode 900, mean 100-episode return -0.04 +-11.26
Episode 1000, mean 100-episode return -0.82 +-14.19
```
- `python3 td_algorithms.py --mode=expected_sarsa --n=4 --off_policy`
```
Episode 100, mean 100-episode return -544.86 +-194.10
Episode 200, mean 100-episode return -340.38 +-142.35
Episode 300, mean 100-episode return -244.34 +-183.61
Episode 400, mean 100-episode return -183.99 +-191.23
Episode 500, mean 100-episode return -100.23 +-170.85
Episode 600, mean 100-episode return -86.90 +-166.61
Episode 700, mean 100-episode return -84.99 +-159.75
Episode 800, mean 100-episode return -45.56 +-117.89
Episode 900, mean 100-episode return -33.16 +-95.41
Episode 1000, mean 100-episode return -2.89 +-11.83
```
- `python3 td_algorithms.py --mode=tree_backup --n=1`
```
Episode 100, mean 100-episode return -361.82 +-191.18
Episode 200, mean 100-episode return -223.74 +-89.97
Episode 300, mean 100-episode return -185.13 +-105.88
Episode 400, mean 100-episode return -133.16 +-92.85
Episode 500, mean 100-episode return -96.37 +-93.79
Episode 600, mean 100-episode return -63.07 +-71.77
Episode 700, mean 100-episode return -58.80 +-74.54
Episode 800, mean 100-episode return -40.60 +-51.91
Episode 900, mean 100-episode return -34.29 +-49.64
Episode 1000, mean 100-episode return -17.55 +-33.08
```
- `python3 td_algorithms.py --mode=tree_backup --n=1 --off_policy`
```
Episode 100, mean 100-episode return -361.52 +-183.75
Episode 200, mean 100-episode return -225.36 +-100.96
Episode 300, mean 100-episode return -187.48 +-97.69
Episode 400, mean 100-episode return -134.88 +-96.69
Episode 500, mean 100-episode return -100.00 +-80.72
Episode 600, mean 100-episode return -66.45 +-67.23
Episode 700, mean 100-episode return -41.29 +-48.97
Episode 800, mean 100-episode return -33.64 +-44.79
Episode 900, mean 100-episode return -19.59 +-32.56
Episode 1000, mean 100-episode return -16.06 +-30.43
```
- `python3 td_algorithms.py --mode=tree_backup --n=4`
```
Episode 100, mean 100-episode return -529.04 +-227.39
Episode 200, mean 100-episode return -319.63 +-101.42
Episode 300, mean 100-episode return -213.11 +-159.49
Episode 400, mean 100-episode return -87.48 +-134.82
Episode 500, mean 100-episode return -41.35 +-95.06
Episode 600, mean 100-episode return -24.18 +-72.25
Episode 700, mean 100-episode return -5.93 +-22.41
Episode 800, mean 100-episode return -0.28 +-14.11
Episode 900, mean 100-episode return -0.52 +-9.41
Episode 1000, mean 100-episode return -0.08 +-10.96
```
- `python3 td_algorithms.py --mode=tree_backup --n=4 --off_policy`
```
Episode 100, mean 100-episode return -544.07 +-208.75
Episode 200, mean 100-episode return -319.63 +-70.77
Episode 300, mean 100-episode return -213.71 +-161.24
Episode 400, mean 100-episode return -114.55 +-141.32
Episode 500, mean 100-episode return -25.77 +-69.23
Episode 600, mean 100-episode return -8.91 +-32.95
Episode 700, mean 100-episode return -1.57 +-17.71
Episode 800, mean 100-episode return -1.08 +-8.79
Episode 900, mean 100-episode return -0.50 +-10.60
Episode 1000, mean 100-episode return -0.60 +-11.27
```
#### Examples End:
