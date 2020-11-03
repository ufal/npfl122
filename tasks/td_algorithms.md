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
Episode 100, mean 100-episode return -360.33 +-184.47
Episode 200, mean 100-episode return -224.77 +-98.00
Episode 300, mean 100-episode return -182.97 +-100.50
Episode 400, mean 100-episode return -138.60 +-96.61
Episode 500, mean 100-episode return -106.91 +-82.49
Episode 600, mean 100-episode return -88.05 +-79.26
Episode 700, mean 100-episode return -57.03 +-53.84
Episode 800, mean 100-episode return -33.09 +-54.11
Episode 900, mean 100-episode return -30.81 +-45.75
Episode 1000, mean 100-episode return -21.21 +-35.98
```
- `python3 td_algorithms.py --mode=sarsa --n=1 --off_policy`
```
Episode 100, mean 100-episode return -368.15 +-184.96
Episode 200, mean 100-episode return -207.59 +-101.46
Episode 300, mean 100-episode return -170.73 +-100.77
Episode 400, mean 100-episode return -143.05 +-97.48
Episode 500, mean 100-episode return -93.66 +-90.10
Episode 600, mean 100-episode return -66.25 +-68.43
Episode 700, mean 100-episode return -38.15 +-56.84
Episode 800, mean 100-episode return -25.82 +-44.67
Episode 900, mean 100-episode return -18.04 +-37.85
Episode 1000, mean 100-episode return -14.56 +-34.09
```
- `python3 td_algorithms.py --mode=sarsa --n=4`
```
Episode 100, mean 100-episode return -516.63 +-256.06
Episode 200, mean 100-episode return -205.93 +-160.51
Episode 300, mean 100-episode return -169.65 +-165.20
Episode 400, mean 100-episode return -68.71 +-131.53
Episode 500, mean 100-episode return -15.79 +-45.34
Episode 600, mean 100-episode return -8.01 +-38.65
Episode 700, mean 100-episode return -6.21 +-30.64
Episode 800, mean 100-episode return -5.69 +-16.12
Episode 900, mean 100-episode return 0.68 +-8.99
Episode 1000, mean 100-episode return -1.56 +-10.94
```
- `python3 td_algorithms.py --mode=sarsa --n=4 --off_policy`
```
Episode 100, mean 100-episode return -524.26 +-195.11
Episode 200, mean 100-episode return -345.41 +-181.73
Episode 300, mean 100-episode return -286.07 +-165.83
Episode 400, mean 100-episode return -249.51 +-187.19
Episode 500, mean 100-episode return -112.83 +-158.33
Episode 600, mean 100-episode return -80.56 +-145.49
Episode 700, mean 100-episode return -20.16 +-71.73
Episode 800, mean 100-episode return -17.42 +-62.07
Episode 900, mean 100-episode return -5.14 +-27.98
Episode 1000, mean 100-episode return -1.83 +-12.61
```
- `python3 td_algorithms.py --mode=expected_sarsa --n=1`
```
Episode 100, mean 100-episode return -361.33 +-186.01
Episode 200, mean 100-episode return -214.54 +-104.67
Episode 300, mean 100-episode return -179.69 +-103.63
Episode 400, mean 100-episode return -147.74 +-92.59
Episode 500, mean 100-episode return -109.10 +-89.53
Episode 600, mean 100-episode return -79.89 +-75.51
Episode 700, mean 100-episode return -59.05 +-57.01
Episode 800, mean 100-episode return -40.03 +-44.50
Episode 900, mean 100-episode return -25.21 +-38.41
Episode 1000, mean 100-episode return -19.67 +-34.80
```
- `python3 td_algorithms.py --mode=expected_sarsa --n=1 --off_policy`
```
Episode 100, mean 100-episode return -358.93 +-187.30
Episode 200, mean 100-episode return -221.93 +-91.20
Episode 300, mean 100-episode return -176.05 +-110.42
Episode 400, mean 100-episode return -124.69 +-92.91
Episode 500, mean 100-episode return -98.44 +-86.99
Episode 600, mean 100-episode return -64.75 +-69.56
Episode 700, mean 100-episode return -51.46 +-52.95
Episode 800, mean 100-episode return -28.69 +-44.46
Episode 900, mean 100-episode return -17.27 +-30.60
Episode 1000, mean 100-episode return -10.83 +-25.23
```
- `python3 td_algorithms.py --mode=expected_sarsa --n=4`
```
Episode 100, mean 100-episode return -555.15 +-204.64
Episode 200, mean 100-episode return -261.06 +-131.13
Episode 300, mean 100-episode return -144.66 +-157.24
Episode 400, mean 100-episode return -88.66 +-144.94
Episode 500, mean 100-episode return -25.55 +-69.55
Episode 600, mean 100-episode return -6.82 +-30.54
Episode 700, mean 100-episode return -2.32 +-18.24
Episode 800, mean 100-episode return -0.09 +-10.35
Episode 900, mean 100-episode return -0.06 +-14.05
Episode 1000, mean 100-episode return -0.28 +-11.60
```
- `python3 td_algorithms.py --mode=expected_sarsa --n=4 --off_policy`
```
Episode 100, mean 100-episode return -526.36 +-202.40
Episode 200, mean 100-episode return -306.17 +-167.38
Episode 300, mean 100-episode return -258.25 +-180.35
Episode 400, mean 100-episode return -146.21 +-174.19
Episode 500, mean 100-episode return -120.67 +-167.93
Episode 600, mean 100-episode return -85.25 +-153.25
Episode 700, mean 100-episode return -23.43 +-92.30
Episode 800, mean 100-episode return -21.92 +-70.71
Episode 900, mean 100-episode return -4.94 +-22.51
Episode 1000, mean 100-episode return -5.79 +-26.25
```
- `python3 td_algorithms.py --mode=tree_backup --n=1`
```
Episode 100, mean 100-episode return -361.33 +-186.01
Episode 200, mean 100-episode return -214.54 +-104.67
Episode 300, mean 100-episode return -179.69 +-103.63
Episode 400, mean 100-episode return -147.74 +-92.59
Episode 500, mean 100-episode return -109.10 +-89.53
Episode 600, mean 100-episode return -79.89 +-75.51
Episode 700, mean 100-episode return -59.05 +-57.01
Episode 800, mean 100-episode return -40.03 +-44.50
Episode 900, mean 100-episode return -25.21 +-38.41
Episode 1000, mean 100-episode return -19.67 +-34.80
```
- `python3 td_algorithms.py --mode=tree_backup --n=1 --off_policy`
```
Episode 100, mean 100-episode return -358.93 +-187.30
Episode 200, mean 100-episode return -221.93 +-91.20
Episode 300, mean 100-episode return -176.05 +-110.42
Episode 400, mean 100-episode return -124.69 +-92.91
Episode 500, mean 100-episode return -98.44 +-86.99
Episode 600, mean 100-episode return -64.75 +-69.56
Episode 700, mean 100-episode return -51.46 +-52.95
Episode 800, mean 100-episode return -28.69 +-44.46
Episode 900, mean 100-episode return -17.27 +-30.60
Episode 1000, mean 100-episode return -10.83 +-25.23
```
- `python3 td_algorithms.py --mode=tree_backup --n=4`
```
Episode 100, mean 100-episode return -522.36 +-226.08
Episode 200, mean 100-episode return -264.75 +-136.85
Episode 300, mean 100-episode return -163.50 +-168.74
Episode 400, mean 100-episode return -54.18 +-105.95
Episode 500, mean 100-episode return -27.66 +-70.12
Episode 600, mean 100-episode return -9.05 +-23.62
Episode 700, mean 100-episode return -4.76 +-31.53
Episode 800, mean 100-episode return -2.57 +-12.74
Episode 900, mean 100-episode return 0.58 +-12.08
Episode 1000, mean 100-episode return 1.17 +-9.07
```
- `python3 td_algorithms.py --mode=tree_backup --n=4 --off_policy`
```
Episode 100, mean 100-episode return -519.80 +-233.81
Episode 200, mean 100-episode return -302.58 +-123.70
Episode 300, mean 100-episode return -203.98 +-153.41
Episode 400, mean 100-episode return -95.12 +-136.49
Episode 500, mean 100-episode return -25.28 +-65.11
Episode 600, mean 100-episode return -4.79 +-19.20
Episode 700, mean 100-episode return -8.53 +-29.38
Episode 800, mean 100-episode return -5.13 +-19.44
Episode 900, mean 100-episode return -1.98 +-12.35
Episode 1000, mean 100-episode return -1.59 +-11.99
```
#### Examples End:
