### Assignment: trace_algorithms
#### Date: Deadline: Feb 14, 23:59
#### Points: 4 points
#### Examples: trace_algorithms_examples

Starting with the [trace_algorithms.py](https://github.com/ufal/npfl122/tree/master/labs/09/trace_algorithms.py)
template, implement the following state value estimations:
- use $n$-step estimates for a given $n$;
- if requested, use eligibility traces with a given $λ$;
- allow off-policy correction using importance sampling, optionally
  clipping the importance sampling ratios by a given threshold.

#### Examples Start: trace_algorithms_examples
Note that your results may sometimes be slightly different (for example because of varying floating point arithmetic on your CPU).
- `python3 trace_algorithms.py --n=1`
```
Episode 100, mean 100-episode return -96.61 +-96.27
Episode 200, mean 100-episode return -30.95 +-57.27
Episode 300, mean 100-episode return -27.00 +-47.28
Episode 400, mean 100-episode return -11.97 +-32.78
Episode 500, mean 100-episode return -10.74 +-34.76
Episode 600, mean 100-episode return -6.70 +-32.46
Episode 700, mean 100-episode return -3.87 +-18.23
Episode 800, mean 100-episode return 0.72 +-11.11
Episode 900, mean 100-episode return -1.34 +-24.02
Episode 1000, mean 100-episode return 2.68 +-9.17
The mean 1000-episode return after evaluation -38.58 +-87.25
```
- `python3 trace_algorithms.py --n=4`
```
Episode 100, mean 100-episode return -81.76 +-98.03
Episode 200, mean 100-episode return -3.72 +-19.53
Episode 300, mean 100-episode return 0.59 +-10.51
Episode 400, mean 100-episode return 0.99 +-8.76
Episode 500, mean 100-episode return -0.35 +-9.10
Episode 600, mean 100-episode return 1.39 +-8.22
Episode 700, mean 100-episode return 2.42 +-7.80
Episode 800, mean 100-episode return 2.38 +-8.33
Episode 900, mean 100-episode return 2.79 +-7.16
Episode 1000, mean 100-episode return 0.42 +-8.51
The mean 1000-episode return after evaluation -9.03 +-57.13
```
- `python3 trace_algorithms.py --n=8`
```
Episode 100, mean 100-episode return -107.63 +-113.99
Episode 200, mean 100-episode return -3.57 +-16.96
Episode 300, mean 100-episode return 0.17 +-10.35
Episode 400, mean 100-episode return 0.20 +-8.34
Episode 500, mean 100-episode return 0.27 +-12.30
Episode 600, mean 100-episode return 1.45 +-8.57
Episode 700, mean 100-episode return 2.39 +-8.68
Episode 800, mean 100-episode return 1.92 +-8.32
Episode 900, mean 100-episode return -2.12 +-15.31
Episode 1000, mean 100-episode return -5.06 +-28.00
The mean 1000-episode return after evaluation -69.59 +-100.82
```
- `python3 trace_algorithms.py --n=4 --trace_lambda=0.6`
```
Episode 100, mean 100-episode return -87.36 +-95.03
Episode 200, mean 100-episode return -10.61 +-28.93
Episode 300, mean 100-episode return -3.48 +-15.93
Episode 400, mean 100-episode return -2.11 +-12.50
Episode 500, mean 100-episode return 1.09 +-8.20
Episode 600, mean 100-episode return 1.40 +-8.85
Episode 700, mean 100-episode return 3.78 +-7.59
Episode 800, mean 100-episode return 1.77 +-8.44
Episode 900, mean 100-episode return 0.53 +-9.03
Episode 1000, mean 100-episode return 1.73 +-7.72
The mean 1000-episode return after evaluation 7.63 +-2.44
```
- `python3 trace_algorithms.py --n=8 --trace_lambda=0.6`
```
Episode 100, mean 100-episode return -110.14 +-107.12
Episode 200, mean 100-episode return -18.70 +-45.52
Episode 300, mean 100-episode return -4.57 +-23.40
Episode 400, mean 100-episode return 1.17 +-8.73
Episode 500, mean 100-episode return 1.57 +-8.19
Episode 600, mean 100-episode return 2.46 +-8.84
Episode 700, mean 100-episode return 1.47 +-8.32
Episode 800, mean 100-episode return 0.11 +-9.15
Episode 900, mean 100-episode return 1.59 +-8.02
Episode 1000, mean 100-episode return 0.85 +-9.86
The mean 1000-episode return after evaluation 6.63 +-16.25
```
- `python3 trace_algorithms.py --n=1 --off_policy`
```
Episode 100, mean 100-episode return -74.33 +-71.96
Episode 200, mean 100-episode return -24.48 +-32.66
Episode 300, mean 100-episode return -19.26 +-26.23
Episode 400, mean 100-episode return -10.81 +-22.29
Episode 500, mean 100-episode return -10.40 +-19.60
Episode 600, mean 100-episode return -2.12 +-14.89
Episode 700, mean 100-episode return -3.98 +-17.19
Episode 800, mean 100-episode return -0.89 +-11.64
Episode 900, mean 100-episode return 0.04 +-9.86
Episode 1000, mean 100-episode return 1.02 +-7.64
The mean 1000-episode return after evaluation -22.17 +-73.86
```
- `python3 trace_algorithms.py --n=4 --off_policy`
```
Episode 100, mean 100-episode return -117.24 +-230.82
Episode 200, mean 100-episode return -51.57 +-168.31
Episode 300, mean 100-episode return -21.17 +-70.69
Episode 400, mean 100-episode return -9.15 +-54.49
Episode 500, mean 100-episode return -16.76 +-69.23
Episode 600, mean 100-episode return -28.78 +-85.01
Episode 700, mean 100-episode return -2.70 +-35.51
Episode 800, mean 100-episode return 2.04 +-12.42
Episode 900, mean 100-episode return 2.11 +-7.18
Episode 1000, mean 100-episode return 3.09 +-7.54
The mean 1000-episode return after evaluation 2.90 +-31.92
```
- `python3 trace_algorithms.py --n=8 --off_policy`
```
Episode 100, mean 100-episode return -149.37 +-113.62
Episode 200, mean 100-episode return -122.90 +-271.17
Episode 300, mean 100-episode return -59.51 +-152.97
Episode 400, mean 100-episode return -144.60 +-352.50
Episode 500, mean 100-episode return -194.83 +-485.96
Episode 600, mean 100-episode return -318.01 +-648.72
Episode 700, mean 100-episode return -257.36 +-592.33
Episode 800, mean 100-episode return -161.44 +-463.22
Episode 900, mean 100-episode return -264.20 +-595.78
Episode 1000, mean 100-episode return -160.56 +-493.25
The mean 1000-episode return after evaluation -151.62 +-453.81
```
- `python3 trace_algorithms.py --n=1 --off_policy --vtrace_clip=1`
```
Episode 100, mean 100-episode return -68.42 +-71.79
Episode 200, mean 100-episode return -29.45 +-42.22
Episode 300, mean 100-episode return -18.72 +-26.28
Episode 400, mean 100-episode return -13.66 +-27.16
Episode 500, mean 100-episode return -6.64 +-19.08
Episode 600, mean 100-episode return -3.34 +-14.91
Episode 700, mean 100-episode return -5.72 +-16.65
Episode 800, mean 100-episode return -0.49 +-11.18
Episode 900, mean 100-episode return -0.67 +-10.13
Episode 1000, mean 100-episode return -0.11 +-11.14
The mean 1000-episode return after evaluation -18.13 +-69.41
```
- `python3 trace_algorithms.py --n=4 --off_policy --vtrace_clip=1`
```
Episode 100, mean 100-episode return -70.81 +-85.32
Episode 200, mean 100-episode return -2.69 +-16.06
Episode 300, mean 100-episode return 1.57 +-12.96
Episode 400, mean 100-episode return 3.57 +-7.22
Episode 500, mean 100-episode return 2.22 +-7.57
Episode 600, mean 100-episode return 1.69 +-7.65
Episode 700, mean 100-episode return 1.91 +-8.33
Episode 800, mean 100-episode return 1.88 +-7.36
Episode 900, mean 100-episode return 3.57 +-7.46
Episode 1000, mean 100-episode return 3.07 +-7.15
The mean 1000-episode return after evaluation 7.88 +-2.62
```
- `python3 trace_algorithms.py --n=8 --off_policy --vtrace_clip=1`
```
Episode 100, mean 100-episode return -86.03 +-103.23
Episode 200, mean 100-episode return -5.39 +-13.53
Episode 300, mean 100-episode return -0.05 +-8.32
Episode 400, mean 100-episode return 0.49 +-9.04
Episode 500, mean 100-episode return 0.83 +-9.05
Episode 600, mean 100-episode return 0.38 +-9.25
Episode 700, mean 100-episode return 2.90 +-7.25
Episode 800, mean 100-episode return 3.79 +-7.86
Episode 900, mean 100-episode return 1.85 +-8.17
Episode 1000, mean 100-episode return -0.75 +-10.54
The mean 1000-episode return after evaluation -3.56 +-47.91
```
#### Examples End:
