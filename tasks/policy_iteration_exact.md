### Assignment: policy_iteration_exact
#### Date: Deadline: Oct 27, 23:59
#### Points: 2 points
#### Examples: policy_iteration_exact_examples

Starting with [policy_iteration_exact.py](https://github.com/ufal/npfl122/tree/master/labs/02/policy_iteration_exact.py),
extend the `policy_iteration` assignment to perform policy evaluation
exactly by solving a system of linear equations.

#### Examples Start: policy_iteration_exact_examples
Note that your results may sometimes be slightly different (for example because of varying floating point arithmetic on your CPU).
- `python3 policy_iteration_exact.py --gamma=0.95 --steps=1`
```
   -0.00→    0.00→    0.00↑    0.00↑
   -0.00↑           -12.35←  -12.35↑
   -0.85←   -8.10←  -19.62← -100.71←
```
- `python3 policy_iteration_exact.py --gamma=0.95 --steps=2`
```
    0.00→    0.00→    0.00→    0.00↑
    0.00↑             0.00←  -11.05←
   -0.00↑   -0.00→    0.00←  -12.10↓
```
- `python3 policy_iteration_exact.py --gamma=0.95 --steps=3`
```
   -0.00↓   -0.00←   -0.00↓   -0.00↑
   -0.00↑             0.00←    0.69←
   -0.00←   -0.00←   -0.00→    6.21↓
```
- `python3 policy_iteration_exact.py --gamma=0.95 --steps=8`
```
   12.12↓   11.37←    9.19←    6.02↑
   13.01↓             9.92←    9.79←
   13.87→   14.89→   15.87→   16.60↓
```
- `python3 policy_iteration_exact.py --gamma=0.95 --steps=9`
```
   12.24↓   11.49←   10.76←    7.05↑
   13.14↓            10.60←   10.42←
   14.01→   15.04→   16.03→   16.71↓
```
- `python3 policy_iteration_exact.py --gamma=0.95 --steps=10`
```
   12.24↓   11.49←   10.76←    7.05↑
   13.14↓            10.60←   10.42←
   14.01→   15.04→   16.03→   16.71↓
```
#### Examples End:
