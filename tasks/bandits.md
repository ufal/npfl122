### Assignment: bandits
#### Date: Deadline: Oct 17, 23:59
#### Points: 3 points
#### Tests: bandits_tests

Implement the $ε$-greedy strategy for solving multi-armed bandits.

Start with the [bandits.py](https://github.com/ufal/npfl122/tree/master/labs/01/bandits.py)
template, which defines `MultiArmedBandits` environment, which has the following
two methods:
- `reset()`: reset the environment
- `step(action) → reward`: perform the chosen action in the environment,
  obtaining a reward
- `greedy(epsilon)`: return `True` with probability 1-`epsilon`

Your goal is to implement the following solution variants:
- `alpha`$=0$: perform $ε$-greedy search, updating the estimated using
  averaging.
- `alpha`$≠0$: perform $ε$-greedy search, updating the estimated using
  a fixed learning rate `alpha`.

Note that the initial estimates should be set to a given value and `epsilon` can
be zero, in which case purely greedy actions are used.

#### Tests Start: bandits_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 bandits.py --alpha=0 --epsilon=0.1 --initial=0`
```
1.39 0.08
```
- `python3 bandits.py --alpha=0 --epsilon=0 --initial=1`
```
1.48 0.22
```
- `python3 bandits.py --alpha=0.15 --epsilon=0.1 --initial=0`
```
1.37 0.09
```
- `python3 bandits.py --alpha=0.15 --epsilon=0 --initial=1`
```
1.52 0.04
```
#### Tests End:
