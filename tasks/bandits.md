### Assignment: bandits
#### Date: Deadline: Oct 20, 23:59
#### Points: 4 points

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
