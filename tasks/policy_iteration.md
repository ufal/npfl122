### Assignment: policy_iteration
#### Date: Deadline: Oct 27, 23:59
#### Points: 4 points
#### Examples: policy_iteration_examples

Consider the following gridworld:

![Gridworld example](//ufal.mff.cuni.cz/~straka/courses/npfl122/2021/tasks/figures/policy_iteration.svgz)

Start with [policy_iteration.py](https://github.com/ufal/npfl122/tree/master/labs/02/policy_iteration.py),
which implements the gridworld mechanics, by providing the following methods:
- `GridWorld.states`: return number of states (`11`)
- `GridWorld.actions`: return lists with labels of the actions (`["↑", "→", "↓", "←"]`)
- `GridWorld.step(state, action)`: return possible outcomes of performing the
  `action` in a given `state`, as a list of triples containing
  - `probability`: probability of the outcome
  - `reward`: reward of the outcome
  - `new_state`: new state of the outcome

Implement policy iteration algorithm, with `--steps` steps of policy
evaluation/policy improvement. During policy evaluation, use the current value
function and perform `--iterations` applications of the Bellman equation.
Perform the policy evaluation synchronously (i.e., do not overwrite the current
value function when computing its improvement). Assume the initial policy is
“go North” and initial value function is zero.

#### Examples Start: policy_iteration_examples
Note that your results may sometimes be slightly different (for example because of varying floating point arithmetic on your CPU).
- `python3 policy_iteration.py --gamma=0.95 --iterations=1 --steps=1`
```
    0.00↑    0.00↑    0.00↑    0.00↑
    0.00↑           -10.00←  -10.00↑
    0.00↑    0.00→    0.10←  -79.90←
```
- `python3 policy_iteration.py --gamma=0.95 --iterations=1 --steps=2`
```
    0.00↑    0.00↑    0.00↑    0.00↑
    0.00↑            -7.59←  -11.90←
    0.00→    0.08←   -0.94←  -18.36←
```
- `python3 policy_iteration.py --gamma=0.95 --iterations=1 --steps=3`
```
    0.00↑    0.00↑    0.00↑    0.00↑
    0.00↓            -5.86←   -7.41←
    0.06↓    0.01←   -0.75←  -13.49↓
```
- `python3 policy_iteration.py --gamma=0.95 --iterations=1 --steps=10`
```
    0.04↓    0.04←    0.01↑    0.00↑
    0.04↓            -0.95←   -1.00←
    0.04↓    0.04←   -0.10→   -0.52↓
```
- `python3 policy_iteration.py --gamma=0.95 --iterations=10 --steps=10`
```
   11.79↓   11.03←   10.31←    6.54↑
   12.69↓            10.14←    9.95←
   13.56→   14.59→   15.58→   16.26↓
```
- `python3 policy_iteration.py --gamma=1 --iterations=1 --steps=100`
```
   66.54↓   65.53←   64.42←   56.34↑
   67.68↓            63.58←   62.97←
   68.69→   69.83→   70.84→   71.75↓
```
#### Examples End:
