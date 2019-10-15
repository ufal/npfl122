### Assignment: policy_iteration
#### Date: Deadline: Nov 03, 23:59
#### Points: 5 points

Consider the following gridworld:

![gridworld example](https://raw.githubusercontent.com/ufal/npfl122/master/tasks/policy_iteration.svg?sanitize=true)

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

After given number of steps and iterations, print the resulting value function
and resulting policy. For example, the output after 4 steps and 4 iterations
should be:
```
    9.15→   10.30→   11.32→   12.33↑
    8.12↑             3.35←    2.58←
    6.95↑    5.90←    4.66←   -4.93↓
```
