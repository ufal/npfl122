### Assignment: policy_iteration
#### Date: Deadline: Sun 21, 23:59
#### Points: **compulsory**

Consider the following gridworld:

<img src="https://raw.githubusercontent.com/ufal/npfl122/master/tasks/policy_iteration.svg?sanitize=true" style="width: 100%">

Start with [policy_iteration.py](https://github.com/ufal/npfl122/tree/master/labs/02/policy_iteration.py),
which implements the gridworld mechanics, by providing the following methods:
- `GridWorld.states()`: return number of states (`11`)
- `GridWorld.actions()`: return lists with labels of the actions (`["↑", "→", "↓", "←"]`)
- `GridWorld.state(state, action)`: return possible outcomes of performing the
  `action` in a given `state`, as a list of triples containing
  - `probability`: probability of the outcome
  - `reward`: reward of the outcome
  - `new_state`: new state of the outcome

Implement policy iteration algorithm, with `--steps` steps of policy
evaluation/policy improvement. During policy evaluation, use the current value
function and perform `--iterations` applications of the Bellman equation.
Perform the policy evaluation synchronously (i.e., do not overwrite the current
value function when computing its improvement).

After given number of iterations, print the resulting value function
and resulting policy.
