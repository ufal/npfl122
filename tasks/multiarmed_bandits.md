### Assignment: multiarmed_bandits
#### Date: Deadline: Oct 21, 23:59
#### Points: **compulsory**

Perform a parameter study of various approaches to solving multiarmed bandits.
For every hyperparameter choice, perform 1000 episodes, each consisting of
1000 trials, and report averaged return (a single number).

Start with the [multiarmed_bandits.py](https://github.com/ufal/npfl122/tree/master/labs/01/multiarmed_bandits.py)
template, which defines `MultiArmedBandits` environment. We use API based on
[OpenAI Gym](https://gym.openai.com/) `Environment` class, notably the following
two methods:
- `reset() → new_state`: starts a new episode
- `step(action) → new_state, reward, done, info`: perform the chosen action
  in the environment, returning the new state, obtained reward, a boolean
  flag indicating an end of episode, and additional environment-specific
  information
Of course, the states are not used by the multiarmed bandits (`None` is
returned).

Your goal is to implement the following modes of calculation. For each mode
evaluate the performance given specified hyperparameters and plot the results
for all modes together in a single graph.
- `greedy`: perform $ε$-greedy search with parameter `epsilon`, computing the
  value function using averaging. Plot results for
  $ε ∈ \{1/64, 1/32, 1/16, 1/8, 1/4\}$.
- `greedy` and `alpha`$≠0$: perform $ε$-greedy search with parameter `epsilon` and
  initial function estimate of 0, using fixed learning rate `alpha`. Plot
  results for $α=0.15$ and $ε ∈ \{1/64, 1/32, 1/16, 1/8, 1/4\}$
- `greedy`, `alpha`$≠0$ and `initial`$≠0$: perform $ε$-greedy search with
  parameter `epsilon`, given `initial` value as starting value function and
  fixed learning rate `alpha`. Plot results for `initial`$=1$, $α=0.15$ and
  $ε ∈ \{1/128, 1/64, 1/32, 1/16\}$.
- `ucb`: perform UCB search with confidence level `c` and computing the value
  function using averaging. Plot results for $c ∈ \{1/4, 1/2, 1, 2, 4\}$.
- `gradient`: choose actions according to softmax distribution, updating the
  parameters using SGD to maximize expected reward. Plot results for
  $α ∈ \{1/16, 1/8, 1/4, 1/2\}$.

This task will be evaluated manually and you should submit the Python source and
the generated graph.
