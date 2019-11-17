### Assignment: multiarmed_bandits
#### Date: Deadline: Oct 20, 23:59
#### Points: 8 points

Perform a parameter study of various approaches to solving multiarmed bandits.
For every hyperparameter choice, perform 100 episodes, each consisting of
1000 trials, and report the average and standard deviation of the 100 episode
returns.

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

Your goal is to implement the following modes of calculation. In addition
to submitting the solution to ReCodEx, you should use
[multiarmed_bandits_draw.py](https://github.com/ufal/npfl122/tree/master/labs/01/multiarmed_bandits_draw.py)
to plots the results in a graph.
- `greedy` _[2 points]_: perform $ε$-greedy search with parameter `epsilon`, computing the
  value function using averaging. (Results for $ε ∈ \{1/64, 1/32, 1/16, 1/8, 1/4\}$ are plotted.)
- `greedy` and `alpha`$≠0$ _[1 point]_: perform $ε$-greedy search with parameter `epsilon` and
  initial function estimate of 0, using fixed learning rate `alpha`. (Results
  for $α=0.15$ and $ε ∈ \{1/64, 1/32, 1/16, 1/8, 1/4\}$ are plotted.)
- `greedy`, `alpha`$≠0$ and `initial`$≠0$ _[1 point]_: perform $ε$-greedy search with
  parameter `epsilon`, given `initial` value as starting value function and
  fixed learning rate `alpha`. (Results for `initial`$=1$, $α=0.15$ and
  $ε ∈ \{1/128, 1/64, 1/32, 1/16\}$ are plotted.)
- `ucb` _[2 points]_: perform UCB search with confidence level `c` and computing the value
  function using averaging. (Results for $c ∈ \{1/4, 1/2, 1, 2, 4\}$ are
  plotted.)
- `gradient` _[2 points]_: choose actions according to softmax distribution, updating the
  parameters using SGD to maximize expected reward. (Results for
  $α ∈ \{1/16, 1/8, 1/4, 1/2\}$ are plotted.)
