- **General RL Settings, Value Iteration**
  Define reinforcement learning as a Markov Decision Process, define
  a policy, value function and an action-value function (and show
  how value and action-value functions can be computed from one another).
  Then, define optimal value and action-value function, and show, how
  optimal value function can be computed using Bellman backup operator,
  i.e., the value iteration algorithm (including a proof of convergence).

- **General RL Settings, Policy Iteration**
  Define reinforcement learning as a Markov Decision Process, define
  a policy, value function and an action-value function (and show
  how value and action-value functions can be computed from one another).
  Then, define optimal value and action-value function, and show, how
  optimal policy can be computed using policy iteration algorithm
  (ideally including proofs).

- **TD Methods**
  Describe temporal difference methods and formulate Sarsa, Q-learning,
  Expected Sarsa, Double Q-learning and $n$-step Sarsa in tabular settings.

- **Off-policy Methods**
  Describe difference between on-policy and off-policy methods, and show
  an off-policy variant of Monte Carlo algorithm in tabular settings,
  both with ordinary and weighted importance sampling. Then describe
  Expected Sarsa as an off-policy algorithm not using importance sampling.
  Finally, describe off-policy $n$-step Sarsa.

- **Function Approximation**
  Assuming function approximation, define the usual mean squared value error
  and describe gradient Monte Carlo and Semi-gradient TD algorithms. Then show
  how off-policy methods in function approximation settings may diverge.
  Finally, sketch Deep Q Network architecture, especially the experience replay
  and target network.

- **DQN**
  Describe Deep Q Network, and especially experience replay, target network and
  reward clipping tricks. Then, describe at least three improvements present in
  Rainbow algorithm (i.e., something of DDQN, prioritized replay, duelling
  architecture, noisy nets and distributional RL).

- **Policy Gradient Methods, REINFORCE**
  Describe policy gradient methods, prove policy gradient theorem, and describe
  REINFORCE, REINFORCE with baseline (including the proof of the baseline).

- **Policy Gradient Methods, PAAC**
  Describe policy gradient methods, prove policy gradient theorem, and describe
  Parallel advantage actor-critic (PAAC) algorithm.

- **Gradient Methods with Continuous Actions, DDPG**
  Show how continuous actions can be incorporated in policy gradient algorithms
  (i.e., in a REINFORCE algorithm, without proving the policy gradient theorem).
  Then formulate and prove deterministic policy gradient theorem. Finally,
  describe the DDPG algorithm.

- **Gradient Methods with Continuous Actions, TD3**
  Formulate and prove deterministic policy gradient theorem. Then, describe the
  DDPG and TD3 algorithms.

- **AlphaZero**
  Describe the algorithm used by AlphaZero – Monte Carlo tree search, overall
  neural network architecture, and training and inference procedures.

- **V-trace and PopArt Normalization**
  Describe the V-trace algorithm and sketch population-based training used in the
  IMPALA algorithm. Then, show how the rewards can be normalized using the PopArt
  normalization approach.
