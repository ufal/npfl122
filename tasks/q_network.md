### Assignment: q_network
#### Date: Deadline: Nov 07, 23:59
#### Points: 5 points

Solve the [CartPole-v1 environment](https://gym.openai.com/envs/CartPole-v1)
environment from the [OpenAI Gym](https://gym.openai.com/) using Q-learning
with neural network as a function approximation.

You can start with the [q_network.py](https://github.com/ufal/npfl122/tree/master/labs/04/q_network.py)
template, which provides a simple network implementation in TensorFlow. Feel
free to use PyTorch instead, if you like.

The continuous environment is very similar to a discrete one, except
that the states are vectors of real-valued observations with shape
`env.observation_space.shape`.

Use Q-learning with neural network as a function approximation, which for
a given state returns state-action values for all actions. You can use any
network architecture, but one hidden layer of several dozens ReLU units is a good start.
Your goal is to reach an average return of 450 during 100 evaluation episodes.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 10 minutes (so you can train in ReCodEx, but you can also pretrain your
network if you like).
