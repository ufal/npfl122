### Assignment: q_network
#### Date: Deadline: Nov 25, 23:59
#### Points: **compulsory**

Solve the [CartPole-v1 environment](https://gym.openai.com/envs/CartPole-v1)
environment from the [OpenAI Gym](https://gym.openai.com/) using Q-learning
with neural network as a function approximation.

The supplied [cart_pole_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/02/cart_pole_evaluator.py)
module (depending on [gym_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/02/gym_evaluator.py))
can also create a continuous environment using `environment(discrete=False)`.
The continuous environment is very similar to the discrete environment, except
that the states are vectors of real-valued observations with shape `environment.state_shape`.

Use Q-learning with neural network as a function approximation, which for
a given states returns state-action values for all actions. You can use any
network architecture, but two hidden layers of 20 ReLU units are a good start.

Your goal is to reach an average return of 400 during 100 evaluation episodes.

You can start with the [q_network.py](https://github.com/ufal/npfl122/tree/master/labs/05/q_network.py)
template, which provides a simple network implementation in TensorFlow.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 10 minutes (so you can train in ReCodEx, but you can also pretrain your
network if you like).
