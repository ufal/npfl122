### Assignment: reinforce
#### Date: Deadline: Dec 08, 23:59
#### Points: 5 points

Solve the [CartPole-v1 environment](https://gym.openai.com/envs/CartPole-v1)
environment from the [OpenAI Gym](https://gym.openai.com/) using the REINFORCE
algorithm.

The supplied [cart_pole_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/07/cart_pole_evaluator.py)
module (depending on [gym_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/07/gym_evaluator.py))
can create a continuous environment using `environment(discrete=False)`.
The continuous environment is very similar to the discrete environment, except
that the states are vectors of real-valued observations with shape `environment.state_shape`.

Your goal is to reach an average return of 490 during 100 evaluation episodes.

You can start with the [reinforce.py](https://github.com/ufal/npfl122/tree/master/labs/07/reinforce.py)
template, which provides a simple network implementation in TensorFlow.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 5 minutes.
