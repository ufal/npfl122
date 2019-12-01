### Assignment: ddpg
#### Date: Deadline: Dec 15, 23:59
#### Points: 7 points

Using the [ddpg.py](https://github.com/ufal/npfl122/tree/master/labs/08/ddpg.py)
template, solve the [Pendulum-v0 environment](https://gym.openai.com/envs/Pendulum-v0)
environment using deep deterministic policy gradient algorithm.

To create the evaluator, use
[gym_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/08/gym_evaluator.py)`.GymEvaluator("Pendulum-v0")`.
The environment is continuous, states and actions are described at
[OpenAI Gym Wiki](https://github.com/openai/gym/wiki/Pendulum-v0).

Your goal is to reach an average return of -200 during 100 evaluation episodes.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 10 minutes.
