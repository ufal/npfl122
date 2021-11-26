### Assignment: ddpg
#### Date: Deadline: ~~Nov 28~~ Dec 05, 23:59
#### Points: 6 points

Solve the [Pendulum-v0 environment](https://gym.openai.com/envs/Pendulum-v0)
environment using deep deterministic policy gradient algorithm.
The environment is continuous, states and actions are described at
[OpenAI Gym Wiki](https://github.com/openai/gym/wiki/Pendulum-v0).

Your goal is to reach an average return of -200 during 100 evaluation episodes.

Start with the [ddpg.py](https://github.com/ufal/npfl122/tree/master/labs/07/ddpg.py)
template, which provides a simple network implementation in TensorFlow. Feel
free to use PyTorch instead, if you like.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 10 minutes.
