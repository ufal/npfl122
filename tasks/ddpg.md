### Assignment: ddpg
#### Date: Deadline: Nov 28, 7:59 a.m.
#### Points: 6 points

Solve the continuous [Pendulum-v1 environment](https://www.gymlibrary.dev/environments/classic_control/pendulum/)
using deep deterministic policy gradient algorithm.

Your goal is to reach an average return of -200 during 100 evaluation episodes.

Start with the [ddpg.py](https://github.com/ufal/npfl122/tree/master/labs/07/ddpg.py)
template, which provides a simple network implementation in TensorFlow. Feel
free to use PyTorch instead, if you like.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 10 minutes.
