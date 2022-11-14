### Assignment: paac
#### Date: Deadline: Nov 28, 7:59 a.m.
#### Points: 4 points

Solve the [CartPole-v1 environment](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)
using parallel actor-critic algorithm, employing the vectorized
environment described in the `car_racing` assignment.

Your goal is to reach an average return of 450 during 100 evaluation episodes.

Start with the [paac.py](https://github.com/ufal/npfl122/tree/master/labs/07/paac.py)
template, which provides a simple network implementation in TensorFlow. Feel
free to use PyTorch or JAX instead, if you like.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 10 minutes.
