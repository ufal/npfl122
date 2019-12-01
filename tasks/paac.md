### Assignment: paac
#### Date: Deadline: Dec 15, 23:59
#### Points: 5 points

Using the [paac.py](https://github.com/ufal/npfl122/tree/master/labs/08/paac.py)
template, solve the [CartPole-v1 environment](https://gym.openai.com/envs/CartPole-v1)
environment using parallel actor-critic algorithm. Use the `parallel_init`
and `parallel_step` methods described in `car_racing` assignment.

Your goal is to reach an average return of 450 during 100 evaluation episodes.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 10 minutes.
