### Assignment: car_racing
#### Date: Deadline: Dec 2, 23:59
#### Points: **10** bonus only

<span class="text-danger">Nov 27: The evaluator has been returning a reference
to the same numpy array with the state, which could have caused problems if you
did not create a copy (when stacking images or resizing it). It has now been
fixed.</span>

In this bonus-only exercise to play with Deep Q Network and its variants,
try solving the [CarRacing-v0 environment](https://gym.openai.com/envs/CarRacing-v0)
environment from the [OpenAI Gym](https://gym.openai.com/).

Use the supplied [car_racing_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/06/car_racing_evaluator.py)
module (depending on [gym_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/02/gym_evaluator.py)
to interact with the environment. The environment is continuous and states are
RGB images of size $96×96×3$, but you can downsample them even more. The actions
are also continuous and consist of an array with the following three elements:
- `steer` in range [-1, 1]
- `gas` in range [0, 1]
- `brake` in range [0, 1]

Internally you should generate discrete actions and convert them to the required
representation before the `step` call. Good initial action space is to use
9 actions – a Cartesian product of 3 steering actions (left/right/none) and
3 driving actions (gas/brake/none).

<span class="text-success">The environment supports frame skipping without
rendering the skipped frames – the second argument to `env.step` determines
how many time is the given action repeated.</span>

The task is a _competition_ and at most 10 points will be awarded according to
relative ordering of your solution performances. In ReCodEx, your solution
is evaluated on 15 different tracks with a total time limit of 15 minutes.
If your average return is at least 100, ReCodEx shows the solution as correct.

The [car_racing.py](https://github.com/ufal/npfl122/tree/master/labs/06/car_racing.py)
template parses several useful parameters and creates the environment.
Note that the [car_racing_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/06/car_racing_evaluator.py)
can be executed directly and in that case you can drive the car using arrows.
