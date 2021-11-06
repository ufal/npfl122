### Assignment: car_racing
#### Date: Deadline: Nov 14, 23:59
#### Points: 8 points + 8 bonus

The goal of this competition is to use Deep Q Networks (and any of Rainbow improvements)
on a more real-world [CarRacing-v0 environment](https://gym.openai.com/envs/CarRacing-v0)
from the [OpenAI Gym](https://gym.openai.com/).

The supplied [car_racing_environment.py](https://github.com/ufal/npfl122/tree/master/labs/05/car_racing_environment.py)
provides the environment. It is continuous and states are RGB images of size
$96×96×3$, but you can downsample them even more. The actions
are also continuous and consist of an array with the following three elements:
- `steer` in range [-1, 1]
- `gas` in range [0, 1]
- `brake` in range [0, 1]; note that full brake is quite aggressive, so you
  might consider using less force when braking

Internally you should probably generate discrete actions and convert them to the
required representation before the `step` call. The smallest set is probably
left, right, gas, brake and no-op, but you can use a more fine-grained one if
you like.

The environment also support frame skipping, which improves its performance (only
some frames need to be rendered).

In ReCodEx, you are expected to submit an already trained model,
which is evaluated on 15 different tracks with a total time
limit of 15 minutes. If your average return is at least 300, you obtain
8 points. The task is also a [_competition_](https://ufal.mff.cuni.cz/courses/npfl122/2122-winter#competitions)
and at most 8 points will be awarded according to relative ordering of your
solution performances.

The [car_racing.py](https://github.com/ufal/npfl122/tree/master/labs/05/car_racing.py)
template parses several useful parameters and creates the environment.
Note that the [car_racing_environment.py](https://github.com/ufal/npfl122/tree/master/labs/05/car_racing_environment.py)
can be executed directly and in that case you can drive the car using arrows.

Also, you might want to use a **vectorized version of the environment** for
training, which runs several individual environments in separate processes.
The template contains instructions how to create it. The vectorized environment
expects a vector of actions and returns a vector of observations, rewards, dones
and infos. When one of the environments is `done`, **it is immediately reset** and
`state` is the initial state of a new episode.
