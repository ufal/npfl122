### Assignment: car_racing
#### Date: Deadline: Dec 01, 23:59
#### Points: 8 points + 10 bonus

The goal of this competition is to use Deep Q Networks and its improvements
on a more real-world  [CarRacing-v0 environment](https://gym.openai.com/envs/CarRacing-v0)
environment from the [OpenAI Gym](https://gym.openai.com/).

Use the supplied [car_racing_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/06/car_racing_evaluator.py)
module (depending on [gym_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/06/gym_evaluator.py)
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

<div class="text-success">

Nov 22: The frame skipping support was changed. The
evironment supports frame skipping without rendering the skipped frames, by
passing `frame_skip` parameter to `car_racing_evaluator.environment(frame_skip=1)`
method – the value of `frame_skip` determines how many times is an action
repeated.

Nov 19: The environment also supports parallel
execution (use multiple CPU threads to simulate several environments in parallel
during training), by providing the following two methods:
- `parallel_init(num_workers) → initial_states`, which initializes the given
  number of parallel workers and returns their environment initial states.
  This method can be called at most once.
- `parallel_step(actions) → List[next_state, reward, done, info]`, which
  performs given action in respective environment, and return the usual
  information with one exception: **If `done=True`, then `next_state` is
  already an initial state of newly started episode.**

</div>

In ReCodEx, you are expected to submit an already trained model,
which is evaluated on 15 different tracks with a total time
limit of 15 minutes. If your average return is at least 200, you obtain
8 points. The task is also a _competition_ and at most 10 points will be awarded
according to relative ordering of your solution performances.

The [car_racing.py](https://github.com/ufal/npfl122/tree/master/labs/06/car_racing.py)
template parses several useful parameters and creates the environment.
Note that the [car_racing_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/06/car_racing_evaluator.py)
can be executed directly and in that case you can drive the car using arrows.
