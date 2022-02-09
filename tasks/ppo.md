### Assignment: ppo
#### Date: Deadline: Feb 27, 23:49
#### Points: 3 points; not required for passing the exam with grade 1 by solving all assignments

Implement the PPO algorithm in a single-agent settings. Notably, solve
the `SingleCollect` environment implemented by the
[multi_collect_environment.py](https://github.com/ufal/npfl122/tree/master/labs/13/multi_collect_environment.py)
module. To familiarize with it, you can [watch a trained agent](https://ufal.mff.cuni.cz/~straka/courses/npfl122/2122/videos/single_collect.mp4)
and you can run the module directly, controlling the agent with the arrow keys.
In the environment, your goal is to reach a known place, obtaining rewards
based on the agent's distance. If the agent is continuously occupying the place
for some period of time, it gets a large reward and the place is moved randomly.
The environment runs for 250 steps and it is considered solved if you obtain
a return of at least 500.

The [ppo.py](https://github.com/ufal/npfl122/tree/master/labs/13/ppo.py)
template contains a skeleton implementation of the PPO algorithm.
Regarding the unspecified hyperparameters, I would consider the following ranges:
- `batch_size` between 64 and 512
- `clip_epsilon` between 0.1 and 0.2
- `epochs` between 1 and 10
- `gamma` between 0.97 and 1.0
- `trace_lambda` is usually 0.95
- `workers` between 16 and 128
- `worker_steps` between tens and hundreds

My implementation trains in approximately a minute on a dual-core CPU.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the average return of 450 on all of them. Time limit for each test
is 10 minutes.
