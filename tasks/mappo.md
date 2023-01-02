### Assignment: mappo
#### Date: Deadline: Feb 12, 23:59
#### Points: 4 points

**The template and the environment will be available soon.**

Implement MAPPO in a multi-agent settings. Notably, solve a multi-agent
extension of `SingleCollect` with 2 agents, the `MultiCollect2` environment
implemented again by the `multi_collect_environment.py`
module (you can [watch the trained agents](https://ufal.mff.cuni.cz/~straka/courses/npfl122/2223/videos/multi_collect.mp4)).
The environment is a generalization of `SingleCollect`. If there are
$A$ agents, there are also $A$ target places, and each place rewards
the closest agent. Additionally, any agent colliding with others gets
a negative reward, and the environment reward is the average of the agents'
rewards (to keep the rewards less dependent on the number of agents).
Again, the environment runs for 250 steps and is considered solved
when reaching return of at least 500.

The `mappo.py`
template contains a skeleton of the MAPPO algorithm implementation.
I use hyperparameter values quite similar to the `ppo` assignment, with
a notable exception of a smaller `learning_rate=3e-4`, which is already
specified in the template.

My implementation successfully converges in only circa 50% of the cases,
and trains in roughly 10-20 minutes.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the average return of 450 on all of them. Time limit for each test
is 10 minutes.
