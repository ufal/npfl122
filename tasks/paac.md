### Assignment: paac
#### Date: Deadline: Dec 16, 23:59
#### Points: **compulsory**

Using the [paac.py](https://github.com/ufal/npfl122/tree/master/labs/08/paac.py)
template, solve the [CartPole-v1 environment](https://gym.openai.com/envs/CartPole-v1)
environment using parallel actor-critic algorithm.

The `gym_environment` now provides the following two methods:
- `parallel_init(num_workers) → initial_states`, which initializes the given
  number of parallel workers and returns their environment initial states.
  This method can be called at most once.
- `parallel_step(actions) → List[next_state, reward, done, info]`, which
  performs given action in respective environment, and return the usual
  information with one exception: **If `done=True`, then `next_state` is
  already a new state of newly started episode.**

Your goal is to reach an average return of 450 during 100 evaluation episodes.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 10 minutes.
