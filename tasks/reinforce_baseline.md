### Assignment: reinforce_baseline
#### Date: Deadline: Nov 21, 23:59
#### Points: 4 points

This is a continuation of `reinforce` assignment.

Using the [reinforce_baseline.py](https://github.com/ufal/npfl122/tree/master/labs/06/reinforce_baseline.py)
template, solve the [CartPole-v1 environment](https://gym.openai.com/envs/CartPole-v1)
environment using the REINFORCE with baseline algorithm.

Using a baseline lowers the variance of the value function gradient estimator,
which allows faster training and decreases sensitivity to hyperparameter values.
To reflect this effect in ReCodEx, note that the evaluation phase will
_automatically start after 200 episodes_. Using only 200 episodes for training
in this setting is probably too little for the REINFORCE algorithm, but
suffices for the variant with a baseline. **In this assignment, you must train
your agent in ReCodEx using the provided environment only.**

Your goal is to reach an average return of 490 during 100 evaluation episodes.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 5 minutes.
