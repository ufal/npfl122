### Assignment: vtrace
#### Date: Deadline: Jan 20, 23:59
#### Points: **compulsory**

Using the [vtrace.py](https://github.com/ufal/npfl122/tree/master/labs/11/vtrace.py)
template, implement the V-trace algorithm.

The template uses the `CartPole-v1` environment and a replay buffer to more
thoroughly test the off-policy capability of the V-trace algorithm.

However, the evaluation in ReCodEx will be performed by calling only the
`vtrace` method and comparing its results to a reference implementation.
Several values of hyperparameters will be used, each test has a time limit
of 1 minute, and all tests must pass.
