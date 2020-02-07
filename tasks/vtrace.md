### Assignment: vtrace
#### Date: Deadline: Mar 08, 23:59
#### Points: 6 points

Using the [vtrace.py](https://github.com/ufal/npfl122/tree/master/labs/10/vtrace.py)
template, implement the V-trace algorithm.

The evaluation in ReCodEx is performed in two ways:
- your solution is imported as a module and the output of
  `Network.vtrace` is compared to the referential implementation;
- your solution is evaluated on `CartPole-v1` and should achieve an
  average return of 490 (two seeds are tried, each with 10-minute limit).

You can perform the test of your `Network.vtrace` implementation yourself using the
[vtrace_test.py](https://github.com/ufal/npfl122/tree/master/labs/10/vtrace_test.py)
module, which loads reference data from
[vtrace_test.pickle](https://github.com/ufal/npfl122/tree/master/labs/10/vtrace_test.pickle)
and then evaluates `Network.vtrace` implementation from a given module.

**Note that you must not submit `gym_environment.py` to ReCodEx.**
