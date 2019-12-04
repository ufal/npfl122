### Assignment: cart_pole_pixels
#### Date: Deadline: Dec 08, 23:59
#### Points: 6 points + 6 bonus

The supplied [cart_pole_pixels_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/07/cart_pole_pixels_evaluator.py)
module (depending on [gym_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/02/gym_evaluator.py))
generates a pixel representation of the `CartPole` environment
as an $80×80$ image with three channels, with each channel representing one time step
(i.e., the current observation and the two previous ones).

To pass the compulsory part of the assignment, you need to reach an average
return of 200 during 100 evaluation episodes. During evaluation in ReCodEx, two
different random seeds will be employed, and you need to reach the required
return on all of them. Time limit for each test is 10 minutes.

The task is additionally a _competition_ and at most 5 points will be awarded
according to relative ordering of your solution performances.

The [cart_pole_pixels.py](https://github.com/ufal/npfl122/tree/master/labs/07/cart_pole_pixels.py)
template parses several parameters and creates the environment.
You are again supposed to train the model beforehand and submit
only the trained neural network.
