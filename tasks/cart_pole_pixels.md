### Assignment: cart_pole_pixels
#### Date: Deadline: Nov 24, 23:59
#### Points: 5 points + 6 bonus

The supplied [cart_pole_pixels_environment.py](https://github.com/ufal/npfl122/tree/master/labs/06/cart_pole_pixels_environment.py)
generates a pixel representation of the `CartPole` environment
as an $80×80$ image with three channels, with each channel representing one time step
(i.e., the current observation and the two previous ones).

In ReCodEx, you are expected to submit an already trained model,
which is evaluated on 15 different tracks with a total time
limit of 15 minutes. If your average return is at least 300, you obtain
8 points. The task is also a _competition_ and at most 10 points will be awarded
according to relative ordering of your solution performances.

To pass the compulsory part of the assignment, you need to reach an average
return of 300 during 100 evaluation episodes.

During evaluation in ReCodEx, three different random seeds will be employed,
each with time limit of 10 minutes, and if you reach an average return at least
300 on all of them, you obtain 5 points. The task is also a _competition_ and
at most 6 points will be awarded according to relative ordering of your
solution performances.

The [cart_pole_pixels.py](https://github.com/ufal/npfl122/tree/master/labs/06/cart_pole_pixels.py)
template parses several parameters and creates the environment.
You are again supposed to train the model beforehand and submit
only the trained neural network.
