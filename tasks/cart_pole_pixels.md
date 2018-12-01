### Assignment: cart_pole_pixels
#### Date: Deadline: Dec 09, 23:59
#### Points: **compulsory** & **7 bonus**

The supplied [cart_pole_pixels_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/07/cart_pole_pixels_evaluator.py)
module (depending on [gym_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/02/gym_evaluator.py))
generates a pixel representation of the `CartPole` environment
as an $80×80$ image with three channels, with each channel representing one time step
(i.e., the current observation and the two previous ones).

To pass the compulsory part of the assignment, you need to reach an average
return of 50 during 100 evaluation episodes. During evaluation in ReCodEx, two
different random seeds will be employed, and you need to reach the required
return on all of them. Time limit for each test is 10 minutes.

The task is additionally a _competition_ and at most 7 points will be awarded
according to relative ordering of your solution performances.

The [cart_pole_pixels.py](https://github.com/ufal/npfl122/tree/master/labs/07/cart_pole_pixels.py)
template parses several parameters, creates the environment
and shows how to save and load neural networks in TensorFlow.
To upload the trained model to ReCodEx, you need to embed the
trained model files using [embed.py](https://github.com/ufal/npfl122/blob/master/labs/embed.py),
submit the resulting `embedded_data.py` along your solution, and
in your solution you need to `import embedded_data` and then
`embedded_data.extract()` (the template does this for you).
