### Assignment: az_quiz
#### Date: Deadline: Jan 06, 23:59
#### Points: **10** bonus only

In this bonus-only exercise, use Monte Carlo Tree Search to learn
an agent for a simplified version of [AZ-kvíz](https://cs.wikipedia.org/wiki/AZ-kv%C3%ADz).
In our version, the agent does not have to answer questions and we assume
that **all answers are correct**.

The game itself is implemented in the
[az_quiz.py](https://github.com/ufal/npfl122/tree/master/labs/10/az_quiz.py)
module, using `randomized=False` constructor argument.

The evaluation in ReCodEx will be implemented by utilizing an interface
described in
[az_quiz_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/10/az_quiz_evaluator.py).

For inspiration, use the [official pseudocode for AlphaZero](http://science.sciencemag.org/highwire/filestream/719481/field_highwire_adjunct_files/1/aar6404_DataS1.zip).
