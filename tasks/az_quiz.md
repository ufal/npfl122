### Assignment: az_quiz
#### Date: Deadline: Jan 13, 23:59
#### Points: **10** bonus only

In this bonus-only exercise, use Monte Carlo Tree Search to learn
an agent for a simplified version of [AZ-kvíz](https://cs.wikipedia.org/wiki/AZ-kv%C3%ADz).
In our version, the agent does not have to answer questions and we assume
that **all answers are correct**.

The game itself is implemented in the
[az_quiz.py](https://github.com/ufal/npfl122/tree/master/labs/10/az_quiz.py)
module, using `randomized=False` constructor argument.

The evaluation in ReCodEx should be implemented by importing a module
`az_quiz_evaluator_recodex` and calling its `evaluate` function. The argument
this functions is an object providing a method `play` which given an AZ-kvíz
instance returns the chosen move. The illustration of the interface is in the
[az_quiz_evaluator_recodex.py](https://github.com/ufal/npfl122/tree/master/labs/10/az_quiz_evaluator_recodex.py)
module.

Your solution in ReCodEx is automatically evaluated only against a random player
[az_quiz_player_random.py](https://github.com/ufal/npfl122/tree/master/labs/10/az_quiz_player_random.py)
and a very simple heuristic
[az_quiz_player_simple_heuristic.py](https://github.com/ufal/npfl122/tree/master/labs/10/az_quiz_player_simple_heuristic.py),
playing against each of them 10 games as a starting player and 10 games
as a non-starting player. The time limit for the games is 10 minutes and you
should see win rate directly in ReCodEx. The final evaluation will be
performed after the deadline by a round-robin tournament, utilizing your latest
submission with non-zero win rate.

For inspiration, use the [official pseudocode for AlphaZero](http://science.sciencemag.org/highwire/filestream/719481/field_highwire_adjunct_files/1/aar6404_DataS1.zip).
