### Assignment: az_quiz_randomized
#### Date: Deadline: Jan 13, 23:59
#### Points: **5** bonus only

Extend the `az_quiz` assignment to handle the possibility of wrong
answers. Therefore, when choosing a field, the agent might answer
incorrectly.

To instantiate this randomized game variant, pass `randomized=True`
to the `AZQuiz` class of [az_quiz.py](https://github.com/ufal/npfl122/tree/master/labs/10/az_quiz.py).

The Monte Carlo Tree Search has to be slightly modified to handle stochastic
MDP. The information about distribution of possible next states is provided
by the `AZQuiz.all_moves` method, which returns a list of `(probability,
az_quiz_instance)` next states (in our environment, there are always two
possible next states).

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
