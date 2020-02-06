### Assignment: az_quiz
#### Date: Deadline: Jan 05, 23:59
#### Points: 10 points + 10 bonus

In this competition assignment, use Monte Carlo Tree Search to learn
an agent for a simplified version of [AZ-kvíz](https://cs.wikipedia.org/wiki/AZ-kv%C3%ADz).
In our version, the agent does not have to answer questions and we assume
that **all answers are correct**.

The game itself is implemented in the
[az_quiz.py](https://github.com/ufal/npfl122/tree/master/labs/09/az_quiz.py)
module, using `randomized=False` constructor argument.

The evaluation in ReCodEx should be implemented by importing a module
`az_quiz_evaluator_recodex` and calling its `evaluate` function. The argument
this functions is an object providing a method `play` which given an AZ-kvíz
instance returns the chosen move. The illustration of the interface is in the
[az_quiz_evaluator_recodex.py](https://github.com/ufal/npfl122/tree/master/labs/09/az_quiz_evaluator_recodex.py)
module, a simple random player implementing the interface is the
[az_quiz_player_random.py](https://github.com/ufal/npfl122/tree/master/labs/09/az_quiz_player_random.py).

Your solution in ReCodEx is automatically evaluated against a very simple heuristic
[az_quiz_player_simple_heuristic.py](https://github.com/ufal/npfl122/tree/master/labs/09/az_quiz_player_simple_heuristic.py),
playing 50 games as a starting player and 50 games as a non-starting player. The
time limit for the games is 15 minutes and you should see the win rate directly in
ReCodEx. If you achieve at least 75%, you will pass the assignment.

The final competition evaluation will be performed after the deadline by
a round-robin tournament.

Note that [az_quiz_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/09/az_quiz_evaluator.py)
can be used to evaluate any two given implementations and there are two
interactive players available, 
[az_quiz_player_interactive_mouse.py](https://github.com/ufal/npfl122/tree/master/labs/09/az_quiz_player_interactive_mouse.py)
and [az_quiz_player_interactive_keyboard.py](https://github.com/ufal/npfl122/tree/master/labs/09/az_quiz_player_interactive_keyboard.py).

For inspiration, use the [official pseudocode for AlphaZero](http://science.sciencemag.org/highwire/filestream/719481/field_highwire_adjunct_files/1/aar6404_DataS1.zip). However, note that there are some errors in it.
- On line 258, value of the children should be inverted, resulting in:
  ```python
  value_score = 1 - child.value()
  ```
- On line 237, next action should be sampled according to a distribution
  of normalized visit counts, not according to a _softmax_ of visit counts.
- Below line 287, the sampled gamma random variables should be normalized
  to produce Dirichlet random sample:
  ```python
  noise /= np.sum(noise)
  ```

**Note that you must not submit `az_quiz.py` nor `az_quiz_evaluator_recodex.py` to ReCodEx.**
