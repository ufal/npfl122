### Assignment: az_quiz
#### Date: Deadline: Jan 02, 23:59 (competition); Feb 13, 23:49 (regular points)
#### Points: 10 points + 10 bonus

In this competition assignment, use Monte Carlo Tree Search to learn
an agent for a simplified version of [AZ-kvíz](https://cs.wikipedia.org/wiki/AZ-kv%C3%ADz).
In our version, the agent does not have to answer questions and we assume
that **all answers are correct**.

The game itself is implemented in the
[az_quiz.py](https://github.com/ufal/npfl122/tree/master/labs/10/az_quiz.py)
module, using `randomized=False` constructor argument.

The evaluation in ReCodEx should be implemented by returning an object
implementing a method `play`, which given an AZ-kvíz instance returns the chosen
move. The illustration of the interface is in the
[az_quiz_player_random.py](https://github.com/ufal/npfl122/tree/master/labs/10/az_quiz_player_random.py)
module, which implements a random agent.

Your solution in ReCodEx is automatically evaluated against a very simple heuristic
[az_quiz_player_simple_heuristic.py](https://github.com/ufal/npfl122/tree/master/labs/10/az_quiz_player_simple_heuristic.py),
playing 56 games as a starting player and 56 games as a non-starting player. The
time limit for the games is 10 minutes and you should see the win rate directly in
ReCodEx. If you achieve at least 80%, you will pass the assignment.

The final competition evaluation will be performed after the deadline by
a round-robin tournament. In this tournament, we also consider games
where the first move is chosen for the first player (`FirstChosen` label
in ReCodEx, `--first_chosen` option of the evaluator).

The [az_quiz_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/10/az_quiz_evaluator.py)
can be used to evaluate any two given implementations and there are two
interactive players available,
[az_quiz_player_interactive_mouse.py](https://github.com/ufal/npfl122/tree/master/labs/10/az_quiz_player_interactive_mouse.py)
and [az_quiz_player_interactive_keyboard.py](https://github.com/ufal/npfl122/tree/master/labs/10/az_quiz_player_interactive_keyboard.py).

The starting template is available in the [az_quiz_agent.py](https://github.com/ufal/npfl122/tree/master/labs/10/az_quiz_agent.py)
module. Additionally, you might also want to read the
[official pseudocode for AlphaZero](https://www.science.org/doi/suppl/10.1126/science.aar6404/suppl_file/aar6404_datas1.zip),
which however contains several errors:
- Below line 215, the following line should be inserted
  ```python
  root.visit_count = 1
  ```
  Otherwise the `visit_count` is 0, `ucb_score` will return all zeros for all
  actions and during the first simulation, the last valid action in the root
  will always be chosen.
- On line 237, next action should be sampled according to a distribution
  of normalized visit counts, not according to a _softmax_ of visit counts.
- On line 258, the value of a child should be inverted, if the player to play in
  the current node is the other one than in the child (which is almost always
  true). If the assume the values are in $[-1, 1]$ range, the fixed line should be
  ```python
  value_score = - child.value()
  ```
- On line 279, a value is inverted using `1 - value`; however, for values in
  $[-1, 1]$, it should be inverted as `- value`.
- Below line 287, the sampled gamma random variables should be normalized
  to produce a Dirichlet random sample:
  ```python
  noise /= np.sum(noise)
  ```
