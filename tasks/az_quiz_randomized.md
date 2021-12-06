### Assignment: az_quiz_randomized
#### Date: Deadline: Jan 02, 23:59
#### Points: 5 bonus; not required for passing the exam with grade 1 by solving all assignments

Extend the `az_quiz` assignment to handle the possibility of wrong
answers. Therefore, when choosing a field, the agent might answer
incorrectly.

To instantiate this randomized game variant, pass `randomized=True`
to the `AZQuiz` class of [az_quiz.py](https://github.com/ufal/npfl122/tree/master/labs/10/previous_year/az_quiz.py).

The Monte Carlo Tree Search has to be slightly modified to handle stochastic
MDP. The information about distribution of possible next states is provided
by the `AZQuiz.all_moves` method, which returns a list of `(probability,
az_quiz_instance)` next states (in our environment, there are always two
possible next states).
