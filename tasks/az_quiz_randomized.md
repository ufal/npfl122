### Assignment: az_quiz_randomized
#### Date: Deadline: Jan 05, 23:59
#### Points: 10 bonus

Extend the `az_quiz` assignment to handle the possibility of wrong
answers. Therefore, when choosing a field, the agent might answer
incorrectly.

To instantiate this randomized game variant, pass `randomized=True`
to the `AZQuiz` class of [az_quiz.py](https://github.com/ufal/npfl122/tree/past-1819/labs/10/az_quiz.py).

The Monte Carlo Tree Search has to be slightly modified to handle stochastic
MDP. The information about distribution of possible next states is provided
by the `AZQuiz.all_moves` method, which returns a list of `(probability,
az_quiz_instance)` next states (in our environment, there are always two
possible next states).

**Note that you must not submit `az_quiz.py` nor `az_quiz_evaluator_recodex.py` to ReCodEx.**
