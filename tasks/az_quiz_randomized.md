### Assignment: az_quiz_randomized
#### Date: Deadline: Jan 06, 23:59
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

The evaluation in ReCodEx will be implemented by utilizing an interface
described in
[az_quiz_evaluator.py](https://github.com/ufal/npfl122/tree/master/labs/10/az_quiz_evaluator.py).

For inspiration, use the [official pseudocode for AlphaZero](http://science.sciencemag.org/highwire/filestream/719481/field_highwire_adjunct_files/1/aar6404_DataS1.zip).
