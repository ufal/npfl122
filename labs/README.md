You should submit the assignments in the [ReCodEx Code
Examiner](https://recodex.mff.cuni.cz/), where they will be either automatically
or manually evaluated (depending on the assignment).
The evaluation is performed using Python 3.6, TensorFlow 1.11.0, NumPy 1.15.2
and OpenAI Gym 0.9.5. For those using PyTorch, CPU version 0.4.1 is available.

You can install TensorFlow and Gym either to user packages using
`pip3 install --user tensorflow==1.11.0 gym==0.9.5 scipy box2d-py atari-py`
(with the last three backages being optinal dependencies of `gym`),
or create a virtual environment using `python3 -m venv VENV_DIR` and then installing
the packages inside it by running
`VENV_DIR/bin/pip3 install tensorflow==1.11.0 gym==0.9.5 scipy box2d-py atari-py`.
On Windows, you can use third-party precompiled versions of
[box2d-py](https://www.lfd.uci.edu/~gohlke/pythonlibs/)
and [atary-py](https://github.com/Kojoley/atari-py/releases).

### Teamwork

Working in teams of size 2 (or at most 3) is encouraged. All members of the team
must submit in ReCodEx individually, but can have exactly the same
sources/models/results. **However, each such solution must explicitly list all
members of the team to allow plagiarism detection using
[this template](https://github.com/ufal/npfl122/tree/master/labs/team_description.py).**
