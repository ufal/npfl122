You should submit the assignments in the [ReCodEx Code
Examiner](https://recodex.mff.cuni.cz/), where they will be either automatically
or manually evaluated (depending on the assignment).
The evaluation is performed using Python 3.6, TensorFlow 1.11.0, NumPy 1.15.2
and OpenAI Gym 0.9.5. For those using PyTorch, CPU version 1.0.0 is available.

You can install TensorFlow and Gym either to user packages using
`pip3 install --user tensorflow==1.11.0 gym==0.9.5 scipy box2d-py atari-py`
(with the last three backages being optinal dependencies of `gym`),
or create a virtual environment using `python3 -m venv VENV_DIR` and then installing
the packages inside it by running
`VENV_DIR/bin/pip3 install tensorflow==1.11.0 gym==0.9.5 scipy box2d-py atari-py`.
On Windows, you can use third-party precompiled versions of
[box2d-py](https://www.lfd.uci.edu/~gohlke/pythonlibs/)
and [atari-py](https://github.com/Kojoley/atari-py/releases).
Note that when your CPU does not support AVX, you need to install TensorFlow 1.5.

### Submitting Data Files to ReCodEx

Even if [ReCodEx](https://recodex.mff.cuni.cz/) allows submitting data files
beside Python sources, the data files are not available during evaluation.
Therefore, in order to submit models, you need to embed them in Python sources.
You can use [the `embed.py` script](https://github.com/ufal/npfl122/blob/master/labs/embed.py),
which compressed and embeds given files and directories into a Python module
providing an `extract()` method.

### Teamwork

Working in teams of size 2 (or at most 3) is encouraged. All members of the team
must submit in ReCodEx individually, but can have exactly the same
sources/models/results. **However, each such solution must explicitly list all
members of the team to allow plagiarism detection using
[this template](https://github.com/ufal/npfl122/tree/master/labs/team_description.py).**
