The tasks are evaluated automatically using the [ReCodEx Code
Examiner](https://recodex.mff.cuni.cz/). The evaluation is
performed using Python 3.6, TensorFlow 2.0.0, NumPy 1.17.2
and OpenAI Gym 0.14.0. For those using PyTorch, CPU version
1.2.0 is available.

You can install TensorFlow and Gym either to user packages using
`pip3 install --user tensorflow==2.0.0 gym==0.14.0 scipy box2d-py atari-py`
(with the last three backages being optinal dependencies of `gym`),
or create a virtual environment using `python3 -m venv VENV_DIR` and then installing
the packages inside it by running
`VENV_DIR/bin/pip3 install ...`.
On Windows, you can use third-party precompiled versions of
[box2d-py](https://www.lfd.uci.edu/~gohlke/pythonlibs/).


### Teamwork

Working in teams of size 2 (or at most 3) is encouraged. All members of the team
must submit in ReCodEx individually, but can have exactly the same
sources/models/results. **However, each such solution must explicitly list all
members of the team to allow plagiarism detection using
[this template](https://github.com/ufal/npfl122/tree/master/labs/team_description.py).**


### Submitting Data Files to ReCodEx

Because [ReCodEx](https://recodex.mff.cuni.cz/) allows submitting only Python
sources in our settings, we need to embed models and other non-Python data
into Python sources. You can use
[the `embed.py` script](https://github.com/ufal/npfl122/blob/master/labs/embed.py),
which compresses the given files and directories and embeds them into a Python
module, which extracts them when imported or executed.
