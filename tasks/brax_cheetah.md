### Assignment: brax_cheetah
#### Date: Deadline: Dec 19, 23:59
#### Points: 4 points; not required for passing the exam with grade 1 by solving all assignments

In this optional exercise, try using the DDPG/TD3/SAC algorithm to solve the
[HalfCheetah](https://gym.openai.com/envs/HalfCheetah-v2/) environment, but
using the [Halfcheetah](https://github.com/google/brax/blob/main/brax/envs/halfcheetah.py)
environment from the [Brax](https://github.com/google/brax) engine.

You will need additional packages for this assignment, namely
`brax==0.0.8 jax==0.2.25 jaxlib==0.1.74 typing-extensions~=3.7.4`, where
the versions are chosen to be compatible with the other course packages.
Unfortunately, the binary packages are available only for Linux and OS X;
the Windows users should use Windows Subsystem for Linux according to the
[JAX installation instructions](https://github.com/google/jax#installation).

The template [brax_cheetah.py](https://github.com/ufal/npfl122/tree/master/labs/09/brax_cheetah.py)
shows how to
- create a single Brax environment
- create a vectorized Brax environment
- render an episode in the Brax HTML visualizer

In ReCodEx, you are expected to submit an already trained model, which is
evaluated with two seeds, each for 100 episodes with a time limit of 10 minutes.
If your average return is at least 5000 on all of them, you obtain 4 bonus points.
