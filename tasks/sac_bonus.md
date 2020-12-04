### Assignment: sac_bonus
#### Date: Deadline: Feb 14, 23:59
#### Points: 8 bonus

In this bonus-only exercise, try using the SAC algorithm to solve the
[Hopper](https://gym.openai.com/envs/Hopper-v2/) environment, but using
`HopperBulletEnv-v0` from open-source [PyBullet framework](https://github.com/bulletphysics/bullet3).
Basic information about Gym interface is in
[PyBullet Quickstart Guide](https://usermanual.wiki/Document/pybullet20quickstart20guide.479068914/view#48);
generally you just need to import `pybullet_envs`, which will register the
Gym environments.

You can install PyBullet by `pip3 install pybullet`. However, precompiled
binaried are available for Linux only, on Windows the library must be compiled
by a suitable Microsoft Visual C++ compiler.

In ReCodEx, you are expected to submit an already trained model, which is
evaluated on the `HopperBulletEnv-v0` environment with two seeds, each for 100
episodes with a time limit of 15 minutes. If your average return is at
least 1000, you obtain 6 bonus points.

A template for the SAC algorithm will be made available.
