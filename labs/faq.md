### TOC: FAQ

### TOCEntry: Install

- _Installing to central user packages repository_

  You can install all required packages to central user packages repository using
 `pip3 install --user tensorflow==2.10.0 tensorflow_probability==0.18.0 numpy==1.23.3 gym==0.26.0 pygame==2.1.2 mujoco==2.2.2 ufal.pybox2d==2.3.10.2`.

- _Installing to a virtual environment_

  Python supports virtual environments, which are directories containing
  independent sets of installed packages. You can create a virtual environment
  by running `python3 -m venv VENV_DIR` followed by
  `VENV_DIR/bin/pip3 install tensorflow==2.10.0 tensorflow_probability==0.18.0 numpy==1.23.3 gym==0.26.0 pygame==2.1.2 mujoco==2.2.2 ufal.pybox2d==2.3.10.2`.
  (or `VENV_DIR/Scripts/pip3` on Windows).

- _**Windows** installation_

  - On Windows, it can happen that `python3` is not in PATH, while `py` command
    is – in that case you can use `py -m venv VENV_DIR`, which uses the newest
    Python available, or for example `py -3.9 -m venv VENV_DIR`, which uses
    Python version 3.9.

  - If your Windows TensorFlow fails with `ImportError: DLL load failed`,
    you are probably missing
    [Visual C++ 2019 Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe).

  - If you encounter a problem creating the logs in the `args.logdir` directory,
    a possible cause is that the path is longer than 260 characters, which is
    the default maximum length of a complete path on Windows. However, you can
    increase this limit on Windows 10, version 1607 or later, by following
    the [instructions](https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation).

- _**macOS** installation_

  - With an **Intel** processor, you should not need anything special.

  - If you have **Apple Silicon**, use the package `tensorflow-macos` instead of
    `tensorflow`. As of Sep 11, the dependency package `grpcio` needs to be
    compiled during the installation (automatically, but you need working Xcode);
    the installation worked fine on my testing macOS. Furthermore, according to
    [this issue](https://github.com/grpc/grpc/issues/29262), a binary wheel for
    `grpcio` could be provided soon.

- _**GPU** support on Linux and Windows_

  TensorFlow 2.10 supports NVIDIA GPU out of the box, but you need to install
  CUDA 11.2 and cuDNN 8.1 libraries yourself.

- _**GPU** support on macOS_

  The AMD and Apple Silicon GPUs can be used by installing a plugin providing
  the GPU acceleration using:
  ```
  python -m pip install tensorflow-metal
  ```

- _Errors when running with a GPU_

  If you encounter errors when running with a GPU:
  - if you are using the GPU also for displaying, try using the following
    environment variable: `export TF_FORCE_GPU_ALLOW_GROWTH=true`
  - you can rerun with `export TF_CPP_MIN_LOG_LEVEL=0` environmental variable,
    which increases verbosity of the log messages.

### TOCEntry: MetaCentrum

- _How to install TensorFlow dependencies on MetaCentrum?_

  To install CUDA, cuDNN, and Python 3.10 on MetaCentrum, it is enough to run
  in every session the following command:
  ```
  module add python/python-3.10.4-gcc-8.3.0-ovkjwzd cuda/cuda-11.2.0-intel-19.0.4-tn4edsz cudnn/cudnn-8.1.0.77-11.2-linux-x64-intel-19.0.4-wx22b5t
  ```

- _How to install TensorFlow on MetaCentrum?_

  Once you have the required dependencies, you can create a virtual environment
  and install TensorFlow in it. However, note that by default the MetaCentrum
  jobs have a little disk space, so read about
  [how to ask for scratch storage](https://wiki.metacentrum.cz/wiki/Scratch_storage)
  when submitting a job, and about [quotas](https://wiki.metacentrum.cz/wiki/Quotas),

  TL;DR:
  - Run an interactive CPU job, asking for 16GB scratch space:
    ```
    qsub -l select=1:ncpus=1:mem=8gb:scratch_local=16gb -I
    ```

  - In the job, use the allocated scratch space as a temporary directory:
    ```
    export TMPDIR=$SCRATCHDIR
    ```

  - Finally, create the virtual environment and install TensorFlow in it:
    ```
    module add python/python-3.10.4-gcc-8.3.0-ovkjwzd cuda/cuda-11.2.0-intel-19.0.4-tn4edsz cudnn/cudnn-8.1.0.77-11.2-linux-x64-intel-19.0.4-wx22b5t
    python3 -m venv CHOSEN_VENV_DIR
    CHOSEN_VENV_DIR/bin/pip install --no-cache-dir tensorflow==2.10.0 tensorflow_probability==0.18.0 numpy==1.23.3 gym==0.26.0 pygame==2.1.2 mujoco==2.2.2 ufal.pybox2d==2.3.10.2
    ```

- _How to run a GPU computation on MetaCentrum?_

  First, read the official MetaCentrum documentation:
  [Beginners guide](https://wiki.metacentrum.cz/wiki/Beginners_guide),
  [About scheduling system](https://wiki.metacentrum.cz/wiki/About_scheduling_system),
  [GPU clusters](https://wiki.metacentrum.cz/wiki/GPU_clusters).

  TL;DR: To run an interactive GPU job with 1 CPU, 1 GPU, 16GB RAM, and 8GB scatch
  space, run:
  ```
  qsub -q gpu -l select=1:ncpus=1:ngpus=1:mem=16gb:scratch_local=8gb -I
  ```

  To run a script in a non-interactive way, replace the `-I` option with the script to be executed.

  If you want to run a CPU-only computation, remove the `-q gpu` and `ngpus=1:`
  from the above commands.

### TOCEntry: AIC

- _How to install TensorFlow dependencies on [AIC](https://aic.ufal.mff.cuni.cz)?_

  To install CUDA, cuDNN and Python 3.9 on AIC, you should add the following to
  your `.profile`:
  ```
  export PATH="/lnet/aic/data/python/3.9.9/bin:$PATH"
  export LD_LIBRARY_PATH="/lnet/aic/opt/cuda/cuda-11.2/lib64:/lnet/aic/opt/cuda/cuda-11.2/cudnn/8.1.1/lib64:/lnet/aic/opt/cuda/cuda-11.2/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
  ```

- _How to run a GPU computation on AIC?_

  First, read the official AIC documentation:
  [Submitting CPU Jobs](https://aic.ufal.mff.cuni.cz/index.php/Submitting_CPU_Jobs),
  [Submitting GPU Jobs](https://aic.ufal.mff.cuni.cz/index.php/Submitting_GPU_Jobs).

  TL;DR: To run an interactive GPU job with 1 CPU, 1 GPU, and 16GB RAM, run:
  ```
  qrsh -q gpu.q -l gpu=1,mem_free=16G,h_data=16G -pty yes bash -l
  ```

  To run a script requiring a GPU in a non-interactive way, use
  ```
  qsub -q gpu.q -l gpu=1,mem_free=16G,h_data=16G -cwd -b y SCRIPT_PATH
  ```

  If you want to run a CPU-only computation, remove the `-q gpu.q` and `gpu=1,`
  from the above commands.

### TOCEntry: ReCodEx

- _What files can be submitted to ReCodEx?_

  You can submit multiple files of any type to ReCodEx. There is a limit of
  **20** files per submission, with a total size of **20MB**.

- _What file does ReCodEx execute and what arguments does it use?_

  Exactly one file with `py` suffix must contain a line starting with `def main(`.
  Such a file is imported by ReCodEx and the `main` method is executed
  (during the import, `__name__ == "__recodex__"`).

  The file must also export an argument parser called `parser`. ReCodEx uses its
  arguments and default values, but it overwrites some of the arguments
  depending on the test being executed – the template should always indicate which
  arguments are set by ReCodEx and which are left intact.

- _What are the time and memory limits?_

  The memory limit during evaluation is **1.5GB**. The time limit varies, but it should
  be at least 10 seconds and at least twice the running time of my solution.

- _Do agents need to be trained directly in ReCodEx?_

  No, you can pre-train your agent locally (unless specified otherwise in the task
  description).
