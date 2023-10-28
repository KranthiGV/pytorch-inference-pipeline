# ML Task

This is a short interview coding task at Coram.AI for the machine learning candidates.

The task tests whether the candidates are familiar with pytorch and libtorch.

## Setup

1. Install [docker](https://docs.docker.com/engine/install/). Recommended version is `23.0.3`.
2. Run the code using `./run_stack.sh`. Underhood, this command builds the docker image, launches the container, and executes the binaries for your test.
   - This setup has been tested on both Ubuntu and MacOS with M1 chips.

## Tasks

1. Finish the code in `src/model_generation_main.py` so that it generates a torchscript model for ResNet18 inference.
2. Finish the code in `src/pytorch_main.cc` so that it loads the torchscript model from the step above and runs inference on some test data.

You are free to edit the code in any way you want to finish the tasks above.

## Tips

To quickly iterate on your code, you can:

1. manually log into the container
2. iterate on your code outside the container
3. test your command by manually executing `build_and_execute.sh`

Refer to `run_stack.sh` for more full details.
