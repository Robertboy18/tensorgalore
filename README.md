# Tensor-GaLore: Memory-Efficient Training via Gradient Tensor Decomposition
This repository contains code to run all experiments reported in our paper. To build, first start in a new conda env. Then:

```
(venv) $ conda install pytorch torchvision pytorch-cuda={VERSION} -c pytorch -c nvidia

(venv) $ pip install -r requirements.txt
```

The configuration files are managed via `configmypy` on Pip. Check the docs to see how to configure scripts in both the command line and associated YAML files.

Directory structure:

1. `tensor_galore`: contains our optimizer, as well as code to compute gradient projections for matrices and tensors, as well as some utilities for profiling and running large-scale experiments.

2. scripts at the top-level (`train_EXP_galore.py`) contain code to train our models end-to-end

3. `./scripts` contains bash scripts to run specific configurations of experiments. 