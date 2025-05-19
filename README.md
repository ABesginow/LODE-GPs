This is the implementation for the Linear Ordinary Differential Equations Gaussian Processes (LODE-GPs), published in [1].

# Installation

First of all, [install SageMath](https://doc.sagemath.org/html/en/installation/index.html).
Additional required libraries are GPyTorch and PyTorch.

# Running
`MWE.py` should run out of the box calling `python MWE.py`, after switching to the sagemath conda environment using `conda activate <sage-env-name>`.

# Options
By now the following features are implemented:
- Arbitrary homogenuous ODEs with constant coefficients through the `A` matrix
- Trainable parameters through SageMath `FunctionField`s (compare for the `Heating` system or the `Bipendulum Parameterized` in `LODEGP.py`)
- SE, Mat52 and Mat32 base kernels

[1] https://proceedings.neurips.cc/paper_files/paper/2022/hash/bcef27c5825d1ed8757290f237b2d851-Abstract-Conference.html 