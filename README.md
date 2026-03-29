# Code to run time-dependent quantum Henon-Heiles calculations with Rothe's method

This repo contains the code required to reproduce the calculations in [this paper](https://doi.org/10.1063/5.0247732) ([arXiv link](https://arxiv.org/abs/2411.05459)).

With this code written in Python, it is possible to run arbitrary-dimensional quantum calculations using Rothe's method. It makes use of `mpi4py` for effective parallelization.

This code has previously been published in a slightly altered form on [zenodo](https://zenodo.org/records/14051077).

## Requirements

1. Python packages (`numpy`, `scipy`, `h5py`, `matplotlib`, `sympy`, `mpi4py`)
2. A system MPI installation (e.g. `OpenMPI`) for `mpi4py`

## Setup (Linux)

```bash
python -m venv .venv
source .venv/bin/activate

# System dependency (Ubuntu/Debian example)
sudo apt update
sudo apt install -y libopenmpi-dev openmpi-bin

# Python dependencies
pip install -r requirements.txt
```

## Execution example
