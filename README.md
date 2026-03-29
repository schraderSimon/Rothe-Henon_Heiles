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

Run from the repository root.

If you are already inside `code/`, remove the `code/` prefix from the script path.

Example without using Strang Splitting:
```bash
mpiexec -n 4 python code/run_optimization.py \
	--t0 0 \
	--dimension 2 \
	--timestep 0.01 \
	--epsilon 10000 \
	--len-initial 10 \
	--start-pos 2.0 \
	--lambda 0.111803
```

Same command from inside `code/`:

```bash
mpiexec -n 4 python run_optimization.py \
	--t0 0 \
	--dimension 2 \
	--timestep 0.01 \
	--epsilon 10000 \
	--len-initial 10 \
	--start-pos 2.0 \
	--lambda 0.111803
```

Example with using Strang Splitting:

```bash
mpiexec -n 4 python code/run_optimization.py \
	--t0 0 \
	--dimension 2 \
	--timestep 0.01 \
	--epsilon 10000 \
	--len-initial 10 \
	--start-pos 2.0 \
	--lambda 0.111803 \
	--nokinetic
```

Resume from the latest saved step in the same output file:

```bash
mpiexec -n 4 python code/run_optimization.py --t0 last
```

Parameter meaning:

1. `--t0`: starting time (`0` for fresh run, `last` to continue from latest saved step)
2. `--dimension`: system dimension (for example `2`, `3`, `4`)
3. `--timestep` / `--dt`: time step size
4. `--epsilon`: total error budget used in adaptive behavior. Unused in practice (i.e. set this very high)
5. `--len-initial`: initial number of Gaussians
6. `--start-pos`: initial center position used when constructing the initial basis: defines energy of initial Gaussian
7. `--lambda`: Henon-Heiles coupling strength
8. `--tfinal`: final simulation time
9. `--nokinetic`: switch to Strang-splitting propagation
10. `--output`: optional explicit output file path (otherwise auto-generated in `outputs/`)
