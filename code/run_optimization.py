import os

nthread = "1"
os.environ["OMP_NUM_THREADS"] = nthread  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = nthread  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = nthread  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = nthread  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = nthread  # export NUMEXPR_NUM_THREADS=6
import sys

import h5py
import numpy as np
from numpy import pi, sqrt
from optimization import *

x = 43
np.random.seed(x)

Tfinal = 100
try:
    t0 = float(sys.argv[1])
    epsilon = float(sys.argv[2])
    dimension = int(sys.argv[3])
    timestep = float(sys.argv[4])
    len_initial = int(sys.argv[5])
    start_pos = int(sys.argv[6])
except:
    print("Error in input, going to default values")
    t0 = 0
    timestep = 0.025
    err_t0 = 0
    epsilon = 1
    dimension = 2
    len_initial = 1
    start_pos = 2
try:
    lambda_ = float(sys.argv[7])
except:
    lambda_ = 0.111803
    print("Lambda set to default, %f" % lambda_)

if dimension > 1:
    lin_params = np.zeros(len_initial)
    lin_params[0] = 1
    nonlin_params = np.zeros((len_initial, dimension * (dimension + 3)))
    k = -1
    for i in range(1, dimension + 1):
        k += i
        nonlin_params[:, k] = sqrt(1 / 2)
    startparam = dimension * (dimension + 1)
    ds = 0.5  # 2 for 25 Gaussians, 0.5 otherwise
    for i in range(dimension):
        np.random.seed(x + i)
        nonlin_params[0, startparam + i] = start_pos
        nonlin_params[1:, startparam + i] = (
            ds * (0.5 - np.random.rand(len_initial - 1)) + start_pos
        )
else:
    print("Error in setting up nonline_params / params")
    raise NotImplementedError()
filename = (
    "../outputs/data_dimension=%d_dt=%.3f_epsilon=%.3f_lambda=%.3f_initlen=%d_startpos=%.2f.h5"
    % (
        dimension,
        timestep,
        epsilon,
        lambda_,
        len_initial,
        start_pos,
    )
)
err_t0 = 0
if t0 >= timestep - 0.00001:
    try:
        with h5py.File(filename, "r") as data_file:
            times = np.array(data_file["times"])
            t0_index = np.argmin(abs(times - t0))
            nonlin_params = np.array(data_file["parameters_t=%.3f" % t0])
            lin_params = np.array(data_file["coefficients_t=%.3f" % t0])
            err_t0 = np.array(data_file["rothe_error"])[t0_index]
    except KeyError:
        print("Some data does not exist.")
        sys.exit(1)
    except FileNotFoundError:
        print("The file does not exist.")
        sys.exit(1)

wave_function = WF(nonlin_params, lin_params, lambda_=lambda_)
print("E0:")
print(lin_params.T @ wave_function.calculate_Hamiltonian() @ lin_params)
timeEvolver = TimeEvolution(
    wave_function,
    h=timestep,
    T=Tfinal,
    epsilon=epsilon,
    t0=t0,
    error_t0=err_t0,
    filename=filename,
    dimension=dimension,
)
timeEvolver.time_evolver()
