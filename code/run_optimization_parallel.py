import os
import sys

import h5py
import numpy as np
from mpi4py import MPI
from numpy import sqrt
from optimization_parallel import TimeEvolution
from WF_HenonHeiles_parallel import WF

x = 43
np.random.seed(x)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Tfinal = 100
try:
    t0 = sys.argv[1]
    epsilon = float(sys.argv[2])
    dimension = int(sys.argv[3])
    timestep = float(sys.argv[4])
    len_initial = int(sys.argv[5])
    start_pos = int(sys.argv[6])
except:
    print("Error in input, going to default values")
    t0 = 0
    timestep = 0.01
    err_t0 = 0
    epsilon = 1000
    dimension = 2
    len_initial = 30
    start_pos = 2
if not t0 == "last":
    t0 = float(t0)
try:
    lambda_ = float(sys.argv[7])
except:
    lambda_ = 0.111803
    if rank == 0:
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
    "../outputs/paralleldata_dimension=%d_dt=%.3f_epsilon=%.3f_lambda=%.3f_initlen=%d_startpos=%.2f.h5"
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
energy_t0 = 0
norm_t0 = 1


if t0 == 0:
    if rank == 0:
        try:
            os.remove(filename)
        except:
            pass
    comm.Barrier()
elif t0 == "last" and (not os.path.exists(filename)):
    t0 = 0


if t0 == "last" or t0 >= timestep - 0.0001:
    try:
        with h5py.File(filename, "r") as data_file:
            times = np.array(data_file["times"])
            if t0 == "last":
                t0 = times[-1]
                t0_index = len(times) - 1
            else:
                t0_index = np.argmin(abs(times - t0))
            nonlin_params = np.array(data_file["parameters_t=%.3f" % t0])
            lin_params = np.array(data_file["coefficients_t=%.3f" % t0])
            err_t0 = np.array(data_file["rothe_error"])[t0_index]
            norm_t0 = np.array(data_file["normalizations"])[t0_index]
            energy_t0 = np.array(data_file["energies"])[t0_index]
    except KeyError:
        print("Some data does not exist.")
        sys.exit(1)
    except FileNotFoundError:
        print("The file does not exist.")
        sys.exit(1)
wave_function = WF(nonlin_params, lin_params, lambda_=lambda_)
if rank == 0:
    print("E0:%.10f" % np.real(np.conj(lin_params).T @ wave_function.H @ lin_params))
timeEvolver = TimeEvolution(
    wave_function,
    h=timestep,
    T=Tfinal,
    epsilon=epsilon,
    t0=t0,
    error_t0=err_t0,
    filename=filename,
    dimension=dimension,
    norm=norm_t0,
    energy=energy_t0,
)
timeEvolver.time_evolver()
