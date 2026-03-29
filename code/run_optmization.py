import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from mpi4py import MPI
from numpy import sqrt
from optimization_parallel import TimeEvolution
from WF_HenonHeiles_parallel import WF


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0", default="0", help="float or 'last'")
    parser.add_argument("--epsilon", type=float, default=10000)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--timestep", "--dt", dest="timestep", type=float, default=0.01)
    parser.add_argument("--len-initial", type=int, default=10)
    parser.add_argument("--start-pos", type=float, default=2.0)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.111803)
    parser.add_argument("--tfinal", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--nokinetic", action="store_true")
    args = parser.parse_args()
    return args


def init_params(dimension, len_initial, start_pos, seed):
    if dimension <= 1:
        raise NotImplementedError("dimension must be > 1")

    lin = np.zeros(len_initial)
    lin[0] = 1
    nonlin = np.zeros((len_initial, dimension * (dimension + 3)))

    k = -1
    for i in range(1, dimension + 1):
        k += i
        nonlin[:, k] = sqrt(1 / 2)

    start_param = dimension * (dimension + 1)
    for i in range(dimension):
        np.random.seed(seed + i)
        nonlin[0, start_param + i] = start_pos
        if len_initial > 1:
            nonlin[1:, start_param + i] = (
                0.5 * (0.5 - np.random.rand(len_initial - 1)) + start_pos
            )
    return nonlin, lin


def output_path(args):
    if args.output:
        return Path(args.output)
    prefix = "nokineticdata" if args.nokinetic else "paralleldata"
    outputs = Path(__file__).resolve().parent.parent / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    name = (
        f"{prefix}_dimension={args.dimension}_dt={args.timestep:.3f}_epsilon={args.epsilon:.3f}"
        f"_lambda={args.lambda_:.3f}_initlen={args.len_initial}_startpos={args.start_pos:.2f}.h5"
    )
    return outputs / name


def load_state(filename, t0):
    with h5py.File(filename, "r") as f:
        times = np.array(f["times"])
        idx = len(times) - 1 if t0 == "last" else int(np.argmin(abs(times - t0)))
        tsel = float(times[idx])
        key = f"{tsel:.3f}"
        nonlin = np.array(f[f"parameters_t={key}"])
        lin = np.array(f[f"coefficients_t={key}"])
        err = float(np.array(f["rothe_error"])[idx])
        norm = float(np.array(f["normalizations"])[idx]) if "normalizations" in f else 1.0
        energy = float(np.array(f["energies"])[idx]) if "energies" in f else 0.0
        return nonlin, lin, tsel, err, norm, energy


def run_parallel(args, filename, nonlin, lin):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    t0 = args.t0 if args.t0 == "last" else float(args.t0)
    err_t0, norm_t0, energy_t0 = 0.0, 1.0, 0.0

    if t0 == 0.0:
        if rank == 0 and filename.exists():
            filename.unlink()
        comm.Barrier()
    elif t0 == "last" and (not filename.exists()):
        t0 = 0.0

    if t0 == "last" or (isinstance(t0, float) and t0 >= args.timestep - 1e-5):
        try:
            nonlin, lin, t0, err_t0, norm_t0, energy_t0 = load_state(filename, t0)
        except FileNotFoundError:
            print("The file does not exist.")
            sys.exit(1)
        except KeyError:
            print("Some data does not exist.")
            sys.exit(1)

    wf = WF(nonlin, lin, lambda_=args.lambda_)
    if rank == 0:
        print(f"E0:{np.real(np.conj(lin).T @ wf.H @ lin):.10f}")

    evolver = TimeEvolution(
        wf,
        h=args.timestep,
        T=args.tfinal,
        epsilon=args.epsilon,
        t0=t0,
        error_t0=err_t0,
        filename=str(filename),
        dimension=args.dimension,
        norm=norm_t0,
        energy=energy_t0,
    )
    evolver.time_evolver(type="nokinetic" if args.nokinetic else "full")


def main() -> None:
    args = parse_args()
    nonlin, lin = init_params(args.dimension, args.len_initial, args.start_pos, args.seed)
    filename = output_path(args)
    run_parallel(args, filename, nonlin, lin)


if __name__ == "__main__":
    main()
