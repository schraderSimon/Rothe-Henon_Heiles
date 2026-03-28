import sys

sys.path.insert(1, "../")

from optimization_parallel import *
import matplotlib.pyplot as plt

dimension = n = int(sys.argv[1])
timestep = float(sys.argv[2])
len_initial1 = int(sys.argv[3])
len_initial2 = int(sys.argv[5])
parallel1 = sys.argv[4]
parallel2 = sys.argv[6]
lambda_ = float(sys.argv[7])
epsilon = 1000
startpos = 2

try:
    max_T_steps = int(sys.argv[8])

except:
    max_T_steps = 100000
t0 = 0
if not parallel1 == "parallel":
    print("Not parallel: %s" % parallel1)
    filename1 = "../../outputs/data_dimension=%d_dt=%.3f_epsilon=%.3f_lambda=%.3f_initlen=%d_startpos=%.2f.h5" % (
        dimension,
        timestep,
        epsilon,
        lambda_,
        len_initial1,
        startpos,
    )
else:
    filename1 = (
        "../../outputs/paralleldata_dimension=%d_dt=%.3f_epsilon=%.3f_lambda=%.3f_initlen=%d_startpos=%.2f.h5"
        % (dimension, timestep, epsilon, lambda_, len_initial1, startpos)
    )
if parallel2 == "parallel":
    print("Not parallel: %s" % parallel2)
    filename2 = (
        "../../outputs/paralleldata_dimension=%d_dt=%.3f_epsilon=%.3f_lambda=%.3f_initlen=%d_startpos=%.2f.h5"
        % (dimension, timestep, epsilon, lambda_, len_initial2, startpos)
    )
elif parallel2 == "nokinetic" or parallel2 == "noKinetic":
    filename2 = (
        "../../outputs/nokineticdata_dimension=%d_dt=%.3f_epsilon=%.3f_lambda=%.3f_initlen=%d_startpos=%.2f.h5"
        % (dimension, timestep, epsilon, lambda_, len_initial2, startpos)
    )


print(filename1)
print(filename2)
all_nonlin_params1 = []

all_lin_params1 = []
try:
    with h5py.File(filename1, "r") as data_file:
        times = np.array(data_file["times"])
        rotheerrors = np.array(data_file["rothe_error"])
        for i, timev in enumerate(times):
            timeindex = np.argmin(abs(times - timev))
            nonlin_params = np.array(data_file["parameters_t=%.3f" % timev])
            lin_params = np.array(data_file["coefficients_t=%.3f" % timev])
            if i == 0:
                print(i)
                lin_params = np.zeros(nonlin_params.shape[0])
                lin_params[0] = 1

            # print(lin_params)

            err_t0 = rotheerrors[timeindex]
            all_nonlin_params1.append(nonlin_params)
            all_lin_params1.append(lin_params)
except KeyError:
    print("Some data does not exist.")
    sys.exit(1)
except FileNotFoundError:
    print("The file does not exist.")
    sys.exit(1)

all_nonlin_params2 = []

all_lin_params2 = []
try:
    with h5py.File(filename2, "r") as data_file:
        times = np.array(data_file["times"])
        rotheerrors = np.array(data_file["rothe_error"])
        for i, timev in enumerate(times):
            timeindex = np.argmin(abs(times - timev))
            nonlin_params = np.array(data_file["parameters_t=%.3f" % timev])
            lin_params = np.array(data_file["coefficients_t=%.3f" % timev])
            if i == 0:
                print(i)
                lin_params = np.zeros(nonlin_params.shape[0])
                lin_params[0] = 1

            # print(lin_params)

            err_t0 = rotheerrors[timeindex]
            all_nonlin_params2.append(nonlin_params)
            all_lin_params2.append(lin_params)
except KeyError:
    print("Some data does not exist.")
    sys.exit(1)
except FileNotFoundError:
    print("The file does not exist.")
    sys.exit(1)
fidelity = []
fid = 1
for i, timev in enumerate(times[:max_T_steps]):
    if i % 100 == 0:
        print(i, fid)
    nonlin_params1 = all_nonlin_params1[i]
    lin_params1 = all_lin_params1[i]
    nonlin_params2 = all_nonlin_params2[i]
    lin_params2 = all_lin_params2[i]
    nonlin_params2_new = nonlin_params2.copy()
    indices_to_change_sign = np.concatenate(
        (np.arange(n * (n + 1) // 2, n * (n + 1)), np.arange(n * (n + 1) + n, n * (n + 1) + 2 * n))
    )

    for index in indices_to_change_sign:
        """
        Turn around the sign of the b and q parameters -> This corresponds to complex conjugating
        each basis function. In addition, we also complex conjugate the linear coefficients.
        """
        nonlin_params2_new[:, index] = -nonlin_params2_new[:, index]
    full_nonlin = np.concatenate((nonlin_params2.flatten(), nonlin_params1.flatten()))
    full_nonlin = np.reshape(full_nonlin, (-1, n * (n + 3)))
    full_lin = np.concatenate((np.conj(lin_params2), lin_params1))
    optimization_WF = WF(full_nonlin, full_lin, lambda_=lambda_, calculate_Gradient=False)

    bra_lin = np.concatenate((np.conj(lin_params2), np.zeros_like(lin_params1)))
    ket_lin = np.concatenate((np.zeros_like(lin_params2), lin_params1))
    S = optimization_WF.overlap
    norm1 = abs(bra_lin.T @ S @ np.conj(bra_lin))
    norm2 = abs(np.conj(ket_lin).T @ S @ ket_lin)
    fid = abs(bra_lin.T @ S @ ket_lin) ** 2 / norm2 / norm1
    fidelity.append(fid)
plt.plot(times[:max_T_steps], fidelity)
plt.show()
if parallel2 == "noKinetic" or parallel2 == "nokinetic":
    outname = (
        "../../outputs/nokineticfidelity_dimension=%d_dt=%.3f_epsilon=%.3f_lambda=%.3f_len1=%d_len2=%d_startpos=%.2f_noKinetic.h5"
        % (dimension, timestep, epsilon, lambda_, len_initial1, len_initial2, startpos)
    )
else:
    outname = (
        "../../outputs/fidelity_dimension=%d_dt=%.3f_epsilon=%.3f_lambda=%.3f_len1=%d_len2=%d_startpos=%.2f.h5"
        % (dimension, timestep, epsilon, lambda_, len_initial1, len_initial2, startpos)
    )
np.savez(outname, time=times[:max_T_steps], autocorrelation=fidelity)

sys.exit(0)
