import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.fft import ifft, fftfreq
import matplotlib

dimension = 2
timestep = 0.01
epsilon = 1000
lambda_ = 0.111803
startpos = 2
reference = 30
i = 20
autocorrelation_file = np.load(
    "../../outputs/autocorrelation_dimension=3_dt=0.010_epsilon=1000.000_lambda=0.112_initlen=20_startpos=2.00.h5.npz"
)
acorr = autocorrelation_file["autocorrelation"]
times = autocorrelation_file["time"]
simen_reference = np.load("reference_data/hehe_split_step_fourier_acorr_dim_3_xmax_25_ng_128_dt_0.01.npz")
acorr_simen = simen_reference["acorr2"]
font = {"size": 18}
matplotlib.rcParams["axes.titlepad"] = 20

matplotlib.rc("font", **font)
matplotlib.rcParams["lines.linewidth"] = 2
alpha = 0.45
gridcolor = "black"
gridstyle = "solid"
gridlinewidth = 2
fig, ax = plt.subplots(1, 1, layout="constrained")
if timestep == 0.01 and dimension <= 4:
    acorr_simen = acorr_simen[: len(times)]

    x_simen = times * 2
    NSimen = len(times)
    TSimen = times[-1] / NSimen
    xf_Simen = fftfreq(NSimen, TSimen)[: NSimen // 2] * np.pi
    yf_Simen = ifft((acorr_simen) * np.exp(-x_simen / 30))
N = len(times)
T = times[-1] / N
x = np.array(times) * 2
yfs = []
y = np.array(acorr) * np.exp(-x[: len(acorr)] / 30)
yf = ifft((y))

xf = fftfreq(N, T)[: N // 2] * np.pi

ax.plot(xf, np.real(yf[: N // 2]), label="20 Gaussians")
ax.plot(
    xf_Simen,
    np.real(yf_Simen[: NSimen // 2]),
    linestyle=gridstyle,
    linewidth=gridlinewidth,
    label="Grid",
    color=gridcolor,
    alpha=alpha,
)
ax.set_ylim(-0.0005, 0.01)
ax.set_xlim(0, 20)
ax.set_xlabel("Energy")
ax.set_ylabel("Intensity")
plt.legend()
plt.savefig("5minpres.pdf")
plt.show()
