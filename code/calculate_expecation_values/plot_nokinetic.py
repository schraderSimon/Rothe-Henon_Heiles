import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.fft import ifft, fftfreq

dimension=2
timestep=0.01
epsilon=1000
lambda_=0.111803
startpos=2
reference=40
i=20
#"../../outputs/nokineticfidelity_dimension=2_dt=0.010_epsilon=1000.000_lambda=0.112_len1=40_len2=20_startpos=2.00_noKinetic.h5.npz"
fidelity_file_nokinetic="../../outputs/nokineticfidelity_dimension=%d_dt=%.3f_epsilon=%.3f_lambda=%.3f_len1=%d_len2=%d_startpos=%.2f_noKinetic.h5.npz"%(dimension,timestep,epsilon,lambda_,reference,i,startpos)
fidelity_file_kinetic="../../outputs/fidelity_dimension=%d_dt=%.3f_epsilon=%.3f_lambda=%.3f_len1=%d_len2=%d_startpos=%.2f.h5.npz"%(dimension,timestep,epsilon,lambda_,reference,i,startpos)
fidelity=np.load(fidelity_file_nokinetic)["autocorrelation"]
autocorrelation_file=np.load("../../outputs/autocorrelation_dimension=2_dt=0.010_epsilon=1000.000_lambda=0.112_initlen=20_startpos=2.00_nokinetic.h5.npz")
acorr=autocorrelation_file["autocorrelation"]
times = autocorrelation_file["time"]
simen_reference = np.load("reference_data/hehe_split_step_fourier_acorr_dim_2_xmax_25_ng_512_dt_0.01.npz")
acorr_simen = simen_reference["acorr2"]
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "text.usetex": True,
    'lines.linewidth': 0.9,  # Slightly thicker lines
})
alpha=0.45
gridcolor = 'black'
gridstyle = 'solid'
gridlinewidth=0.85
figsize=(3.4, 1.5*3.5)
fig, axes = plt.subplots(3, 1, figsize=figsize, constrained_layout=True)  # Reduced height
plt.suptitle("Strang Splitting propagation for $D=2$, $N=20$",size=11)
axes[0].plot(2*times, np.abs(acorr),linestyle=gridstyle,linewidth=gridlinewidth, label="20 Gaussians")
axes[0].plot(2*times, np.abs(acorr_simen),linestyle=gridstyle,linewidth=gridlinewidth, label="Grid", color=gridcolor, alpha=alpha)
if timestep == 0.01 and dimension <= 4:
    acorr_simen = acorr_simen[:len(times)]

    x_simen = times * 2
    NSimen = len(times)
    TSimen = times[-1] / NSimen
    xf_Simen = fftfreq(NSimen, TSimen)[:NSimen // 2] * np.pi
    yf_Simen = ifft((acorr_simen) * np.exp(-x_simen / 30))
N = len(times)
T = times[-1] / N
x = np.array(times) * 2
yfs = []
y = np.array(acorr) * np.exp(-x[:len(acorr)] / 30)
yf = ifft((y))

xf = fftfreq(N, T)[:N // 2] * np.pi

axes[1].plot(xf, np.real(yf[:N // 2]), label="20 Gaussians")
axes[1].plot(xf_Simen, np.real(yf_Simen[:NSimen // 2]),linestyle=gridstyle,linewidth=gridlinewidth, label="Grid", color=gridcolor, alpha=alpha)
axes[1].set_ylim(-0.0005, 0.019)
axes[1].set_xlim(0, 15)
axes[2].plot(times, fidelity, label="20 Gaussians")
axes[2].set_ylim(0.6, 1.05)
titles=[r"Autocorrelation $|\langle \Psi^*(t/2)|\Psi(t/2)\rangle|$",r"Spectrum $S(\omega)$", "Fidelity $F^2_{20,40}(t)$"]
xlabels=["Time","Energy","Time"]
locs=["upper right","upper right","lower right"]
for i in range(3):
    axes[i].legend(loc=locs[i],labelspacing=0.1, borderpad=0.2, handletextpad=0.4)
    axes[i].set_title(titles[i])
    axes[i].set_xlabel(xlabels[i])
axes[1].set_ylabel("Intensity (arb. units)")
axes[2].set_ylabel("Fidelity")
axes[0].set_ylabel("Autocorrelation")
plt.savefig("nokinetic_data_D=2_N=20.pdf")
plt.savefig("nokinetic_data_D=2_N=20.png")
plt.show()