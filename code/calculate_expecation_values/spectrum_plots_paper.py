import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.fft import ifft, fftfreq

# Parsing command line arguments
dimension = int(sys.argv[1])
ng = [int(arg) for arg in sys.argv[2:]]

# Constants
lambda_ = 0.111803
timestep = 0.01
plot_simen = True
spectra_simon = []
times_simon = []

# General formatting (No serif fonts, larger text)
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
    'lines.linewidth': 0.7,  # Slightly thicker lines
})
alpha=0.45
gridcolor = 'black'
gridstyle = 'solid'
gridlinewidth=0.6
# Load reference data
if lambda_ == 0.111803:
    if dimension==2:
        simen_reference = np.load("reference_data/hehe_split_step_fourier_acorr_dim_2_xmax_25_ng_512_dt_0.01.npz")
        acorr_simen = simen_reference["acorr2"]
    elif dimension==3:
        simen_reference = np.load("reference_data/hehe_split_step_fourier_acorr_dim_3_xmax_25_ng_128_dt_0.01.npz")
        acorr_simen = simen_reference["acorr2"]
    elif dimension==4:
        simen_reference = np.load("reference_data/hehe_split_step_fourier_acorr_dim_4_xmax_25_ng_64_dt_0.01.npz")
        acorr_simen = simen_reference["acorr2"]
    timesteps_simen = np.linspace(0, 100, acorr_simen.shape[0])

# Load calculated data
for i in range(len(ng)):
    outname = "../../outputs/autocorrelation_dimension=%d_dt=%.3f_epsilon=%.3f_lambda=%.3f_initlen=%d_startpos=%.2f.h5.npz" % (
        dimension, 0.01, 1000, lambda_, ng[i], 2)
    simon_data = np.load(outname)
    autocorrelation = simon_data["autocorrelation"]
    times = simon_data["time"]
    spectra_simon.append(autocorrelation)
    times_simon.append(times)

# Plot section
n_plots = len(ng)
figsize=(3.4, 1.5*n_plots)
fig, axes = plt.subplots(n_plots, 1, figsize=figsize, constrained_layout=True, sharex=True,sharey=True)  # Reduced height

colors = ['mediumvioletred','red', 'green', '#CC8400', '#1f77b4']  # '#1f77b4' is a lighter blue
#colors = colors[1:]

colors=colors[::-1]  # Reverse the order of the colors
for i in range(n_plots):
    axes[i].plot(times_simon[i]*2, np.abs(spectra_simon[i]),label="%d Gaussians" % ng[i], color=colors[i])
    if plot_simen:
        axes[i].plot(2*timesteps_simen, np.abs(acorr_simen),linestyle=gridstyle,linewidth=gridlinewidth, label="Grid", color=gridcolor, alpha=alpha)
    axes[i].legend(labelspacing=0.1, borderpad=0.2, handletextpad=0.4)  # Adjust these values as needed

    #axes[i].set_ylabel(r'$|\langle \Psi^*(t/2)|\Psi(t/2)\rangle|$')
    axes[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))  # Format tick labels to two digits

axes[-1].set_xlabel(r'Time t')
fig.suptitle(r'Autocorrelation $|\langle \Psi^*(t/2)|\Psi(t/2)\rangle|$ for D=%d'%dimension)


# Save the first plot
fig.savefig(f'autocorrelation_dim={dimension}.pdf', format='pdf')
fig.savefig(f'autocorrelation_dim={dimension}.png', format='png', dpi=300)

plt.show()

# FFT plot section
if timestep == 0.01 and dimension <= 4:
    acorr_simen = acorr_simen[:len(timesteps_simen)]

    x_simen = timesteps_simen * 2
    NSimen = len(timesteps_simen)
    TSimen = timesteps_simen[-1] / NSimen
    xf_Simen = fftfreq(NSimen, TSimen)[:NSimen // 2] * np.pi
    yf_Simen = ifft((acorr_simen) * np.exp(-x_simen / 30))
N = len(times)
T = times[-1] / N
x = np.array(times) * 2
yfs = []
for i in range(len(ng)):
    y = np.array(spectra_simon[i]) * np.exp(-x[:len(spectra_simon[i])] / 30)
    yf = ifft((y))
    yfs.append(yf)

xf = fftfreq(N, T)[:N // 2] * np.pi
fig, axes = plt.subplots(n_plots, 1, figsize=figsize, constrained_layout=True, sharex=True,sharey=True)  # Reduced height

for i in range(n_plots):
    axes[i].plot(xf, np.real(yfs[i][:N // 2]), label="%d Gaussians" % ng[i], color=colors[i])
    if plot_simen:
        axes[i].plot(xf_Simen, np.real(yf_Simen[:NSimen // 2]),linestyle=gridstyle,linewidth=gridlinewidth, label="Grid", color=gridcolor, alpha=alpha)
    axes[i].legend(labelspacing=0.1, borderpad=0.2, handletextpad=0.4)  # Adjust these values as needed

    axes[i].set_ylabel(r'Intensity (arb. units)')
    axes[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))  # Format tick labels to two digits

if lambda_ == 0.111803 and dimension == 2:
    for ax in axes:
        ax.set_ylim(-0.0005, 0.019)
        ax.set_xlim(0, 15)
elif lambda_ == 0.111803 and dimension == 3:
    for ax in axes:
        ax.set_ylim(-0.0005, 0.01)
        ax.set_xlim(0, 20)

elif lambda_ == 0.111803 and dimension == 4:
    for ax in axes:
        ax.set_ylim(-0.0005, 0.0062)
        ax.set_xlim(0, 20)

axes[-1].set_xlabel(r'Energy')
fig.suptitle(fr'Spectra $S(\omega)$ for D={dimension}')

# Save the second plot
fig.savefig(f'spectrum_dim={dimension}.pdf', format='pdf')
fig.savefig(f'spectrum_dim={dimension}.png', format='png', dpi=300)

plt.show()
# Absolute Difference Plot Section

plt.figure(figsize=(5.4, 2.5))  # Reduced height

for i in range(n_plots):
    abs_diff = np.abs(np.real(yfs[i][:N // 2]) - np.real(yf_Simen[:NSimen // 2])) / max(np.abs(np.real(yf_Simen[:NSimen // 2])))
    plt.plot(xf, abs_diff, label="%d Gaussians" % ng[i], color=colors[i])

# Move legend outside the plot
plt.legend(labelspacing=0.1, borderpad=0.2, handletextpad=0.4, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylabel('Abs. Diff.')
plt.xlabel('Energy')
plt.title(fr'Absolute Difference in Spectra for D={dimension}')

if lambda_ == 0.111803 and dimension == 2:
    plt.xlim(0, 15)
elif lambda_ == 0.111803 and dimension == 3:
    plt.xlim(0, 20)
elif lambda_ == 0.111803 and dimension == 4:
    plt.xlim(0, 20)

plt.ylim(1e-6, 1)
plt.yscale('log')
plt.tight_layout()  # Adjust rect to leave space for the legend
plt.savefig(f'absolute_difference_spectrum_dim={dimension}.pdf', format='pdf')
plt.savefig(f'absolute_difference_spectrum_dim={dimension}.png', format='png', dpi=300)

plt.show()


plt.figure(figsize=(5.4,2.5))  # Reduced height

for i in range(n_plots):
    abs_diff = np.abs(np.real(yfs[i][:N // 2]) - np.real(yf_Simen[:NSimen // 2]))
    rel_diff=abs_diff/np.abs(np.real(yf_Simen[:NSimen //2]))
    plt.plot(xf, rel_diff, label="%d Gaussians" % ng[i], color=colors[i])
plt.legend(labelspacing=0.1, borderpad=0.2, handletextpad=0.4, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylabel('Rel. Diff.')

plt.xlabel('Energy')
plt.title(fr'Relative Difference in Spectra for D={dimension}')
if lambda_ == 0.111803 and dimension == 2:
    for ax in axes:
        plt.xlim(0, 15)
elif lambda_ == 0.111803 and dimension == 3:
    for ax in axes:
        plt.xlim(0, 20)

elif lambda_ == 0.111803 and dimension == 4:
    for ax in axes:
        plt.xlim(0, 20)

# Save the third plot
plt.ylim(1e-4, 10)
plt.yscale('log')
plt.tight_layout()

plt.savefig(f'relative_difference_spectrum_dim={dimension}.pdf', format='pdf')
plt.savefig(f'relative_difference_spectrum_dim={dimension}.png', format='png', dpi=300)

plt.show()
