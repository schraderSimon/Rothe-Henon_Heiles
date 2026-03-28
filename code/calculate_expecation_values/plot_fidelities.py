import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
dimension=int(sys.argv[1])
timestep=float(sys.argv[2])
epsilon=1000
lambda_=0.111803
startpos=2
reference=int(sys.argv[3])
existing_files=[]
fidelities=[]
rotheerrors=[]
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "text.usetex": True
})

for i in range(1,reference+1):
    try:
        datafile="../../outputs/fidelity_dimension=%d_dt=%.3f_epsilon=%.3f_lambda=%.3f_len1=%d_len2=%d_startpos=%.2f.h5.npz"%(dimension,timestep,epsilon,lambda_,reference,i,startpos)
        fidelity=np.load(datafile)
        fidelities.append(fidelity["autocorrelation"])
        times=fidelity["time"]
        existing_files.append(i)
        print(i)
    except FileNotFoundError:
        continue
expp1=existing_files.copy()
expp1.append(reference)
for i in expp1:
    try:
        filename="../../outputs/paralleldata_dimension=%d_dt=%.3f_epsilon=%.3f_lambda=%.3f_initlen=%d_startpos=%.2f.h5"%(dimension,timestep,epsilon,lambda_,i,startpos)
        with h5py.File(filename, "r") as data_file:
                times=np.array(data_file["times"])
                rotheerror=np.array(data_file["rothe_error"])
                rotheerrors.append(rotheerror)
    except FileNotFoundError:
        continue
colors = ["mediumvioletred",'red', 'green', '#CC8400', '#1f77b4']  # '#1f77b4' is a lighter blue
#colors = colors[1:len(colors)]
colors=colors[::-1]  # Reverse the order of the colors
fig, axes = plt.subplots(2, 1, figsize=(3.4, 2*1.7), constrained_layout=True, sharex=True,sharey=False)  # Reduced height
ax=axes[0]
ax2=axes[1]
for num,i in enumerate(existing_files):
    ax.plot(times,fidelities[num],label="N=%d"%(i),color=colors[num])
for num,i in enumerate(expp1):
    ax2.plot(times,rotheerrors[num],label="N=%d"%(i),color=colors[num])

ax.set_xlabel('Time')
ax.set_ylabel(r"$\text{Fidelity } F^4_{N,40}(t)$")
#ax.set_title("Fidelities for D=%d with $N_{max}=%d$"%(dimension,reference))
ax2.set_xlabel('Time')
ax2.set_ylabel("Cumulative Rothe error")
ax2.set_yscale("log")
yticks = [1e-3, 1e-2, 1e-1, 1e0, 1e1,1e2]
ax2.set_yticks(yticks)
ax2.set_ylim(1e-3, 1e2)
ax2.grid(True)
ax.grid(True)

# Format the ytick labels for ax2
def exp_formatter(x, pos):
    return f'$10^{{{int(np.log10(x))}}}$'
from matplotlib.ticker import FuncFormatter

ax2.yaxis.set_major_formatter(FuncFormatter(exp_formatter))
ax2.yaxis.set_minor_formatter(plt.NullFormatter())
for ytick in yticks:
    pass
    #ax2.axhline(y=ytick, color='gray', linestyle=':', alpha=0.7, zorder=0)

ax.legend()
ax2.legend()
plt.tight_layout()
plt.savefig("Fidelities_D=%d_Mmax=%d.png"%(dimension,reference))
plt.savefig("Fidelities_D=%d_Mmax=%d.pdf"%(dimension,reference))

plt.show()