import sys
sys.path.insert(1, '../')

from optimization_parallel import *
epsilon=float(sys.argv[1])

dimension=n=int(sys.argv[2])
timestep=float(sys.argv[3])
len_initial=int(sys.argv[4])
startpos=float(sys.argv[5])
lambda_=float(sys.argv[6])
parallel=sys.argv[7]
try:
    max_T_steps=int(sys.argv[8])

except:
    max_T_steps=100000
t0=0
if not parallel=="parallel" and not parallel=="noKinetic":
    print("Not parallel: %s"%parallel)
    filename="../../outputs/data_dimension=%d_dt=%.3f_epsilon=%.3f_lambda=%.3f_initlen=%d_startpos=%.2f.h5"%(dimension,timestep,epsilon,lambda_,len_initial,startpos)
elif parallel=="noKinetic":
    filename="../../outputs/nokineticdata_dimension=%d_dt=%.3f_epsilon=%.3f_lambda=%.3f_initlen=%d_startpos=%.2f.h5"%(dimension,timestep,epsilon,lambda_,len_initial,startpos)
    print("No kinetic: %s"%parallel)
else:
    filename="../../outputs/paralleldata_dimension=%d_dt=%.3f_epsilon=%.3f_lambda=%.3f_initlen=%d_startpos=%.2f.h5"%(dimension,timestep,epsilon,lambda_,len_initial,startpos)

print(filename)
all_nonlin_params=[]
all_lin_params=[]
plot_simen=False
if dimension==2 and lambda_==0.111803:
    simen_reference_2D=np.load("reference_data/hehe_split_step_fourier_acorr_dim_2_xmax_25_ng_512_dt_0.01.npz")
    acorr_simen=simen_reference_2D["acorr2"]
    timesteps_simen=np.linspace(0,100,acorr_simen.shape[0])
if dimension==3 and lambda_==0.111803:
    simen_reference_3D=np.load("reference_data/hehe_split_step_fourier_acorr_dim_3_xmax_25_ng_128_dt_0.01.npz")
    acorr_simen=simen_reference_3D["acorr2"]
if dimension==4 and lambda_==0.111803:
    simen_reference_4D=np.load("reference_data/hehe_split_step_fourier_acorr_dim_4_xmax_25_ng_64_dt_0.01.npz")
    acorr_simen=simen_reference_4D["acorr2"]
if lambda_==0.111803 and dimension<=4:
    timesteps_simen=np.linspace(0,100,acorr_simen.shape[0])
    plot_simen=True
try:
    with h5py.File(filename, "r") as data_file:
        times=np.array(data_file["times"])
        rotheerrors=np.array(data_file["rothe_error"])
        for i,timev in enumerate(times):
            timeindex=np.argmin(abs(times-timev))
            nonlin_params=np.array(data_file["parameters_t=%.3f"%timev])
            lin_params=np.array(data_file["coefficients_t=%.3f"%timev])
            if i==0:
                print(i)
                lin_params=np.zeros(nonlin_params.shape[0])
                lin_params[0]=1
            
            
            err_t0=rotheerrors[timeindex]
            all_nonlin_params.append(nonlin_params)
            all_lin_params.append(lin_params)
except KeyError:
    print("Some data does not exist.")
    sys.exit(1)
except FileNotFoundError:
    print("The file does not exist.")
    sys.exit(1)
print("Rothe error at final time: %f"%err_t0)
nonlin_params_0=all_nonlin_params[0].copy()
lin_params_0=all_lin_params[0].copy()
autocorrelations=[]
norms=[]
overlap_with_cc=[]
i=-1
nonlin_params=all_nonlin_params[i]
lin_params=all_lin_params[i]
full_nonlin=np.concatenate((nonlin_params_0.flatten(),nonlin_params.flatten()))
full_nonlin=np.reshape(full_nonlin,(-1,n*(n+3)))
full_lin=np.concatenate((lin_params_0,lin_params))
full_lin_0=np.concatenate((lin_params_0,np.zeros_like(lin_params)))
full_lin_1=np.concatenate((np.zeros_like(lin_params_0),lin_params))
optimization_WF=WF(full_nonlin,full_lin,lambda_=lambda_,calculate_Gradient=False)
S=optimization_WF.overlap
autocorr=np.conj(full_lin_0).T@S@full_lin_1
norm=np.conj(full_lin_1).T@S@full_lin_1
print("Norm at final timestep: %f"%norm)

for i,timev in enumerate(times[:max_T_steps]):
    if i%100==0:
        print(i)
    nonlin_params=all_nonlin_params[i]
    lin_params=all_lin_params[i]
    #print(lin_params)
    if timev==times[:max_T_steps][-1]:
        full_nonlin=np.concatenate((nonlin_params_0.flatten(),nonlin_params.flatten()))
        full_nonlin=np.reshape(full_nonlin,(-1,n*(n+3)))
        full_lin=np.concatenate((lin_params_0,lin_params))
        full_lin_0=np.concatenate((lin_params_0,np.zeros_like(lin_params)))
        full_lin_1=np.concatenate((np.zeros_like(lin_params_0),lin_params))
        optimization_WF=WF(full_nonlin,full_lin,lambda_=lambda_,calculate_Gradient=False)
        S=optimization_WF.overlap
        autocorr=np.conj(full_lin_0).T@S@full_lin_1
        norm=np.conj(full_lin_1).T@S@full_lin_1
        autocorrelations.append(autocorr)
        norms.append(norm)
        print("Norm at final timestep: %f"%norm)
    nonlin_params_new=nonlin_params.copy()
    indices_to_change_sign=np.concatenate((np.arange(n*(n+1)//2,n*(n+1)),np.arange(n*(n+1)+n,n*(n+1)+2*n)))
    for index in indices_to_change_sign:
        """
        Turn around the sign of the b and q parameters -> This corresponds to complex conjugating
        each basis function. In addition, we also complex conjugate the linear coefficients.
        """
        nonlin_params_new[:,index]=-nonlin_params_new[:,index]
    full_nonlin=np.concatenate((nonlin_params_new.flatten(),nonlin_params.flatten()))
    full_nonlin=np.reshape(full_nonlin,(-1,n*(n+3)))
    full_lin=np.concatenate((np.conj(lin_params),lin_params))
    optimization_WF=WF(full_nonlin,full_lin,lambda_=lambda_,calculate_Gradient=False)
    S=optimization_WF.overlap
    full_lin_0=np.concatenate((np.conj(lin_params),np.zeros_like(lin_params)))
    full_lin_1=np.concatenate((np.zeros_like(lin_params),lin_params))
    autocorr=np.conj(full_lin_0).T@S@full_lin_1
    overlap_with_cc.append(autocorr)
if not parallel=="noKinetic":
    outname="../../outputs/autocorrelation_dimension=%d_dt=%.3f_epsilon=%.3f_lambda=%.3f_initlen=%d_startpos=%.2f.h5"%(dimension,timestep,epsilon,lambda_,len_initial,startpos)
else:
    outname="../../outputs/autocorrelation_dimension=%d_dt=%.3f_epsilon=%.3f_lambda=%.3f_initlen=%d_startpos=%.2f_nokinetic.h5"%(dimension,timestep,epsilon,lambda_,len_initial,startpos)

np.savez(outname, time=times[:max_T_steps], autocorrelation=overlap_with_cc)

import sys
import matplotlib.pyplot as plt
times=times[:max_T_steps]

if timestep==0.01 and dimension<4:
    timesteps_simen=times
    acorr_simen=acorr_simen[:len(times)]
else:
    timesteps_simen=timesteps_simen[timesteps_simen<=times[-1]]
    acorr_simen=acorr_simen[:len(timesteps_simen)]

if plot_simen:
    plt.plot(timesteps_simen,np.abs(acorr_simen),label="Simen")
max_T_steps=len(times)
RE=rotheerrors[1:max_T_steps]-rotheerrors[:max_T_steps-1]
plt.plot(times[:max_T_steps],np.abs(overlap_with_cc)[:max_T_steps],label="Simon")
plt.plot(times[1:max_T_steps],100*RE,label="100 times Rothe error")
plt.legend()
plt.show()

from scipy.fft import ifft, fftfreq
import numpy as np
# Number of sample points
N = len(times)
T = times[-1]/N
x = np.array(times)*2
y = np.array(overlap_with_cc)*np.exp(-x/30)
yf = ifft((y))
xf = fftfreq(N, T)[:N//2]*np.pi
if plot_simen:
    x_simen=timesteps_simen*2
    NSimen=len(timesteps_simen)
    TSimen=timesteps_simen[-1]/NSimen
    xf_Simen = fftfreq(NSimen, TSimen)[:NSimen//2]*np.pi
    yf_Simen =  ifft((acorr_simen)*np.exp(-x_simen/30))
import matplotlib.pyplot as plt

try:
    plt.plot(xf_Simen,np.real(yf_Simen[0:NSimen//2]),label="Grid refence")
except:
    pass
    print("Something went wrong")
plt.plot(xf,np.real(yf[0:N//2]),label="Rothe")
ylimmax=1
if lambda_==0.111803 and dimension==3:
    ylimmax=0.01

elif lambda_==0.111803 and dimension==2:
    ylimmax=0.02
elif lambda_==0.111803 and dimension==4:
    ylimmax=0.007
else:
    ylimmax=np.max(np.real(yf[0:N//2]))*1.05
if np.max(np.real(yf[0:N//2]))>ylimmax:
    ylimmax=np.max(np.real(yf[0:N//2]))*1.05
if dimension==2:
    plt.xlim(0,15)
elif dimension<5:
    plt.xlim(0,20)
else:
    plt.xlim(0,30)
plt.ylim(0,ylimmax)
plt.grid()
if len_initial==1:
    plt.title(r"N=%d, %d Gaussian, $\lambda=%.6f$, dt=%f, pos=%.1f"%(dimension, len_initial,lambda_,timestep,startpos))
else:
    plt.title(r"N=%d, %d Gaussians, $\lambda=%.6f$, dt=%f, pos=%.1f"%(dimension, len_initial,lambda_,timestep,startpos))
plt.xlabel("Energy")
plt.ylabel("Intensity")
plt.legend()
if parallel=="noKinetic":
    plt.savefig("HHG_dim=%d_Lambda=%.3f_nd=%dstartpos=%d_dt=%.3f_noKinetic.pdf"%(dimension,lambda_,len_initial,startpos,timestep))
    plt.savefig("HHG_dim=%d_Lambda=%.3f_nd=%dstartpos=%d_dt=%.3f_noKinetic.png"%(dimension,lambda_,len_initial,startpos,timestep))
else:
    plt.savefig("HHG_dim=%d_Lambda=%.3f_nd=%dstartpos=%d_dt=%.3f.pdf"%(dimension,lambda_,len_initial,startpos,timestep))
    plt.savefig("HHG_dim=%d_Lambda=%.3f_nd=%dstartpos=%d_dt=%.3f.png"%(dimension,lambda_,len_initial,startpos,timestep))

plt.show()
