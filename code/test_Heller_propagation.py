import numpy as np
import matplotlib.pyplot as plt
import sys
from WF_HenonHeiles_parallel import *
from numpy import array
sys.path.append('../')
from time import time
lambda_=0.111803
timestep=1
epsilon=1000
dimension=3
len_initial=5
start_pos=2
#np.random.seed(int(time()))
if dimension>1:
    lin_params=np.zeros(len_initial,dtype=complex);lin_params[0]=1
    nonlin_params=np.zeros((len_initial,dimension*(dimension+3)))
    k=-1
    for i in range(1,dimension+1):
      k+=i
      nonlin_params[:,k]=sqrt(1/2)
    startparam=dimension*(dimension+1)
    ds=5 #2 for 25 Gaussians, 0.5 otherwise
    for i in range(dimension):
        nonlin_params[0,startparam+i]=start_pos
        nonlin_params[1:,startparam+i]=ds*(0.5-np.random.rand(len_initial-1))+start_pos
lin_params=np.random.rand(len_initial)-0.5+1j*(np.random.rand(len_initial)-0.5)
nonlin_params+=np.random.rand(nonlin_params.shape[0],nonlin_params.shape[1])*1-0.5
print(nonlin_params)


print("Old normalization: ")
wave_function=WF(nonlin_params,lin_params,lambda_=lambda_)
print(np.conj(lin_params).T@wave_function.overlap@lin_params)
nonlin_new,lin_new=propagate_kinetic_analytical(nonlin_params,lin_params,timestep)
wave_function=WF(nonlin_new,lin_new,lambda_=lambda_)
lin_new=np.array(lin_new)
print("New normalization")
print(np.conj(lin_new).T@wave_function.overlap@lin_new)
