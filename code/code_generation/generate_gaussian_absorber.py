from numpy import pi, exp,sqrt
import matplotlib.pyplot as plt
#Generate code to the text in the figure bigger
plt.rcParams.update({'font.size': 18})

import numpy as np
from scipy import linalg
from scipy.special import erf
from scipy.optimize import minimize, brute
import sys
xmin=7
xmax=15
a_vec_init=np.linspace(-5,0,20)
#a_vec_init=[-6.9944409244115135, -7.155904252780918, -6.756536552344995, -6.122911066390372, -6.473481690750272, -6.169816544278547, -5.822566237240751, -5.531790621285681, -5.217924981431635, -4.921459558757821, -4.605389113731694, -4.278185232232926, -3.981571074149685, -3.6544991563915117, -3.3448138598222426, -2.975417921820227, -2.6423231583038977, -2.248053754985816, -1.8955975062971175, -1.6144815686078082, -1.4223581057510657]
#a_vec_init=np.array(a_vec_init)+1e-3*2*(np.random.rand(len(a_vec_init))-0.5)
atol=1e-5
def ffit_function(x):
    return (x<=xmax)*(x>xmin)*np.nan_to_num(np.cos((x-xmin)*(np.pi/2)/(xmax-xmin))**(1/8)) + np.array(x<=xmin,dtype=float)
plotx=np.linspace(0,100,5000)
x=np.concatenate((np.linspace(0,xmin,2000),np.linspace(xmax,80,50)))
error=ffit_function(x)
ploterror=ffit_function(plotx)
def err_function_alpha(a_vec,y,gridpoints):
    Phi=getPhi(a_vec,gridpoints)
    pseudoinverse=linalg.pinv(Phi)
    lin_err=y-Phi@pseudoinverse@y
    error=lin_err.T@lin_err
    return error
def error_vector_alpha(a_vec,y,gridpoints):
    Phi=getPhi(a_vec,gridpoints)
    pseudoinverse=linalg.pinv(Phi)
    lin_err=y-Phi@pseudoinverse@y
    return lin_err
def fit_function(a,x):
    return np.exp(-np.exp(a)*x**2)

def getPhi(a_vec,gridpoints):
    L=len(gridpoints) #Number of grid points
    N=len(a_vec) #Number of gaussians
    Phi=np.zeros((L,N))
    for i in range(L):
        for j in range(N):
            Phi[i,j]=fit_function(a_vec[j],gridpoints[i])
    return Phi
def findC(a_vec,y,gridpoints):
    Phi=getPhi(a_vec,gridpoints)
    pseudoinverse=linalg.pinv(Phi,atol=atol)
    return pseudoinverse@y
    #return linalg.inv(Phi.T@Phi+np.eye(3)*1e-16)@Phi.T@y
def func(x,c,a):
    result=np.zeros(len(x))
    for i in range(len(c)):
        result+=c[i]*fit_function(a[i],x)
    return result

grid=x
y=error #The target function
#a_vec_init=np.linspace(-5,10,4)
#a_vec_init=np.linspace(-4.45,-1.13,5)
#a_vec_init=[-3.509098496176515, -3.735150415076037, -3.3970521564975926, -3.7029111390400757, -3.6725847295147496, -2.4003956201891934, -2.8786760651636665, -3.658776319531985, -3.343482328051444]
#a_vec_init=[-3.0421164218273256, -2.726552512717564, -3.418739191125739, -2.800301338842608, -3.1155491026865634, -2.791359385014596, -3.038718683796809, -2.9489234868530776, -2.4980753376676694, -2.5732703753378017, -3.3616621966370044, -2.753564244234464]
#a_vec_init=np.array(a_vec_init)+0.05*(2*np.random.rand(len(a_vec_init))-1)
def jacobian(a_vec,y, gridpoints):
    jac=np.zeros((len(y),len(a_vec)))
    A=np.zeros_like(jac)
    B=np.zeros_like(jac)
    Phi=getPhi(a_vec,gridpoints)

    U,Sigma,Vt=linalg.svd(Phi,check_finite=False,full_matrices=False)
    Sigma_inv=np.linalg.inv(np.diag(Sigma))
    pseudoinverse=linalg.pinv(Phi,atol=atol)
    c=pseudoinverse@y
    r_w=y-Phi@c
    grid2=gridpoints**2
    ea=np.exp(a_vec)
    multiple_D_k=np.zeros((len(y),len(a_vec)))
    multiple_Dkc=np.zeros((len(y),len(a_vec)))
    for k in range(len(a_vec)):
        multiple_D_k[:,k]=-ea[k]*grid2*Phi[:,k]
        multiple_Dkc[:,k]=multiple_D_k[:,k]*c[k]
    AB=multiple_Dkc-U@(U.T@(multiple_Dkc)-(Sigma_inv@(Vt@ np.diag(multiple_D_k.T@r_w) ) ) )
    """
    for k in range(len(a_vec)):
        D_k=np.zeros((len(y),len(a_vec))) #Reset D_k to zero
        D_k[:,k]=-ea[k]*grid2*Phi[:,k]
        Dkc=D_k@c
        A[:,k]=Dkc-U@(U.T@(Dkc))
        B[:,k]=U@(Sigma_inv@(Vt@(D_k.T@r_w)))
    """
    jac=-AB
    #jac=-(A+B)
    return jac

from scipy.optimize import least_squares
errtol=1e-15
sol=least_squares(error_vector_alpha,a_vec_init,jac=jacobian,method="lm",args=(y,grid),xtol=errtol,gtol=errtol,ftol=errtol)
print("a")
print(list(sol.x))
error=ffit_function(np.linspace(0,xmin-1,2000))

print(err_function_alpha(sol.x,error,np.linspace(0,xmin-1,2000)))
#print(sol.optimality)
sol_a=sol.x
sol_c=findC(sol_a,y,grid)
print("c")
print(list(sol_c))
v2=np.zeros(2000)
v=np.zeros(len(plotx))
for i in range(len(sol_c)):
    v+=sol_c[i]*np.exp(-np.exp(sol_a[i])*plotx**2)
    v2+=sol_c[i]*np.exp(-np.exp(sol_a[i])*np.linspace(0,xmin-1,2000)**2)
print(max((v2-error)[:2000]))
plt.plot(plotx,func(plotx,sol_c,sol_a))
#plt.plot(plotx,v)
#plt.plot(plotx,ploterror,label="cos^(1/8) absorber")
#plt.xlim(0,11)
#plt.ylim(0.9999,1.0001)
#plt.plot(x,error-func(x,sol_c,sol_a))
#plt.legend()
plt.tight_layout()
plt.xlim(0,30)

# plot the zoomed portion
#X_detail = np.linspace(0, 10, 1024)
#Y_detail = func(X_detail,sol_c,sol_a)

 # location for the zoomed portion 
#sub_axes = plt.axes([.1, .1, .25, .25]) 

# plot the zoomed portion
#sub_axes.plot(X_detail, Y_detail, c = 'k') 

plt.xlabel("r")
plt.ylabel("Mask M(r)")
plt.savefig("mask.pdf")
plt.show()

print("lin")
print(list(sol_c))
print("Nonlin")
print(list(np.exp(sol_a)))