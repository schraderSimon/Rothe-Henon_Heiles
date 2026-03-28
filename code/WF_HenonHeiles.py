import warnings
warnings.filterwarnings("ignore", message="Derivative was zero.") #Ignore error message 
import time
import sys
import itertools as it

import numpy as np
import scipy as sp
from numpy import sqrt, pi, einsum, log, exp, real, imag, conj
np.set_printoptions(linewidth=300,precision=5)
import os
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm,svd, det,inv, eig, eigh
import numpy as np
import sys
from scipy.optimize import root_scalar, newton,bisect, minimize

from utils import solve
dx=1e-7
from calculate_all_integrals_at_once import calculate_all_expectation_values_full_allij, get_x7_x8_contraction_allij





def update_sum_matrices(matrices,pairs_of_indices,sum_matrices,sum_matrices_inv):
    poi=pairs_of_indices
    sum_matrix_temp_list=[]
    for pair in poi:
        i=pair[0]
        j=pair[1]
        sum_matrix=conj(matrices[i])+matrices[j]
        sum_matrix_temp_list.append(sum_matrix)
    inverts=np.linalg.inv(sum_matrix_temp_list)
    for i in range(len(poi)):
        sum_matrices[poi[i][0],poi[i][1],:,:]=sum_matrix_temp_list[i]
        sum_matrices[poi[i][1],poi[i][0],:,:]=conj(sum_matrix_temp_list[i])
        sum_matrices_inv[poi[i][0],poi[i][1],:,:]=inverts[i]
        sum_matrices_inv[poi[i][1],poi[i][0],:,:]=conj(inverts[i])
    return sum_matrices,sum_matrices_inv

def update_overlap_matrix(overlap_matrix,pairs_of_indices,matrices,vectors,sum_matrices,sum_matrices_inv):
    poi_array = np.array(pairs_of_indices)  # Convert to NumPy array if not already
    row_indices = poi_array[:, 0]  # Extract row indices
    col_indices = poi_array[:, 1]  # Extract column indices

    num_gaussians=len(overlap_matrix)
    vmips=np.einsum("ia,iab,ib->i",vectors,matrices,vectors) #Very cheap to calculate
    diagonal_determinants=np.zeros(num_gaussians,dtype=complex)
    diagonal_exponents=np.zeros(num_gaussians,dtype=complex)
    diagonal_sum_matrix = np.einsum('iijk->ijk', sum_matrices)
    diagonal_sum_matrices_inv = np.einsum('iijk->ijk', sum_matrices_inv)
    diagonal_determinants=sqrt(det(diagonal_sum_matrix))
    bvecs_list=np.einsum("ak,akl->al",vectors,matrices)
    bvec_calc=2*np.real(bvecs_list)
    diagonal_exponents=np.einsum("ak,akl,al->a",bvec_calc,diagonal_sum_matrices_inv,bvec_calc)-vmips-vmips.conj()    
    sum_matrix_temp_list=sum_matrices[row_indices,col_indices,:,:]
    eigvals_temp_list=np.linalg.eigvals(sum_matrix_temp_list)
    sqrt_eigvals_temp_list=sqrt(eigvals_temp_list)
    determinants_temp_list=np.prod(sqrt_eigvals_temp_list,axis=1) 
    inv_dets_temp_list=1/determinants_temp_list
    vmips_outer=np.conj(vmips[:,np.newaxis])+vmips[np.newaxis,:]
    diagonal_exponents_outer=(diagonal_exponents[:,np.newaxis]+diagonal_exponents[np.newaxis,:])
    bvec_outer= np.conj(bvecs_list)[:,np.newaxis,:]+(bvecs_list)[np.newaxis,:,:]
    bvecs = bvec_outer[row_indices, col_indices,:]
    
    exp_constants = vmips_outer[row_indices, col_indices]+0.5*diagonal_exponents_outer[row_indices, col_indices]
    exponents=np.einsum("ia,iab,ib->i",bvecs,sum_matrices_inv[row_indices, col_indices, :, :],bvecs)-exp_constants
    diag_determinants_sqrt=np.sqrt(diagonal_determinants)
    diagonal_determinants_list = diag_determinants_sqrt[row_indices]*diag_determinants_sqrt[col_indices]
    overlap_update=inv_dets_temp_list*diagonal_determinants_list*np.exp(exponents)
    overlap_matrix[row_indices,col_indices]=overlap_update
    overlap_matrix[col_indices,row_indices]=conj(overlap_update)
    return overlap_matrix



def make_sum_matrices(matrices,sum_matrices,sum_matrices_inv):
    num_gauss=len(matrices)
    #sum_matrices=np.einsum()-
    for i in range(num_gauss):
        for j in range(0,i+1):
            sum_matrix=conj(matrices[i])+matrices[j]
            sum_matrices[i,j,:,:]=sum_matrix
            sum_matrices[j,i,:,:]=conj(sum_matrix)
    inverts=np.linalg.inv(sum_matrices)
    sum_matrices_inv=inverts
    return sum_matrices,sum_matrices_inv
class WF:
    def __init__(self,nonlin_params,lin_params,lambda_=0.1,normalization=1,calculate_Gradient=True,h=1e-3,onlyX1X2=False):
        start=time.time()
        """
        nonlin_params: list of list or 2D array all parameters. The general form is (ngauss,n(n+3))), where n is the dimensionality of the problem
        lin_params: list of same length as nonlin_params
        lambda: Strength of the Henon-Heiles interaction
        normalization: The normalization of the WF. 
        """
        self.onlyX1X2=onlyX1X2
        self.threshold_overlap=1e-6
        self.num_gaussians=len(lin_params)
        self.num_dimensions=np.rint(0.5*(-3+sqrt(4*len(nonlin_params[0])+9))).astype(int) #This is the invert formula of nparams = n*(n+3) for the number of dimensions
        self.nonlin_params=np.asarray(nonlin_params)
        self.lin_params=lin_params
        self.matrices=None
        self.vectors=None
        self.lambda_=lambda_
        self.lambda_sq=lambda_**2
        self.normalization=normalization 
        self.calculate_Gradient=calculate_Gradient
        self.h=h
        self.setupintermediates()
        
        ng=self.num_gaussians
        nd=self.num_dimensions
        self.x1_expectation_values=np.zeros((ng,ng,nd),dtype=complex)
        self.x2_expectation_values=np.zeros((ng,ng,nd,nd),dtype=complex)
        self.x3_expectation_values=np.zeros((ng,ng,nd,nd,nd),dtype=complex)
        self.x4_expectation_values=np.zeros((ng,ng,nd,nd,nd,nd),dtype=complex)
        self.x5_expectation_values=np.zeros((ng,ng,nd,nd,nd,nd,nd),dtype=complex)
        self.x6_expectation_values=np.zeros((ng,ng,nd,nd,nd,nd,nd,nd),dtype=complex)
        self.x8_c1=np.zeros((ng,ng,nd,nd),dtype=complex)
        self.x8_c2=np.zeros((ng,ng,nd,nd),dtype=complex)
        self.x8_c3=np.zeros((ng,ng,nd,nd),dtype=complex)
        self.x7_c1=np.zeros((ng,ng,nd,nd),dtype=complex)
        self.x7_c2=np.zeros((ng,ng,nd,nd),dtype=complex)
        self.x7_c3=np.zeros((ng,ng,nd,nd,nd,nd),dtype=complex)
        self.x7_c4=np.zeros((ng,ng,nd,nd,nd,nd),dtype=complex)
        self.x7_c1v=np.zeros((ng,ng,nd),dtype=complex)
        self.x7_c2v=np.zeros((ng,ng,nd),dtype=complex)
        self.x7_c3v=np.zeros((ng,ng,nd),dtype=complex)
        
        self.root=None
        mu_list=[]
        cross_mat_inv_list=[]#np.empty((ng*(ng+1)//2,nd,nd),dtype=complex)
        ovlp_list=[]#np.empty(ng*(ng+1)//2,dtype=complex)
        counter=0
        relevant_pairs=[]
        for i in range(self.num_gaussians):
            for j in range(0,i+1):
                if abs(self.overlap[i,j])>self.threshold_overlap:
                    relevant_pairs.append((i,j))
                    cross_mat_inv_list.append(self.sum_matrices_inv[i,j,:,:])
                    bvec=self.vectors[j].T@self.matrices[j]+conj(self.vectors[i].T@self.matrices[i])
                    newShift=self.sum_matrices_inv[i,j,:,:]@bvec
                    mu_list.append(newShift)
                    ovlp_list.append(self.overlap[i,j])
                    counter+=1
        mu_list=np.array(mu_list)
        cross_mat_inv_list=np.array(cross_mat_inv_list)
        ovlp_list=np.array(ovlp_list)
        #start=time.time()
        if self.h>=0:
            if self.onlyX1X2:
                x1_temp,x2_temp=calculate_all_expectation_values_full_allij(mu_list,cross_mat_inv_list,ovlp_list,order=2)
            else:
                x1_temp,x2_temp,x3_temp,x4_temp,x5_temp,x6_temp=calculate_all_expectation_values_full_allij(mu_list,cross_mat_inv_list,ovlp_list)
            #end=time.time()
            #print("Time for full calculation: ",end-start)
            #start=time.time()
            if self.calculate_Gradient and not self.onlyX1X2:
                x8_c1,x8_c2,x8_c3,x7_c1,x7_c2,x7_c1v,x7_c2v,x7_c3v,x7_c3,x7_c4=get_x7_x8_contraction_allij(mu_list,cross_mat_inv_list,ovlp_list)
            counter=0
                        
            for i,j in relevant_pairs:
                self.x1_expectation_values[i,j]=x1_temp[counter]
                self.x2_expectation_values[i,j]=x2_temp[counter]
                self.x1_expectation_values[j,i]=conj(x1_temp[counter])
                self.x2_expectation_values[j,i]=conj(x2_temp[counter])

                if not self.onlyX1X2:
                    self.x3_expectation_values[i,j]=x3_temp[counter]
                    self.x4_expectation_values[i,j]=x4_temp[counter]
                    self.x5_expectation_values[i,j]=x5_temp[counter]
                    self.x6_expectation_values[i,j]=x6_temp[counter]
                    self.x3_expectation_values[j,i]=conj(x3_temp[counter])
                    self.x4_expectation_values[j,i]=conj(x4_temp[counter])
                    self.x5_expectation_values[j,i]=conj(x5_temp[counter])
                    self.x6_expectation_values[j,i]=conj(x6_temp[counter])
                    if self.calculate_Gradient:
                        self.x8_c1[i,j,:]=x8_c1[counter]
                        self.x8_c2[i,j]=x8_c2[counter]
                        self.x8_c3[i,j]=x8_c3[counter]
                        self.x7_c1[i,j]=x7_c1[counter]
                        self.x7_c2[i,j]=x7_c2[counter]
                        self.x7_c3[i,j]=x7_c3[counter]
                        self.x7_c4[i,j]=x7_c4[counter]
                        self.x7_c1v[i,j]=x7_c1v[counter]
                        self.x7_c2v[i,j]=x7_c2v[counter]
                        self.x7_c3v[i,j]=x7_c3v[counter]
                        self.x8_c1[j,i]=np.conj(x8_c1[counter])
                        self.x8_c2[j,i]=np.conj(x8_c2[counter])
                        self.x8_c3[j,i]=np.conj(x8_c3[counter])
                        self.x7_c1[j,i]=np.conj(x7_c1[counter])
                        self.x7_c2[j,i]=np.conj(x7_c2[counter])
                        self.x7_c3[j,i]=np.conj(x7_c3[counter])
                        self.x7_c4[j,i]=np.conj(x7_c4[counter])
                        self.x7_c1v[j,i]=np.conj(x7_c1v[counter])
                        self.x7_c2v[j,i]=np.conj(x7_c2v[counter])
                        self.x7_c3v[j,i]=np.conj(x7_c3v[counter])
                counter+=1
        end=time.time()
        self.setupMatrices()

        self.fodmd=self.calculate_overlap_matrix_log_deriv()
    
    def calculate_gaussian_distances_from_origo(self):
        ng=self.num_gaussians
        centers=[]
        for i in range(ng):
            mu=real(self.vectors[i])
            q=imag(self.vectors[i])
            A=real(self.matrices[i,:,:])
            B=imag(self.matrices[i,:,:])
            z=((-q.T@B+mu.T@A)@np.linalg.inv(A)).T
            centers.append(sqrt(z.T@z))
        return centers
    def setupintermediates(self):
        nd=self.num_dimensions
        ng=self.num_gaussians
        #self.vas=np.eye(nd)
        self.L_matrices=np.zeros((ng,nd,nd))
        self.K_matrices=np.zeros((ng,nd,nd))
        self.Iab=np.zeros((nd,nd,nd,nd))
        self.IabLj=np.zeros((ng,nd,nd,nd,nd),dtype=complex)#Here, self.IabLj[j,a,b,:,:] #Gives me the IabLj matrix for a given j, a, b
        self.Kab=np.zeros((ng,nd,nd,nd,nd))#Here, self.Kab[j,a,b,:,:] #Gives me the Kab matrix for a given a,b

        for a in range(nd):
            for b in range(a+1):
                self.Iab[a,b,a,b]=1
                self.Kab[:,a,b,a,b]+=1#;self.Kab[:,a,b,b,a]+=1
        self.matrices=np.einsum("ijk,ilk->ijl",self.L_matrices,self.L_matrices)+1j*(self.K_matrices+np.einsum("ijk->ikj",self.K_matrices))
        for i in range(ng):
            self.L_matrices[i,:,:][np.tril_indices(nd)]=self.nonlin_params[i][:int(0.5*nd*(nd+1))]
            self.K_matrices[i,:,:][np.tril_indices(nd)]=self.nonlin_params[i][int(0.5*nd*(nd+1)):(nd*(nd+1))]
        self.vectors=self.nonlin_params[:,(nd*(nd+1)):(nd*(nd+1))+nd]+1j*self.nonlin_params[:,nd*(nd+1)+nd:]
        self.matrices=np.einsum("ijk,ilk->ijl",self.L_matrices,self.L_matrices)-1j
        self.matrices+=1j
        self.matrices+=1j*(self.K_matrices+np.einsum("ijk->ikj",self.K_matrices))
        #print(self.matrices)
        self.matrices_squared=np.einsum("ikx,ixl->ikl",self.matrices,self.matrices)
        self.matrices_squared_conj=conj(self.matrices_squared)
        self.vmmips=np.einsum("ia,iab,ib->i",self.vectors,self.matrices_squared,self.vectors)
        self.cjs=np.einsum("jii->j",self.matrices)-2*self.vmmips
        self.cis=np.conj(self.cjs)
        self.omega_T=np.einsum("ik,ikl->il",self.vectors,self.matrices_squared)
        self.rho_T_conjs=conj(self.omega_T)
        self.Kab+=+np.swapaxes(self.Kab,-1,-2)           
        self.IabLj=np.einsum("abkl,jml->jabkm",self.Iab,self.L_matrices)
        self.IabLj=self.IabLj+np.swapaxes(self.IabLj,-1,-2)
        self.Kabmuj=np.einsum("jablk,jk->jabl",self.Kab,self.vectors)
        self.IabLjmuj=np.einsum("jablk,jk->jabl",self.IabLj,self.vectors)
        self.mKm=np.einsum("jk,jabk->jab",self.vectors,self.Kabmuj)
        self.mLm=np.einsum("jk,jabk->jab",self.vectors,self.IabLjmuj)
        IabLj_list=[]
        IabLjmuj_list=[]
        Kabmuj_list=[]
        Kab_list=[]
        self.ab_iter=0
        
        for a in range(nd):
             for b in range(a+1):
                self.ab_iter+=1
                IabLj_list.append(self.IabLj[:,a,b])
                IabLjmuj_list.append(self.IabLjmuj[:,a,b])
                Kabmuj_list.append(self.Kabmuj[:,a,b])
                Kab_list.append(self.Kab[:,a,b])
        self.Kab_list=np.array(Kab_list)
        self.Kabmuj_list=np.array(Kabmuj_list)

        self.IabLj_list=np.array(IabLj_list)
        #self.IabLj_list=self.IabLj.reshape()
        self.IabLjmuj_list=np.array(IabLjmuj_list)
        
        
        self.sum_matrices_inv=np.empty((ng,ng,nd,nd),dtype=complex)
        self.sum_matrices,self.sum_matrices_inv=make_sum_matrices(self.matrices,np.empty((ng,ng,nd,nd),dtype=complex),None) #Those matrices are changed in-place
        self.overlap=self.calculate_overlap_matrix_normalized()
        #self.overlap[abs(self.overlap)<1e-5]=0
        self.coefficients=np.asarray(self.lin_params)
    def update_params_overlapFitting(self,indices,new_nonlin_params,new_lin_params=None,onlyX1X2=False):
        """
        This function, which is not implemented so far, "updates" the Gaussians at indices "indices" and then recalculates 
        all expectation values and derivatives REQUIRED for the overlap fitting (such as e.g. a mask).
        By "updating" the Gaussians, we mean that we change the parameters of the Gaussians at the given indices.
        Naturally, the number of indices must be equal to the number of new_params.

        """
        #Step 0: Set up the matrices and vectors of the new Gaussians.
        nd=self.num_dimensions
        ng=self.num_gaussians
        num_new_gauss=len(indices)
        new_L_matrices=np.zeros((num_new_gauss,nd,nd))
        new_K_matrices=np.zeros((num_new_gauss,nd,nd))
        for i in range(num_new_gauss):
            new_L_matrices[i,:,:][np.tril_indices(nd)]=new_nonlin_params[i][:int(0.5*nd*(nd+1))]
            new_K_matrices[i,:,:][np.tril_indices(nd)]=new_nonlin_params[i][int(0.5*nd*(nd+1)):(nd*(nd+1))]
            self.nonlin_params[indices[i]]=new_nonlin_params[i]
        if new_lin_params is not None:
            for i in range(num_new_gauss):
                self.lin_params[indices[i]]=new_lin_params[i]
        new_vectors=new_nonlin_params[:,(nd*(nd+1)):(nd*(nd+1))+nd]+1j*new_nonlin_params[:,nd*(nd+1)+nd:]
        new_matrices=np.einsum("ijk,ilk->ijl",new_L_matrices,new_L_matrices)-1j
        new_matrices+=1j
        new_matrices+=1j*(new_K_matrices+np.einsum("ijk->ikj",new_K_matrices))
        for i in range(num_new_gauss):
            self.L_matrices[indices[i]]=new_L_matrices[i]
            self.K_matrices[indices[i]]=new_K_matrices[i]
            self.vectors[indices[i]]=new_vectors[i]
            self.matrices[indices[i]]=new_matrices[i]
  
        self.matrices_squared=np.einsum("ikx,ixl->ikl",self.matrices,self.matrices)
        self.matrices_squared_conj=conj(self.matrices_squared)
        self.vmmips=np.einsum("ia,iab,ib->i",self.vectors,self.matrices_squared,self.vectors)
        self.cjs=np.einsum("jii->j",self.matrices)-2*self.vmmips
        self.cis=np.conj(self.cjs)
        self.omega_T=np.einsum("ik,ikl->il",self.vectors,self.matrices_squared)
        self.rho_T_conjs=conj(self.omega_T)
        self.IabLj=np.einsum("abkl,jml->jabkm",self.Iab,self.L_matrices)
        self.IabLj=self.IabLj+np.swapaxes(self.IabLj,-1,-2)
        self.Kabmuj=np.einsum("jablk,jk->jabl",self.Kab,self.vectors)
        self.IabLjmuj=np.einsum("jablk,jk->jabl",self.IabLj,self.vectors)
        self.mKm=np.einsum("jk,jabk->jab",self.vectors,self.Kabmuj)
        self.mLm=np.einsum("jk,jabk->jab",self.vectors,self.IabLjmuj)
        IabLj_list=[]
        IabLjmuj_list=[]
        Kabmuj_list=[]
        Kab_list=[]
        self.ab_iter=0
        
        for a in range(nd):
             for b in range(a+1):
                self.ab_iter+=1
                IabLj_list.append(self.IabLj[:,a,b])
                IabLjmuj_list.append(self.IabLjmuj[:,a,b])
                Kabmuj_list.append(self.Kabmuj[:,a,b])
                Kab_list.append(self.Kab[:,a,b])
        self.Kab_list=np.array(Kab_list)
        self.Kabmuj_list=np.array(Kabmuj_list)

        self.IabLj_list=np.array(IabLj_list)
        self.IabLjmuj_list=np.array(IabLjmuj_list)
        """
        Notes for myself:
        - The overlap matrix is an ng x ng matrix, where ng is the number of Gaussians.
        - When updating indices [i,j,k,...], we need to recalculate the overlap matrix for all relevant pairs of Gaussians.
        """
        pairs=[]
        for i in range(len(indices)):
            for j in range(self.num_gaussians):
                index_pair=(indices[i],j)
                inverse_index_pair=(j,indices[i])
                if index_pair not in pairs and inverse_index_pair not in pairs:
                    pairs.append(index_pair)
        self.sum_matrices,self.sum_matrices_inv=update_sum_matrices(self.matrices,pairs,self.sum_matrices,self.sum_matrices_inv)
        self.overlap=update_overlap_matrix(self.overlap,pairs,self.matrices,self.vectors,self.sum_matrices,self.sum_matrices_inv)
        mu_list=[]
        cross_mat_inv_list=[]
        ovlp_list=[]

        counter=0
        nonzero_pairs=[]
        for i,j in pairs:
            if abs(self.overlap[i,j])>self.threshold_overlap:
                nonzero_pairs.append((i,j))
                cross_mat_inv_list.append(self.sum_matrices_inv[i,j,:,:])
                bvec=self.vectors[j].T@self.matrices[j]+conj(self.vectors[i].T@self.matrices[i])
                newShift=self.sum_matrices_inv[i,j,:,:]@bvec
                mu_list.append(newShift)
                ovlp_list.append(self.overlap[i,j])
        # Step 2: Recalculate the expectation values
        mu_list=np.array(mu_list)
        cross_mat_inv_list=np.array(cross_mat_inv_list)
        ovlp_list=np.array(ovlp_list)
        #start=time.time()
        if not onlyX1X2:
            x1_temp,x2_temp,x3_temp,x4_temp,x5_temp,x6_temp=calculate_all_expectation_values_full_allij(mu_list,cross_mat_inv_list,ovlp_list)
        else:
            x1_temp,x2_temp=calculate_all_expectation_values_full_allij(mu_list,cross_mat_inv_list,ovlp_list,order=2)
        if self.calculate_Gradient and not onlyX1X2:
            x8_c1,x8_c2,x8_c3,x7_c1,x7_c2,x7_c1v,x7_c2v,x7_c3v,x7_c3,x7_c4=get_x7_x8_contraction_allij(mu_list,cross_mat_inv_list,ovlp_list)
        counter=0
        for i,j in nonzero_pairs:
            self.x1_expectation_values[i,j]=x1_temp[counter]
            self.x2_expectation_values[i,j]=x2_temp[counter]
            self.x1_expectation_values[j,i]=conj(x1_temp[counter])
            self.x2_expectation_values[j,i]=conj(x2_temp[counter])
            if not onlyX1X2:
                self.x3_expectation_values[i,j]=x3_temp[counter]
                self.x4_expectation_values[i,j]=x4_temp[counter]
                
                self.x3_expectation_values[j,i]=conj(x3_temp[counter])
                self.x4_expectation_values[j,i]=conj(x4_temp[counter])
                self.x5_expectation_values[i,j]=x5_temp[counter]
                self.x6_expectation_values[i,j]=x6_temp[counter]
                self.x5_expectation_values[j,i]=conj(x5_temp[counter])
                self.x6_expectation_values[j,i]=conj(x6_temp[counter])
                if self.calculate_Gradient:
                    self.x8_c1[i,j,:]=x8_c1[counter]
                    self.x8_c2[i,j]=x8_c2[counter]
                    self.x8_c3[i,j]=x8_c3[counter]
                    self.x7_c1[i,j]=x7_c1[counter]
                    self.x7_c2[i,j]=x7_c2[counter]
                    self.x7_c3[i,j]=x7_c3[counter]
                    self.x7_c4[i,j]=x7_c4[counter]
                    self.x7_c1v[i,j]=x7_c1v[counter]
                    self.x7_c2v[i,j]=x7_c2v[counter]
                    self.x7_c3v[i,j]=x7_c3v[counter]
                    self.x8_c1[j,i]=np.conj(x8_c1[counter])
                    self.x8_c2[j,i]=np.conj(x8_c2[counter])
                    self.x8_c3[j,i]=np.conj(x8_c3[counter])
                    self.x7_c1[j,i]=np.conj(x7_c1[counter])
                    self.x7_c2[j,i]=np.conj(x7_c2[counter])
                    self.x7_c3[j,i]=np.conj(x7_c3[counter])
                    self.x7_c4[j,i]=np.conj(x7_c4[counter])
                    self.x7_c1v[j,i]=np.conj(x7_c1v[counter])
                    self.x7_c2v[j,i]=np.conj(x7_c2v[counter])
                    self.x7_c3v[j,i]=np.conj(x7_c3v[counter])
            counter+=1
        #end=time.time()
        #print("Time for calculating all expectation values: ",end-start)
        self.setupMatrices()
        #end=time.time()
        #print("Time for calculating all expectation values: ",end-start)
        self.fodmd=self.calculate_overlap_matrix_log_deriv()
        #end=time.time()
        #print("Time for calculating all expectation values: ",end-start)
    def calculate_overlap_matrix_log_deriv(self):
        #This calculates a matrix, where the element D[i,n] is the derivative \frac{\partial}{\partial n_i}log(<g_i|g_i>), where $n_i$ is one of the
        #n(n+3) parameters specifying the Gaussian. 
        ng=self.num_gaussians
        nd=self.num_dimensions
        D_tensor=np.zeros((ng,nd*(nd+3)),dtype=float)
        As=real(self.matrices)
        Bs=imag(self.matrices)
        Ainvs=np.linalg.inv(As)
        for i in range(ng):
            A=As[i]
            Ainv=Ainvs[i]
            L=self.L_matrices[i]
            B=Bs[i]
            q=imag(self.vectors[i])
            counter=0
            for a in range(nd):
                for b in range(a+1):
                    Aderiv=real(self.IabLj[i,a,b])
                    if a==b:
                        D_tensor[i,counter]+=-1/L[a,a]
                    D_tensor[i,counter]+=2*q.T@(-B@Ainv@Aderiv@Ainv@B+Aderiv)@q
                    Bderiv=real(self.Kab[i,a,b])
                    D_tensor[i,counter+(nd*(nd+1))//2]=2*(q.T@B@Ainv@Bderiv@q+q.T@Bderiv@Ainv@B@q)
                    counter+=1
            D_tensor[i,nd*(nd+2):]=4*(B@Ainv@B+A)@q
        return D_tensor
    def calculate_overlap_matrix_diagonal_log(self):
        ng=self.num_gaussians
        nd=self.num_dimensions
        logmat=np.zeros(ng)
        invs=np.linalg.inv(real(self.matrices))
        for i in range(ng):
            A=real(self.matrices[i])
            L=self.L_matrices[i]
            B=imag(self.matrices[i])
            q=imag(self.vectors[i])
            logdet=-0.5*np.sum(log(np.diagonal(L)**2))
            summy=2*(q.T@B@invs[i]@B@q+q.T@A@q)
            logmat[i]=summy+logdet+nd/2*log(pi/2)
        return logmat

    def calculate_overlap_matrix_normalized(self):
        self.overlap=np.ones((self.num_gaussians,self.num_gaussians),dtype=np.complex128)
        vmips=np.einsum("ia,iab,ib->i",self.vectors,self.matrices,self.vectors)
        diagonal_determinants=np.zeros(self.num_gaussians,dtype=complex)
        diagonal_sum_matrix =np.diagonal(self.sum_matrices,axis1=0,axis2=1).transpose([2,0,1]) #Same as np.einsum('iijk->ijk', self.sum_matrices)
        diagonal_sum_matrices_inv = np.diagonal(self.sum_matrices_inv,axis1=0,axis2=1).transpose([2,0,1])
        diagonal_determinants=[]
        diagonal_determinants=sqrt(det(diagonal_sum_matrix))
        bvecs_list=np.einsum("ak,akl->al",self.vectors,self.matrices)
        bvec_calc=2*np.real(bvecs_list)
        bvec_outer= np.conj(bvecs_list)[:,np.newaxis,:]+(bvecs_list)[np.newaxis,:,:]
        vmips_outer=np.conj(vmips[:,np.newaxis])+vmips[np.newaxis,:]

        diagonal_exponents=np.einsum("ak,akl,al->a",bvec_calc,diagonal_sum_matrices_inv,bvec_calc)-vmips-vmips.conj()
        diagonal_exponents_outer=(diagonal_exponents[:,np.newaxis]+diagonal_exponents[np.newaxis,:])
        eigvals=np.linalg.eigvals(self.sum_matrices)
        sqrt_eigvals=sqrt(eigvals) #Principal square root of eigenvalues
        inv_dets=1/np.prod(sqrt_eigvals,axis=2)
        sqrt_diagonal_dets=sqrt(diagonal_determinants)
        self.overlap=np.einsum("ij,i,j->ij",inv_dets,sqrt_diagonal_dets,sqrt_diagonal_dets)
        exponents=np.einsum("ija,ijab,ijb->ij",bvec_outer,self.sum_matrices_inv,bvec_outer)-vmips_outer-0.5*diagonal_exponents_outer
        self.overlap*=np.exp(exponents)
        return self.overlap

    def calculate_full_overlap_deriv_matrix(self,start_index_WFnew):
        #This calculates a 3-tensor, where the element D[i,j,n] is the derivative <g_i|\frac{\partial}{\partial n_j}|g_j>, where $n_j$ is one of the
        #n(n+3) parameters specifying the Gaussian. 
        #Returns the matrix deriv.shape = [num_gaussians_tot,num_gaussians_newWF,num_params_per_gaussians]
        ng=self.num_gaussians
        nd=self.num_dimensions
        ngMs=start_index_WFnew
        D_tensor=np.zeros((ng,ng-start_index_WFnew,nd*(nd+3)),dtype=complex)
        D_tensor_mu=np.zeros((ng,ng-start_index_WFnew,nd),dtype=complex)
        D_tensor[:,:,:self.ab_iter]=-einsum("ijkl,cjkl->ijc",self.x2_expectation_values[:,ngMs:],self.IabLj_list[:,ngMs:])
        D_tensor[:,:,:self.ab_iter]+=2*einsum("ijk,cjk->ijc",self.x1_expectation_values[:,ngMs:],self.IabLjmuj_list[:,ngMs:])
        D_tensor[:,:,:self.ab_iter]+=-1*einsum("ij,jk,cjk->ijc",self.overlap[:,ngMs:],self.vectors[ngMs:],self.IabLjmuj_list[:,ngMs:])
        D_tensor[:,:,self.ab_iter:2*self.ab_iter]=-1j*einsum("ijab,cjab->ijc",self.x2_expectation_values[:,ngMs:],self.Kab_list[:,ngMs:])
        D_tensor[:,:,self.ab_iter:2*self.ab_iter]+=2j*einsum("ijk,cjk->ijc",self.x1_expectation_values[:,ngMs:],self.Kabmuj_list[:,ngMs:])
        D_tensor[:,:,self.ab_iter:2*self.ab_iter]+=-1j*einsum("ij,jk,cjk->ijc",self.overlap[:,ngMs:],self.vectors[ngMs:],self.Kabmuj_list[:,ngMs:])
        D_tensor_mu=-2*einsum("jak,jk,ij->ija",self.matrices[ngMs:,:,:],self.vectors[ngMs:,:],self.overlap[:,ngMs:])
        D_tensor_mu+=2*einsum("jak,ijk->ija",self.matrices[ngMs:,:,:],self.x1_expectation_values[:,ngMs:,:,])
        D_tensor[:,:,nd*(nd+1):nd*(nd+2)]=D_tensor_mu
        D_tensor[:,:,nd*(nd+2):nd*(nd+3)]=1j*D_tensor_mu
        fodmd=self.fodmd
        D_tensor+=-0.5*np.einsum("ij,jk->ijk",self.overlap[:,ngMs:],fodmd[ngMs:,:])

                    
        return D_tensor


    def calculate_potential(self):
        self.potential=0.5*einsum("ijkk->ij",self.x2_expectation_values)
        self.potential+=-self.lambda_/3*einsum("ijkkk->ij",self.x3_expectation_values[:,:,1:,1:,1:]) #First type of third order terms
        self.potential+=self.lambda_*einsum("ijkkk->ij",self.x3_expectation_values[:,:,0:-1,0:-1,1:])
        return self.potential
    def calculate_full_potential_deriv_matrix(self,start_index_WFnew):
        #This calculates a 3-tensor, where the element D[i,j,n] is the derivative <g_i|V(\vec{x})\frac{\partial}{\partial n_j}|g_j>, where $n_j$ is one of the
        #n(n+3) parameters specifying the Gaussian. 
        ng=self.num_gaussians
        nd=self.num_dimensions
        ngMs=start_index_WFnew
        D_tensor=np.zeros((ng,ng-start_index_WFnew,nd*(nd+3)),dtype=complex)

        potential=self.potential
        x5_contr1=einsum("ijlmkkk->ijlm",self.x5_expectation_values[:,ngMs:,:,:,1:,1:,1:])
        x5_contr2=einsum("ijlmkkk->ijlm",self.x5_expectation_values[:,ngMs:,:,:,0:-1,0:-1,1:])
        x5_contr=self.lambda_/3*x5_contr1-self.lambda_*x5_contr2
        x4_contr=einsum("ijlkkk->ijl",-self.lambda_/3*2*self.x4_expectation_values[:,ngMs:,:,1:,1:,1:])
        x4_contr+=einsum("ijlkkk->ijl",2*self.lambda_*self.x4_expectation_values[:,ngMs:,:,0:-1,0:-1,1:])
        contr1=einsum("ij,jk->ijk",potential[:,ngMs:],self.vectors[ngMs:])
        contr2=x5_contr-0.5*einsum("ijabkk->ijab",self.x4_expectation_values[:,ngMs:])
        contr3=einsum("ijlkk->ijl",self.x3_expectation_values[:,ngMs:])-contr1+x4_contr

        D_tensor[:,:,:self.ab_iter]+=einsum("ijl,cjl->ijc",contr3,self.IabLjmuj_list[:,ngMs:])
        D_tensor[:,:,:self.ab_iter]+=einsum("cjab,ijab->ijc",self.IabLj_list[:,ngMs:],contr2)
        D_tensor[:,:,self.ab_iter:2*self.ab_iter]+=einsum("ijl,cjl->ijc",contr3,self.Kabmuj_list[:,ngMs:])
        D_tensor[:,:,self.ab_iter:2*self.ab_iter]+=einsum("cjab,ijab->ijc",self.Kab_list[:,ngMs:],contr2)
        D_tensor[:,:,self.ab_iter:2*self.ab_iter]*=1j
                        
        #sys.exit(1)
        D_tensor[:,:,nd*(nd+1):nd*(nd+2)]+=einsum("jak,ijkll->ija",self.matrices[ngMs:,:,:],self.x3_expectation_values[:,ngMs:,:,:,:])
        D_tensor[:,:,nd*(nd+1):nd*(nd+2)]+=einsum("jak,ijk->ija",self.matrices[ngMs:,:,:],x4_contr)
        D_tensor[:,:,nd*(nd+1):nd*(nd+2)]+=-2*einsum("jak,jk,ij->ija",self.matrices[ngMs:,:,:],self.vectors[ngMs:,:],potential[:,ngMs:])
        D_tensor[:,:,nd*(nd+2):nd*(nd+3)]=1j*D_tensor[:,:,nd*(nd+1):nd*(nd+2)]

        fodmd=self.fodmd
        D_tensor+=-0.5*np.einsum("ij,jk->ijk",self.potential[:,ngMs:],fodmd[ngMs:,:])


        return D_tensor

    def calculate_kinetic_energy(self):
        cj=np.einsum("jii->j",self.matrices)-2*self.vmmips
        j_linear=4*np.einsum("ji,jik->jk",self.vectors,self.matrices_squared)
        self.kinetic=einsum("j,ij->ij",cj,self.overlap)
        self.kinetic+=einsum("jk,ijk->ij",j_linear,self.x1_expectation_values)
        self.kinetic+=-2*einsum("ijab,jab->ij",self.x2_expectation_values,self.matrices_squared)
        return self.kinetic
    def calculate_full_kinetic_energy_deriv(self,start_index_WFnew): 
        #start with the bra expression, e.g. nabla^2 acting on the bra, that's a hack to not have to derive the ket 
        #This calculates a 3-tensor, where the element D[i,j,n] is the derivative <g_i|-(1/2 laplace)\frac{\partial}{\partial n_j}|g_j>, where $n_j$ is one of the
        #n(n+3) parameters specifying the Gaussian. 
        ng=self.num_gaussians
        nd=self.num_dimensions
        ngMs=start_index_WFnew
        D_tensor=np.zeros((ng,ng-start_index_WFnew,nd*(nd+3)),dtype=complex)
        kinetic=self.kinetic
        cjs=self.cjs
        cis=self.cis
        matrices_squared_conj=self.matrices_squared_conj
        rho_T_conjs=self.rho_T_conjs
        C2=-einsum("i,ijnm->ijnm",cis,self.x2_expectation_values[:,ngMs:])
        C1=-einsum("i,ij,jk->ijk",cis,self.overlap[:,ngMs:],self.vectors[ngMs:])
        C1+=2*einsum("i,ijk->ijk",cis,self.x1_expectation_values[:,ngMs:])
        C1+=-einsum("ij,jk->ijk",kinetic[:,ngMs:],self.vectors[ngMs:])
        C1+=einsum("i,ij,jk->ijk",cis,self.overlap[:,ngMs:],self.vectors[ngMs:])
        C2+=2*einsum("ijnmkl,imn->ijkl",self.x4_expectation_values[:,ngMs:],matrices_squared_conj)
        C1+=-4*einsum("ijnmk,imn->ijk",self.x3_expectation_values[:,ngMs:],matrices_squared_conj)
        C2+=-4*einsum("ijnkl,in->ijkl",self.x3_expectation_values[:,ngMs:],rho_T_conjs)
        C1+=8*einsum("ijnk,ik->ijn",self.x2_expectation_values[:,ngMs:],rho_T_conjs)
        D_tensor[:,:,:self.ab_iter]+=einsum("ijnm,cjnm->ijc",C2,self.IabLj_list[:,ngMs:])
        D_tensor[:,:,:self.ab_iter]+=einsum("ijk,cjk->ijc",C1,self.IabLjmuj_list[:,ngMs:])
        D_tensor[:,:,self.ab_iter:2*self.ab_iter]+=einsum("ijnm,cjnm->ijc",C2,self.Kab_list[:,ngMs:])
        D_tensor[:,:,self.ab_iter:2*self.ab_iter]+=einsum("ijk,cjk->ijc",C1,self.Kabmuj_list[:,ngMs:])

        D_tensor[:,:,self.ab_iter:2*self.ab_iter]*=1j
        D_tensor_mu=2*einsum("i,jaf,ijf->ija",cis,self.matrices[ngMs:,:,:],self.x1_expectation_values[:,ngMs:,:])
        D_tensor_mu+=-2*einsum("ij,jaf,jf->ija",kinetic[:,ngMs:],self.matrices[ngMs:],self.vectors[ngMs:,:])
        D_tensor_mu+=8*einsum("jaf,ig,ijfg->ija",self.matrices[ngMs:,:,:],self.rho_T_conjs,self.x2_expectation_values[:,ngMs:])
        D_tensor_mu+=-4*einsum("ifg,jak,ijfgk->ija",matrices_squared_conj,self.matrices[ngMs:,:,:],self.x3_expectation_values[:,ngMs:])

        D_tensor[:,:,nd*(nd+1):nd*(nd+2)]=D_tensor_mu
        D_tensor[:,:,nd*(nd+2):nd*(nd+3)]=1j*D_tensor[:,:,nd*(nd+1):nd*(nd+2)]
 
        fodmd=self.fodmd
        D_tensor+=-0.5*np.einsum("ij,jk->ijk",kinetic[:,ngMs:],fodmd[ngMs:,:])
        return D_tensor


    def calculate_kinetic_energy_squared(self):
        self.kinetic_energy_squared=np.empty((self.num_gaussians,self.num_gaussians),dtype=np.complex128)
        cjs=self.cjs
        cis=self.cis
        rho_T_conjs=self.rho_T_conjs
        self.kinetic_energy_squared=-2*einsum("a,bij,abij->ab",cis,self.matrices_squared,self.x2_expectation_values)
        self.kinetic_energy_squared+=-2*einsum("b,aij,abij->ab",cjs,self.matrices_squared_conj,self.x2_expectation_values)
        self.kinetic_energy_squared+=16*einsum("ai,bj,abij->ab",rho_T_conjs,self.omega_T,self.x2_expectation_values)
        self.kinetic_energy_squared+=4*einsum("a,bj,abj->ab",cis,self.omega_T,self.x1_expectation_values)
        self.kinetic_energy_squared+=4*einsum("b,aj,abj->ab",cjs,rho_T_conjs,self.x1_expectation_values)
        self.kinetic_energy_squared-=8*einsum("bi,ajm,abijm->ab",self.omega_T,self.matrices_squared_conj,self.x3_expectation_values)
        self.kinetic_energy_squared-=8*einsum("ai,bjm,abijm->ab",rho_T_conjs,(self.matrices_squared),self.x3_expectation_values)

        self.kinetic_energy_squared+=einsum("a,b,ab->ab",cis,cjs,self.overlap)
        self.kinetic_energy_squared+=4*einsum("aij,bmn,abijmn->ab",self.matrices_squared_conj,self.matrices_squared,self.x4_expectation_values)
        return self.kinetic_energy_squared
    def calculate_full_kinetic_energy_squared_deriv(self,start_index_WFnew):
        #start with the bra expression, e.g. nabla^2 acting on the bra, that's a hack to not have to derive the ket 
        #This calculates a 3-tensor, where the element D[i,j,n] is the derivative <g_i|-(1/2 laplace)\frac{\partial}{\partial n_j}|g_j>, where $n_j$ is one of the
        #n(n+3) parameters specifying the Gaussian. 
        ng=self.num_gaussians
        nd=self.num_dimensions
        ngMs=start_index_WFnew
        D_tensor=np.zeros((ng,ng-start_index_WFnew,nd*(nd+3)),dtype=complex)
        cjs=self.cjs
        cis=self.cis
        matrices_squared_conj=self.matrices_squared_conj

        x5rhomat=einsum("abiklmn,ai->abklmn",self.x5_expectation_values,self.rho_T_conjs)
        x6_eAisqconjmat=einsum("abklijmn,aij->abklmn",self.x6_expectation_values,matrices_squared_conj)
        x5_eAisqconjmat=einsum("abkijmn,aij->abkmn",self.x5_expectation_values,matrices_squared_conj)
        tempAjsqx6_eAisqconj=-4*einsum("jkl,ijklmn->ijmn",self.matrices_squared,x6_eAisqconjmat)
        tempx5_eAjsqAisqconj=8*einsum("abmkl,bkl->abm",x5_eAisqconjmat,self.matrices_squared)
        rho_T_conjs=self.rho_T_conjs
        Asquared_derivs_I=einsum("jabkx,jxl->jabkl",self.IabLj[ngMs:],self.matrices[ngMs:])
        Asquared_derivs_I+=einsum("jkx,jabxl->jabkl",self.matrices[ngMs:],self.IabLj[ngMs:])
        Asquared_derivs_K=einsum("jabkx,jxl->jabkl",self.Kab[ngMs:],self.matrices[ngMs:])
        Asquared_derivs_K+=einsum("jkx,jabxl->jabkl",self.matrices[ngMs:],self.Kab[ngMs:])

        cj_derivs_I=einsum("jabkk->jab",self.IabLj[ngMs:],dtype=complex)
        cj_derivs_I+=-2*einsum("jk,jabkl,jl->jab",self.vectors[ngMs:],Asquared_derivs_I,self.vectors[ngMs:])
        cj_derivs_K=einsum("jabkk->jab",self.Kab[ngMs:],dtype=complex)
        cj_derivs_K+=-2*einsum("jk,jabkl,jl->jab",self.vectors[ngMs:],Asquared_derivs_K,self.vectors[ngMs:])
        omega_derivs_K=einsum("jk,jablk->jabl",self.vectors[ngMs:],Asquared_derivs_K)
        omega_derivs_I=einsum("jk,jablk->jabl",self.vectors[ngMs:],Asquared_derivs_I)
        cj_derivs_I_list=[]
        cj_derivs_K_list=[]
        omega_derivs_K_list=[]
        omega_derivs_I_list=[]
        mKm_list=[]
        mLm_list=[]
        Asquared_derivs_I_list=[]
        Asquared_derivs_K_list=[]
        self.ab_iter=0
        for a in range(nd):
             for b in range(a+1):
                self.ab_iter+=1
                cj_derivs_I_list.append(cj_derivs_I[:,a,b])
                cj_derivs_K_list.append(cj_derivs_K[:,a,b])
                omega_derivs_K_list.append(omega_derivs_K[:,a,b,:])
                omega_derivs_I_list.append(omega_derivs_I[:,a,b,:])
                mLm_list.append(self.mLm[:,a,b])
                mKm_list.append(self.mKm[:,a,b])
                Asquared_derivs_I_list.append(Asquared_derivs_I[:,a,b])
                Asquared_derivs_K_list.append(Asquared_derivs_K[:,a,b])

        cj_derivs_I_list=np.array(cj_derivs_I_list)
        cj_derivs_K_list=np.array(cj_derivs_K_list)
        omega_derivs_K_list=np.array(omega_derivs_K_list)
        mLm_list=np.array(mLm_list)
        mKm_list=np.array(mKm_list)
        omega_derivs_I_list=np.array(omega_derivs_I_list)
        Asquared_derivs_K_list=np.array(Asquared_derivs_K_list)
        Asquared_derivs_I_list=np.array(Asquared_derivs_I_list)
        temp1=einsum("i,ij->ij",cis,self.overlap[:,ngMs:])
       
        temp1+=4*einsum("ik,ijk->ij",rho_T_conjs,self.x1_expectation_values[:,ngMs:])
        temp1+=-2*einsum("ijkl,ikl->ij",self.x2_expectation_values[:,ngMs:],matrices_squared_conj)

        temp3=4*einsum("i,ijk->ijk",cis,self.x1_expectation_values[:,ngMs:])
        temp3+=16*einsum("il,ijkl->ijk",rho_T_conjs,self.x2_expectation_values[:,ngMs:])
        temp3+=-8*einsum("ilm,ijklm->ijk",matrices_squared_conj,self.x3_expectation_values[:,ngMs:])
        temp4=-2*einsum("i,ijnm->ijnm",cis,self.x2_expectation_values[:,ngMs:])
        temp4+=-8*einsum("ik,ijnmk->ijnm",rho_T_conjs,self.x3_expectation_values[:,ngMs:])
        temp4+=4*einsum("ikl,ijnmkl->ijnm",matrices_squared_conj,self.x4_expectation_values[:,ngMs:])

        temp2=-einsum("j,i,ij->ij",cjs[ngMs:],cis,self.overlap[:,ngMs:])
        

        temp2+=-4*einsum("j,ijk,ik->ij",cjs[ngMs:],self.x1_expectation_values[:,ngMs:],rho_T_conjs)
        temp2+=-4*einsum("i,jk,ijk->ij",cis,self.omega_T[ngMs:],self.x1_expectation_values[:,ngMs:])
        temp2+=2*einsum("j,ikl,ijkl->ij",cjs[ngMs:],matrices_squared_conj,self.x2_expectation_values[:,ngMs:])
        temp2+=2*einsum("i,jkl,ijkl->ij",cis,self.matrices_squared[ngMs:],self.x2_expectation_values[:,ngMs:])
        temp2+=-16*einsum("jk,ijkl,il->ij",self.omega_T[ngMs:],self.x2_expectation_values[:,ngMs:],rho_T_conjs)
        temp2+=8*einsum("jk,ijklm,ilm->ij",self.omega_T[ngMs:],self.x3_expectation_values[:,ngMs:],matrices_squared_conj)
        temp2+=8*einsum("ijmkl,im,jkl->ij",self.x3_expectation_values[:,ngMs:],rho_T_conjs,self.matrices_squared[ngMs:])
        temp2+=-4*einsum("jkl,ijklmn,imn->ij",self.matrices_squared[ngMs:],self.x4_expectation_values[:,ngMs:],matrices_squared_conj)

        temp5=2*einsum("j,i,ijm->ijm",cjs[ngMs:],cis,self.x1_expectation_values[:,ngMs:])
        temp5+=8*einsum("j,il,ijml->ijm",cjs[ngMs:],rho_T_conjs,self.x2_expectation_values[:,ngMs:])
        temp5+=-4*einsum("j,ijmkl,ikl->ijm",cjs[ngMs:],self.x3_expectation_values[:,ngMs:],matrices_squared_conj)
        temp5+=8*einsum("i,jk,ijmk->ijm",cis,self.omega_T[ngMs:],self.x2_expectation_values[:,ngMs:])
        temp5+=-4*einsum("i,jkl,ijmkl->ijm",cis,self.matrices_squared[ngMs:],self.x3_expectation_values[:,ngMs:])
        temp5+=32*einsum("ijmkl,jk,il->ijm",self.x3_expectation_values[:,ngMs:],self.omega_T[ngMs:],rho_T_conjs)
        temp5+=-16*einsum("ijmkln,jk,iln->ijm",self.x4_expectation_values[:,ngMs:],self.omega_T[ngMs:],matrices_squared_conj)
        temp5+=-16*einsum("ijmnkl,in,jkl->ijm",self.x4_expectation_values[:,ngMs:],rho_T_conjs,self.matrices_squared[ngMs:])
        temp5+=tempx5_eAjsqAisqconj[:,ngMs:]

        temp6=2*einsum("j,ikl,ijklmn->ijmn",cjs[ngMs:],matrices_squared_conj,self.x4_expectation_values[:,ngMs:])
        temp6+=-16*einsum("jk,ijklmn,il->ijmn",self.omega_T[ngMs:],self.x4_expectation_values[:,ngMs:],rho_T_conjs)
        temp6+=8*einsum("jk,ijkmn->ijmn",self.omega_T[ngMs:],x5_eAisqconjmat[:,ngMs:])
        temp6+=8*einsum("ijklmn,jkl->ijmn",x5rhomat[:,ngMs:],self.matrices_squared[ngMs:])
        temp6+=tempAjsqx6_eAisqconj[:,ngMs:]
        temp6+=-einsum("j,i,ijkl->ijkl",cjs[ngMs:],cis,self.x2_expectation_values[:,ngMs:])
        temp6+=-4*einsum("j,ijlmn,il->ijmn",cjs[ngMs:],self.x3_expectation_values[:,ngMs:],rho_T_conjs)
        temp6+=-4*einsum("i,jk,ijkmn->ijmn",cis,self.omega_T[ngMs:],self.x3_expectation_values[:,ngMs:])
        temp6+=2*einsum("i,jkl,ijklmn->ijmn",cis,self.matrices_squared[ngMs:],self.x4_expectation_values[:,ngMs:])
        D_tensor[:,:,:self.ab_iter]+=einsum("cj,ij->ijc",mLm_list[:,ngMs:],temp2)
        D_tensor[:,:,:self.ab_iter]+=einsum("cjnm,ijnm->ijc",Asquared_derivs_I_list[:,:],temp4)
        D_tensor[:,:,:self.ab_iter]+=einsum ("ij,cj->ijc",temp1,cj_derivs_I_list[:,:])
        D_tensor[:,:,:self.ab_iter]+= einsum("cjk,ijk->ijc",omega_derivs_I_list,temp3)

        D_tensor[:,:,:self.ab_iter]+=einsum("cjm,ijm->ijc",self.IabLjmuj_list[:,ngMs:],temp5)
        D_tensor[:,:,:self.ab_iter]+=einsum("ijmn,cjnm->ijc",temp6,self.IabLj_list[:,ngMs:])

        D_tensor[:,:,self.ab_iter:2*self.ab_iter]+=einsum("cj,ij->ijc",mKm_list[:,ngMs:],temp2)
        D_tensor[:,:,self.ab_iter:2*self.ab_iter]+=einsum("cjnm,ijnm->ijc",Asquared_derivs_K_list[:,:],temp4)
        D_tensor[:,:,self.ab_iter:2*self.ab_iter]+=einsum ("ij,cj->ijc",temp1,cj_derivs_K_list[:,:])
        D_tensor[:,:,self.ab_iter:2*self.ab_iter]+= einsum("cjk,ijk->ijc",omega_derivs_K_list,temp3)

        D_tensor[:,:,self.ab_iter:2*self.ab_iter]+=einsum("cjm,ijm->ijc",self.Kabmuj_list[:,ngMs:],temp5)
        D_tensor[:,:,self.ab_iter:2*self.ab_iter]+=einsum("ijmn,cjnm->ijc",temp6,self.Kab_list[:,ngMs:])

        D_tensor[:,:,self.ab_iter:2*self.ab_iter]*=1j
        
        temp7=-8*einsum("j,ik,ijk->ij",cjs[ngMs:],rho_T_conjs,self.x1_expectation_values[:,ngMs:])
        temp7+=4*einsum("j,ifg,ijfg->ij",cjs[ngMs:],matrices_squared_conj,self.x2_expectation_values[:,ngMs:])
        temp7+=-32*einsum("jk,if,ijkf->ij",self.omega_T[ngMs:],rho_T_conjs,self.x2_expectation_values[:,ngMs:])
        temp7+=16*einsum("jk,ifg,ijkfg->ij",self.omega_T[ngMs:],matrices_squared_conj,self.x3_expectation_values[:,ngMs:])
        temp7+=16*einsum("if,jkl,ijfkl->ij",rho_T_conjs,self.matrices_squared[ngMs:],self.x3_expectation_values[:,ngMs:])
        temp7+=-8*einsum("ikl,jfg,ijfgkl->ij",matrices_squared_conj,self.matrices_squared[ngMs:],self.x4_expectation_values[:,ngMs:])
        temp7+=-2*einsum("i,j,ij->ij",cis,cjs[ngMs:],self.overlap[:,ngMs:])
        temp7+=-8*einsum("i,jk,ijk->ij",cis,self.omega_T[ngMs:],self.x1_expectation_values[:,ngMs:])
        temp7+=4*einsum("i,jkl,ijkl->ij",cis,self.matrices_squared[ngMs:],self.x2_expectation_values[:,ngMs:])
        temp8=8*einsum("j,if,ijfn->ijn",cjs[ngMs:],rho_T_conjs,self.x2_expectation_values[:,ngMs:])
        temp8+=-4*einsum("j,ifg,ijfgn->ijn",cjs[ngMs:],matrices_squared_conj,self.x3_expectation_values[:,ngMs:])
        temp8+=8*einsum("i,jk,ijkn->ijn",cis,self.omega_T[ngMs:],self.x2_expectation_values[:,ngMs:])
        temp8+=-4*einsum("i,jfg,ijfgn->ijn",cis,self.matrices_squared[ngMs:],self.x3_expectation_values[:,ngMs:])
        temp8+=2*einsum("j,i,ijn->ijn",cjs[ngMs:],cis,self.x1_expectation_values[:,ngMs:])
        temp8+=-16*einsum("jk,ifg,ijknfg->ijn",self.omega_T[ngMs:],matrices_squared_conj,self.x4_expectation_values[:,ngMs:])
        temp8+=32*einsum("jk,ig,ijkgn->ijn",self.omega_T[ngMs:],rho_T_conjs,self.x3_expectation_values[:,ngMs:])
        temp8+=-16*einsum("if,jkl,ijfnkl->ijn",rho_T_conjs,self.matrices_squared[ngMs:],self.x4_expectation_values[:,ngMs:])
        temp8+=8*einsum("jkl,ijnkl->ijn",self.matrices_squared[ngMs:],x5_eAisqconjmat[:,ngMs:])
        temp9=-4*einsum("i,ij->ij",cis,self.overlap[:,ngMs:])
        temp9+=-16*einsum("ik,ijk->ij",rho_T_conjs,self.x1_expectation_values[:,ngMs:])
        temp9+=8*einsum("ijkl,ikl->ij",self.x2_expectation_values[:,ngMs:],matrices_squared_conj)
        temp10=4*einsum("i,ijk->ijk",cis,self.x1_expectation_values[:,ngMs:])
        temp10+=16*einsum("il,ijlk->ijk",rho_T_conjs,self.x2_expectation_values[:,ngMs:])
        temp10+=-8*einsum("ilm,ijlmk->ijk",matrices_squared_conj,self.x3_expectation_values[:,ngMs:])
        zs=einsum("jak,jk->ja",self.matrices[ngMs:],self.vectors[ngMs:])
        ck_derivs=einsum("jal,jl->ja",self.matrices_squared[ngMs:],self.vectors[ngMs:])
        vaAsquareds=self.matrices_squared[ngMs:]#einsum("ak,jkl->jal",self.vas,self.matrices_squared[ngMs:])
        
        D_tensor[:,:,nd*(nd+1):nd*(nd+2)]+=einsum("ijn,jan->ija",temp8,self.matrices[ngMs:])
        D_tensor[:,:,nd*(nd+1):nd*(nd+2)]+=einsum("ij,ja->ija",temp7,zs)
        D_tensor[:,:,nd*(nd+1):nd*(nd+2)]+=einsum("ij,ja->ija",temp9,ck_derivs)
        D_tensor[:,:,nd*(nd+1):nd*(nd+2)]+=einsum("ijk,jak->ija",temp10,vaAsquareds)

        D_tensor[:,:,nd*(nd+2):nd*(nd+3)]=1j*D_tensor[:,:,nd*(nd+1):nd*(nd+2)]

        fodmd=self.fodmd
        D_tensor+=-0.5*np.einsum("ij,jk->ijk",self.kinetic_energy_squared[:,ngMs:],fodmd[ngMs:,:])
        return D_tensor

    def calculate_potential_times_kinetic_energy(self):
        cj=np.einsum("jii->j",self.matrices)-2*self.vmmips
        self.potential_times_kinetic=2/3*self.lambda_*einsum("abmmmij,bij->ab",self.x5_expectation_values[:,:,1:,1:,1:,:,:],self.matrices_squared)
        self.potential_times_kinetic+=-2*self.lambda_*einsum("abmmmij,bij->ab",self.x5_expectation_values[:,:,:-1,:-1,1:,:,:],self.matrices_squared)
        self.potential_times_kinetic+=-4/3*self.lambda_*einsum("abijjj,bi->ab",self.x4_expectation_values[:,:,:,1:,1:,1:],self.omega_T)
        self.potential_times_kinetic+=4*self.lambda_*einsum("abijjj,bi->ab",self.x4_expectation_values[:,:,:,:-1,:-1,1:],self.omega_T)
        self.potential_times_kinetic+=-einsum("abijmm,bij->ab",self.x4_expectation_values,self.matrices_squared)
        self.potential_times_kinetic+=-1/3*self.lambda_*einsum("b,abiii->ab",cj,self.x3_expectation_values[:,:,1:,1:,1:])
        self.potential_times_kinetic+=self.lambda_*einsum("b,abiii->ab",cj,self.x3_expectation_values[:,:,0:-1,0:-1,1:])
        self.potential_times_kinetic+=0.5*np.einsum("b,abii->ab",cj,self.x2_expectation_values)
        self.potential_times_kinetic+=2*einsum("bi,abijj->ab",self.omega_T,self.x3_expectation_values)
        self.potential_times_kinetic+=conj(self.potential_times_kinetic).T
        return self.potential_times_kinetic
    def calculate_full_potential_times_kinetic_deriv(self,start_index_WFnew):
        ng=self.num_gaussians
        nd=self.num_dimensions
        ngMs=start_index_WFnew
        D_tensor=np.zeros((ng,ng-start_index_WFnew,nd*(nd+3)),dtype=complex)
        cjs=self.cjs
        cis=self.cis
        rho_T_conjs=self.rho_T_conjs
        matrices_squared_conj=self.matrices_squared_conj
        x6_of_interest=self.x6_expectation_values[:,ngMs:]
        x6e_c1_mat=einsum("abijkmmm->abijk",x6_of_interest[:,:,:,:,:,:-1,:-1,1:])
        x6_e_c2_mat=einsum("abijkmmm->abijk",x6_of_interest[:,:,:,:,:,1:,1:,1:])
        x6_e_c3_mat=einsum("abijklmm->abijkl",x6_of_interest[:,:,:,:,:,:,:,:])
        x5e_c1_mat=einsum("abijmmm->abij",self.x5_expectation_values[:,ngMs:,:,:,:-1,:-1,1:])
        x5e_c2_mat=einsum("abijmmm->abij",self.x5_expectation_values[:,ngMs:,:,:,1:,1:,1:])
        x5_e_c3_mat=einsum("abiklmm->abikl",self.x5_expectation_values[:,ngMs:])
        x3e_c1_mat=einsum("abiii->ab",self.x3_expectation_values[:,ngMs:,:-1,:-1,1:]) 
        x3e_c2_mat=einsum("abiii->ab",self.x3_expectation_values[:,ngMs:,1:,1:,1:])
        x4e_cV1_mat=einsum("abijjj->abi",self.x4_expectation_values[:,ngMs:,:,:-1,:-1,1:])
        x4e_cV2_mat=einsum("abijjj->abi",self.x4_expectation_values[:,ngMs:,:,1:,1:,1:])
        x2e_c_mat=einsum("abii->ab",self.x2_expectation_values[:,ngMs:])
        temp_Aisqconjx6e_mat=einsum("aij,abijkl->abkl",matrices_squared_conj,x6_e_c3_mat)
        x4e_c1_mat=einsum("ai,abi->ab",self.rho_T_conjs,x4e_cV1_mat)
        temp_Ajsqx6e=einsum("bij,abijkl->abkl",self.matrices_squared[ngMs:],x6_e_c3_mat)

        Asquared_derivs_I =einsum("jabkx,jxl->jabkl",self.IabLj[ngMs:],self.matrices[ngMs:])
        Asquared_derivs_I+=einsum("jabxl,jkx->jabkl",self.IabLj[ngMs:],self.matrices[ngMs:])
        Asquared_derivs_K =einsum("jabkx,jxl->jabkl",self.Kab[ngMs:],self.matrices[ngMs:])
        Asquared_derivs_K+=einsum("jkx,jabxl->jabkl",self.matrices[ngMs:],self.Kab[ngMs:])

        cj_derivs_I=einsum("jabkk->jab",self.IabLj[ngMs:],dtype=complex)
        cj_derivs_I+=-2*einsum("jk,jabkl,jl->jab",self.vectors[ngMs:],Asquared_derivs_I,self.vectors[ngMs:])
        cj_derivs_K=einsum("jabkk->jab",self.Kab[ngMs:],dtype=complex)
        cj_derivs_K+=-2*einsum("jk,jabkl,jl->jab",self.vectors[ngMs:],Asquared_derivs_K,self.vectors[ngMs:])
        omega_derivs_K=einsum("jk,jablk->jabl",self.vectors[ngMs:],Asquared_derivs_K)
        omega_derivs_I=einsum("jk,jablk->jabl",self.vectors[ngMs:],Asquared_derivs_I)
        cj_derivs_I_list=[]
        cj_derivs_K_list=[]
        omega_derivs_K_list=[]
        omega_derivs_I_list=[]
        mKm_list=[]
        mLm_list=[]
        Asquared_derivs_I_list=[]
        Asquared_derivs_K_list=[]
        self.ab_iter=0
        for a in range(nd):
             for b in range(a+1):
                self.ab_iter+=1
                cj_derivs_I_list.append(cj_derivs_I[:,a,b])
                cj_derivs_K_list.append(cj_derivs_K[:,a,b])
                omega_derivs_K_list.append(omega_derivs_K[:,a,b,:])
                omega_derivs_I_list.append(omega_derivs_I[:,a,b,:])
                mLm_list.append(self.mLm[:,a,b])
                mKm_list.append(self.mKm[:,a,b])
                Asquared_derivs_I_list.append(Asquared_derivs_I[:,a,b])
                Asquared_derivs_K_list.append(Asquared_derivs_K[:,a,b])

        cj_derivs_I_list=np.array(cj_derivs_I_list)
        cj_derivs_K_list=np.array(cj_derivs_K_list)
        omega_derivs_K_list=np.array(omega_derivs_K_list)
        mLm_list=np.array(mLm_list)
        mKm_list=np.array(mKm_list)
        omega_derivs_I_list=np.array(omega_derivs_I_list)
        Asquared_derivs_K_list=np.array(Asquared_derivs_K_list)
        Asquared_derivs_I_list=np.array(Asquared_derivs_I_list)
        temp_contr6=4/3*self.lambda_*x6_e_c2_mat- 4*self.lambda_*x6e_c1_mat
        temp_contr5=self.lambda_*1/3*x5e_c2_mat- self.lambda_*x5e_c1_mat
        temp_contr4=+ self.lambda_*2*x4e_cV1_mat- self.lambda_*2/3*x4e_cV2_mat
        temp_contr7=+ 2*self.lambda_*self.x7_c3[:,ngMs:]- 2/3*self.lambda_*self.x7_c4[:,ngMs:]

        temp1=4/3*self.lambda_*einsum("ik,ijk->ij",rho_T_conjs,x4e_cV2_mat)
        temp1+=einsum("ikl,ijkl->ij",matrices_squared_conj,-2*temp_contr5)
        temp1+=- 2*einsum("ik,ijkll->ij",rho_T_conjs,self.x3_expectation_values[:,ngMs:])
        temp1+=einsum("imn,ijmnkk->ij",matrices_squared_conj,self.x4_expectation_values[:,ngMs:])
        temp1+=einsum("jk,ijk->ij",self.omega_T[ngMs:],-2*temp_contr4)
        temp1+=-2*einsum("jkl,ijkl->ij",self.matrices_squared[ngMs:],temp_contr5)
        temp1+=-2*einsum("jk,ijkll->ij",self.omega_T[ngMs:],self.x3_expectation_values[:,ngMs:])
        temp1+=einsum("jmn,ijmnkk->ij",self.matrices_squared[ngMs:],self.x4_expectation_values[:,ngMs:])
        temp1+=einsum("i,ij->ij",cis,-self.lambda_*x3e_c1_mat+1/3*self.lambda_*x3e_c2_mat- 0.5*x2e_c_mat)
        temp1+=-einsum("ij->ij", 4*self.lambda_*x4e_c1_mat)
        temp1+=einsum("j,ij->ij",cjs[ngMs:],+self.lambda_*1/3* x3e_c2_mat- 0.5*x2e_c_mat-self.lambda_*x3e_c1_mat)
        temp2=einsum("i,ijm->ijm",cis,temp_contr4)
        temp2+=einsum("i,ijmkk->ijm",cis,self.x3_expectation_values[:,ngMs:])
        temp2+=-8*einsum("il,ijml->ijm",rho_T_conjs,temp_contr5)
        temp2+=einsum("ilk,ijmlk->ijm",matrices_squared_conj,temp_contr6)
        temp2+=4*einsum("ik,ijmkll->ijm",rho_T_conjs,self.x4_expectation_values[:,ngMs:])
        temp2+=-2*einsum("ikl,ijmkl->ijm",matrices_squared_conj,x5_e_c3_mat)
        temp2+=4*einsum("jk,ijmkll->ijm",self.omega_T[ngMs:],self.x4_expectation_values[:,ngMs:])
        temp2+=-2*einsum("jkl,ijmkl->ijm",self.matrices_squared[ngMs:],x5_e_c3_mat)
        temp2+=einsum("jl,ijml->ijm",self.omega_T[ngMs:],-8*temp_contr5)
        temp2+=einsum("jlk,ijmlk->ijm",self.matrices_squared[ngMs:],+temp_contr6)
        temp2+=einsum("j,ijm->ijm",cjs[ngMs:],temp_contr4)
        temp2+=einsum("j,ijmkk->ijm",cjs[ngMs:],self.x3_expectation_values[:,ngMs:])
        temp3=einsum("i,ijkl->ijkl",cis,temp_contr5)
        temp3+=-0.5*einsum("i,ijklxx->ijkl",cis,self.x4_expectation_values[:,ngMs:])
        temp3+=einsum("im,ijmkl->ijkl",rho_T_conjs,temp_contr6)
        temp3+=einsum("imn,ijmnkl->ijkl",matrices_squared_conj,temp_contr7)
        temp3+=-2*einsum("im,ijmkl->ijkl",rho_T_conjs,x5_e_c3_mat)
        temp3+=temp_Aisqconjx6e_mat
        temp3+=einsum("j,ijkl->ijkl",cjs[ngMs:],temp_contr5)
        temp3+=-0.5*einsum("j,ijklxx->ijkl",cjs[ngMs:],self.x4_expectation_values[:,ngMs:])
        temp3+=einsum("jm,ijmkl->ijkl",self.omega_T[ngMs:],temp_contr6)
        temp3+=einsum("jmn,ijmnkl->ijkl",self.matrices_squared[ngMs:],temp_contr7)
        temp3+=-2*einsum("jm,ijmkl->ijkl",self.omega_T[ngMs:],x5_e_c3_mat)
        temp3+=temp_Ajsqx6e
        temp4=2*temp_contr4
        temp4+=2*einsum("ijkll->ijk",self.x3_expectation_values[:,ngMs:])
        temp5=2*temp_contr5
        temp5+=-einsum("ijklmm->ijkl",self.x4_expectation_values[:,ngMs:])
        temp6=self.lambda_*x3e_c1_mat+ 0.5*x2e_c_mat-1/3*self.lambda_*x3e_c2_mat
        D_tensor[:,:,:self.ab_iter]+=einsum("cj,ij->ijc",mLm_list[:,ngMs:],temp1)
        D_tensor[:,:,:self.ab_iter]+=einsum("cjm,ijm->ijc",self.IabLjmuj_list[:,ngMs:],temp2)
        D_tensor[:,:,:self.ab_iter]+=einsum("cjkl,ijkl->ijc",self.IabLj_list[:,ngMs:],temp3)
        D_tensor[:,:,:self.ab_iter]+=einsum("cjk,ijk->ijc",omega_derivs_I_list[:,:],temp4)
        D_tensor[:,:,:self.ab_iter]+=einsum("cjkl,ijkl->ijc",Asquared_derivs_I_list[:,:],temp5)
        D_tensor[:,:,:self.ab_iter]+=einsum("cj,ij->ijc",cj_derivs_I_list[:,:],temp6)

        D_tensor[:,:,self.ab_iter:2*self.ab_iter]+=einsum("cj,ij->ijc",mKm_list[:,ngMs:],temp1)
        D_tensor[:,:,self.ab_iter:2*self.ab_iter]+=einsum("cjm,ijm->ijc",self.Kabmuj_list[:,ngMs:],temp2)
        D_tensor[:,:,self.ab_iter:2*self.ab_iter]+=einsum("cjkl,ijkl->ijc",self.Kab_list[:,ngMs:],temp3)
        D_tensor[:,:,self.ab_iter:2*self.ab_iter]+=einsum("cjk,ijk->ijc",omega_derivs_K_list[:,:],temp4)
        D_tensor[:,:,self.ab_iter:2*self.ab_iter]+=einsum("cjkl,ijkl->ijc",Asquared_derivs_K_list[:,:],temp5)
        D_tensor[:,:,self.ab_iter:2*self.ab_iter]+=einsum("cj,ij->ijc",cj_derivs_K_list[:,:],temp6)
        D_tensor[:,:,self.ab_iter:2*self.ab_iter]*=1j

        omega_derivs=self.matrices_squared[ngMs:]#einsum("ak,jlk->jal",self.vas,self.matrices_squared[ngMs:])
        cj_derivs=-4*einsum("jak,jk->ja",omega_derivs,self.vectors[ngMs:])
        zs=einsum("jak,jk->ja",self.matrices[ngMs:],self.vectors[ngMs:])


        temp7=einsum("i,ijfgg->ijf",cis,self.x3_expectation_values[:,ngMs:])
        temp7+=einsum("i,ijf->ijf",cis,temp_contr4)
        temp7+=einsum("igk,ijfgk->ijf",matrices_squared_conj,temp_contr6)
        temp7+=- 2*einsum("ikg,ijfkg->ijf",matrices_squared_conj,x5_e_c3_mat)
        temp7+=einsum("jgk,ijfgk->ijf",self.matrices_squared[ngMs:],temp_contr6)
        temp7+=- 2*einsum("jkg,ijfkg->ijf",self.matrices_squared[ngMs:],x5_e_c3_mat)
        temp7+= einsum("j,ijf->ijf",cjs[ngMs:],temp_contr4)
        temp7+=einsum("j,ijfgg->ijf",cjs[ngMs:],self.x3_expectation_values[:,ngMs:])
        temp7+=4*einsum("jx,ijxfkk->ijf",self.omega_T[ngMs:],self.x4_expectation_values[:,ngMs:])
        temp7+=-8*einsum("ix,ijxf->ijf",rho_T_conjs,temp_contr5)
        temp7+=-8*einsum("jx,ijxf->ijf",self.omega_T[ngMs:],temp_contr5)
        temp7+=4*einsum("ix,ijxfkk->ijf",rho_T_conjs,self.x4_expectation_values[:,ngMs:])
        D_tensor[:,:,nd*(nd+1):nd*(nd+2)]+=einsum("ijf,jaf->ija",temp7,self.matrices[ngMs:])
        D_tensor[:,:,nd*(nd+1):nd*(nd+2)]+= 2*einsum("jaf,ijf->ija",omega_derivs,temp_contr4)
        D_tensor[:,:,nd*(nd+1):nd*(nd+2)]+=2*einsum("jaf,ijfgg->ija",omega_derivs,self.x3_expectation_values[:,ngMs:])


        temp8=-self.lambda_*2*einsum("j,ij->ij",cjs[ngMs:],x3e_c1_mat)
        temp8+=self.lambda_*2/3*einsum("j,ij->ij",cjs[ngMs:],x3e_c2_mat)
        temp8+=2*einsum("ifg,ijfgkk->ij",matrices_squared_conj,self.x4_expectation_values[:,ngMs:])
        temp8+=-4*einsum("if,ijfgg->ij",rho_T_conjs,self.x3_expectation_values[:,ngMs:])
        temp8+=-2*self.lambda_*einsum("i,ij->ij",cis,x3e_c1_mat)
        temp8+=2/3*self.lambda_*einsum("i,ij->ij",cis,x3e_c2_mat)
        temp8+=-8*self.lambda_*x4e_c1_mat#einsum("ij->ij",x4e_c1_mat)
        temp8+=8/3 *self.lambda_*einsum("if,ijf->ij",rho_T_conjs,x4e_cV2_mat)
        temp8+=-4*einsum("ifg,ijfg->ij",matrices_squared_conj,temp_contr5)
        temp8+=- einsum("i,ij->ij",cis,x2e_c_mat)
        temp8+=- 4*einsum("jf,ijf->ij",self.omega_T[ngMs:],temp_contr4)
        temp8+=- 4*einsum("jfg,ijfg->ij",self.matrices_squared[ngMs:],temp_contr5)
        temp8+=- einsum("j,ij->ij",cjs[ngMs:],x2e_c_mat)
        temp8+=- 4*einsum("jf,ijfgg->ij",self.omega_T[ngMs:],self.x3_expectation_values[:,ngMs:])
        temp8+=+ 2*einsum("jfg,ijfgkk->ij",self.matrices_squared[ngMs:],self.x4_expectation_values[:,ngMs:])
        temp9=self.lambda_*x3e_c1_mat- self.lambda_*1/3*x3e_c2_mat+ 0.5*x2e_c_mat
        D_tensor[:,:,nd*(nd+1):nd*(nd+2)]+=einsum("ij,ja->ija",temp8,zs)
        D_tensor[:,:,nd*(nd+1):nd*(nd+2)]+=einsum("ij,ja->ija",temp9,cj_derivs)

        D_tensor[:,:,nd*(nd+2):nd*(nd+3)]=1j*D_tensor[:,:,nd*(nd+1):nd*(nd+2)]
        fodmd=self.fodmd
        D_tensor+=-0.5*np.einsum("ij,jk->ijk",self.potential_times_kinetic[:,ngMs:],fodmd[ngMs:,:])
        return D_tensor
           
    def calculate_potential_squared(self):
        self.potential_squared=0.25*einsum("abiijj->ab",self.x4_expectation_values)
        self.potential_squared+=self.lambda_*einsum("abiijjj->ab",self.x5_expectation_values[:,:,:,:,:-1,:-1,1:])
        self.potential_squared+=-self.lambda_/3*einsum("abiijjj->ab",self.x5_expectation_values[:,:,:,:,1:,1:,1:])
        self.potential_squared+=self.lambda_sq/9*einsum("abiiijjj->ab",self.x6_expectation_values[:,:,1:,1:,1:,1:,1:,1:])
        self.potential_squared+=self.lambda_sq*einsum("abiiijjj->ab",self.x6_expectation_values[:,:,:-1,:-1,1:,:-1,:-1,1:])
        self.potential_squared+=-2/3*self.lambda_sq*einsum("abiiijjj->ab",self.x6_expectation_values[:,:,1:,1:,1:,0:-1,0:-1,1:])
        return self.potential_squared
    def calculate_full_potential_squared_deriv_matrix(self,start_index_WFnew):
        #This calculates a 3-tensor, where the element D[i,j,n] is the derivative <g_i|V^2 (\vec{x})\frac{\partial}{\partial n_j}|g_j>, where $n_j$ is one of the
        #n(n+3) parameters specifying the Gaussian. 
        ng=self.num_gaussians
        nd=self.num_dimensions
        ngMs=start_index_WFnew
        D_tensor=np.zeros((ng,ng-start_index_WFnew,nd*(nd+3)),dtype=complex)
        potentialsquared=self.calculate_potential_squared()
        x5e_contracted_fullV=einsum("abkiijj->abk",self.x5_expectation_values[:,ngMs:])
        x6_e_contracted_full=einsum("abkliijj->abkl",self.x6_expectation_values[:,ngMs:]) #Candidate for wrongness (only enters super_contr2)
        x6_e_contracted_firstV=einsum("abkiijjj->abk",self.x6_expectation_values[:,ngMs:,:,:,:,:-1,:-1,1:])
        x6_e_contracted_secondV=einsum("abkiijjj->abk",self.x6_expectation_values[:,ngMs:,:,:,:,1:,1:,1:])
        super_contr1=1/2*x5e_contracted_fullV+2*x6_e_contracted_firstV*self.lambda_+x6_e_contracted_secondV*(-2/3*self.lambda_)
        super_contr1+=2/9*self.lambda_sq*self.x7_c1v[:,ngMs:]+2*self.lambda_sq*self.x7_c2v[:,ngMs:]+-4/3*self.lambda_sq*self.x7_c3v[:,ngMs:]
        super_contr2=x6_e_contracted_full*(-0.25)
        super_contr2+=-self.lambda_*self.x7_c1[:,ngMs:]
        super_contr2+=+self.lambda_/3*self.x7_c2[:,ngMs:]
        super_contr2+=-self.lambda_sq/9*self.x8_c1[:,ngMs:]
        super_contr2+=+2/3*self.lambda_sq*self.x8_c3[:,ngMs:]
        super_contr2+=-self.lambda_sq*self.x8_c2[:,ngMs:]
        #Only reasonable canditates for wrongness are: super_contr2 and everything that enters it (x7_c1, x7_c2, x8_c1, x8_c2, x8_c3)
        #As well at the actual expressions. 
        #Another possible obvious source might be that the matrix isn't set up correctly. Should check for that. 
        D_tensor[:,:,:self.ab_iter]+=einsum("cjkl,ijkl->ijc",self.IabLj_list[:,ngMs:],super_contr2)
        D_tensor[:,:,:self.ab_iter]+=einsum("cjk,ijk->ijc",self.IabLjmuj_list[:,ngMs:],super_contr1)
        D_tensor[:,:,:self.ab_iter]+=-einsum("ij,jk,cjk->ijc",potentialsquared[:,ngMs:],self.vectors[ngMs:],self.IabLjmuj_list[:,ngMs:])
        D_tensor[:,:,self.ab_iter:2*self.ab_iter]+=einsum("cjkl,ijkl->ijc",self.Kab_list[:,ngMs:],super_contr2)
        D_tensor[:,:,self.ab_iter:2*self.ab_iter]+=einsum("cjk,ijk->ijc",self.Kabmuj_list[:,ngMs:],super_contr1)
        D_tensor[:,:,self.ab_iter:2*self.ab_iter]+=-einsum("ij,jk,cjk->ijc",potentialsquared[:,ngMs:],self.vectors[ngMs:],self.Kabmuj_list[:,ngMs:])
        D_tensor[:,:,self.ab_iter:2*self.ab_iter]*=1j
        D_tensor[:,:,nd*(nd+1):nd*(nd+2)]+=einsum("jck,ijk->ijc",self.matrices[ngMs:,:],super_contr1)
        D_tensor[:,:,nd*(nd+1):nd*(nd+2)]+=-2*einsum("ij,jck,jk->ijc",potentialsquared[:,ngMs:],self.matrices[ngMs:,:],self.vectors[ngMs:])
        D_tensor[:,:,nd*(nd+2):nd*(nd+3)]=1j*D_tensor[:,:,nd*(nd+1):nd*(nd+2)]
        fodmd=self.fodmd
        D_tensor+=-0.5*np.einsum("ij,jk->ijk",self.potential_squared[:,ngMs:],fodmd[ngMs:,:])

        return D_tensor
    def calculate_Hamiltonian(self):
        return (self.calculate_potential()+self.calculate_kinetic_energy())
    def calculate_Hamiltonian_deriv(self,start_index_WFnew): 
        return self.calculate_full_potential_deriv_matrix(start_index_WFnew)+self.calculate_full_kinetic_energy_deriv(start_index_WFnew)
    def calculate_Hamiltonian_squared(self):
        return (self.calculate_kinetic_energy_squared()+self.calculate_potential_squared()+self.calculate_potential_times_kinetic_energy())

    def calculate_Hamiltonian_squared_deriv(self,start_index_WFnew):
        returnval=self.calculate_full_kinetic_energy_squared_deriv(start_index_WFnew) #Takes second most time
        returnval+=self.calculate_full_potential_squared_deriv_matrix(start_index_WFnew)
        returnval+=self.calculate_full_potential_times_kinetic_deriv(start_index_WFnew) #Takes most time
        return returnval
    def calculate_S_mat(self,start_index_WFnew,h):
        if h==0:
            self.S_mat=self.overlap[start_index_WFnew:,start_index_WFnew:]
        else:
            self.S_mat=(self.overlap +h**2/4*self.Hsquared )[start_index_WFnew:,start_index_WFnew:]
        return self.S_mat
    def calculate_rho_vec(self,start_index_WFnew,h):
        if h==0:
            rho_mat=self.overlap[:start_index_WFnew,start_index_WFnew:] #Upper right corner
        else:
            rho_mat=(self.overlap-h**2/4*self.Hsquared +1j*h*self.H)[:start_index_WFnew,start_index_WFnew:] #Upper right corner
        self.rho_mat=rho_mat
        self.rho_vec=einsum("i,ij->j",self.coefficients[:start_index_WFnew],conj(rho_mat))
        return self.rho_vec
    def rothe_optimal_c(self,start_index_WFnew,h):
        rho=self.calculate_rho_vec(start_index_WFnew,h)
        S=self.calculate_S_mat(start_index_WFnew,h)
        self.Smat=S
        self.opt_c=solve(S,rho)
        self.rho=rho
    def rothe_optimal_c_normalization(self,start_index_WFnew,h):
        rho=self.calculate_rho_vec(start_index_WFnew,h)
        S=self.calculate_S_mat(start_index_WFnew,h)
        self.Smat=S
        self.rho=rho
        self.opt_c=solve(S,rho)
        #ovlp_mat=self.overlap[start_index_WFnew:,start_index_WFnew:]
        #def find_x(x):
        #    c_kand=solve(S+x*ovlp_mat,rho)
        #    return real(conj(c_kand)@ovlp_mat@c_kand)-self.normalization #Normalization condition
        #sol=root_scalar(find_x,x0=0)
        #root=sol.root
        #self.opt_c=solve(S+root*ovlp_mat,rho)
        #self.root=root
    def euler_optimal_c(self,si,h):
        S_mat=(self.overlap )[si:,si:]
        rho_mat=(self.overlap+1j*h*self.H)[:si,si:] #Upper right corner
        rho_vec=einsum("i,ij->j",self.coefficients[:si],conj(rho_mat))
        return solve(S_mat,rho_vec)
    def calculate_euler_error(self,start_index_WFnew,h):
        si=start_index_WFnew
        self.rothe_optimal_c_normalization(si,h=h)
        new_lin_params=self.euler_optimal_c(si,h)
        old_lin_params=self.coefficients[:si]
        O=self.overlap
        H=self.calculate_Hamiltonian()
        H2=self.calculate_Hamiltonian_squared()
        inner_ovlp=np.conj(new_lin_params).T@O[si:,si:]@new_lin_params
        old_old_term=np.conj(old_lin_params).T@(O+h**2*H2)[:si,:si]@old_lin_params
        cross_terms=-2*(np.conj(new_lin_params).T@(O-h*1j*H)[si:,:si]@old_lin_params)
        return real(inner_ovlp+old_old_term+cross_terms)
    def calculate_euler_deriv(self,si,h):
        opt_c=self.euler_optimal_c(si,h)
        self.setUpDerivs(si,Hsquared=False)
        S_deriv_vecs=self.ovlp_deriv[si:]
        rho_deriv_vecs=self.ovlp_deriv+1j*self.H_deriv*h
        rho_indices= einsum("abc,a->bc",rho_deriv_vecs[:si],conj(self.coefficients[:si]))
        opt_c_rep=np.repeat(opt_c[:, np.newaxis], self.num_dimensions*(self.num_dimensions+3), axis=1)
        rho_optc_prods=real((rho_indices)*opt_c_rep+conj((rho_indices)*opt_c_rep))
        optc_smat_prod=einsum("a,aij,i->ij",conj(opt_c),S_deriv_vecs,opt_c)+einsum("i,aij,a->ij",conj(opt_c),conj(S_deriv_vecs),opt_c)
        jacobian=real(optc_smat_prod-rho_optc_prods).flatten()
        return jacobian
    def rothe_jacobian_noNormalization(self,start_index_WFnew,h):
        #if "opt_c" not in dir():
        self.rothe_optimal_c_normalization(start_index_WFnew,h)
        self.setUpDerivs(start_index_WFnew)
        S_deriv_vecs=self.calculate_all_S_deriv_vecs(start_index_WFnew,h)
        rho_indices=self.calculate_all_rho_derivs(start_index_WFnew,h)
        opt_c=self.opt_c
        opt_c_rep=np.repeat(opt_c[:, np.newaxis], self.num_dimensions*(self.num_dimensions+3), axis=1)
        rho_optc_prods=real((rho_indices)*opt_c_rep+conj((rho_indices)*opt_c_rep))
        optc_smat_prod=einsum("a,aij,i->ij",conj(opt_c),S_deriv_vecs,opt_c)+einsum("i,aij,a->ij",conj(opt_c),conj(S_deriv_vecs),opt_c)
        jacobian=real(optc_smat_prod-rho_optc_prods).flatten()
        return jacobian
    def setupMatrices(self):
        if self.h>0:
            self.Hsquared=self.calculate_Hamiltonian_squared()
            self.H=self.calculate_Hamiltonian()
        else:
            self.Hsquared=np.zeros_like(self.overlap)
            self.H=np.zeros_like(self.overlap)
    def setUpDerivs(self,start_index_WFnew,Hsquared=True):
        self.ovlp_deriv=self.calculate_full_overlap_deriv_matrix(start_index_WFnew)
        if self.h>0:
            if Hsquared:
                self.Hsquared_deriv=self.calculate_Hamiltonian_squared_deriv(start_index_WFnew)
            self.H_deriv=self.calculate_Hamiltonian_deriv(start_index_WFnew)
        else:
            self.H_deriv=np.zeros_like(self.ovlp_deriv)
            self.Hsquared_deriv=np.zeros_like(self.ovlp_deriv)
    def calculate_all_S_deriv_vecs(self,start_index_WFnew,h):
        S_deriv_vecs=self.ovlp_deriv+self.Hsquared_deriv*h**2/4
        return S_deriv_vecs[start_index_WFnew:]
    def calculate_all_rho_derivs(self,start_index_WFnew,h):
        rho_deriv_vecs=self.ovlp_deriv-h**2/4*self.Hsquared_deriv+1j*self.H_deriv*h
        return einsum("abc,a->bc",rho_deriv_vecs[:start_index_WFnew],conj(self.coefficients[:start_index_WFnew]))


    def calculate_overlap_tildePhim(self,start_index_WFnew,h): #Calculate <\tilde \Phi_m|\tilde \Phi_m>
        S=self.overlap[:start_index_WFnew,:start_index_WFnew].copy() # NEEDS TO BE A FUCKING COPY OTHERWISE THE WHOLE SHIT CRASHES LIKE THE FUCKING MALAYSIA AIRLINES
        contribution=self.Hsquared[:start_index_WFnew,:start_index_WFnew]
        S+=0.25*h**2*contribution
        return conj(self.coefficients[:start_index_WFnew]).T@S@self.coefficients[:start_index_WFnew]
    def rothe_error(self,start_index_WFnew,h):
        overlap_term=self.calculate_overlap_tildePhim(start_index_WFnew,h)
        self.rothe_optimal_c_normalization(start_index_WFnew,h)
        projection_term=2*conj(self.rho).T@self.opt_c-conj(self.opt_c).T@self.Smat@self.opt_c
        difference=overlap_term-projection_term
        if np.isnan(difference):
            return 100
        else:
            return abs(real(difference)) #numerical noise makes this not exactly real
        
    def rothe_jacobian(self,start_index_WFnew,h):
        if False:# self.root is not None:
            self.rothe_optimal_c(start_index_WFnew,h)   
            c=self.opt_c
            rho=self.rho
            S=self.Smat
            ovlp_mat=self.overlap[start_index_WFnew:,start_index_WFnew:]
            ngsi=self.num_gaussians-start_index_WFnew
            self.setUpDerivs(start_index_WFnew)
            S_deriv_vecs=self.calculate_all_S_deriv_vecs(start_index_WFnew,h)
            ovlp_deriv_vecs=self.ovlp_deriv[start_index_WFnew:]
            rho_deriv_val=self.calculate_all_rho_derivs(start_index_WFnew,h)
            jac=np.zeros((ngsi,self.num_dimensions*(self.num_dimensions+3)),dtype=complex)
            M=S+self.root*ovlp_mat
            v_1=solve(M,rho)        
            for j in range (ngsi): #For each Gaussian
                for a in range(self.num_dimensions*(self.num_dimensions+3)): #For each parameter
                    S_deriv_mat=np.zeros((ngsi,ngsi),dtype=complex)
                    ovlp_deriv_mat=np.zeros((ngsi,ngsi),dtype=complex)
                    rho_deriv_vec=np.zeros(ngsi,dtype=complex)
                    S_deriv_mat[:,j]+=S_deriv_vecs[:,j,a] #not sure about this on
                    S_deriv_mat[j,:]+=conj(S_deriv_vecs[:,j,a])
                    ovlp_deriv_mat[:,j]+=ovlp_deriv_vecs[:,j,a] #not sure about this on
                    ovlp_deriv_mat[j,:]+=conj(ovlp_deriv_vecs[:,j,a])
                    rho_deriv_vec[j]=conj(rho_deriv_val[j,a])
                    v_2=solve(M,rho_deriv_vec)
                    def find_x(x):
                        M_x=S_deriv_mat+self.root*ovlp_deriv_mat+x*ovlp_mat
                        c_deriv_cand=-solve(M,M_x@v_1)+v_2
                        return  real(np.conj(c_deriv_cand).T@ovlp_mat@c+np.conj(c)@ovlp_mat@c_deriv_cand+np.conj(c)@ovlp_deriv_mat@c)
                    sol=root_scalar(find_x,x0=0)
                    root_deriv=sol.root
                    c_deriv=-solve(M,(S_deriv_mat+self.root*ovlp_deriv_mat+root_deriv*ovlp_mat)@v_1)+v_2
                    jac[j,a]=(np.conj(c_deriv).T@S@c+np.conj(c).T@S@c_deriv+np.conj(c)@S_deriv_mat@c-2*(np.conj(c_deriv)@rho+np.conj(c).T@rho_deriv_vec))
            jac_sol=real(jac.flatten())
        else:
            jac_sol=self.rothe_jacobian_noNormalization(start_index_WFnew,h)
        return jac_sol

def test_full_rothe_error_implementation():
    dim=3
    #params1=np.asarray([[sqrt(0.5),0.7,sqrt(0.5),0.2,-0.2,0.2,0.40,-0.15,0.3,0.1]])
    cvalsv=[-4.531396-1.810139j , 2.505948-0.34591j ,  2.030252+1.668j , 1.47  ]
    params1=[ 6.929664e-01  ,3.909027e-02,  7.515107e-01 , 3.050364e-02,  3.847112e-02,  7.227180e-01 , 1.506672e-02 , 2.774190e-02, -1.415030e-02, -1.766089e-02 , 4.679727e-02, -3.541376e-02 , 2.214731e+00 , 2.080399e+00 , 2.042120e+00, -6.505042e-03  ,2.629362e-02, -3.654459e-03]
    params2=[ 7.387447e-01 , 9.086572e-02 , 7.616151e-01,  6.371750e-02 , 4.517546e-02 , 7.266459e-01, -1.223442e-02 ,-2.018877e-02 ,-3.551452e-02, -4.674869e-02 , 7.833209e-03 ,-4.309553e-02  ,1.755156e+00 , 1.754918e+00 , 1.832218e+00 , 2.333246e-02  ,5.743388e-02 ,-1.032477e-02]
    params3=[ 7.642810e-01,  1.375058e-01,  7.790654e-01 , 9.136050e-02 , 6.472280e-02,  7.303435e-01  ,1.614382e-03,  2.119395e-02, -1.155141e-02, -2.526436e-02 , 5.832423e-02, -3.697802e-02 , 2.604403e+00  ,2.341630e+00 , 2.212544e+00, -1.965453e-01 ,-6.320133e-02, -3.411345e-02]
    params4=list((np.array(params1)+np.array(params2))/2)
    params=np.concatenate((np.array(params1),np.array(params2),params1,params2)).reshape(4,-1)
    #params=np.concatenate((np.array(params1),np.array(params1)+1e-3)).reshape(2,-1)
    num=len(params)//2
    h=0.25
    cvals=np.zeros(2*num,dtype=complex)
    cvals[:num]=cvalsv[:num]
    wf=WF(params,np.asarray(cvals))
    re=wf.rothe_error(num,h=h)
    print(re)
    rj=wf.rothe_jacobian(num,h= h)
    for i in range(dim*(dim+3)):
        for j in range(num):
            dparams=params.copy()
            dparams[num+j][i]+=dx
            dwf=WF(dparams,np.asarray(cvals))
            print((dwf.rothe_error(num,h=h)-re)/dx,rj[dim*(dim+3)*j+i])

def test_derivs(typeInd=0,parInd=0,dim=3,lambda_=0.1):
    params=np.random.rand(2,2*dim+dim*(dim+1))*1e-1
    k=-1
    for i in range(1,dim+1):
        k+=i
        params[:,k]=sqrt(1/2)+0.1*(np.random.rand()-0.5) #Set diagonal of A matrix equal to 1/2 plus some random perturbation    
    wf=WF(params,[1,0],lambda_=lambda_)
    types=["Kinetic","Potential","Kin^2","Pot^2","Kin cross Pot"]
    type=types[typeInd]
    if type==types[4]:
        all_derivs=wf.calculate_full_potential_times_kinetic_deriv(1)
    if type==types[3]:
        all_derivs=wf.calculate_full_potential_squared_deriv_matrix(1)
    if type==types[2]:
        all_derivs=wf.calculate_full_kinetic_energy_squared_deriv(1)
    if type==types[1]:
        all_derivs=wf.calculate_full_potential_deriv_matrix(1)
    if type==types[0]:
        all_derivs=wf.calculate_full_kinetic_energy_deriv(1)
    j=1
    h=1e-8
    dparams=params.copy();dparams[1,parInd]+=h
    dwf=WF(dparams,[1,0],lambda_=lambda_)
    if type==types[4]:
        numerical=(dwf.calculate_potential_times_kinetic_energy()[0,1]-wf.calculate_potential_times_kinetic_energy()[0,1])
    if type==types[3]:
        numerical=(dwf.calculate_potential_squared()[0,1]-wf.calculate_potential_squared()[0,1])
    if type==types[2]:
        numerical=(dwf.calculate_kinetic_energy_squared()[0,1]-wf.calculate_kinetic_energy_squared()[0,1])
    if type==types[1]:
        numerical=(dwf.calculate_potential()[0,1]-wf.calculate_potential()[0,1])
    if type==types[0]:
        numerical=(dwf.calculate_kinetic_energy()[0,1]-wf.calculate_kinetic_energy()[0,1])
    numerical/=h
    
    AE=abs(all_derivs[0,0,parInd]-numerical)
    if abs(numerical)>1e-12:
        RE=AE/abs(numerical)
    else:
        RE=np.nan
    if AE>1e-5 or RE>1e-5:
        print(type,parInd)
        print("RE: %.1e, AE: %.1e, Numerical value: %.1e"%(RE,AE,abs(numerical)))
        print("--------------")
def test_update_procedure(dim,num_gauss,m=2):
    cvals=np.random.rand(num_gauss*m)-0.5+1j*(np.random.rand(num_gauss*m)-0.5)
    params=np.random.rand(num_gauss*m,dim*(dim+3))*1e-1
    k=-1
    for i in range(1,dim+1):
        k+=i
        params[:,k]=sqrt(1/2)+0.1*(np.random.rand()-0.5) #Set diagonal of A matrix equal to 1/2 plus some random perturbation    
    print(params.shape)
    print(cvals.shape)

    wf1=WF(params,cvals)
    params_updated=params.copy()
    params_new=(np.random.rand(num_gauss,dim*(dim+3))-0.5)*1e-1
    #Procedure 1: Make a new WF object
    params_updated[(m-1)*num_gauss:]=params_new
    start=time.time()
    wf=WF(params_updated,cvals)
    end=time.time()
    print("Time for new WF object: ",end-start)
    #Procedure 2: Update the WF object
    start=time.time()
    indices=np.arange((m-1)*num_gauss,m*num_gauss)
    print(indices)
    wf1.update_params_overlapFitting(indices,params_new,new_lin_params=None)
    end=time.time()
    print("Time for update procedure: ",end-start)
    #print(wf1.H-wf.H)
    print("Difference in Rothe error: ")
    print(wf1.rothe_error((m-1)*num_gauss,1e-2)-wf.rothe_error((m-1)*num_gauss,1e-2))
    print("Difference in Rothe Jacobian: ")
    print(sum(abs(wf1.rothe_jacobian((m-1)*num_gauss,1e-2)-wf.rothe_jacobian((m-1)*num_gauss,1e-2))))
    #print(wf.fodmd-wf1.fodmd) #FODMD IS WRONG...
    #print(wf.Kab-wf1.Kab) #Kab IS WRONG, which is not surprising since fodmd is wrong

    pass
def test_speedup_derivs(dim,num_gauss,m=2):
    cvals=np.random.rand(num_gauss*m)-0.5+1j*(np.random.rand(num_gauss*m)-0.5)
    params=np.random.rand(num_gauss*m,dim*(dim+3))*1e-1
    k=-1
    for i in range(1,dim+1):
        k+=i
        params[:,k]=sqrt(1/2)+0.1*(np.random.rand()-0.5) #Set diagonal of A matrix equal to 1/2 plus some random perturbation    
    print(params.shape)
    print(cvals.shape)

    wf1=WF(params,cvals)
    start=time.time()
    first=wf1.calculate_full_kinetic_energy_squared_deriv(m)
    end=time.time()
    print("Time for first: ",end-start)


if __name__=="__main__":
    from numpy import array
    rank = 0
    if rank==0:
        start=time.time()

    dim=4
    h=0.025
    num=num_gauss=5
    m=2
    np.random.seed(42) #Use same seed acreoss all processes

    cvals=np.random.rand(num_gauss*m)-0.5+1j*(np.random.rand(num_gauss*m)-0.5)
    params=np.random.rand(num_gauss*m,dim*(dim+3))*1e-1
    k=-1
    for i in range(1,dim+1):
        k+=i
        params[:,k]=sqrt(1/2)+0.3*(np.random.rand()-0.5) #Set diagonal of A matrix equal to 1/2 plus some random perturbation    

    params[num:,:]=params[:num,:]+0.1*(np.random.rand(params[:num,:].shape[0],params[:num,:].shape[1])-0.5) #make it symmetric
    if rank==0:
        start=time.time()
        wf=WF(params,np.asarray(cvals),calculate_Gradient=True,h=0.2) 
        deriv=wf.calculate_Hamiltonian_deriv(num)
        end=time.time()
        pass
        np.save("reference_H_size=%d,num_gauss=%d,dim=%d.npy"%(1,num_gauss,dim),deriv)
        reference=np.load("reference_H_size=%d,num_gauss=%d,dim=%d.npy"%(1,num_gauss,dim))
        print(np.allclose(deriv,reference))
        #print(overlap_deriv)
        print("Time taken: ",end-start)
