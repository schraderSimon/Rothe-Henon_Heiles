import itertools as it
import numpy as np
from numpy import einsum
from mpi4py import MPI
import sys
def group(iterable, n=2):
    return zip(*([iter(iterable)] * n))

def set_partition(iterable, n=2):
    set_partitions = set()

    for permutation in it.permutations(iterable):
        grouped = group(list(permutation), n)
        sorted_group = tuple(sorted([tuple(sorted(partition)) for partition in grouped]))
        set_partitions.add(sorted_group)

    return set_partitions
fourth_partitions = set_partition(['i','j','k','l'], 2)
six_partitions = set_partition(['i','j','k','l','m','n'], 2)
eight_partitions = set_partition(['i','j','k','l','m','n','o','p'], 2)
fourth_partition_strings=[]
six_partition_strings=[]
eight_partition_strings=[]
for i,partition in enumerate(fourth_partitions):
    fourth_partition_strings.append('a%s%s%s%s'%(partition[0][0],partition[0][1],partition[1][0],partition[1][1]))
for i,partition in enumerate(six_partitions):
    six_partition_strings.append('a%s%s%s%s%s%s'%(partition[0][0],partition[0][1],partition[1][0],partition[1][1],partition[2][0],partition[2][1]))
six_partition_strings=['aimjkln', 'ainjlkm', 'aijklmn', 'ailjnkm', 'ailjmkn', 'ailjkmn', 'aimjlkn', 'ainjklm', 'ainjmkl', 'aikjmln', 'aikjnlm', 'aimjnkl', 'aikjlmn', 'aijkmln', 'aijknlm']

stringy='aijklmnop'
strings_4=['aij,akl->aijkl','akl,aij->aijkl','aik,ajl->aijkl','ail,ajk->aijkl','ajk,ail->aijkl','ajl,aik->aijkl']
strings5_3=['aijk,alm->aijklm','aklm,aij->aijklm','aijl,akm->aijklm','aijm,akl->aijklm','aikl,ajm->aijklm','aikm,ajl->aijklm','ailm,ajk->aijklm','ajkl,aim->aijklm','ajkm,ail->aijklm','ajlm,aik->aijklm']
strings5_1=['ai,ajklm->aijklm','aj,aiklm->aijklm','ak,aijlm->aijklm','al,aijkm->aijklm','am,aijkl->aijklm']
strings5_3_altaltalt=[(0,1,2,3,4,5),(0,3,4,5,1,2),(0,1,2,4,3,5),(0,1,2,5,3,4),(0,1,3,4,2,5),(0,1,3,5,2,4),(0,1,4,5,2,3),(0,2,3,4,1,5),(0,2,3,5,1,4),(0,2,4,5,1,3)]
strings5_1_altaltalt=[(0,1,2,3,4,5),(0,2,1,3,4,5),(0,3,1,2,4,5),(0,4,1,2,3,5),(0,5,1,2,3,4)]
strings5_1_alt=['ajiklm->aijklm','akijlm->aijklm','alijkm->aijklm','amijkl->aijklm']
strings5_3_alt=['aklmij->aijklm','aijlkm->aijklm','aijmkl->aijklm','aikljm->aijklm','aikmjl->aijklm','ailmjk->aijklm','ajklim->aijklm','ajkmil->aijklm','ajlmik->aijklm']
strings5_1_altalt=['ajiklm','ajkilm','ajklim','ajklmi']
strings5_1_altalt=[[0,2,1,3,4,5],[0,2,3,1,4,5],
                    [0,2,3,4,1,5],[0,2,3,4,5,1]]
strings5_1_altalt = [tuple(l) for l in strings5_1_altalt]

strings5_3_altalt=['almijk',
                   'aijlkm',
                   'aijlmk',
                   'ailjkm',
                   'ailjmk',
                   'ailmjk',
                   'alijkm',
                   'alijmk',
                   'alimjk']
strings5_3_altalt=[[0,4,5,1,2,3],
                   [0,1,2,4,3,5],
                   [0,1,2,4,5,3],
                   [0,1,4,2,3,5],
                   [0,1,4,2,5,3],
                   [0,1,4,5,2,3],
                   [0,4,1,2,3,5],
                   [0,4,1,2,5,3],
                   [0,4,1,5,2,3]]
strings5_3_altalt = [tuple(l) for l in strings5_3_altalt]

strings6_4=['aijkl,amn->aijklmn', 'aijkm,aln->aijklmn', 'aijkn,alm->aijklmn', 'aijlm,akn->aijklmn', 'aijln,akm->aijklmn', 'aijmn,akl->aijklmn', 'aiklm,ajn->aijklmn', 'aikln,ajm->aijklmn', 'aikmn,ajl->aijklmn', 'ailmn,ajk->aijklmn', 'ajklm,ain->aijklmn', 'ajkln,aim->aijklmn', 'ajkmn,ail->aijklmn', 'ajlmn,aik->aijklmn', 'aklmn,aij->aijklmn']
strings6_2=['aij,aklmn->aijklmn', 'aik,ajlmn->aijklmn', 'ail,ajkmn->aijklmn', 'aim,ajkln->aijklmn', 'ain,ajklm->aijklmn', 'ajk,ailmn->aijklmn', 'ajl,aikmn->aijklmn', 'ajm,aikln->aijklmn', 'ajn,aiklm->aijklmn', 'akl,aijmn->aijklmn', 'akm,aijln->aijklmn', 'akn,aijlm->aijklmn', 'alm,aijkn->aijklmn', 'aln,aijkm->aijklmn', 'amn,aijkl->aijklmn']
strings6_4_alt=['aijkmln->aijklmn', 'aijknlm->aijklmn', 'aijlmkn->aijklmn', 'aijlnkm->aijklmn', 'aijmnkl->aijklmn', 'aiklmjn->aijklmn', 'aiklnjm->aijklmn', 'aikmnjl->aijklmn', 'ailmnjk->aijklmn', 'ajklmin->aijklmn', 'ajklnim->aijklmn', 'ajkmnil->aijklmn', 'ajlmnik->aijklmn', 'aklmnij->aijklmn']
strings6_2_alt=['aikjlmn->aijklmn', 'ailjkmn->aijklmn', 'aimjkln->aijklmn', 'ainjklm->aijklmn', 'ajkilmn->aijklmn', 'ajlikmn->aijklmn', 'ajmikln->aijklmn', 'ajniklm->aijklmn', 'aklijmn->aijklmn', 'akmijln->aijklmn', 'aknijlm->aijklmn', 'almijkn->aijklmn', 'alnijkm->aijklmn', 'amnijkl->aijklmn']
strings_6_2_altalt=[ [0,1,3,2,4,5,6],[0,1,3,4,2,5,6],
                     [0,1,3,4,5,2,6],[0,1,3,4,5,6,2],
                     [0,3,1,2,4,5,6],[0,3,1,4,2,5,6],
                     [0,3,1,4,5,2,6],[0,3,1,4,5,6,2],
                     [0,3,4,1,2,5,6],[0,3,4,1,5,2,6],
                     [0,3,4,1,5,6,2],[0,3,4,5,1,2,6],
                     [0,3,4,5,1,6,2],[0,3,4,5,6,1,2]
                     ]
strings_6_4_altalt=[ [0,1,2,3,5,4,6],[0,1,2,3,5,6,4],
                     [0,1,2,5,3,4,6],[0,1,2,5,3,6,4],
                     [0,1,2,5,6,3,4],[0,1,5,2,3,4,6],
                     [0,1,5,2,3,6,4],[0,1,5,2,6,3,4],
                     [0,1,5,6,2,3,4],[0,5,1,2,3,4,6],
                     [0,5,1,2,3,6,4],[0,5,1,2,6,3,4],
                     [0,5,1,6,2,3,4],[0,5,6,1,2,3,4]]
strings_6_2_altalt = [tuple(l) for l in strings_6_2_altalt]
strings_6_4_altalt = [tuple(l) for l in strings_6_4_altalt]
strings_6_4_altaltalt=((0,1,2,3,4,5,6),(0,1,2,3,5,4,6),(0,1,2,3,6,4,5),
                    (0,1,2,4,5,3,6),(0,1,2,4,6,3,5),
                    (0,1,2,5,6,3,4),(0,1,3,4,5,2,6),
                    (0,1,3,4,6,2,5),(0,1,3,5,6,2,4),
                    (0,1,4,5,6,2,3),(0,2,3,4,5,1,6),
                    (0,2,3,4,6,1,5),(0,2,3,5,6,1,4),
                    (0,2,4,5,6,1,3),(0,3,4,5,6,1,2))
strings_6_2_altaltalt=((0,1,2,3,4,5,6),(0,1,3,2,4,5,6),(0,1,4,2,3,5,6),
                       (0,1,5,2,3,4,6),(0,1,6,2,3,4,5),
                       (0,2,3,1,4,5,6),(0,2,4,1,3,5,6),
                       (0,2,5,1,3,4,6),(0,2,6,1,3,4,5),
                       (0,3,4,1,2,5,6),(0,3,5,1,2,4,6),
                       (0,3,6,1,2,4,5),(0,4,5,1,2,3,6),
                       (0,4,6,1,2,3,5),(0,5,6,1,2,3,4))
six_partition_string_transpose=[[0,1,3,4,5,2,6],[0,1,3,5,4,6,2],[0,1,3,5,2,6,4],[0,1,3,5,2,4,6],[0,1,3,4,2,5,6],[0,1,3,5,4,2,6],[0,1,3,4,5,6,2],
[0,1,3,5,6,4,2],[0,1,3,2,5,4,6],[0,1,3,2,5,6,4],[0,1,3,5,6,2,4],[0,1,3,2,4,5,6],[0,1,2,3,5,4,6],[0,1,2,3,5,6,4]]

dims=[2,3,4,5,6]
reduced_indices6_list=[]
reduced_indices5_list=[]

for dim in dims:
       indices_6=[]
       indices_5=[]
       reduced_indices6=[]
       reduced_indices5=[]

       for i in range(dim):
              for j in range(dim):
                     for k in range(dim):
                            for l in range(dim):
                                   for m in range(dim):
                                          indices_5.append((i,j,k,l,m))
                                          for n in range(dim):
                                                 indices_6.append((i,j,k,l,m,n))
       for index in indices_6:
              index_sorted=tuple(sorted(index))
              if not index_sorted in reduced_indices6:
                     reduced_indices6.append(tuple(index_sorted))
       for index in indices_5:
              index_sorted=tuple(sorted(index))
              if not index_sorted in reduced_indices5:
                     reduced_indices5.append(tuple(index_sorted))
       reduced_indices5_list.append(reduced_indices5)
       reduced_indices6_list.append(reduced_indices6)
transposed_indices_64_lists=[]
transposed_indices_62_lists=[]
transposed_indices_53_lists=[]
transposed_indices_51_lists=[]

for dim in dims:
    transposed_indices_64_list=[]
    transposed_indices_62_list=[]
    transposed_indices_53_list=[]
    transposed_indices_51_list=[]
    for x in range(len(strings_6_2_altaltalt)):
        transposed_indices_1=[]
        transposed_indices_2=[]
        for indices in reduced_indices6_list[dim-2]:
            transposed_indices_1.append([indices[z-1] for z in strings_6_4_altaltalt[x][1:]])
            transposed_indices_2.append([indices[z-1] for z in strings_6_2_altaltalt[x][1:]])
        transpose_indices_64=np.array(transposed_indices_1).T
        transpose_indices_62=np.array(transposed_indices_2).T
        transposed_indices_64_list.append(transpose_indices_64)
        transposed_indices_62_list.append(transpose_indices_62)
    transposed_indices_64_lists.append(transposed_indices_64_list)
    transposed_indices_62_lists.append(transposed_indices_62_list)
    for x in range(len(strings5_3_altaltalt)):
        transposed_indices_3=[]
        for indices in reduced_indices5_list[dim-2]:
            transposed_indices_3.append([indices[z-1] for z in strings5_3_altaltalt[x][1:]])
        transpose_indices_53=np.array(transposed_indices_3).T
        transposed_indices_53_list.append(transpose_indices_53)
    for x in range(len(strings5_1_altaltalt)):
        transposed_indices_1=[]
        for indices in reduced_indices5_list[dim-2]:
            transposed_indices_1.append([indices[z-1] for z in strings5_1_altaltalt[x][1:]])
        transpose_indices_51=np.array(transposed_indices_1).T
        transposed_indices_51_list.append(transpose_indices_51)

    transposed_indices_53_lists.append(transposed_indices_53_list)
    transposed_indices_51_lists.append(transposed_indices_51_list)

all_permutations_lists6=[]
all_permutations_lists5=[]

from itertools import permutations
for dim in dims:
    all_permutations_list6=[]
    for index,(i,j,k,l,m,n) in enumerate(reduced_indices6_list[dim-2]):
        all_permutations6=set(permutations([i,j,k,l,m,n]))
        all_permutations_list6.append(tuple(all_permutations6))
    all_permutations_lists6.append(all_permutations_list6)
    all_permutations_list5=[]
    for index,(i,j,k,l,m) in enumerate(reduced_indices5_list[dim-2]):
        all_permutations5=set(permutations([i,j,k,l,m]))
        all_permutations_list5.append(tuple(all_permutations5))
    all_permutations_lists5.append(all_permutations_list5)
all_permutations_new_lists6=[]
all_permutations_new_lists5=[]

for dim in dims:
    all_permutations_list_new6=[]
    for index,ap in enumerate(all_permutations_lists6[dim-2]):
        length=len(ap)
        ap=np.array(ap).reshape(length,-1)
        all_permutations_list_new6.append(ap)
    all_permutations_new_lists6.append(all_permutations_list_new6)
    all_permutations_list_new5=[]
    for index,ap in enumerate(all_permutations_lists5[dim-2]):
        length=len(ap)
        ap=np.array(ap).reshape(length,-1)
        all_permutations_list_new5.append(ap)
    all_permutations_new_lists5.append(all_permutations_list_new5)
def make_integrals(matrix,order=2):
    input_matrix=0.5*matrix #This is just the covariance matrix
    
    if order==4:
        fourth_order_sum_elements = input_matrix[:, :, :, np.newaxis, np.newaxis] * input_matrix[:, np.newaxis, np.newaxis, :, :]
        fourth_order_integrals=summaker(fourth_order_sum_elements,fourth_partition_strings)
        return fourth_order_integrals
    sixth_order_sum_elements=input_matrix[:, :, :, np.newaxis, np.newaxis,np.newaxis,np.newaxis] * input_matrix[:, np.newaxis, np.newaxis, :, :,np.newaxis,np.newaxis]*input_matrix[:, np.newaxis,np.newaxis,np.newaxis, np.newaxis, :, :]
    if order==6:
        sixth_order_integrals=sixth_order_sum_elements.copy()
        for i in range(len(six_partition_string_transpose)):
            sixth_order_integrals+=(sixth_order_sum_elements.transpose(six_partition_string_transpose[i]))
        return sixth_order_integrals
    eight_order_sum_elements=einsum('aijklmn,aop->aijklmnop',sixth_order_sum_elements,input_matrix)
    if order==8:
        return summaker(eight_order_sum_elements,eight_partition_strings)
    return sixth_order_integrals
def summaker(sum_elements,partition_strings):
    expectations=np.zeros_like(sum_elements)
    length=len(sum_elements.shape)

    for i in range(len(partition_strings)):
        permutation=partition_strings[i]
        expectations+=einsum('%s->%s'%(permutation,stringy[:length]),sum_elements)
    return expectations
def find_returnval6_reduced(mu4,mu2,inp,fourth_expecs,reduced_indices6,ti64,ti62):
       i, j, k, l, m, n = np.array(reduced_indices6).T  # extract i, j, k, l, m, and n from indices
       returnval_reduced = mu4[:, i,j,k,l]*mu2[:, m,n]
       for x in range(0,len(ti64)):
              ia,ja,ka,la,ma,na=ti64[x]
              returnval_reduced+=mu4[:,ia,ja,ka,la]*inp[:,ma,na]
       for x in range(0,len(ti62)):
              ia,ja,ka,la,ma,na=ti62[x]
              returnval_reduced+=mu2[:,ia,ja]*fourth_expecs[:,ka,la,ma,na]
       return returnval_reduced

def makeFulltensor5(tensor_reduced, all_permutations_list, dim=4):
    tensor_reduced = tensor_reduced.reshape(tensor_reduced.shape[0], -1, 1)
    full_tensors = np.zeros((tensor_reduced.shape[0], dim, dim, dim, dim, dim), dtype=tensor_reduced.dtype)
    for index, all_permutations in enumerate(all_permutations_list):
        ap = np.array(all_permutations) 
        full_tensors[(np.s_[:], ap[:, 0], ap[:, 1], ap[:, 2], ap[:, 3], ap[:, 4])] = tensor_reduced[:, index]
    return full_tensors
def find_returnval5_reduced(mu,mu3,inp,fourth_expecs,reduced_indices5,ti53,ti51):
       indices = np.array(reduced_indices5)
       i, j, k, l, m = indices.T  # extract i, j, k, l, m, and n from indices
       returnval_reduced = mu3[:, i,j,k]*mu[:, l]*mu[:,m]
       for x in range(0,len(ti53)):
              ia,ja,ka,la,ma=ti53[x]
              returnval_reduced+=mu3[:,ia,ja,ka]*inp[:,la,ma]
       ###THIS ONE IS WRONG - ti53 is correct, ti51 is not
       for x in range(0,len(ti51)):
              ia,ja,ka,la,ma=ti51[x]
              returnval_reduced+=mu[:,ia]*fourth_expecs[:,ja,ka,la,ma]
       return returnval_reduced

def makeFulltensor6(tensor_reduced, all_permutations_list, dim=4):
    tensor_reduced = tensor_reduced.reshape(tensor_reduced.shape[0], -1, 1)
    full_tensors = np.zeros((tensor_reduced.shape[0], dim, dim, dim, dim, dim, dim), dtype=tensor_reduced.dtype)
    for index, all_permutations in enumerate(all_permutations_list):
        ap = np.array(all_permutations) 
        full_tensors[(np.s_[:], ap[:, 0], ap[:, 1], ap[:, 2], ap[:, 3], ap[:, 4], ap[:, 5])] = tensor_reduced[:, index]
    return full_tensors

def calculate_all_expectation_values_full_allij(mu,matrix,ovlps,order=6):
    dimension=mu.shape[1]
    inp=matrix/2
    returnvals=[]
    returnvals.append(einsum('n,ni->ni',ovlps,mu))
    mu2=einsum('ni,nj->nij',mu,mu)
    returnvals.append(einsum('n,nij->nij',ovlps,inp+mu2))
    if order==2:
        return returnvals
    mu3=einsum('ni,njk->nijk',mu,mu2)
    mu3_norm=einsum('n,ni,njk->nijk',ovlps,mu,mu2)
    returnvals.append(mu3_norm+einsum('n,ni,njk->nijk',ovlps,mu,inp)+einsum('n,nj,nik->nijk',ovlps,mu,inp)+einsum('n,nk,nij->nijk',ovlps,mu,inp))
    fourth_expecs=make_integrals(matrix,order=4)
    mu4=einsum('ni,njkl->nijkl',mu,mu3)
    returnval4=mu4+fourth_expecs
    for stringy in strings_4:
        returnval4+= +einsum('%s'%stringy,mu2,inp) 
    returnval4=einsum('n,nijkl->nijkl',ovlps,returnval4)
    returnvals.append(returnval4)
    if order==4:
        return returnvals
    if dimension>6:
        #Only slower in high dimensions
        mu5=mu[:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]*mu4[:,np.newaxis,...]
        
        returnval5=mu5.copy()      
        temp1=einsum('%s'%strings5_3[0],mu3,inp)
        for stringy in strings5_3_altalt:
            returnval5+= np.transpose(temp1, stringy)
        returnval5+=temp1
        temp2=einsum('%s'%strings5_1[0],mu,fourth_expecs)
        for stringy in strings5_1_altalt:
            returnval5+= np.transpose(temp2, stringy)
        returnval5+=temp2
    else:
        #slower in few dimensions
        ind=dimension-2
        returnval5_temp=find_returnval5_reduced(mu,mu3,inp,fourth_expecs,reduced_indices5_list[ind],transposed_indices_53_lists[ind],transposed_indices_51_lists[ind])
        returnval5=makeFulltensor5(returnval5_temp,all_permutations_new_lists5[ind],dim=dimension) 
    returnval5=einsum('n,nijklm->nijklm',ovlps,returnval5)
    returnvals.append(returnval5)
    if order==5:
        return returnvals 
    
    returnval6=make_integrals(matrix,order=6)
    if dimension>6:
        #print("Slower")
        mu5=mu[:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]*mu4[:,np.newaxis,...]
        returnval6+=mu[:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis]*mu5[:,np.newaxis,...]
        temp1=mu4[..., np.newaxis, np.newaxis] * inp[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]
        temp2=mu2[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis] * fourth_expecs[:, np.newaxis, np.newaxis, ...] 
        returnval6+=temp1
        returnval6+=temp2
        for i in range(len(strings_6_4_altalt[:])):
            returnval6+= +np.transpose(temp1, strings_6_4_altalt[i])
            returnval6+= +np.transpose(temp2, strings_6_2_altalt[i])
    else:
        #print("Quicker")
        ind=dimension-2
        
        returnval6_temp=find_returnval6_reduced(mu4,mu2,inp,fourth_expecs,reduced_indices6_list[ind],transposed_indices_64_lists[dimension-2],transposed_indices_62_lists[ind])
        
        returnval6+=makeFulltensor6(returnval6_temp,all_permutations_new_lists6[ind],dim=dimension)
    returnval6=einsum('a,aijklmn->aijklmn',ovlps,returnval6)
    returnvals.append(returnval6)
    return returnvals
def get_x7_x8_contraction_allij(mu,matrix,ovlps):
    inp=matrix/2
    mum1=mu[:,:-1]
    mu1=mu[:,1:]
    mu_m_contr_m1m11=einsum("am,am,am->a",mum1,mum1,mu1)
    mu_m_contr_111=einsum("am,am,am->a",mu1,mu1,mu1)
    fourth_expecs=make_integrals(matrix,order=4)
    sixth_expecs=make_integrals(matrix,order=6)
    fourth_111_temp=einsum("almmm->al",fourth_expecs[:,:,1:,1:,1:])
    fourth_m1m11_temp=einsum("almmm->al",fourth_expecs[:,:,:-1,:-1,1:])
    sixth_111_temp=einsum("alknmmm->alkn",sixth_expecs[:,:,:,:,1:,1:,1:])
    sixth_m1m11_temp=einsum("alknmmm->alkn",sixth_expecs[:,:,:,:,:-1,:-1,1:])
    mum1_fem11contr=einsum('ak,aijkk->aij',mum1,fourth_expecs[:,:,:,:-1,1:])
    mu1_fem1m1contr=einsum('ak,aijkk->aij',mu1,fourth_expecs[:,:,:,:-1,:-1])
    mu1_fe11contr=einsum('ak,aijkk->aij',mu1,fourth_expecs[:,:,:,1:,1:])
    mu1_inp11_temp=einsum("am,amm->a",mu1,inp[:,1:,1:])
    mu1mu1fe1_temp=einsum('am,am,ajklm->ajkl',mu1,mu1,fourth_expecs[:,:,:,:,1:])
    contraction_1=einsum( 'ak,al,a,a->akl' ,mu,mu,mu_m_contr_111,mu_m_contr_111)  
    contraction_1+=6*einsum( 'ai,aj,a,a->aij' ,mu,mu,mu_m_contr_111,mu1_inp11_temp)
    contraction_1+=9*einsum( 'ai,aj,ak,ak,al,al,akl->aij' ,mu,mu,mu1,mu1,mu1,mu1,inp[:,1:,1:])
    temp=einsum( 'ai,al,a,al,ajl->aij' ,mu,mu1,mu_m_contr_111,mu1,inp[:,:,1:])
    contraction_1+=6*temp
    contraction_1+=6*np.transpose(temp,[0,2,1])
    contraction_1+=einsum( 'a,a,aij->aij' ,mu_m_contr_111,mu_m_contr_111,inp)
    contraction_1+= 3*einsum( 'ai,aj,ak,ak,ak->aij' ,mu,mu,mu1,mu1,fourth_111_temp[:,1:])
    mu1_fe11_temp=einsum("ak,ajkkl->ajl",mu1,fourth_expecs[:,:,1:,1:,:]) 
    contraction_1+= 9*einsum( 'ai,aj,al,all->aij' ,mu,mu,mu1,mu1_fe11_temp[:,1:,1:]) 
    fe_lll_contr=einsum("alllk->ak",fourth_expecs[:,1:,1:,1:,:])

    contraction_1+= 3*einsum( 'ai,aj,al,al,al->aij' ,mu,mu,mu1,mu1,fe_lll_contr[:,1:]) 
    contraction_1+= 1*einsum( 'ai,a,aj->aij' ,mu,mu_m_contr_111,fourth_111_temp) 
    temp=6*einsum( 'ai,ak,ak,al,ajkll->aij' ,mu,mu1,mu1,mu1,fourth_expecs[:,:,1:,1:,1:]) 
    contraction_1+= temp+np.transpose(temp,[0,2,1])
    temp=einsum( 'ai,ak,al,al,ajkkl->aij' ,mu,mu1,mu1,mu1,fourth_expecs[:,:,1:,1:,1:]) 
    contraction_1+=9*temp+7*np.transpose(temp,[0,2,1])
    contraction_1+=6*einsum( 'a,aij->aij' ,mu_m_contr_111,mu1_fe11contr)
    temp=einsum( 'ai,a,aj->aij' ,mu,mu_m_contr_111,fourth_111_temp)
    contraction_1+=temp+np.transpose(temp,[0,2,1])
    contraction_1+=9*einsum( 'ak,ak,al,al,aijkl->aij' ,mu1,mu1,mu1,mu1,fourth_expecs[:,:,:,1:,1:])
    contraction_1+=einsum( 'ak,ak,aj,ak,ai->aij' ,mu1,mu1,mu,mu1,fourth_111_temp)
    temp=3*einsum( 'ak,ak,ai,al,ajkll->aij' ,mu1,mu1,mu,mu1,fourth_expecs[:,:,1:,1:,1:])
    contraction_1+=temp+np.transpose(temp,[0,2,1])
    contraction_1+=2*einsum( 'aj,al,al,ail->aij' ,mu,mu1,mu1,mu1_fe11_temp[:,:,1:])
    contraction_1+=einsum( 'ai,aj,akkk->aij' ,mu,mu,sixth_111_temp[:,1:,1:,1:])
    temp=3*einsum( 'ai,ak,ajkk->aij' ,mu,mu1,sixth_111_temp[:,:,1:,1:])
    contraction_1+=temp+np.transpose(temp,[0,2,1])
    sixth_111kkk_contr=einsum("ajkkklm->ajlm",sixth_expecs[:,:,1:,1:,1:,:,:])
    sixth_m1m11kkk_contr=einsum("ajkkklm->ajlm",sixth_expecs[:,:,:-1,:-1,1:,:,:])
    temp=3*einsum( 'ai,al,ajll->aij' ,mu,mu1,sixth_111kkk_contr[:,:,1:,1:])
    contraction_1+=temp+np.transpose(temp,[0,2,1])
    temp=3*einsum( 'ak,ak,aijk->aij' ,mu1,mu1,sixth_111_temp[:,:,:,1:])
    contraction_1+=temp+np.transpose(temp,[0,2,1])
    contraction_1+=9*einsum( 'ak,al,aijkkll->aij' ,mu1,mu1,sixth_expecs[:,:,:,1:,1:,1:,1:])
    akklall_contr=einsum("akl,all->ak",inp[:,1:,1:],inp[:,1:,1:])
    contraction_1+=9*einsum( 'aij,akk,ak->aij' ,inp,inp[:,1:,1:],akklall_contr)
    contraction_1+=36*einsum( 'aik,ajk,ak->aij' ,inp[:,:,1:],inp[:,:,1:],akklall_contr)
    contraction_1+=18*einsum( 'aik,ajl,akk,all->aij' ,inp[:,:,1:],inp[:,:,1:],inp[:,1:,1:],inp[:,1:,1:])
    contraction_1+=36*einsum( 'aik,ajl,akl,akl->aij' ,inp[:,:,1:],inp[:,:,1:],inp[:,1:,1:],inp[:,1:,1:])
    contraction_1+=6*einsum( 'aij,akl,akl,akl->aij' ,inp,inp[:,1:,1:],inp[:,1:,1:],inp[:,1:,1:])
    mum1_in_m11_contr=einsum("am,amm->a",mum1,inp[:,:-1,1:])
    fourth_kkk_contr=einsum("akkkl->al",fourth_expecs[:,:-1,:-1,1:,:])
    t_contraction_2=einsum('a,a->a',mu_m_contr_m1m11,mu_m_contr_m1m11)
    t_contraction_2+=2*einsum('a,a->a',mu_m_contr_m1m11,mum1_in_m11_contr)
    t_contraction_2+=einsum('a,al,all->a',mu_m_contr_m1m11,mu1,inp[:,:-1,:-1])
    t_contraction_2+=einsum('ak,ak,al,al,akl->a',mum1,mum1,mum1,mum1,inp[:,1:,1:])
    t_contraction_2+=2*einsum('ak,ak,al,al,akl->a',mum1,mum1,mum1,mu1,inp[:,1:,:-1])
    t_contraction_2+=einsum('ak,ak,al,al,akl->a',mum1,mu1,mum1,mum1,inp[:,:-1,1:])
    t_contraction_2+=4*einsum('ak,ak,al,al,akl->a',mum1,mu1,mum1,mu1,inp[:,:-1,:-1])
    t_contraction_2+=2*einsum('a,a->a',mu_m_contr_m1m11,mum1_in_m11_contr)
    t_contraction_2+=einsum('ak,al,al,ak,akl->a',mum1,mum1,mum1,mu1,inp[:,:-1,1:])
    t_contraction_2+=einsum('ak,a,akk->a',mu1,mu_m_contr_m1m11,inp[:,:-1,:-1])
    t_contraction_2+=einsum('ak,ak,ak->a',mum1,mum1,fourth_m1m11_temp[:,1:])
    t_contraction_2+=2*einsum('ak,ak,ak->a',mum1,mu1,fourth_m1m11_temp[:,:-1])
    t_contraction_2+=4*einsum('ak,al,akkll->a',mum1,mum1,fourth_expecs[:,:-1,1:,:-1,1:])
    t_contraction_2+=2*einsum('ak,al,akkll->a',mum1,mu1,fourth_expecs[:,:-1,1:,:-1,:-1])
    t_contraction_2+=2*einsum('ak,al,akkll->a',mu1,mum1,fourth_expecs[:,:-1,:-1,:-1,1:])
    t_contraction_2+=einsum('ak,al,akkll->a',mu1,mu1,fourth_expecs[:,:-1,:-1,:-1,:-1])
    t_contraction_2+=einsum('al,al,al->a',mum1,mum1,fourth_kkk_contr[:,1:])
    t_contraction_2+=2*einsum('al,al,al->a',mum1,mu1,fourth_kkk_contr[:,:-1])
    t_contraction_2+=einsum('akkk->a',sixth_m1m11_temp[:,:-1,:-1,1:])

    contraction_2=einsum('ai,aj,a->aij',mu,mu,t_contraction_2)
    temp=einsum('ai,ak,ak,a,ajk->aij',mu,mum1,mum1,mu_m_contr_m1m11,inp[:,:,1:])
    contraction_2+=temp+np.transpose(temp,[0,2,1])
    temp=2*einsum('ai,ak,ak,a,ajk->aij',mu,mum1,mu1,mu_m_contr_m1m11,inp[:,:,:-1])
    contraction_2+=temp+np.transpose(temp,[0,2,1])
    contraction_2+=einsum('a,a,aij->aij',mu_m_contr_m1m11,mu_m_contr_m1m11,inp)
    temp=2*einsum('ai,a,al,al,ajl->aij',mu,mu_m_contr_m1m11,mum1,mu1,inp[:,:,:-1])
    contraction_2+=temp+np.transpose(temp,[0,2,1])
    temp=einsum('ai,a,al,al,ajl->aij',mu,mu_m_contr_m1m11,mum1,mum1,inp[:,:,1:])
    contraction_2+=temp+np.transpose(temp,[0,2,1])
    temp=einsum('ai,a,aj->aij',mu,mu_m_contr_m1m11,fourth_m1m11_temp)
    contraction_2+=np.transpose(temp,[0,2,1])
    contraction_2+=2*temp
    temp=einsum('ai,ak,ak,al,ajkll->aij',mu,mum1,mu1,mu1,fourth_expecs[:,:,:-1,:-1,:-1])
    contraction_2+=temp+np.transpose(temp,[0,2,1])
    temp=4*einsum('ai,ak,al,al,ajkkl->aij',mu,mum1,mum1,mu1,fourth_expecs[:,:,:-1,1:,:-1])
    contraction_2+=temp+np.transpose(temp,[0,2,1])
    contraction_2+=2*einsum('a,aij->aij',mu_m_contr_m1m11,mu1_fem1m1contr)
    contraction_2+=2*einsum('ak,ak,al,al,aijkl->aij',mum1,mum1,mum1,mu1,fourth_expecs[:,:,:,1:,:-1])
    contraction_2+=4*einsum('ak,ak,al,al,aijkl->aij',mum1,mu1,mum1,mu1,fourth_expecs[:,:,:,:-1,:-1])
    contraction_2+=2*einsum('a,aij->aij',mu_m_contr_m1m11,mum1_fem11contr)
    temp=einsum('ai,ak,al,al,ajkkl->aij',mu,mu1,mum1,mu1,fourth_expecs[:,:,:-1,:-1,:-1])
    contraction_2+=temp+2*np.transpose(temp,[0,2,1])
    contraction_2+=2*einsum('a,aij->aij',mu_m_contr_m1m11,mum1_fem11contr)
    contraction_2+=2*einsum('ak,ak,al,al,aijkl->aij',mum1,mu1,mum1,mum1,fourth_expecs[:,:,:,:-1,1:])
    contraction_2+=einsum('ak,ak,aj,ak,ai->aij',mum1,mu1,mu,mum1,fourth_m1m11_temp)
    temp=2*einsum('ak,ak,ai,al,ajkll->aij',mum1,mu1,mu,mum1,fourth_expecs[:,:,:-1,:-1,1:])
    contraction_2+=temp+np.transpose(temp,[0,2,1])
    temp=einsum('ak,ak,ai,al,ajkll->aij',mum1,mu1,mu,mu1,fourth_expecs[:,:,:-1,:-1,:-1])
    contraction_2+=temp+np.transpose(temp,[0,2,1])
    temp=2*einsum('ai,ak,al,al,ajkkl->aij',mu,mum1,mum1,mum1,fourth_expecs[:,:,:-1,1:,1:])
    contraction_2+=temp+np.transpose(temp,[0,2,1])
    contraction_2+=einsum('ak,aj,al,al,aikkl->aij',mu1,mu,mum1,mum1,fourth_expecs[:,:,:-1,:-1,1:])
    contraction_2+=einsum('ak,ak,al,al,aijkl->aij',mum1,mum1,mum1,mum1,fourth_expecs[:,:,:,1:,1:])
    temp=einsum('ai,ak,ak,al,ajkll->aij',mu,mum1,mum1,mu1,fourth_expecs[:,:,1:,:-1,:-1])
    contraction_2+=temp+np.transpose(temp,[0,2,1])
    temp=2*einsum('ai,ak,ak,al,ajkll->aij',mu,mum1,mu1,mum1,fourth_expecs[:,:,:-1,:-1,1:])
    contraction_2+=temp+np.transpose(temp,[0,2,1])
    
    contraction_2+=einsum('ai,ak,al,al,ajkkl->aij',mu,mu1,mum1,mum1,fourth_expecs[:,:,:-1,:-1,1:])
    contraction_2+=einsum('ak,aj,al,al,aikkl->aij',mu1,mu,mum1,mu1,fourth_expecs[:,:,:-1,:-1,:-1])
    mum1fe1m11temp=einsum("al,ajkll->ajk",mum1,fourth_expecs[:,:,1:,:-1,1:])
    temp=2*einsum('ai,ak,ak,ajk->aij',mu,mum1,mum1,mum1fe1m11temp)
    contraction_2+=temp+np.transpose(temp,[0,2,1])





    temp=2*einsum('ai,ak,ajkk->aij',mu,mum1,sixth_m1m11_temp[:,:,:-1,1:])
    contraction_2+=temp+np.transpose(temp,[0,2,1])
    temp=einsum('ai,ak,ajkk->aij',mu,mu1,sixth_m1m11_temp[:,:,:-1,:-1])
    contraction_2+=temp+np.transpose(temp,[0,2,1])

    temp=2*einsum('ai,al,ajll->aij',mu,mum1,sixth_m1m11kkk_contr[:,:,:-1,1:])
    contraction_2+=temp+np.transpose(temp,[0,2,1])
    temp=einsum('ai,al,ajll->aij',mu,mu1,sixth_m1m11kkk_contr[:,:,:-1,:-1])
    contraction_2+=temp+np.transpose(temp,[0,2,1])
    contraction_2+=einsum('ak,ak,aijk->aij',mum1,mum1,sixth_m1m11_temp[:,:,:,1:])
    contraction_2+=2*einsum('ak,ak,aijk->aij',mum1,mu1,sixth_m1m11_temp[:,:,:,:-1])
    contraction_2+=4*einsum('ak,al,aijkkll->aij',mum1,mum1,sixth_expecs[:,:,:,:-1,1:,:-1,1:])
    contraction_2+=4*einsum('ak,al,aijkkll->aij',mum1,mu1,sixth_expecs[:,:,:,:-1,1:,:-1,:-1])
    contraction_2+=einsum('ak,al,aijkkll->aij',mu1,mu1,sixth_expecs[:,:,:,:-1,:-1,:-1,:-1])
    contraction_2+=einsum('al,al,aijl->aij',mum1,mum1,sixth_m1m11kkk_contr[:,:,:,1:])
    contraction_2+=2*einsum('al,al,aijl->aij',mum1,mu1,sixth_m1m11kkk_contr[:,:,:,:-1])
    contraction_2+=2*einsum('aik,ajk,akl,all->aij',inp[:,:,:-1],inp[:,:,1:],inp[:,:-1,1:],inp[:,:-1,:-1])
    temp=4*einsum('ail,ajk,all,akk->aij',inp[:,:,:-1],inp[:,:,:-1],inp[:,:-1,1:],inp[:,:-1,1:])
    contraction_2+=temp+np.transpose(temp,[0,2,1])


    temp=2*einsum('ail,ajk,akl,akl->aij',inp[:,:,1:],inp[:,:,1:],inp[:,:-1,:-1],inp[:,:-1,:-1])
    contraction_2+=temp+np.transpose(temp,[0,2,1])
    aklakk_contr=einsum("akl,akk->al",inp[:,1:,:-1],inp[:,:-1,:-1])

    contraction_2+=2*einsum('ail,ajl,al->aij',inp[:,:,:-1],inp[:,:,1:],aklakk_contr)
    contraction_2+=4*einsum('ail,ajl,al->aij',inp[:,:,1:],inp[:,:,:-1],aklakk_contr)
    contraction_2+=4*einsum('ail,ajl,akl,akk->aij',inp[:,:,:-1],inp[:,:,:-1],inp[:,1:,1:],inp[:,:-1,:-1])
    contraction_2+=8*einsum('ail,ajl,akl,akk->aij',inp[:,:,:-1],inp[:,:,:-1],inp[:,:-1,1:],inp[:,:-1,1:])
    aklakk_contr2=einsum("akl,akk->al",inp[:,:-1,:-1],inp[:,:-1,1:])
    contraction_2+=8*einsum('ail,ajl,al->aij',inp[:,:,:-1],inp[:,:,1:],aklakk_contr2)

    contraction_2+=8*einsum('ail,ajl,al->aij',inp[:,:,1:],inp[:,:,:-1],aklakk_contr2)
    contraction_2+=16*einsum('ail,ajk,akl,akl->aij',inp[:,:,1:],inp[:,:,:-1],inp[:,:-1,:-1],inp[:,1:,:-1])
    contraction_2+=8*einsum('ail,ajk,akl,akl->aij',inp[:,:,:-1],inp[:,:,:-1],inp[:,:-1,1:],inp[:,1:,:-1])
    contraction_2+=8*einsum('ail,ajk,akl,akl->aij',inp[:,:,:-1],inp[:,:,:-1],inp[:,:-1,:-1],inp[:,1:,1:])

    contraction_2+=2*einsum('ail,ajk,all,akk->aij',inp[:,:,1:],inp[:,:,1:] ,inp[:,:-1,:-1],inp[:,:-1,:-1])
    contraction_2+=8*einsum('ail,ajk,all,akk->aij',inp[:,:,1:],inp[:,:,:-1],inp[:,:-1,:-1],inp[:,:-1,1:])
    temp_contr =4*einsum('all,al->a',inp[:,:-1,1:],aklakk_contr2)
    temp_contr+=4*einsum('all,alk,akk->a',inp[:,:-1,1:],inp[:,:-1,1:],inp[:,:-1,:-1])
    temp_contr+=  einsum('all,akl,akk->a',inp[:,:-1,:-1],inp[:,1:,1:],inp[:,:-1,:-1])

    #temp_contr+=2*einsum('akl,akl,akl->a',inp[:,:-1,1:],inp[:,:-1,:-1],inp[:,1:,:-1])
    temp_contr+=4*einsum('akl,akl,akl->a',inp[:,:-1,1:],inp[:,:-1,:-1],inp[:,1:,:-1])
    temp_contr+=2*einsum('akl,akl,akl->a',inp[:,:-1,:-1],inp[:,:-1,:-1],inp[:,1:,1:])
    contraction_2+=einsum("aij,a->aij",inp,temp_contr)



    t_contraction_3=einsum('a,a->a',mu_m_contr_111,mu_m_contr_m1m11)
    t_contraction_3+=2*einsum('a,a->a',mu_m_contr_111,mum1_in_m11_contr)
    t_contraction_3+=einsum('a,al,all->a',mu_m_contr_111,mu1,inp[:,:-1,:-1])
    t_contraction_3+=2*einsum('ak,ak,al,al,akl->a',mu1,mu1,mum1,mum1,inp[:,1:,1:])
    t_contraction_3+=6*einsum('ak,ak,al,al,akl->a',mu1,mu1,mum1,mu1,inp[:,1:,:-1])
    t_contraction_3+=3*einsum('ak,a,akk->a',mu1,mu_m_contr_m1m11,inp[:,1:,1:])
    t_contraction_3+=einsum('ak,al,al,ak,akl->a',mu1,mum1,mum1,mu1,inp[:,1:,1:])
    t_contraction_3+=3*einsum('ak,ak,ak->a',mu1,mu1,fourth_m1m11_temp[:,1:])
    t_contraction_3+=6*einsum('ak,al,akkll->a',mu1,mum1,fourth_expecs[:,1:,1:,:-1,1:])
    t_contraction_3+=3*einsum('ak,al,akkll->a',mu1,mu1,fourth_expecs[:,1:,1:,:-1,:-1])
    t_contraction_3+=einsum('ak,ak,ak->a',mum1,mum1,fe_lll_contr[:,1:])
    t_contraction_3+=2*einsum('ak,ak,ak->a',mum1,mu1,fe_lll_contr[:,:-1])
    t_contraction_3+=einsum('akkk->a',sixth_m1m11_temp[:,1:,1:,1:])
    contraction_3=einsum('ai,aj,a->aij',mu,mu,t_contraction_3)
    temp=3*einsum('ai,ak,ak,a,ajk->aij',mu,mu1,mu1,mu_m_contr_m1m11,inp[:,:,1:])
    contraction_3+=temp+np.transpose(temp,[0,2,1])
    temp=2*einsum('ai,a,al,al,ajl->aij',mu,mu_m_contr_111,mum1,mu1,inp[:,:,:-1])
    contraction_3+=temp+np.transpose(temp,[0,2,1])
    contraction_3+=einsum('a,a,aij->aij',mu_m_contr_111,mu_m_contr_m1m11,inp)
    temp=einsum('ai,a,al,al,ajl->aij',mu,mu_m_contr_111,mum1,mum1,inp[:,:,1:])
    contraction_3+=temp+np.transpose(temp,[0,2,1])


    contraction_3+=einsum('ai,a,aj->aij',mu,mu_m_contr_111,fourth_m1m11_temp)
    
    contraction_3+=12*einsum('ai,ak,al,al,ajkkl->aij',mu,mum1,mu1,mu1,fourth_expecs[:,:,1:,:-1,1:])
    contraction_3+=6*einsum('ai,ak,al,al,ajkkl->aij',mu,mu1,mu1,mu1,fourth_expecs[:,:,:-1,:-1,1:])
    
    contraction_3+=6*einsum('ai,al,al,ajl->aij',mu,mum1,mum1,mu1_fe11_temp[:,:,1:])
    contraction_3+=12*einsum('ai,al,al,ajl->aij',mu,mum1,mu1,mu1_fe11_temp[:,:,:-1])

    contraction_3+=3*einsum('a,aij->aij',mu_m_contr_m1m11,mu1_fe11_temp)
    contraction_3+=einsum('ai,a,aj->aij',mu,mu_m_contr_m1m11,fourth_111_temp)
    contraction_3+=6*einsum('ak,ak,al,al,aijkl->aij',mu1,mu1,mum1,mu1,fourth_expecs[:,:,:,1:,:-1])
    contraction_3+=einsum('ai,a,aj->aij',mu,mu_m_contr_m1m11,fourth_111_temp)
    contraction_3+=2*einsum('a,aij->aij',mu_m_contr_111,mum1_fem11contr)
    contraction_3+=einsum('a,aij->aij',mu_m_contr_111,mu1_fem1m1contr)
    contraction_3+=3*einsum('ak,ak,al,al,aijkl->aij',mu1,mu1,mum1,mum1,fourth_expecs[:,:,:,1:,1:])
    contraction_3+=einsum('ak,ak,aj,ak,ai->aij',mu1,mu1,mu,mu1,fourth_m1m11_temp)

    temp=einsum('ai,al,ajll->aij',mu,mu1,sixth_111kkk_contr[:,:,:-1,:-1])
    contraction_3+=temp+np.transpose(temp,[0,2,1])
    temp=3*einsum('ai,ak,ajkk->aij',mu,mu1,sixth_m1m11_temp[:,:,1:,1:])
    contraction_3+=temp+np.transpose(temp,[0,2,1])
    temp=2*einsum('ai,al,ajll->aij',mu,mum1,sixth_111kkk_contr[:,:,:-1,1:])
    contraction_3+=temp+np.transpose(temp,[0,2,1])
    contraction_3+=3*einsum('ak,ak,aijk->aij',mu1,mu1,sixth_m1m11_temp[:,:,:,1:])
    contraction_3+=6*einsum('ak,al,aijkkll->aij',mu1,mum1,sixth_expecs[:,:,:,1:,1:,:-1,1:])
    contraction_3+=3*einsum('ak,al,aijkkll->aij',mu1,mu1,sixth_expecs[:,:,:,1:,1:,:-1,:-1])
    contraction_3+=einsum('al,al,aijl->aij',mum1,mum1,sixth_111kkk_contr[:,:,:,1:])
    contraction_3+=2*einsum('al,al,aijl->aij',mum1,mu1,sixth_111kkk_contr[:,:,:,:-1])

    contraction_3+=12*einsum('aik,ajl,akl,akl->aij',inp[:,:,1:],inp[:,:,:-1],inp[:,1:,1:],inp[:,1:,:-1])
    contraction_3+=6*einsum('ajl,aik,akl,akl->aij',inp[:,:,1:],inp[:,:,:-1],inp[:,:-1,1:],inp[:,1:,1:])
    contraction_3+=6*einsum('aik,ajl,akl,akl->aij',inp[:,:,1:],inp[:,:,:-1],inp[:,1:,:-1],inp[:,1:,1:])
    temp=          6*einsum('aik,ajl,akl,akl->aij',inp[:,:,1:],inp[:,:,1:],inp[:,1:,:-1],inp[:,1:,:-1])
    contraction_3+=temp+np.transpose(temp,[0,2,1])
    contraction_3+=12*einsum('aik,ajk,all,akl->aij',inp[:,:,1:],inp[:,:,:-1],inp[:,1:,1:],inp[:,:-1,1:])

    contraction_3+=6*einsum('aik,ajk,akl,all->aij',inp[:,:,1:],inp[:,:,1:],inp[:,1:,1:],inp[:,:-1,:-1])
    contraction_3+=6*einsum('aik,ajl,akk,all->aij',inp[:,:,1:],inp[:,:,:-1],inp[:,1:,1:],inp[:,:-1,1:])
    contraction_3+=12*einsum('aik,ajk,akl,all->aij',inp[:,:,1:],inp[:,:,1:],inp[:,1:,:-1],inp[:,:-1,1:])
    
    temp=           3*einsum('aik,ajl,akk,all->aij',inp[:,:,1:],inp[:,:,1:],inp[:,1:,1:],inp[:,:-1,:-1])
    contraction_3+=temp+np.transpose(temp,[0,2,1])
    
    contraction_3+=6*einsum('aik,ajk,ak->aij',inp[:,:,:-1],inp[:,:,:-1],akklall_contr)
    contraction_3+=6*einsum('aik,ajl,akk,all->aij',inp[:,:,:-1],inp[:,:,1:],inp[:,:-1,1:],inp[:,1:,1:])
    
    contr_temp=6*einsum("akl,akl,akl->a",inp[:,1:,:-1],inp[:,1:,1:],inp[:,1:,:-1])
    contr_temp+=3*einsum("al,all->a",akklall_contr,inp[:,:-1,:-1])
    contr_temp+=6*einsum("akk,akl,all->a",inp[:,1:,1:],inp[:,1:,:-1],inp[:,:-1,1:])
    contraction_3+=einsum("aij,a->aij",inp,contr_temp)

    x7_e_contracted_first=einsum('ai,aj,ak,ak,a->aij',mu,mu,mu,mu,mu_m_contr_m1m11)
    x7_e_contracted_first+=2*einsum('ai,aj,ak,ak,a->aij',mu,mu,mu,mu,mum1_in_m11_contr)
    x7_e_contracted_first+=1*einsum('ai,aj,ak,ak,al,all->aij',mu,mu,mu,mu,mu1,inp[:,:-1,:-1])
    x7_e_contracted_first+=2*einsum('ai,aj,ak,al,al,akl->aij',mu,mu,mu,mum1,mum1,inp[:,:,1:])
    x7_e_contracted_first+=4*einsum('ai,aj,ak,al,al,akl->aij',mu,mu,mu,mum1,mu1,inp[:,:,:-1])
    temp=1*einsum('ai,ak,ak,al,al,ajl->aij',mu,mu,mu,mum1,mum1,inp[:,:,1:])
    x7_e_contracted_first+=temp+np.transpose(temp,[0,2,1])
    temp=2*einsum('ai,ak,ak,al,al,ajl->aij',mu,mu,mu,mum1,mu1,inp[:,:,:-1])
    x7_e_contracted_first+=temp+np.transpose(temp,[0,2,1])
    x7_e_contracted_first+=1*einsum('ak,ak,a,aij->aij',mu,mu,mu_m_contr_m1m11,inp)
    x7_e_contracted_first+=1*einsum('ai,aj,a,akk->aij',mu,mu,mu_m_contr_m1m11,inp)
    temp=2*einsum('ai,ak,a,ajk->aij',mu,mu,mu_m_contr_m1m11,inp)
    x7_e_contracted_first+=temp+np.transpose(temp,[0,2,1])    
    x7_e_contracted_first+=2*einsum('ai,aj,ak,ak->aij',mu,mu,mu,fourth_m1m11_temp)
    x7_e_contracted_first+=2*einsum('ai,aj,all->aij',mu,mu,mum1_fem11contr)
    x7_e_contracted_first+=1*einsum('ai,aj,all->aij',mu,mu,mu1_fem1m1contr)
    temp=1*einsum('ai,ak,ak,aj->aij',mu,mu,mu,fourth_m1m11_temp)
    x7_e_contracted_first+=temp+np.transpose(temp,[0,2,1])
    temp=4*einsum('ai,ak,ajk->aij',mu,mu,mum1_fem11contr)
    x7_e_contracted_first+=temp+np.transpose(temp,[0,2,1])
    temp=2*einsum('ai,ak,ajk->aij',mu,mu,mu1_fem1m1contr)
    x7_e_contracted_first+=temp+np.transpose(temp,[0,2,1])
    temp=1*einsum('ai,al,al,ajkkl->aij',mu,mum1,mum1,fourth_expecs[:,:,:,:,1:])
    x7_e_contracted_first+=temp+np.transpose(temp,[0,2,1])
    temp=2*einsum('ai,al,al,ajkkl->aij',mu,mum1,mu1,fourth_expecs[:,:,:,:,:-1])
    x7_e_contracted_first+=temp+np.transpose(temp,[0,2,1])
    x7_e_contracted_first+=2*einsum('ak,ak,aij->aij',mu,mu,mum1_fem11contr)
    x7_e_contracted_first+=1*einsum('ak,ak,aij->aij',mu,mu,mu1_fem1m1contr)
    x7_e_contracted_first+=2*einsum('ak,al,al,aijkl->aij',mu,mum1,mum1,fourth_expecs[:,:,:,:,1:])
    x7_e_contracted_first+=4*einsum('ak,al,al,aijkl->aij',mu,mum1,mu1,fourth_expecs[:,:,:,:,:-1])
    x7_e_contracted_first+=1*einsum('a,aijkk->aij',mu_m_contr_m1m11,fourth_expecs)
    temp=1*einsum('ai,ajkk->aij',mu,sixth_m1m11_temp)
    x7_e_contracted_first+=temp+np.transpose(temp,[0,2,1])
    x7_e_contracted_first+=2*einsum('ak,aijk->aij',mu,sixth_m1m11_temp)
    sixth_kk_kontr=einsum("aijkklm->aijlm",sixth_expecs)
    x7_e_contracted_first+=2*einsum('ak,aijkk->aij',mum1,sixth_kk_kontr[:,:,:,:-1,1:])
    x7_e_contracted_first+=1*einsum('ak,aijkk->aij',mu1,sixth_kk_kontr[:,:,:,:-1,:-1])



    mu1mu1inp1_temp=einsum('am,am,akm->ak',mu1,mu1,inp[:,:,1:])


    x7_e_contracted_second=einsum('ai,aj,ak,ak,a->aij',mu,mu,mu,mu,mu_m_contr_111)
    x7_e_contracted_second+=3*einsum('ai,aj,ak,ak,a->aij',mu,mu,mu,mu,mu1_inp11_temp)
    x7_e_contracted_second+=6*einsum('ai,aj,ak,ak->aij',mu,mu,mu,mu1mu1inp1_temp)
    
    temp=3*einsum('ai,ak,ak,aj->aji',mu,mu,mu,mu1mu1inp1_temp)
    x7_e_contracted_second+=temp+np.transpose(temp,[0,2,1])

    x7_e_contracted_second+=1*einsum('ak,ak,a,aij->aij',mu,mu,mu_m_contr_111,inp)
    x7_e_contracted_second+=1*einsum('ai,aj,a,akk->aij',mu,mu,mu_m_contr_111,inp)

    temp=2*einsum('ai,ak,a,ajk->aij',mu,mu,mu_m_contr_111,inp)
    x7_e_contracted_second+=temp+np.transpose(temp,[0,2,1])
    x7_e_contracted_second+=2*einsum('ai,aj,ak,ak->aij',mu,mu,mu,fourth_111_temp)
    x7_e_contracted_second+=3*einsum('ai,aj,akk->aij',mu,mu,mu1_fe11contr)
    temp=1*einsum('ai,ak,ak,aj->aij',mu,mu,mu,fourth_111_temp)
    x7_e_contracted_second+=temp+np.transpose(temp,[0,2,1])
    x7_e_contracted_second+=6*einsum('ai,ak,ajk->aij',mu,mu,mu1_fe11contr)
    temp=3*einsum('ai,ajkk->aij',mu,mu1mu1fe1_temp)
    x7_e_contracted_second+=temp+np.transpose(temp,[0,2,1])
    x7_e_contracted_second+=6*einsum('ai,ak,ajk->aij',mu,mu,mu1_fe11contr)
    x7_e_contracted_second+=3*einsum('al,al,aij->aij',mu,mu,mu1_fe11contr)
    x7_e_contracted_second+=6*einsum('ak,aijk->aij',mu,mu1mu1fe1_temp)
    x7_e_contracted_second+=1*einsum('a,aijkk->aij',mu_m_contr_111,fourth_expecs)
    temp=1*einsum('ai,ajkk->aij',mu,sixth_111_temp)
    x7_e_contracted_second+=temp+np.transpose(temp,[0,2,1])
    x7_e_contracted_second+=2*einsum('ak,aijk->aij',mu,sixth_111_temp)
    x7_e_contracted_second+=3*einsum('ak,aijkk->aij',mu1,sixth_kk_kontr[:,:,:,1:,1:])


    x7_e_contracted_firstV=einsum('ai,a,a->ai',mu,mu_m_contr_111,mu_m_contr_111)
    x7_e_contracted_firstV+=9*einsum('ai,ak,ak,al,al,akl->ai',mu,mu1,mu1,mu1,mu1,inp[:,1:,1:])
    x7_e_contracted_firstV+=6*einsum('ak,ak,a,aik->ai',mu1,mu1,mu_m_contr_111,inp[:,:,1:])
    x7_e_contracted_firstV+=6*einsum('ai,ak,a,akk->ai',mu,mu1,mu_m_contr_111,inp[:,1:,1:])
    x7_e_contracted_firstV+=3*einsum('ai,ak,ak,ak->ai',mu,mu1,mu1,fourth_111_temp[:,1:])
    x7_e_contracted_firstV+=9*einsum('ai,al,all->ai',mu,mu1,mu1_fe11_temp[:,1:,1:])
    x7_e_contracted_firstV+=3*einsum('ai,al,al,al->ai',mu,mu1,mu1,fe_lll_contr[:,1:])
    x7_e_contracted_firstV+=2*einsum('ak,ak,ak,ai->ai',mu1,mu1,mu1,fourth_111_temp)
    x7_e_contracted_firstV+=9*einsum('ak,ak,al,aikll->ai',mu1,mu1,mu1,fourth_expecs[:,:,1:,1:,1:])
    x7_e_contracted_firstV+=9*einsum('ak,al,al,aikkl->ai',mu1,mu1,mu1,fourth_expecs[:,:,1:,1:,1:])
    x7_e_contracted_firstV+=1*einsum('ai,akkk->ai',mu,sixth_111_temp[:,1:,1:,1:])
    x7_e_contracted_firstV+=3*einsum('ak,aikk->ai',mu1,sixth_111_temp[:,:,1:,1:])
    x7_e_contracted_firstV+=3*einsum('al,aill->ai',mu1,sixth_111kkk_contr[:,:,1:,1:])
    
    x7_e_contracted_secondV =einsum('ai,a,a->ai',mu,mu_m_contr_m1m11,mu_m_contr_m1m11)
    x7_e_contracted_secondV+=2*einsum('ai,a,a->ai',mu,mu_m_contr_m1m11,mum1_in_m11_contr)
    x7_e_contracted_secondV+=1*einsum('ai,a,al,all->ai',mu,mu_m_contr_m1m11,mu1,inp[:,:-1,:-1])
    x7_e_contracted_secondV+=1*einsum('ai,ak,ak,al,al,akl->ai',mu,mum1,mum1,mum1,mum1,inp[:,1:,1:])
    x7_e_contracted_secondV+=2*einsum('ai,ak,ak,al,al,akl->ai',mu,mum1,mum1,mum1,mu1,inp[:,1:,:-1])
    x7_e_contracted_secondV+=2*einsum('ai,ak,ak,al,al,akl->ai',mu,mum1,mu1,mum1,mum1,inp[:,:-1,1:])
    x7_e_contracted_secondV+=4*einsum('ai,ak,ak,al,al,akl->ai',mu,mum1,mu1,mum1,mu1,inp[:,:-1,:-1])
    x7_e_contracted_secondV+=1*einsum('a,al,al,ail->ai',mu_m_contr_m1m11,mum1,mum1,inp[:,:,1:])
    x7_e_contracted_secondV+=2*einsum('a,al,al,ail->ai',mu_m_contr_m1m11,mum1,mu1,inp[:,:,:-1])
    x7_e_contracted_secondV+=2*einsum('ak,ak,a,aik->ai',mum1,mu1,mu_m_contr_m1m11,inp[:,:,:-1])
    x7_e_contracted_secondV+=2*einsum('ai,ak,a,akk->ai',mu,mum1,mu_m_contr_m1m11,inp[:,:-1,1:])
    x7_e_contracted_secondV+=1*einsum('ai,ak,a,akk->ai',mu,mu1,mu_m_contr_m1m11,inp[:,:-1,:-1])
    x7_e_contracted_secondV+=1*einsum('ak,ak,a,aik->ai',mum1,mum1,mu_m_contr_m1m11,inp[:,:,1:])
    x7_e_contracted_secondV+=1*einsum('ai,ak,ak,ak->ai',mu,mum1,mum1,fourth_m1m11_temp[:,1:])
    x7_e_contracted_secondV+=2*einsum('ai,ak,ak,ak->ai',mu,mum1,mu1,fourth_m1m11_temp[:,:-1])
    x7_e_contracted_secondV+=4*einsum('ai,ak,al,akkll->ai',mu,mum1,mum1,fourth_expecs[:,:-1,1:,:-1,1:])
    x7_e_contracted_secondV+=2*einsum('ai,ak,al,akkll->ai',mu,mum1,mu1,fourth_expecs[:,:-1,1:,:-1,:-1])
    x7_e_contracted_secondV+=2*einsum('ai,ak,al,akkll->ai',mu,mu1,mum1,fourth_expecs[:,:-1,:-1,:-1,1:])
    x7_e_contracted_secondV+=1*einsum('ai,ak,al,akkll->ai',mu,mu1,mu1,fourth_expecs[:,:-1,:-1,:-1,:-1])
    x7_e_contracted_secondV+=1*einsum('ai,al,al,al->ai',mu,mum1,mum1,fourth_kkk_contr[:,1:])
    x7_e_contracted_secondV+=2*einsum('ai,al,al,al->ai',mu,mum1,mu1,fourth_kkk_contr[:,:-1])
    x7_e_contracted_secondV+=1*einsum('a,ai->ai',mu_m_contr_m1m11,fourth_m1m11_temp)
    x7_e_contracted_secondV+=2*einsum('ak,ak,aik->ai',mum1,mum1,mum1fe1m11temp)
    x7_e_contracted_secondV+=1*einsum('ak,ak,al,aikll->ai',mum1,mum1,mu1,fourth_expecs[:,:,1:,:-1,:-1])
    x7_e_contracted_secondV+=4*einsum('ak,ak,al,aikll->ai',mum1,mu1,mum1,fourth_expecs[:,:,:-1,:-1,1:])
    x7_e_contracted_secondV+=2*einsum('ak,ak,al,aikll->ai',mum1,mu1,mu1,fourth_expecs[:,:,:-1,:-1,:-1])
    x7_e_contracted_secondV+=2*einsum('ak,al,al,aikkl->ai',mum1,mum1,mum1,fourth_expecs[:,:,:-1,1:,1:])
    x7_e_contracted_secondV+=4*einsum('ak,al,al,aikkl->ai',mum1,mum1,mu1,fourth_expecs[:,:,:-1,1:,:-1])
    x7_e_contracted_secondV+=1*einsum('ak,al,al,aikkl->ai',mu1,mum1,mum1,fourth_expecs[:,:,:-1,:-1,1:])
    x7_e_contracted_secondV+=2*einsum('ak,al,al,aikkl->ai',mu1,mum1,mu1,fourth_expecs[:,:,:-1,:-1,:-1])
    x7_e_contracted_secondV+=1*einsum('a,ai->ai',mu_m_contr_m1m11,fourth_m1m11_temp)
    x7_e_contracted_secondV+=1*einsum('ai,akkk->ai',mu,sixth_m1m11_temp[:,:-1,:-1,1:])
    x7_e_contracted_secondV+=2*einsum('ak,aikk->ai',mum1,sixth_m1m11_temp[:,:,:-1,1:])
    x7_e_contracted_secondV+=1*einsum('ak,aikk->ai',mu1,sixth_m1m11_temp[:,:,:-1,:-1])
    x7_e_contracted_secondV+=2*einsum('al,aill->ai',mum1,sixth_m1m11kkk_contr[:,:,:-1,1:])
    x7_e_contracted_secondV+=1*einsum('al,aill->ai',mu1,sixth_m1m11kkk_contr[:,:,:-1,:-1])

    x7_e_contracted_thirdV =einsum('ai,a,a->ai',mu,mu_m_contr_111,mu_m_contr_m1m11)
    x7_e_contracted_thirdV+=2*einsum('ai,a,a->ai',mu,mu_m_contr_111,mum1_in_m11_contr)
    x7_e_contracted_thirdV+=1*einsum('ai,a,al,all->ai',mu,mu_m_contr_111,mu1,inp[:,:-1,:-1])
    x7_e_contracted_thirdV+=3*einsum('ai,ak,ak,al,al,akl->ai',mu,mu1,mu1,mum1,mum1,inp[:,1:,1:])
    x7_e_contracted_thirdV+=6*einsum('ai,ak,ak,al,al,akl->ai',mu,mu1,mu1,mum1,mu1,inp[:,1:,:-1])
    x7_e_contracted_thirdV+=1*einsum('a,al,al,ail->ai',mu_m_contr_111,mum1,mum1,inp[:,:,1:])
    x7_e_contracted_thirdV+=2*einsum('a,al,al,ail->ai',mu_m_contr_111,mum1,mu1,inp[:,:,:-1])
    x7_e_contracted_thirdV+=3*einsum('ak,ak,a,aik->ai',mu1,mu1,mu_m_contr_m1m11,inp[:,:,1:])
    x7_e_contracted_thirdV+=3*einsum('ai,ak,a,akk->ai',mu,mu1,mu_m_contr_m1m11,inp[:,1:,1:])
    x7_e_contracted_thirdV+=3*einsum('ai,ak,ak,ak->ai',mu,mu1,mu1,fourth_m1m11_temp[:,1:])
    x7_e_contracted_thirdV+=6*einsum('ai,ak,al,akkll->ai',mu,mu1,mum1,fourth_expecs[:,1:,1:,:-1,1:])
    x7_e_contracted_thirdV+=3*einsum('ai,ak,al,akkll->ai',mu,mu1,mu1,fourth_expecs[:,1:,1:,:-1,:-1])
    x7_e_contracted_thirdV+=1*einsum('ai,al,al,al->ai',mu,mum1,mum1,fe_lll_contr[:,1:])
    x7_e_contracted_thirdV+=2*einsum('ai,al,al,al->ai',mu,mum1,mu1,fe_lll_contr[:,:-1])
    x7_e_contracted_thirdV+=1*einsum('a,ai->ai',mu_m_contr_111,fourth_m1m11_temp)
    x7_e_contracted_thirdV+=6*einsum('ak,ak,aik->ai',mu1,mu1,mum1fe1m11temp)
    x7_e_contracted_thirdV+=3*einsum('ak,ak,al,aikll->ai',mu1,mu1,mu1,fourth_expecs[:,:,1:,:-1,:-1])
    x7_e_contracted_thirdV+=3*einsum('ak,al,al,aikkl->ai',mu1,mum1,mum1,fourth_expecs[:,:,1:,1:,1:])
    x7_e_contracted_thirdV+=6*einsum('ak,al,al,aikkl->ai',mu1,mum1,mu1,fourth_expecs[:,:,1:,1:,:-1])
    x7_e_contracted_thirdV+=1*einsum('a,ai->ai',mu_m_contr_m1m11,fourth_111_temp)
    x7_e_contracted_thirdV+=1*einsum('ai,akkk->ai',mu,sixth_m1m11_temp[:,1:,1:,1:])
    x7_e_contracted_thirdV+=3*einsum('ak,aikk->ai',mu1,sixth_m1m11_temp[:,:,1:,1:])
    x7_e_contracted_thirdV+=2*einsum('al,aill->ai',mum1,sixth_111kkk_contr[:,:,:-1,1:])
    x7_e_contracted_thirdV+=1*einsum('al,aill->ai',mu1,sixth_111kkk_contr[:,:,:-1,:-1])
    mum1mum1fe1_temp=einsum('am,am,ajklm->ajkl',mum1,mum1,fourth_expecs[:,:,:,:,1:])
    mum1mu1fem1_temp=einsum('am,am,ajklm->ajkl',mum1,mu1,fourth_expecs[:,:,:,:,:-1])

    t_x7_e_contracted_fourth=einsum('aj,ak,al,a->ajkl',mu,mu,mu,mu_m_contr_m1m11)
    t_x7_e_contracted_fourth+=2*einsum('aj,ak,al,a->ajkl',mu,mu,mu,mum1_in_m11_contr)
    t_x7_e_contracted_fourth+=1*einsum('aj,ak,al,am,amm->ajkl',mu,mu,mu,mu1,inp[:,:-1,:-1])
    t_x7_e_contracted_fourth+=2*einsum('aj,ak,am,am,alm->ajkl',mu,mu,mum1,mum1,inp[:,:,1:])
    t_x7_e_contracted_fourth+=4*einsum('aj,ak,am,am,alm->ajkl',mu,mu,mum1,mu1,inp[:,:,:-1])
    t_x7_e_contracted_fourth+=2*einsum('ak,al,am,am,ajm->ajkl',mu,mu,mum1,mum1,inp[:,:,1:])
    t_x7_e_contracted_fourth+=4*einsum('ak,al,am,am,ajm->ajkl',mu,mu,mum1,mu1,inp[:,:,:-1])
    
    t_x7_e_contracted_fourth+=1*einsum('aj,a,akl->ajkl',mu,mu_m_contr_m1m11,inp)
    t_x7_e_contracted_fourth+=4*einsum('al,a,ajk->ajkl',mu,mu_m_contr_m1m11,inp)
    t_x7_e_contracted_fourth+=2*einsum('aj,ak,al->ajkl',mu,mu,fourth_m1m11_temp)
    t_x7_e_contracted_fourth+=einsum('aj,akl->ajkl',mu,mum1_fem11contr*2+mu1_fem1m1contr)
    t_x7_e_contracted_fourth+=2*einsum('ak,ajl->ajkl',mu,mum1_fem11contr)
    t_x7_e_contracted_fourth+=1*einsum('ak,ajl->ajkl',mu,mu1_fem1m1contr)
    t_x7_e_contracted_fourth+=2*einsum('ajkl->ajkl',sixth_m1m11_temp)

    t_x7_e_contracted_fourth+=2*einsum('ajkl->ajkl',mum1mum1fe1_temp)
    t_x7_e_contracted_fourth+=4*einsum('ajkl->ajkl',mum1mu1fem1_temp)
    t_x7_e_contracted_fourth+=2*einsum('ak,al,aj->ajkl',mu,mu,fourth_m1m11_temp)
    t_x7_e_contracted_fourth+=6*einsum('al,ajk->ajkl',mu,mum1_fem11contr)
    t_x7_e_contracted_fourth+=3*einsum('al,ajk->ajkl',mu,mu1_fem1m1contr)
    x7_e_contracted_fourth=einsum("ai,ajkl->aijkl",mu,t_x7_e_contracted_fourth)
    'ai,'

    x7_e_contracted_fourth+=1*einsum('ak,al,a,aij->aijkl',mu,mu,mu_m_contr_m1m11,inp)

    x7_e_contracted_fourth+=einsum('ak,al,aij->aijkl',mu,mu,2*mum1_fem11contr+mu1_fem1m1contr)
    x7_e_contracted_fourth+=2*einsum('al,aijk->aijkl',mu,mum1mum1fe1_temp+2*mum1mu1fem1_temp)
    x7_e_contracted_fourth+=1*einsum('a,aijkl->aijkl',mu_m_contr_m1m11,fourth_expecs)
    x7_e_contracted_fourth+=2*einsum('ak,aijl->aijkl',mu,sixth_m1m11_temp)
    x7_e_contracted_fourth+=2*einsum('am,aijklmm->aijkl',mum1,sixth_expecs[:,:,:,:,:,:-1,1:])
    x7_e_contracted_fourth+=1*einsum('am,aijklmm->aijkl',mu1,sixth_expecs[:,:,:,:,:,:-1,:-1])


    t_x7_e_contracted_fifth=einsum('aj,ak,al,a->ajkl',mu,mu,mu,mu_m_contr_111)
    t_x7_e_contracted_fifth+=3*einsum('aj,ak,al,a->ajkl',mu,mu,mu,mu1_inp11_temp)
    t_x7_e_contracted_fifth+=3*einsum('aj,ak,al->ajkl',mu,mu,mu1mu1inp1_temp)
    t_x7_e_contracted_fifth+=3*einsum('aj,al,ak->ajkl',mu,mu,mu1mu1inp1_temp)
    t_x7_e_contracted_fifth+=6*einsum('ak,al,aj->ajkl',mu,mu,mu1mu1inp1_temp)
    t_x7_e_contracted_fifth+=1*einsum('aj,a,akl->ajkl',mu,mu_m_contr_111,inp)
    t_x7_e_contracted_fifth+=4*einsum('ak,a,ajl->ajkl',mu,mu_m_contr_111,inp)
    
    t_x7_e_contracted_fifth+=2*einsum('aj,ak,al->ajkl',mu,mu,fourth_111_temp)
    t_x7_e_contracted_fifth+=3*einsum('aj,akl->ajkl',mu,mu1_fe11contr)
    t_x7_e_contracted_fifth+=2*einsum('ak,al,aj->ajkl',mu,mu,fourth_111_temp)
    t_x7_e_contracted_fifth+=12*einsum('ak,ajl->ajkl',mu,mu1_fe11contr)
    
    t_x7_e_contracted_fifth+=6*mu1mu1fe1_temp+2*sixth_111_temp#einsum('ajkl->ajkl',)
    x7_e_contracted_fifth=einsum("ai,ajkl->aijkl",mu,t_x7_e_contracted_fifth)
    x7_e_contracted_fifth+=1*einsum('ak,al,a,aij->aijkl',mu,mu,mu_m_contr_111,inp)
    x7_e_contracted_fifth+=3*einsum('ak,al,aij->aijkl',mu,mu,mu1_fe11contr)
    x7_e_contracted_fifth+=einsum('ak,aijl->aijkl',mu,6*mu1mu1fe1_temp+2*sixth_111_temp)
    x7_e_contracted_fifth+=1*einsum('a,aijkl->aijkl',mu_m_contr_111,fourth_expecs)
    x7_e_contracted_fifth+=3*einsum('am,aijklmm->aijkl',mu1,sixth_expecs[:,:,:,:,:,1:,1:])
    c1=einsum("a,aij->aij",ovlps,contraction_1)
    c2=einsum("a,aij->aij",ovlps,contraction_2)
    c3=einsum("a,aij->aij",ovlps,contraction_3)
    c4=einsum("a,aij->aij",ovlps,x7_e_contracted_first)
    c5=einsum("a,aij->aij",ovlps,x7_e_contracted_second)
    c6=einsum("a,ai->ai",ovlps,x7_e_contracted_firstV)
    c7=einsum("a,ai->ai",ovlps,x7_e_contracted_secondV)
    c8=einsum("a,ai->ai",ovlps,x7_e_contracted_thirdV)
    c9=einsum("a,aijkl->aijkl",ovlps,x7_e_contracted_fourth)
    c10=einsum("a,aijkl->aijkl",ovlps,x7_e_contracted_fifth)
    return c1,c2,c3,c4,c5,c6,c7,c8,c9,c10
def get_x7_x8_contraction_allij_parallel(mu,matrix,ovlps):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_expecs=len(ovlps)
    mynum=int(num_expecs/size)
    myvals=[]
    for i in range(size):
        myvals.append(i*mynum)
    myvals.append(num_expecs+1)
    mymu=mu[myvals[rank]:myvals[rank+1]]
    myovlp=ovlps[myvals[rank]:myvals[rank+1]]
    mymatrices=matrix[myvals[rank]:myvals[rank+1]]
    myvals=list(get_x7_x8_contraction_allij(mymu,mymatrices,myovlp))
    if rank == 0:
        for src in range(1,size):
            rec=comm.recv(source=src, tag=11)
            for i in range(len(rec)):
                myvals[i]=np.concatenate((myvals[i],rec[i]))
        return myvals
    else:
        req=comm.send(myvals,dest=0,tag=11)
    
import time
#from calculate_efficient_tensors import *
if __name__=="__main__":
    ovlps=np.array([1.0])
    dim=6
    muvals=np.zeros((1,dim),dtype=np.complex128)#+np.random.rand(1,dim)
    matrices=np.array([np.eye(3)/np.sqrt(2)])
    matrices=np.random.rand(1,dim,dim)
    #dim=3
    #muvals=np.array(np.arange(maxnum*dim).reshape(maxnum,dim),dtype=np.complex128)
    #matrices=np.array(np.arange(maxnum*dim*dim).reshape(maxnum,dim,dim),dtype=np.complex128)
    #ovlps=np.array(np.arange(maxnum),dtype=np.complex128)
    x8_c1,x8_c2,x8_c3,x7_c1,x7_c2,x7_c1v,x7_c2v,x7_c3v,x7_c3,x7_c4=get_x7_x8_contraction_allij(muvals,matrices,ovlps)
    from calculate_efficient_tensors import *
    sintegrals=calculate_full_expectation_value(matrices[0],muvals[0],order=8,)
    x8_contraction1=einsum("kliiijjj->kl",sintegrals[:,:,1:,1:,1:,1:,1:,1:])
    x8_contraction2=einsum("kliiijjj->kl",sintegrals[:,:,:-1,:-1,1:,:-1,:-1,1:])
    x8_contraction3=einsum("kliiijjj->kl",sintegrals[:,:,1:,1:,1:,0:-1,0:-1,1:])
    sintegrals2=calculate_full_expectation_value(matrices[0],muvals[0],order=7,)
    x7_e_contracted_1=einsum("kliijjj->kl",sintegrals2[:,:,:,:,:-1,:-1,1:])
    x7_e_contracted_2=einsum("kliijjj->kl",sintegrals2[:,:,:,:,1:,1:,1:])
    x7_e_contracted_3=einsum("ikkklll->i",sintegrals2[:,1:,1:,1:,1:,1:,1:])
    x7_e_contracted_4=einsum("kiiijjj->k",sintegrals2[:,:-1,:-1,1:,:-1,:-1,1:])
    x7_e_contracted_5=einsum("kiiijjj->k",sintegrals2[:,1:,1:,1:,0:-1,0:-1,1:])
    x7_e_contracted_6=einsum("ijklmmm->ijkl",sintegrals2[:,:,:,:,0:-1,0:-1,1:])
    x7_e_contracted_7=einsum("ijklmmm->ijkl",sintegrals2[:,:,:,:,1:,1:,1:])
    print("x8")
    print(x8_c1-x8_contraction1)
    print(x8_c2-x8_contraction2)
    print(x8_c3-x8_contraction3)
    print("x7 matrices")
    print(x7_c1-x7_e_contracted_1)
    print(x7_c2-x7_e_contracted_2)
    print("x7 4-tensor")
    print(x7_c3-x7_e_contracted_6)
    print(x7_c4-x7_e_contracted_7)
    print("x7 vectros")
    print(x7_c1v-x7_e_contracted_3)
    print(x7_c2v-x7_e_contracted_4)
    print(x7_c3v-x7_e_contracted_5)
    #print(x8_c1-x8_contraction1)
    #print(x8_c1-x8_contraction1)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    maxnum=nmat=200*200//2
    dim=3
    muvals=np.array(np.arange(maxnum*dim).reshape(maxnum,dim),dtype=np.complex128)
    matrices=np.array(np.arange(maxnum*dim*dim).reshape(maxnum,dim,dim),dtype=np.complex128)
    ovlps=np.array(np.arange(maxnum),dtype=np.complex128)
    
    if rank==0:
        start=time.time()
        final1=calculate_all_expectation_values_full_allij(muvals,matrices,ovlps)
        #final=get_x7_x8_contraction_allij(muvals,matrices,ovlps)
        end=time.time()
        print(end-start)
    sys.exit(1)
    print("derp")
    start=time.time()
    retval=get_x7_x8_contraction_allij_parallel(muvals,matrices,ovlps)
    end=time.time()
    if rank==0:
        print(end-start)
        print(np.sum(final[5]-retval[5]))
    """