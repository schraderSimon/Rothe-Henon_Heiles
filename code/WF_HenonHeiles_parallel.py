import warnings

warnings.filterwarnings("ignore", message="Derivative was zero.")  # Ignore error message
import itertools as it

import numpy as np
from mpi4py import MPI
from numpy import conj, einsum, imag, pi, real, sqrt
from numpy.linalg import eigvals
from scipy.linalg import sqrtm

np.set_printoptions(linewidth=300, precision=5)

from calculate_all_integrals_at_once import (
    calculate_all_expectation_values_full_allij,
    get_x7_x8_contraction_allij,
)
from numpy.linalg import det
from utils import solve

dx = 1e-7


def make_sum_matrices(matrices, sum_matrices, sum_matrices_inv):
    num_gauss = len(matrices)
    # sum_matrices=einsum()-
    for i in range(num_gauss):
        for j in range(0, i + 1):
            sum_matrix = conj(matrices[i]) + matrices[j]
            sum_matrices[i, j, :, :] = sum_matrix
            sum_matrices[j, i, :, :] = conj(sum_matrix)
    inverts = np.linalg.inv(sum_matrices)
    sum_matrices_inv = inverts
    return sum_matrices, sum_matrices_inv


def update_sum_matrices(matrices, pairs_of_indices, sum_matrices, sum_matrices_inv):
    poi = pairs_of_indices
    sum_matrix_temp_list = []
    for pair in poi:
        i = pair[0]
        j = pair[1]
        sum_matrix = conj(matrices[i]) + matrices[j]
        sum_matrix_temp_list.append(sum_matrix)
    inverts = np.linalg.inv(sum_matrix_temp_list)
    for i in range(len(poi)):
        sum_matrices[poi[i][0], poi[i][1], :, :] = sum_matrix_temp_list[i]
        sum_matrices[poi[i][1], poi[i][0], :, :] = conj(sum_matrix_temp_list[i])
        sum_matrices_inv[poi[i][0], poi[i][1], :, :] = inverts[i]
        sum_matrices_inv[poi[i][1], poi[i][0], :, :] = conj(inverts[i])
    return sum_matrices, sum_matrices_inv


class WF:
    def __init__(
        self,
        nonlin_params,
        lin_params,
        lambda_=0.1,
        normalization=1,
        calculate_Gradient=True,
        h=1e-3,
        onlyX1X2=False,
    ):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        rank = self.rank
        self.onlyX1X2 = onlyX1X2
        self.update = False
        self.jvc = None
        self.ivc = None
        self.jvnc = None
        self.ivnc = None
        self.ic = None
        self.inc = None
        self.indices_j_vc = None
        self.indices_j_vnc = None
        """
        nonlin_params: list of list or 2D array all parameters. The general form is (ngauss,n(n+3))), where n is the dimensionality of the problem
        lin_params: list of same length as nonlin_params
        lambda: Strength of the Henon-Heiles interaction
        normalization: The normalization of the WF.
        """
        self.num_gaussians = len(lin_params)
        self.nd = self.num_dimensions = np.rint(
            0.5 * (-3 + sqrt(4 * len(nonlin_params[0]) + 9))
        ).astype(
            int
        )  # This is the invert formula of nparams = n*(n+3) for the number of dimensions
        self.nonlin_params = np.asarray(nonlin_params)
        self.lin_params = lin_params
        self.matrices = None
        self.vectors = None
        self.lambda_ = lambda_
        self.lambda_sq = lambda_**2
        self.normalization = normalization
        self.calculate_Gradient = calculate_Gradient
        self.h = h
        self.intermediateshavebeenupdated = False
        self.setupintermediates()
        self.setupmyintermediates()
        self.overlap = self.calculate_myoverlapmatrix(self.myrow_indices, self.mycol_indices)
        self.comm.Allreduce(MPI.IN_PLACE, [self.overlap, MPI.COMPLEX16], op=MPI.SUM)
        myovlp = self.myovlp
        myvals = self.myvals

        self.root = None

        # First, get the lower triangular indices
        tril_indices = np.tril_indices(self.num_gaussians)
        i_indices, j_indices = tril_indices
        counter = len(i_indices)

        # Access the relevant elements using advanced indexing for cross_mat_inv_list
        cross_mat_inv_list = np.zeros(
            (counter,) + self.sum_matrices_inv.shape[2:], dtype=np.complex128
        )
        mu_list = np.zeros((counter, len(self.vectors[0])), dtype=np.complex128)

        for idx, (i, j) in enumerate(zip(i_indices, j_indices)):
            cross_mat_inv_list[idx] = self.sum_matrices_inv[i, j]
            bvec = self.vectors[j].T @ self.matrices[j] + np.conjugate(
                self.vectors[i].T @ self.matrices[i]
            )
            mu_list[idx] = cross_mat_inv_list[idx] @ bvec

        mymu = mu_list[myvals[rank] : myvals[rank + 1]]
        mymatrices = cross_mat_inv_list[myvals[rank] : myvals[rank + 1]]
        # Screen matrices for small values
        threshold = -1
        myovlp_of_interest = myovlp
        mymu_of_interest = mymu
        mymatrices_of_interest = mymatrices
        myTotnumOfElements = len(myovlp)
        if self.h >= 0:  # Only negative h will lead to this not being calculated
            if self.onlyX1X2:
                x1_temp, x2_temp = calculate_all_expectation_values_full_allij(
                    mymu_of_interest, mymatrices_of_interest, myovlp_of_interest, order=2
                )
                self.x1_temp = np.empty((myTotnumOfElements, self.nd), dtype=np.complex128)
                self.x1_temp = x1_temp
                self.x2_temp = np.empty(
                    (myTotnumOfElements, self.nd, self.nd), dtype=np.complex128
                )
                self.x2_temp = x2_temp

            else:
                x1_temp, x2_temp, x3_temp, x4_temp, x5_temp, x6_temp = (
                    calculate_all_expectation_values_full_allij(
                        mymu_of_interest, mymatrices_of_interest, myovlp_of_interest
                    )
                )
                self.x1_temp = np.empty((myTotnumOfElements, self.nd), dtype=np.complex128)
                self.x1_temp = x1_temp
                self.x2_temp = np.empty(
                    (myTotnumOfElements, self.nd, self.nd), dtype=np.complex128
                )
                self.x2_temp = x2_temp
                self.x3_temp = np.empty(
                    (myTotnumOfElements, *([self.nd] * 3)), dtype=np.complex128
                )
                self.x3_temp = x3_temp
                self.x4_temp = np.empty(
                    (myTotnumOfElements, *([self.nd] * 4)), dtype=np.complex128
                )
                self.x4_temp = x4_temp
                self.x5_temp = np.empty(
                    (myTotnumOfElements, *([self.nd] * 5)), dtype=np.complex128
                )
                self.x5_temp = x5_temp
                self.x6_temp = np.empty(
                    (myTotnumOfElements, *([self.nd] * 6)), dtype=np.complex128
                )
                self.x6_temp = x6_temp
            if self.calculate_Gradient and not self.onlyX1X2:
                x8_c1, x8_c2, x8_c3, x7_c1, x7_c2, x7_c1v, x7_c2v, x7_c3v, x7_c3, x7_c4 = (
                    get_x7_x8_contraction_allij(mymu, mymatrices, myovlp)
                )
                self.x8_c1_temp = x8_c1
                self.x8_c2_temp = x8_c2
                self.x8_c3_temp = x8_c3
                self.x7_c1_temp = x7_c1
                self.x7_c2_temp = x7_c2
                self.x7_c3_temp = x7_c3
                self.x7_c4_temp = x7_c4
                self.x7_c1v_temp = x7_c1v
                self.x7_c2v_temp = x7_c2v
                self.x7_c3v_temp = x7_c3v
        self.setupMatrices()
        self.fodmd = self.calculate_overlap_matrix_log_deriv()

        # Pack all Hamiltonian matrix elements into a single array
        # All of these are not used if onlyX1X2 is True
        if not self.onlyX1X2:
            big_array = np.concatenate(
                (
                    self.kinetic.flatten(),
                    self.potential.flatten(),
                    self.kinetic_energy_squared.flatten(),
                    self.potential_squared.flatten(),
                    self.potential_times_kinetic.flatten(),
                    self.H.flatten(),
                    self.Hsquared.flatten(),
                )
            )
            # Sum all Hamiltonian matrix elements across all processes
            self.comm.Allreduce(MPI.IN_PLACE, [big_array, MPI.COMPLEX16], op=MPI.SUM)
            # Unpack the big array into the individual Hamiltonian matrix elements
            offset = 0
            length = np.prod(self.kinetic.shape)
            for original_array in (
                self.kinetic,
                self.potential,
                self.kinetic_energy_squared,
                self.potential_squared,
                self.potential_times_kinetic,
                self.H,
                self.Hsquared,
            ):
                original_array[:] = big_array[offset : offset + length].reshape(
                    original_array.shape
                )
                offset += length
            self.mykinetic = self.kinetic[self.myrow_indices, self.mycol_indices]
            self.mypotential = self.potential[self.myrow_indices, self.mycol_indices]
            self.mypotential_squared = self.potential_squared[
                self.myrow_indices, self.mycol_indices
            ]
            self.mykinetic_energy_squared = self.kinetic_energy_squared[
                self.myrow_indices, self.mycol_indices
            ]
            self.mypotential_times_kinetic = self.potential_times_kinetic[
                self.myrow_indices, self.mycol_indices
            ]
            self.myH = self.H[self.myrow_indices, self.mycol_indices]
            self.myHsquared = self.Hsquared[self.myrow_indices, self.mycol_indices]

    def update_overlap_and_overlap_derivs(
        self, indices_new, new_nonlin_params, new_lin_params=None
    ):

        ng = self.num_gaussians
        nd = self.num_dimensions
        # This function updates the overlap matrix and the relevant elements for the overlap matrix derivatives, given a new set of parameters.
        # The indices_new are the indices of the new parameters in the old parameter list.
        # The new_nonlin_params are the new non-linear parameters.
        # The new_lin_params are the new linear parameters.
        # If new_lin_params is None, the old linear parameters are used (or this might simply not be required)
        full_indices = np.arange(self.num_gaussians)
        seen_pairs = set()
        indices_to_update = []
        for i, j in it.product(full_indices, indices_new):
            if (i, j) not in seen_pairs and (j, i) not in seen_pairs:
                if i >= j:
                    indices_to_update.append((i, j))
                else:
                    indices_to_update.append((j, i))
                seen_pairs.add((i, j))
                seen_pairs.add((j, i))

        def get_indices(data_length, rank, size):
            """Calculate start and end indices for the given rank."""
            k, m = divmod(data_length, size)
            start = rank * k + min(rank, m)
            end = start + k + (1 if rank < m else 0)
            return start, end

        start_idx, end_idx = get_indices(len(indices_to_update), self.rank, self.size)
        self.myindices = indices_to_update[start_idx:end_idx]
        self.mynewrow_indices = np.array(self.myindices)[:, 0]
        self.mynewcol_indices = np.array(self.myindices)[:, 1]
        # Next step: Find the sum_matrix and mu for the new parameters

        self.updateintermediates(indices_new[0], indices_new[-1], new_nonlin_params)
        update_sum_matrices(
            self.matrices, indices_to_update, self.sum_matrices, self.sum_matrices_inv
        )  # This updates ALL sum matrices, not just the ones of this rank

        mynew_overlap_matrix = self.calculate_myoverlapmatrix(
            self.mynewrow_indices, self.mynewcol_indices
        )
        self.comm.Allreduce(MPI.IN_PLACE, [mynew_overlap_matrix, MPI.COMPLEX16], op=MPI.SUM)
        self.overlap[np.ix_(np.array(indices_new), np.array(full_indices))] = 0
        self.overlap[np.ix_(np.array(full_indices), np.array(indices_new))] = 0
        self.overlap += mynew_overlap_matrix
        myovlp_list = self.overlap[self.mynewrow_indices, self.mynewcol_indices]

        selected_vectors = self.vectors[full_indices]
        selected_matrices = self.matrices[full_indices]
        prods = np.einsum("ij,ijk->ik", selected_vectors, selected_matrices)
        mycross_mat_inv_list = self.sum_matrices_inv[
            self.mynewrow_indices, self.mynewcol_indices
        ]
        bvecs_list = prods[self.mynewcol_indices] + np.conjugate(prods[self.mynewrow_indices])
        mymu_list = np.einsum("ijk,ik->ij", mycross_mat_inv_list, bvecs_list)
        # Next step: From the new sum matrices, calculate the new overlap matrix and the new overlap matrix derivatives

        new_ij_indices = list(zip(self.mynewrow_indices, self.mynewcol_indices))
        # Next step: Update the x1 and x2 values
        """This step only needs to be done one single time if the indices are not changed. Running it every time is unnecessarily| costly"""
        if not self.intermediateshavebeenupdated:
            self.updatemyderivativeintermediates(
                indices_new[0], new_ij_indices
            )  # Definitely the most time consuming step!!
            self.intermediateshavebeenupdated = True
        self.new_x1_temp, self.new_x2_temp = calculate_all_expectation_values_full_allij(
            mymu_list, mycross_mat_inv_list, myovlp_list, order=2
        )
        self.update_overlap_matrix_log_deriv(indices_new)

    def calculate_gaussian_distances_from_origo(self):
        ng = self.num_gaussians
        centers = []
        for i in range(ng):
            mu = real(self.vectors[i])
            q = imag(self.vectors[i])
            A = real(self.matrices[i, :, :])
            B = imag(self.matrices[i, :, :])
            z = ((-q.T @ B + mu.T @ A) @ np.linalg.inv(A)).T
            centers.append(sqrt(z.T @ z))
        return centers

    def setupintermediates(self):
        nd = self.num_dimensions
        ng = self.num_gaussians
        self.L_matrices = np.zeros((ng, nd, nd))
        self.K_matrices = np.zeros((ng, nd, nd))
        self.Iab = np.zeros((nd, nd, nd, nd))
        self.IabLj = np.zeros(
            (ng, nd, nd, nd, nd), dtype=np.complex128
        )  # Here, self.IabLj[j,a,b,:,:] #Gives me the IabLj matrix for a given j, a, b
        self.Kab = np.zeros(
            (ng, nd, nd, nd, nd)
        )  # Here, self.Kab[j,a,b,:,:] #Gives me the Kab matrix for a given a,b

        for a in range(nd):
            for b in range(a + 1):
                self.Iab[a, b, a, b] = 1
                self.Kab[:, a, b, a, b] += 1  # ;self.Kab[:,a,b,b,a]+=1
        self.Kab += +np.swapaxes(self.Kab, -1, -2)
        mytril_indices = np.tril_indices(nd)
        for i in range(ng):
            self.L_matrices[i, :, :][mytril_indices] = self.nonlin_params[i][
                : int(0.5 * nd * (nd + 1))
            ]
            self.K_matrices[i, :, :][mytril_indices] = self.nonlin_params[i][
                int(0.5 * nd * (nd + 1)) : (nd * (nd + 1))
            ]
        self.vectors = (
            self.nonlin_params[:, (nd * (nd + 1)) : (nd * (nd + 1)) + nd]
            + 1j * self.nonlin_params[:, nd * (nd + 1) + nd :]
        )
        self.matrices = einsum("ijk,ilk->ijl", self.L_matrices, self.L_matrices) - 1j
        self.matrices += 1j
        self.matrices += 1j * (self.K_matrices + einsum("ijk->ikj", self.K_matrices))
        self.matrices_squared = einsum("ikx,ixl->ikl", self.matrices, self.matrices)
        self.matrices_squared_conj = conj(self.matrices_squared)
        self.vmmips = einsum("ia,iab,ib->i", self.vectors, self.matrices_squared, self.vectors)
        self.cjs = einsum("jii->j", self.matrices) - 2 * self.vmmips
        self.cis = np.conj(self.cjs)
        self.omega_T = einsum("ik,ikl->il", self.vectors, self.matrices_squared)
        self.rho_T_conjs = conj(self.omega_T)

        self.IabLj = einsum("abkl,jml->jabkm", self.Iab, self.L_matrices)
        self.IabLj = self.IabLj + np.swapaxes(self.IabLj, -1, -2)
        self.Kabmuj = einsum("jablk,jk->jabl", self.Kab, self.vectors)
        self.IabLjmuj = einsum("jablk,jk->jabl", self.IabLj, self.vectors)
        self.mKm = einsum("jk,jabk->jab", self.vectors, self.Kabmuj)
        self.mLm = einsum("jk,jabk->jab", self.vectors, self.IabLjmuj)
        IabLj_list = []
        IabLjmuj_list = []
        Kabmuj_list = []
        Kab_list = []
        self.ab_iter = 0

        for a in range(nd):
            for b in range(a + 1):
                self.ab_iter += 1
                IabLj_list.append(self.IabLj[:, a, b])
                IabLjmuj_list.append(self.IabLjmuj[:, a, b])
                Kabmuj_list.append(self.Kabmuj[:, a, b])
                Kab_list.append(self.Kab[:, a, b])
        self.Kab_list = np.array(Kab_list)
        self.Kabmuj_list = np.array(Kabmuj_list)

        self.IabLj_list = np.array(IabLj_list)
        # self.IabLj_list=self.IabLj.reshape()
        self.IabLjmuj_list = np.array(IabLjmuj_list)

        self.sum_matrices_inv = np.zeros((ng, ng, nd, nd), dtype=np.complex128)
        self.sum_matrices, self.sum_matrices_inv = make_sum_matrices(
            self.matrices, np.zeros((ng, ng, nd, nd), dtype=np.complex128), None
        )  # Those matrices are changed in-place
        # self.overlap[abs(self.overlap)<1e-5]=0
        self.coefficients = np.asarray(self.lin_params)

    def updateintermediates(self, ind_min, ind_max, new_nonlin_params):
        self.update = True
        nd = self.num_dimensions
        ng_full = self.num_gaussians
        ng_new = (
            ind_max - ind_min + 1
        )  # The number of new Gaussians. The +1 is because the range is inclusive.
        new_indices = np.arange(ind_min, ind_max + 1)
        mytril_indices = np.tril_indices(nd)
        # Update L and K matrices at the new indices
        for i in range(ng_new):
            self.L_matrices[ind_min + i, :, :][mytril_indices] = new_nonlin_params[i][
                : int(0.5 * nd * (nd + 1))
            ]
            self.K_matrices[ind_min + i, :, :][mytril_indices] = new_nonlin_params[i][
                int(0.5 * nd * (nd + 1)) : (nd * (nd + 1))
            ]
        self.vectors[ind_min : ind_max + 1] = (
            new_nonlin_params[:, (nd * (nd + 1)) : (nd * (nd + 1)) + nd]
            + 1j * new_nonlin_params[:, nd * (nd + 1) + nd :]
        )
        self.matrices[ind_min : ind_max + 1] = einsum(
            "ijk,ilk->ijl",
            self.L_matrices[ind_min : ind_max + 1],
            self.L_matrices[ind_min : ind_max + 1],
        )
        self.matrices[ind_min : ind_max + 1] += 1j * (
            self.K_matrices[ind_min : ind_max + 1]
            + einsum("ijk->ikj", self.K_matrices[ind_min : ind_max + 1])
        )
        self.matrices_squared[ind_min : ind_max + 1] = einsum(
            "ikx,ixl->ikl",
            self.matrices[ind_min : ind_max + 1],
            self.matrices[ind_min : ind_max + 1],
        )
        self.matrices_squared_conj[ind_min : ind_max + 1] = conj(
            self.matrices_squared[ind_min : ind_max + 1]
        )
        self.vmmips[ind_min : ind_max + 1] = einsum(
            "ia,iab,ib->i",
            self.vectors[ind_min : ind_max + 1],
            self.matrices_squared[ind_min : ind_max + 1],
            self.vectors[ind_min : ind_max + 1],
        )
        self.cjs[ind_min : ind_max + 1] = (
            einsum("jii->j", self.matrices[ind_min : ind_max + 1])
            - 2 * self.vmmips[ind_min : ind_max + 1]
        )
        self.cis[ind_min : ind_max + 1] = np.conj(self.cjs[ind_min : ind_max + 1])
        self.omega_T[ind_min : ind_max + 1] = einsum(
            "ik,ikl->il",
            self.vectors[ind_min : ind_max + 1],
            self.matrices_squared[ind_min : ind_max + 1],
        )
        self.rho_T_conjs[ind_min : ind_max + 1] = conj(self.omega_T[ind_min : ind_max + 1])
        self.IabLj[ind_min : ind_max + 1] = einsum(
            "abkl,jml->jabkm", self.Iab, self.L_matrices[ind_min : ind_max + 1]
        )
        self.IabLj[ind_min : ind_max + 1] = self.IabLj[ind_min : ind_max + 1] + np.swapaxes(
            self.IabLj[ind_min : ind_max + 1], -1, -2
        )
        self.Kabmuj[ind_min : ind_max + 1] = einsum(
            "jablk,jk->jabl",
            self.Kab[ind_min : ind_max + 1],
            self.vectors[ind_min : ind_max + 1],
        )
        self.IabLjmuj[ind_min : ind_max + 1] = einsum(
            "jablk,jk->jabl",
            self.IabLj[ind_min : ind_max + 1],
            self.vectors[ind_min : ind_max + 1],
        )
        self.mKm[ind_min : ind_max + 1] = einsum(
            "jk,jabk->jab",
            self.vectors[ind_min : ind_max + 1],
            self.Kabmuj[ind_min : ind_max + 1],
        )
        self.mLm[ind_min : ind_max + 1] = einsum(
            "jk,jabk->jab",
            self.vectors[ind_min : ind_max + 1],
            self.IabLjmuj[ind_min : ind_max + 1],
        )
        # I think those are relatively cheap to calculate up to here.

        IabLj_list = []
        IabLjmuj_list = []
        Kabmuj_list = []
        Kab_list = []
        self.ab_iter = 0
        for a in range(nd):
            for b in range(a + 1):
                IabLj_list.append(self.IabLj[ind_min : ind_max + 1, a, b])
                IabLjmuj_list.append(self.IabLjmuj[ind_min : ind_max + 1, a, b])
                Kabmuj_list.append(self.Kabmuj[ind_min : ind_max + 1, a, b])
                Kab_list.append(self.Kab[ind_min : ind_max + 1, a, b])
                self.ab_iter += 1
        self.Kab_list[:, ind_min : ind_max + 1] = np.array(Kab_list)
        self.Kabmuj_list[:, ind_min : ind_max + 1] = np.array(Kabmuj_list)
        self.IabLj_list[:, ind_min : ind_max + 1] = np.array(IabLj_list)
        self.IabLjmuj_list[:, ind_min : ind_max + 1] = np.array(IabLjmuj_list)

    def setupmyintermediates(self):
        size = self.size
        rank = self.rank
        ng = self.num_gaussians
        nd = self.num_dimensions
        self.mynum = mynum = int(ng * (ng + 1) // 2 / size)
        self.myvals = []
        for i in range(size):
            self.myvals.append(i * mynum)
        self.myvals.append(ng * (ng + 1) // 2 + 1)
        myvals = self.myvals
        self.ij_zero_indices = (
            []
        )  # The indices at which matrix elements are not calculated, i.e. are set to zero
        self.ij_indices = (
            []
        )  # The indices at which matrix elements are calculated, i.e. are not zero
        counter = 0
        for i in range(self.num_gaussians):
            for j in range(0, i + 1):
                if counter < myvals[rank + 1] and counter >= myvals[rank]:
                    counter += 1
                    self.ij_indices.append([i, j])
                else:
                    counter += 1
                    self.ij_zero_indices.append([i, j])
                    self.ij_zero_indices.append([j, i])
        self.cj_of_interest = []
        self.j_linear_of_interest = []
        self.matrices_squared_of_interest = []
        j_linear = 4 * einsum("ji,jik->jk", self.vectors, self.matrices_squared)
        self.omega_T_of_interest = []
        self.rho_T_conj_of_interest = []
        self.matrices_j_squared_of_interest = []
        self.matrices_i_squared_of_interest = []
        self.ci_of_interest = []
        self.IabLjmuj_of_interest = []
        for i, j in self.ij_indices:
            self.j_linear_of_interest.append(j_linear[j, :])
            self.cj_of_interest.append(self.cjs[j])
            self.ci_of_interest.append(self.cis[i])
            self.matrices_squared_of_interest.append(self.matrices_squared[j, :, :])
            self.omega_T_of_interest.append(self.omega_T[j, :])
            self.rho_T_conj_of_interest.append(np.conj(self.omega_T[i, :]))
            self.matrices_j_squared_of_interest.append(self.matrices_squared[j, :, :])
            self.matrices_i_squared_of_interest.append(np.conj(self.matrices_squared[i, :, :]))
            counter += 1

        self.myrow_indices = np.array(self.ij_indices)[:, 0]  # Extract row indices
        self.mycol_indices = np.array(self.ij_indices)[:, 1]  # Extract column indices

    def calculate_overlap_matrix_log_deriv(self):
        # This calculates a matrix, where the element D[i,n] is the derivative \frac{\partial}{\partial n_i}log(<g_i|g_i>), where $n_i$ is one of the
        # n(n+3) parameters specifying the Gaussian. This is something which I arguably want to have access to in all parallel processes.
        ng = self.num_gaussians
        nd = self.num_dimensions
        D_tensor = np.zeros((ng, nd * (nd + 3)), dtype=float)
        As = real(self.matrices)
        Bs = imag(self.matrices)
        Ainvs = np.linalg.inv(As)
        for i in range(ng):
            A = As[i]
            Ainv = Ainvs[i]
            L = self.L_matrices[i]
            B = Bs[i]
            q = imag(self.vectors[i])
            counter = 0
            BAinv = B @ Ainv
            AinvB = Ainv @ B
            for a in range(nd):
                for b in range(a + 1):
                    Aderiv = real(self.IabLj[i, a, b])
                    if a == b:
                        D_tensor[i, counter] += -1 / L[a, a]
                    D_tensor[i, counter] += 2 * q.T @ (-BAinv @ Aderiv @ AinvB + Aderiv) @ q
                    Bderiv = real(self.Kab[i, a, b])
                    D_tensor[i, counter + (nd * (nd + 1)) // 2] = (
                        2 * q.T @ (BAinv @ Bderiv + Bderiv @ AinvB) @ q
                    )
                    counter += 1
            D_tensor[i, nd * (nd + 2) :] = 4 * (BAinv @ B + A) @ q
        return D_tensor

    def update_overlap_matrix_log_deriv(self, update_indices):
        ng = self.num_gaussians
        nd = self.num_dimensions
        D_tensor = np.zeros((len([update_indices]), nd * (nd + 3)), dtype=float)
        As = real(self.matrices[update_indices])
        Bs = imag(self.matrices[update_indices])
        Ainvs = np.linalg.inv(As)
        for i in range(len([update_indices])):
            A = As[i]
            Ainv = Ainvs[i]
            L = self.L_matrices[i]
            B = Bs[i]
            q = imag(self.vectors[update_indices][i])
            counter = 0
            BAinv = B @ Ainv
            AinvB = Ainv @ B
            for a in range(nd):
                for b in range(a + 1):
                    Aderiv = real(self.IabLj[update_indices[i], a, b])
                    if a == b:
                        D_tensor[i, counter] += -1 / L[a, a]
                    D_tensor[i, counter] += 2 * q.T @ (-BAinv @ Aderiv @ AinvB + Aderiv) @ q
                    Bderiv = real(self.Kab[update_indices[i], a, b])
                    D_tensor[i, counter + (nd * (nd + 1)) // 2] = (
                        2 * q.T @ (BAinv @ Bderiv + Bderiv @ AinvB) @ q
                    )
                    counter += 1
            D_tensor[i, nd * (nd + 2) :] = 4 * (BAinv @ B + A) @ q
        self.fodmd[update_indices] = D_tensor

    def calculate_overlap_matrix_diagonals(self):
        bvecs_list = einsum("ak,akl->al", self.vectors, self.matrices)
        bvec_calc = 2 * np.real(bvecs_list)
        vmips = einsum("ia,iab,ib->i", self.vectors, self.matrices, self.vectors)
        diagonal_sum_matrices_inv = np.diagonal(
            self.sum_matrices_inv, axis1=0, axis2=1
        ).transpose([2, 0, 1])
        diagonal_exponents = (
            einsum("ak,akl,al->a", bvec_calc, diagonal_sum_matrices_inv, bvec_calc)
            - vmips
            - vmips.conj()
        )
        diagonal_sum_matrix = np.diagonal(self.sum_matrices, axis1=0, axis2=1).transpose(
            [2, 0, 1]
        )  # Same as einsum('iijk->ijk', self.sum_matrices)
        diagonal_determinants = sqrt(det(diagonal_sum_matrix))
        return bvecs_list, vmips, diagonal_determinants, diagonal_exponents

    def calculate_myoverlapmatrix(self, row_indices, col_indices):

        bvecs_list, vmips, diagonal_determinants, diagonal_exponents = (
            self.calculate_overlap_matrix_diagonals()
        )
        bvec_outer = np.conj(bvecs_list)[:, np.newaxis, :] + (bvecs_list)[np.newaxis, :, :]
        vmips_outer = np.conj(vmips[:, np.newaxis]) + vmips[np.newaxis, :]
        diagonal_exponents_outer = (
            diagonal_exponents[:, np.newaxis] + diagonal_exponents[np.newaxis, :]
        )
        mybvecouter = bvec_outer[row_indices, col_indices, :]
        myvmipsouter = vmips_outer[row_indices, col_indices]
        mydiagonalexponentsouter = diagonal_exponents_outer[row_indices, col_indices]
        mysummatricesinv = self.sum_matrices_inv[row_indices, col_indices, :, :]
        mysqrt_eigvals_inv = sqrt(
            eigvals(mysummatricesinv)
        )  # Principal square root of eigenvalues
        myinv_dets = np.prod(mysqrt_eigvals_inv, axis=1)
        mysqrt_diagonal_dets = sqrt(diagonal_determinants)
        mydiagonal_dets_prod = einsum("i,j->ij", mysqrt_diagonal_dets, mysqrt_diagonal_dets)[
            row_indices, col_indices
        ]
        myoverlap = einsum("x,x->x", myinv_dets, mydiagonal_dets_prod)
        myexponents = (
            einsum("xa,xab,xb->x", mybvecouter, mysummatricesinv, mybvecouter)
            - myvmipsouter
            - 0.5 * mydiagonalexponentsouter
        )
        myoverlap *= np.exp(myexponents)
        self.myovlp = myoverlap
        self.overlap2 = np.zeros((self.num_gaussians, self.num_gaussians), dtype=np.complex128)
        self.overlap2[row_indices, col_indices] = myoverlap
        self.overlap2[col_indices, row_indices] = np.conj(myoverlap)
        return self.overlap2

    def calculate_overlap_matrix(self):
        """This is from a *very old* version of the program"""
        self.overlap_matrix = np.empty(
            (self.num_gaussians, self.num_gaussians), dtype=np.complex128
        )
        vmips = np.empty(
            (self.num_gaussians), dtype=np.complex128
        )  # VMIP stands for "vector matrix inner product"
        for i in range(self.num_gaussians):
            vmips[i] = self.vectors[i].T @ self.matrices[i] @ self.vectors[i]
        for i in range(self.num_gaussians):
            for j in range(i, self.num_gaussians):
                sum_matrix = self.sum_matrices[i, j, :, :]
                sum_matrix_inv = self.sum_matrices_inv[i, j, :, :]
                squareroot = np.asarray(sqrtm(sum_matrix), dtype=np.complex128)
                determinant = det(squareroot)
                self.overlap_matrix[i, j] = 1 / determinant
                bvec = self.vectors[j].T @ self.matrices[j] + np.conj(
                    self.vectors[i].T @ self.matrices[i]
                )
                exponent = +bvec.T @ sum_matrix_inv @ bvec - vmips[j] - np.conj(vmips[i])
                self.overlap_matrix[i, j] *= np.exp(exponent)
                self.overlap_matrix[j, i] = np.conj(self.overlap_matrix[i, j])
        self.overlap_matrix *= pi ** (self.num_dimensions / 2)
        return self.overlap_matrix

    def setupmyderivintermediates(self, start_index_WFnew):
        j_values_noconj = []
        i_values_noconj = []
        indices_conj = []
        j_values_conj = []
        i_values_conj = []
        indices_noconj = []
        ngMs = start_index_WFnew
        ng = self.num_gaussians
        nd = self.num_dimensions
        for i in range(ng):
            for j in range(ngMs, ng):
                if [j, i] in self.ij_indices:
                    index = self.ij_indices.index([j, i])
                    j_values_conj.append(j)
                    i_values_conj.append(i)
                    indices_conj.append(index)
                elif [i, j] in self.ij_indices:
                    pass
                    index = self.ij_indices.index([i, j])
                    j_values_noconj.append(j)
                    i_values_noconj.append(i)
                    indices_noconj.append(index)

        self.jvc = np.array(j_values_conj)
        self.ivc = np.array(i_values_conj)
        self.jvnc = np.array(j_values_noconj)
        self.ivnc = np.array(i_values_noconj)
        self.ic = np.array(indices_conj)
        self.inc = np.array(indices_noconj)
        self.indices_j_vc = ngMs + self.jvc - ng
        self.indices_j_vnc = ngMs + self.jvnc - ng

    def updatemyderivativeintermediates(self, start_index_WFnew, ij_indices):
        ij_indices = [list(ij) for ij in ij_indices]
        j_values_noconj = []
        i_values_noconj = []
        indices_conj = []
        j_values_conj = []
        i_values_conj = []
        indices_noconj = []
        ngMs = start_index_WFnew
        ng = self.num_gaussians
        for i in range(ng):
            for j in range(ngMs, ng):

                if [j, i] in ij_indices:
                    index = ij_indices.index([j, i])
                    j_values_conj.append(j)
                    i_values_conj.append(i)
                    indices_conj.append(index)
                elif [i, j] in ij_indices:
                    index = ij_indices.index([i, j])
                    j_values_noconj.append(j)
                    i_values_noconj.append(i)
                    indices_noconj.append(index)
        self.jvc = np.array(j_values_conj)
        self.ivc = np.array(i_values_conj)
        self.jvnc = np.array(j_values_noconj)
        self.ivnc = np.array(i_values_noconj)
        self.ic = np.array(indices_conj)
        self.inc = np.array(indices_noconj)
        self.indices_j_vc = ngMs + self.jvc - ng
        self.indices_j_vnc = ngMs + self.jvnc - ng

    def calculate_full_overlap_deriv_matrix(self, start_index_WFnew):
        # This calculates a 3-tensor, where the element D[i,j,n] is the derivative <g_i|\frac{\partial}{\partial n_j}|g_j>, where $n_j$ is one of the
        # n(n+3) parameters specifying the Gaussian.
        # Returns the matrix deriv.shape = [num_gaussians_tot,num_gaussians_newWF,num_params_per_gaussians]
        ng = self.num_gaussians
        nd = self.num_dimensions
        ngMs = start_index_WFnew
        if self.update == False:
            if self.jvc is None:
                self.setupmyderivintermediates(start_index_WFnew)
            x1_temp = self.x1_temp
            x2_temp = self.x2_temp
        else:
            x1_temp = self.new_x1_temp
            x2_temp = self.new_x2_temp
        ic = self.ic
        inc = self.inc
        ivc = self.ivc
        ivnc = self.ivnc
        jvc = self.jvc
        jvnc = self.jvnc
        indices_j_vc = self.indices_j_vc
        indices_j_vnc = self.indices_j_vnc

        def calculate_LK_derivs(indices, j_values, mat, matmuj, conj=True):
            x2 = np.conj(x2_temp[indices]) if conj else x2_temp[indices]
            x1 = np.conj(x1_temp[indices]) if conj else x1_temp[indices]
            ovlp = np.conj(self.myovlp[indices]) if conj else self.myovlp[indices]
            temp = -einsum("xkl,cxkl->xc", x2, mat[:, j_values, :, :])
            temp += 2 * einsum("xk,cxk->xc", x1, matmuj[:, j_values])
            temp += -einsum(
                "x,xk,cxk->xc", ovlp, self.vectors[j_values, :], matmuj[:, j_values]
            )
            return temp

        def calculate_mu_derivs(indices, j_values, conj):
            ovlp = np.conj(self.myovlp[indices]) if conj else self.myovlp[indices]
            x1 = np.conj(x1_temp[indices]) if conj else x1_temp[indices]
            temp = -2 * einsum(
                "xck,xk,x->xc", self.matrices[j_values, :, :], self.vectors[j_values, :], ovlp
            )
            temp += 2 * einsum("xck,xk->xc", self.matrices[j_values, :, :], x1)
            return temp

        D_tensor = np.zeros((ng, ng - start_index_WFnew, nd * (nd + 3)), dtype=np.complex128)

        if ic.size > 0:
            D_tensor[ivc, indices_j_vc - ngMs, : self.ab_iter] = calculate_LK_derivs(
                ic, jvc, self.IabLj_list, self.IabLjmuj_list, conj=True
            )
            D_tensor[ivc, indices_j_vc - ngMs, self.ab_iter : 2 * self.ab_iter] = (
                1j * calculate_LK_derivs(ic, jvc, self.Kab_list, self.Kabmuj_list, conj=True)
            )
            D_tensor[ivc, indices_j_vc - ngMs, nd * (nd + 1) : nd * (nd + 2)] = (
                calculate_mu_derivs(ic, jvc, conj=True)
            )

        if inc.size > 0:
            D_tensor[ivnc, indices_j_vnc - ngMs, : self.ab_iter] = calculate_LK_derivs(
                inc, jvnc, self.IabLj_list, self.IabLjmuj_list, conj=False
            )
            D_tensor[ivnc, indices_j_vnc - ngMs, self.ab_iter : 2 * self.ab_iter] = (
                1j
                * calculate_LK_derivs(inc, jvnc, self.Kab_list, self.Kabmuj_list, conj=False)
            )
            D_tensor[ivnc, indices_j_vnc - ngMs, nd * (nd + 1) : nd * (nd + 2)] = (
                calculate_mu_derivs(inc, jvnc, conj=False)
            )
        D_tensor[:, :, nd * (nd + 2) : nd * (nd + 3)] = (
            1j * D_tensor[:, :, nd * (nd + 1) : nd * (nd + 2)]
        )
        D_tensor_full = np.zeros_like(D_tensor)
        self.comm.Reduce(
            [D_tensor, MPI.COMPLEX16], [D_tensor_full, MPI.COMPLEX16], op=MPI.SUM, root=0
        )
        if self.rank == 0:
            fodmd = self.fodmd

            D_tensor_full += -0.5 * einsum(
                "ij,jk->ijk", self.overlap[:, ngMs:], fodmd[ngMs:, :]
            )

        return D_tensor_full

    def calculate_potential(self):
        potential_temp = 0.5 * einsum("xkk->x", self.x2_temp)
        potential_temp += -self.lambda_ / 3 * einsum("xkkk->x", self.x3_temp[:, 1:, 1:, 1:])
        potential_temp += self.lambda_ * einsum("xkkk->x", self.x3_temp[:, 0:-1, 0:-1, 1:])
        self.potential = np.zeros_like(self.overlap)
        self.potential[self.myrow_indices, self.mycol_indices] = potential_temp
        self.potential[self.mycol_indices, self.myrow_indices] = np.conj(potential_temp)
        return self.potential

    def calculate_full_potential_deriv_matrix(self, start_index_WFnew):
        # This calculates a 3-tensor, where the element D[i,j,n] is the derivative <g_i|V(\vec{x})\frac{\partial}{\partial n_j}|g_j>, where $n_j$ is one of the
        # n(n+3) parameters specifying the Gaussian.
        ng = self.num_gaussians
        nd = self.num_dimensions
        ngMs = start_index_WFnew
        if self.jvc is None:
            self.setupmyderivintermediates(start_index_WFnew)
        ic = self.ic
        inc = self.inc
        ivc = self.ivc
        ivnc = self.ivnc
        jvc = self.jvc
        jvnc = self.jvnc
        indices_j_vc = self.indices_j_vc
        indices_j_vnc = self.indices_j_vnc

        def calculate_temp_mats(indices, j_values, i_values, conj=True):
            x5 = np.conj(self.x5_temp[indices]) if conj else self.x5_temp[indices]
            x4 = np.conj(self.x4_temp[indices]) if conj else self.x4_temp[indices]
            x3 = np.conj(self.x3_temp[indices]) if conj else self.x3_temp[indices]
            mypot = np.conj(self.mypotential[indices]) if conj else self.mypotential[indices]
            myvec = self.vectors[j_values, :]
            x5_contr1 = einsum("xlmkkk->xlm", x5[:, :, :, 1:, 1:, 1:])
            x5_contr2 = einsum("xlmkkk->xlm", x5[:, :, :, 0:-1, 0:-1, 1:])
            x5_contr = self.lambda_ / 3 * x5_contr1 - self.lambda_ * x5_contr2
            x4_contr = einsum("xlkkk->xl", -self.lambda_ / 3 * 2 * x4[:, :, 1:, 1:, 1:])
            x4_contr += einsum("xlkkk->xl", 2 * self.lambda_ * x4[:, :, 0:-1, 0:-1, 1:])
            contr1 = einsum("x,xk->xk", mypot, myvec)
            contr2 = x5_contr - 0.5 * einsum("xabkk->xab", x4)
            contr3 = einsum("xlkk->xl", x3) - contr1 + x4_contr
            return contr3, contr2

        def calculate_LK_derivs(indices, j_values, i_values, mat, matmuj, contr3, contr2):
            return einsum("xnm,cxnm->xc", contr2, mat[:, j_values, :, :]) + einsum(
                "xn,cxn->xc", contr3, matmuj[:, j_values]
            )

        def calculate_mu_derivs(indices, j_values, i_values, conj):
            x4 = np.conj(self.x4_temp[indices]) if conj else self.x4_temp[indices]
            x4_contr = einsum("xlkkk->xl", -self.lambda_ / 3 * 2 * x4[:, :, 1:, 1:, 1:])
            x4_contr += einsum("xlkkk->xl", 2 * self.lambda_ * x4[:, :, 0:-1, 0:-1, 1:])
            x3 = np.conj(self.x3_temp[indices]) if conj else self.x3_temp[indices]
            potential = (
                np.conj(self.mypotential[indices]) if conj else self.mypotential[indices]
            )
            mymatrices = self.matrices[j_values]
            myvec = self.vectors[j_values, :]
            temp = einsum("xak,xkll->xa", mymatrices, x3)
            temp += einsum("xak,xk->xa", mymatrices, x4_contr)
            temp += -2 * einsum("xak,xk,x->xa", mymatrices, myvec, potential)

            return temp

        D_tensor_alt = np.zeros(
            (ng, ng - start_index_WFnew, nd * (nd + 3)), dtype=np.complex128
        )
        if ic.size > 0:
            conj = True
            C1, C2 = calculate_temp_mats(ic, jvc, ivc, conj=conj)
            D_tensor_alt[ivc, indices_j_vc, : self.ab_iter] = calculate_LK_derivs(
                ic, jvc, ivc, self.IabLj_list, self.IabLjmuj_list, C1, C2
            )
            D_tensor_alt[ivc, indices_j_vc, self.ab_iter : 2 * self.ab_iter] = (
                1j * calculate_LK_derivs(ic, jvc, ivc, self.Kab_list, self.Kabmuj_list, C1, C2)
            )
            D_tensor_alt[ivc, indices_j_vc, nd * (nd + 1) : nd * (nd + 2)] = (
                calculate_mu_derivs(ic, jvc, ivc, conj=conj)
            )

        if inc.size > 0:
            conj = False
            C1, C2 = calculate_temp_mats(inc, jvnc, ivnc, conj=conj)
            D_tensor_alt[ivnc, indices_j_vnc, : self.ab_iter] = calculate_LK_derivs(
                inc, jvnc, ivnc, self.IabLj_list, self.IabLjmuj_list, C1, C2
            )
            D_tensor_alt[ivnc, indices_j_vnc, self.ab_iter : 2 * self.ab_iter] = (
                1j
                * calculate_LK_derivs(inc, jvnc, ivnc, self.Kab_list, self.Kabmuj_list, C1, C2)
            )
            D_tensor_alt[ivnc, indices_j_vnc, nd * (nd + 1) : nd * (nd + 2)] = (
                calculate_mu_derivs(inc, jvnc, ivnc, conj=conj)
            )
        D_tensor_alt[:, :, nd * (nd + 2) : nd * (nd + 3)] = (
            1j * D_tensor_alt[:, :, nd * (nd + 1) : nd * (nd + 2)]
        )

        return D_tensor_alt

    def calculate_kinetic_energy(self):
        self.kinetic = np.zeros_like(self.overlap)
        counter = 0
        kinetic_temp = einsum("x,x->x", self.cj_of_interest, self.myovlp)
        kinetic_temp += einsum("xk,xk->x", self.j_linear_of_interest, self.x1_temp)
        kinetic_temp += -2 * einsum(
            "xab,xab->x", self.x2_temp, self.matrices_squared_of_interest
        )
        self.kinetic[self.myrow_indices, self.mycol_indices] = kinetic_temp
        self.kinetic[self.mycol_indices, self.myrow_indices] = np.conj(kinetic_temp)
        return self.kinetic

    def calculate_full_kinetic_energy_deriv(self, start_index_WFnew):
        # start with the bra expression, e.g. nabla^2 acting on the bra, that's a hack to not have to derive the ket
        # This calculates a 3-tensor, where the element D[i,j,n] is the derivative <g_i|-(1/2 laplace)\frac{\partial}{\partial n_j}|g_j>, where $n_j$ is one of the
        # n(n+3) parameters specifying the Gaussian.
        ng = self.num_gaussians
        nd = self.num_dimensions
        ngMs = start_index_WFnew
        if self.jvc is None:
            self.setupmyderivintermediates(start_index_WFnew)
        ic = self.ic
        inc = self.inc
        ivc = self.ivc
        ivnc = self.ivnc
        jvc = self.jvc
        jvnc = self.jvnc
        indices_j_vc = self.indices_j_vc
        indices_j_vnc = self.indices_j_vnc
        D_tensor = np.zeros((ng, ng - start_index_WFnew, nd * (nd + 3)), dtype=np.complex128)

        def calculate_C1C2_mats(indices, j_values, i_values, conj=True):
            x4 = np.conj(self.x4_temp[indices]) if conj else self.x4_temp[indices]
            x3 = np.conj(self.x3_temp[indices]) if conj else self.x3_temp[indices]
            x2 = np.conj(self.x2_temp[indices]) if conj else self.x2_temp[indices]
            x1 = np.conj(self.x1_temp[indices]) if conj else self.x1_temp[indices]
            ovlp = np.conj(self.myovlp[indices]) if conj else self.myovlp[indices]
            kinetic = np.conj(self.mykinetic[indices]) if conj else self.mykinetic[indices]
            mycis = self.cis[i_values]
            mymsc = self.matrices_squared_conj[i_values, :, :]
            myvec = self.vectors[j_values, :]
            myrtc = self.rho_T_conjs[i_values, :]
            C2 = -einsum("x,xkl->xkl", mycis, x2)
            C2 += 2 * einsum("xnmkl,xmn->xkl", x4, mymsc)
            C2 += -4 * einsum("xnkl,xn->xkl", x3, myrtc)
            C1 = -einsum("x,x,xk->xk", mycis, ovlp, myvec)
            C1 += 2 * einsum("x,xk->xk", mycis, x1)
            C1 += -einsum("x,xk->xk", kinetic, myvec)
            C1 += einsum("x,x,xk->xk", mycis, ovlp, myvec)
            C1 += -4 * einsum("xnmk,xmn->xk", x3, mymsc)
            C1 += 8 * einsum("xnk,xk->xn", x2, myrtc)
            return C1, C2

        def calculate_LK_derivs(indices, j_values, i_values, mat, matmuj, C1, C2):
            return einsum("xnm,cxnm->xc", C2, mat[:, j_values, :, :]) + einsum(
                "xn,cxn->xc", C1, matmuj[:, j_values]
            )

        def calculate_mu_derivs(indices, j_values, i_values, conj):
            x3 = np.conj(self.x3_temp[indices]) if conj else self.x3_temp[indices]
            x2 = np.conj(self.x2_temp[indices]) if conj else self.x2_temp[indices]
            x1 = np.conj(self.x1_temp[indices]) if conj else self.x1_temp[indices]
            kinetic = np.conj(self.mykinetic[indices]) if conj else self.mykinetic[indices]
            mycis = self.cis[i_values]
            mymatrices = self.matrices[j_values]
            mymsc = self.matrices_squared_conj[i_values, :, :]
            myvec = self.vectors[j_values, :]
            myrtc = self.rho_T_conjs[i_values, :]
            temp = 2 * einsum("x,xaf,xf->xa", mycis, mymatrices, x1)
            temp += -2 * einsum("x,xaf,xf->xa", kinetic, mymatrices, myvec)
            temp += 8 * einsum("xaf,xg,xfg->xa", mymatrices, myrtc, x2)
            temp += -4 * einsum("xfg,xak,xfgk->xa", mymsc, mymatrices, x3)
            return temp

        if ic.size > 0:
            conj = True
            C1, C2 = calculate_C1C2_mats(ic, jvc, ivc, conj=conj)
            D_tensor[ivc, indices_j_vc, : self.ab_iter] = calculate_LK_derivs(
                ic, jvc, ivc, self.IabLj_list, self.IabLjmuj_list, C1, C2
            )
            D_tensor[ivc, indices_j_vc, self.ab_iter : 2 * self.ab_iter] = (
                1j * calculate_LK_derivs(ic, jvc, ivc, self.Kab_list, self.Kabmuj_list, C1, C2)
            )
            D_tensor[ivc, indices_j_vc, nd * (nd + 1) : nd * (nd + 2)] = calculate_mu_derivs(
                ic, jvc, ivc, conj=conj
            )

        if inc.size > 0:
            conj = False
            C1, C2 = calculate_C1C2_mats(inc, jvnc, ivnc, conj=conj)
            D_tensor[ivnc, indices_j_vnc, : self.ab_iter] = calculate_LK_derivs(
                inc, jvnc, ivnc, self.IabLj_list, self.IabLjmuj_list, C1, C2
            )
            D_tensor[ivnc, indices_j_vnc, self.ab_iter : 2 * self.ab_iter] = (
                1j
                * calculate_LK_derivs(inc, jvnc, ivnc, self.Kab_list, self.Kabmuj_list, C1, C2)
            )
            D_tensor[ivnc, indices_j_vnc, nd * (nd + 1) : nd * (nd + 2)] = calculate_mu_derivs(
                inc, jvnc, ivnc, conj=conj
            )
        D_tensor[:, :, nd * (nd + 2) : nd * (nd + 3)] = (
            1j * D_tensor[:, :, nd * (nd + 1) : nd * (nd + 2)]
        )
        return D_tensor

    def calculate_kinetic_energy_squared(self):
        self.kinetic_energy_squared = np.zeros_like(self.overlap)
        kinetic_squared_temp = -2 * einsum(
            "x,xkl,xkl->x",
            self.ci_of_interest,
            self.matrices_j_squared_of_interest,
            self.x2_temp,
        )
        kinetic_squared_temp += -2 * einsum(
            "x,xkl,xkl->x",
            self.cj_of_interest,
            self.matrices_i_squared_of_interest,
            self.x2_temp,
        )
        kinetic_squared_temp += 16 * einsum(
            "xk,xl,xkl->x", self.rho_T_conj_of_interest, self.omega_T_of_interest, self.x2_temp
        )
        kinetic_squared_temp += 4 * einsum(
            "x,xl,xl->x", self.ci_of_interest, self.omega_T_of_interest, self.x1_temp
        )
        kinetic_squared_temp += 4 * einsum(
            "x,xl,xl->x", self.cj_of_interest, self.rho_T_conj_of_interest, self.x1_temp
        )

        kinetic_squared_temp += -8 * einsum(
            "xk,xlm,xklm->x",
            self.omega_T_of_interest,
            self.matrices_i_squared_of_interest,
            self.x3_temp,
        )
        kinetic_squared_temp += -8 * einsum(
            "xk,xlm,xklm->x",
            self.rho_T_conj_of_interest,
            self.matrices_j_squared_of_interest,
            self.x3_temp,
        )

        kinetic_squared_temp += einsum(
            "x,x,x->x", self.ci_of_interest, self.cj_of_interest, self.myovlp
        )
        kinetic_squared_temp += 4 * einsum(
            "xkl,xmn,xklmn->x",
            self.matrices_i_squared_of_interest,
            self.matrices_j_squared_of_interest,
            self.x4_temp,
        )
        self.kinetic_energy_squared[self.myrow_indices, self.mycol_indices] = (
            kinetic_squared_temp
        )
        self.kinetic_energy_squared[self.mycol_indices, self.myrow_indices] = np.conj(
            kinetic_squared_temp
        )

        return self.kinetic_energy_squared

    def calculate_full_potential_squared_deriv_matrix(self, start_index_WFnew):
        # This calculates a 3-tensor, where the element D[i,j,n] is the derivative <g_i|V^2 (\vec{x})\frac{\partial}{\partial n_j}|g_j>, where $n_j$ is one of the
        # n(n+3) parameters specifying the Gaussian.

        ng = self.num_gaussians
        nd = self.num_dimensions
        ngMs = start_index_WFnew
        if self.jvc is None:
            self.setupmyderivintermediates(start_index_WFnew)
        ic = self.ic
        inc = self.inc
        ivc = self.ivc
        ivnc = self.ivnc
        jvc = self.jvc
        jvnc = self.jvnc
        indices_j_vc = self.indices_j_vc
        indices_j_vnc = self.indices_j_vnc

        def calculate_temp_mats(indices, j_values, i_values, conj=True):
            x6 = np.conj(self.x6_temp[indices]) if conj else self.x6_temp[indices]

            x5 = np.conj(self.x5_temp[indices]) if conj else self.x5_temp[indices]
            x7_c1v = np.conj(self.x7_c1v_temp[indices]) if conj else self.x7_c1v_temp[indices]
            x7_c2v = np.conj(self.x7_c2v_temp[indices]) if conj else self.x7_c2v_temp[indices]
            x7_c3v = np.conj(self.x7_c3v_temp[indices]) if conj else self.x7_c3v_temp[indices]
            x7_c1 = np.conj(self.x7_c1_temp[indices]) if conj else self.x7_c1_temp[indices]
            x7_c2 = np.conj(self.x7_c2_temp[indices]) if conj else self.x7_c2_temp[indices]
            x8_c1 = np.conj(self.x8_c1_temp[indices]) if conj else self.x8_c1_temp[indices]
            x8_c2 = np.conj(self.x8_c2_temp[indices]) if conj else self.x8_c2_temp[indices]
            x8_c3 = np.conj(self.x8_c3_temp[indices]) if conj else self.x8_c3_temp[indices]
            x5e_contracted_fullV = einsum("xkiijj->xk", x5)
            x6_e_contracted_full = einsum("xkliijj->xkl", x6)
            x6_e_contracted_firstV = einsum("xkiijjj->xk", x6[:, :, :, :, :-1, :-1, 1:])
            x6_e_contracted_secondV = einsum("xkiijjj->xk", x6[:, :, :, :, 1:, 1:, 1:])
            super_contr1 = (
                1 / 2 * x5e_contracted_fullV
                + 2 * x6_e_contracted_firstV * self.lambda_
                + x6_e_contracted_secondV * (-2 / 3 * self.lambda_)
            )
            super_contr1 += (
                2 / 9 * self.lambda_sq * x7_c1v
                + 2 * self.lambda_sq * x7_c2v
                + -4 / 3 * self.lambda_sq * x7_c3v
            )
            super_contr2 = x6_e_contracted_full * (-0.25)
            super_contr2 += -self.lambda_ * x7_c1
            super_contr2 += +self.lambda_ / 3 * x7_c2
            super_contr2 += -self.lambda_sq / 9 * x8_c1
            super_contr2 += +2 / 3 * self.lambda_sq * x8_c3
            super_contr2 += -self.lambda_sq * x8_c2
            return super_contr1, super_contr2

        def calculate_LK_derivs(
            indices, j_values, i_values, mat, matmuj, super_contr1, super_contr2
        ):
            mypotsq = (
                np.conj(self.mypotential_squared[indices])
                if conj
                else self.mypotential_squared[indices]
            )
            myvec = self.vectors[j_values, :]
            retval = einsum("xnm,cxnm->xc", super_contr2, mat[:, j_values, :, :])
            retval += einsum("xn,cxn->xc", super_contr1, matmuj[:, j_values])
            retval += -einsum("x,xk,cxk->xc", mypotsq, myvec, matmuj[:, j_values])
            return retval

        def calculate_mu_derivs(indices, j_values, i_values, conj):
            x6 = np.conj(self.x6_temp[indices]) if conj else self.x6_temp[indices]
            x5 = np.conj(self.x5_temp[indices]) if conj else self.x5_temp[indices]
            x7_c1v = np.conj(self.x7_c1v_temp[indices]) if conj else self.x7_c1v_temp[indices]
            x7_c2v = np.conj(self.x7_c2v_temp[indices]) if conj else self.x7_c2v_temp[indices]
            x7_c3v = np.conj(self.x7_c3v_temp[indices]) if conj else self.x7_c3v_temp[indices]
            x5e_contracted_fullV = einsum("xkiijj->xk", x5)
            x6_e_contracted_firstV = einsum("xkiijjj->xk", x6[:, :, :, :, :-1, :-1, 1:])
            x6_e_contracted_secondV = einsum("xkiijjj->xk", x6[:, :, :, :, 1:, 1:, 1:])
            super_contr1 = (
                1 / 2 * x5e_contracted_fullV
                + 2 * x6_e_contracted_firstV * self.lambda_
                + x6_e_contracted_secondV * (-2 / 3 * self.lambda_)
            )
            super_contr1 += (
                2 / 9 * self.lambda_sq * x7_c1v
                + 2 * self.lambda_sq * x7_c2v
                + -4 / 3 * self.lambda_sq * x7_c3v
            )
            mypotsq = (
                np.conj(self.mypotential_squared[indices])
                if conj
                else self.mypotential_squared[indices]
            )
            mymatrices = self.matrices[j_values]
            myvec = self.vectors[j_values, :]
            temp = einsum("xck,xk->xc", mymatrices, super_contr1)
            temp += -2 * einsum("x,xck,xk->xc", mypotsq, mymatrices, myvec)
            return temp

        D_tensor_alt = np.zeros(
            (ng, ng - start_index_WFnew, nd * (nd + 3)), dtype=np.complex128
        )

        if ic.size > 0:
            conj = True
            C1, C2 = calculate_temp_mats(ic, jvc, ivc, conj=conj)
            D_tensor_alt[ivc, indices_j_vc, : self.ab_iter] = calculate_LK_derivs(
                ic, jvc, ivc, self.IabLj_list, self.IabLjmuj_list, C1, C2
            )
            D_tensor_alt[ivc, indices_j_vc, self.ab_iter : 2 * self.ab_iter] = (
                1j * calculate_LK_derivs(ic, jvc, ivc, self.Kab_list, self.Kabmuj_list, C1, C2)
            )
            D_tensor_alt[ivc, indices_j_vc, nd * (nd + 1) : nd * (nd + 2)] = (
                calculate_mu_derivs(ic, jvc, ivc, conj=conj)
            )

        if inc.size > 0:
            conj = False
            C1, C2 = calculate_temp_mats(inc, jvnc, ivnc, conj=conj)
            D_tensor_alt[ivnc, indices_j_vnc, : self.ab_iter] = calculate_LK_derivs(
                inc, jvnc, ivnc, self.IabLj_list, self.IabLjmuj_list, C1, C2
            )
            D_tensor_alt[ivnc, indices_j_vnc, self.ab_iter : 2 * self.ab_iter] = (
                1j
                * calculate_LK_derivs(inc, jvnc, ivnc, self.Kab_list, self.Kabmuj_list, C1, C2)
            )
            D_tensor_alt[ivnc, indices_j_vnc, nd * (nd + 1) : nd * (nd + 2)] = (
                calculate_mu_derivs(inc, jvnc, ivnc, conj=conj)
            )
        D_tensor_alt[:, :, nd * (nd + 2) : nd * (nd + 3)] = (
            1j * D_tensor_alt[:, :, nd * (nd + 1) : nd * (nd + 2)]
        )
        return D_tensor_alt

    def calculate_full_kinetic_energy_squared_deriv(self, start_index_WFnew):
        # start with the bra expression, e.g. nabla^2 acting on the bra, that's a hack to not have to derive the ket
        # This calculates a 3-tensor, where the element D[i,j,n] is the derivative <g_i|-(1/2 laplace)\frac{\partial}{\partial n_j}|g_j>, where $n_j$ is one of the
        # n(n+3) parameters specifying the Gaussian.
        ng = self.num_gaussians
        nd = self.num_dimensions
        if self.jvc is None:
            self.setupmyderivintermediates(start_index_WFnew)
        ic = self.ic
        inc = self.inc
        ivc = self.ivc
        ivnc = self.ivnc
        jvc = self.jvc
        jvnc = self.jvnc
        indices_j_vc = self.indices_j_vc
        indices_j_vnc = self.indices_j_vnc

        def calculate_temp_mats(indices, j_values, i_values, conj=True):
            x6 = np.conj(self.x6_temp[indices]) if conj else self.x6_temp[indices]
            x5 = np.conj(self.x5_temp[indices]) if conj else self.x5_temp[indices]
            x1 = np.conj(self.x1_temp[indices]) if conj else self.x1_temp[indices]
            x2 = np.conj(self.x2_temp[indices]) if conj else self.x2_temp[indices]
            x3 = np.conj(self.x3_temp[indices]) if conj else self.x3_temp[indices]
            x4 = np.conj(self.x4_temp[indices]) if conj else self.x4_temp[indices]

            ovlp = np.conj(self.myovlp[indices]) if conj else self.myovlp[indices]

            mycis = self.cis[i_values]
            mycjs = self.cjs[j_values]
            mymsc = self.matrices_squared_conj[i_values, :, :]
            myms = self.matrices_squared[j_values, :, :]
            myvec = self.vectors[j_values, :]
            myrtc = self.rho_T_conjs[i_values, :]
            myomega = self.omega_T[j_values, :]
            x5rhomat = einsum("xzklmn,xz->xklmn", x5, myrtc)

            x6_eAisqconjmat = einsum("aklxymn,axy->aklmn", x6, mymsc)
            x5_eAisqconjmat = einsum("akxymn,axy->akmn", x5, mymsc)
            tempAjsqx6_eAisqconj = -4 * einsum("akl,aklmn->amn", myms, x6_eAisqconjmat)
            tempx5_eAjsqAisqconj = 8 * einsum("amkl,akl->am", x5_eAisqconjmat, myms)
            temp1 = einsum("x,x->x", mycis, ovlp)

            temp1 += 4 * einsum("xk,xk->x", myrtc, x1)
            temp1 += -2 * einsum("xkl,xkl->x", x2, mymsc)

            temp3 = 4 * einsum("x,xk->xk", mycis, x1)
            temp3 += 16 * einsum("xl,xkl->xk", myrtc, x2)
            temp3 += -8 * einsum("xlm,xklm->xk", mymsc, x3)
            temp4 = -2 * einsum("x,xnm->xnm", mycis, x2)
            temp4 += -8 * einsum("xk,xnmk->xnm", myrtc, x3)
            temp4 += 4 * einsum("xkl,xnmkl->xnm", mymsc, x4)

            temp2 = -einsum("x,x,x->x", mycjs, mycis, ovlp)

            temp2 += -4 * einsum("x,xk,xk->x", mycjs, x1, myrtc)
            temp2 += -4 * einsum("x,xk,xk->x", mycis, myomega, x1)
            temp2 += 2 * einsum("x,xkl,xkl->x", mycjs, mymsc, x2)
            temp2 += 2 * einsum("x,xkl,xkl->x", mycis, myms, x2)
            temp2 += -16 * einsum("xk,xkl,xl->x", myomega, x2, myrtc)
            temp2 += 8 * einsum("xk,xklm,xlm->x", myomega, x3, mymsc)
            temp2 += 8 * einsum("xmkl,xm,xkl->x", x3, myrtc, myms)
            temp2 += -4 * einsum("xkl,xklmn,xmn->x", myms, x4, mymsc)

            temp5 = 2 * einsum("x,x,xm->xm", mycjs, mycis, x1)
            temp5 += 8 * einsum("x,xl,xml->xm", mycjs, myrtc, x2)
            temp5 += -4 * einsum("x,xmkl,xkl->xm", mycjs, x3, mymsc)
            temp5 += 8 * einsum("x,xk,xmk->xm", mycis, myomega, x2)
            temp5 += -4 * einsum("x,xkl,xmkl->xm", mycis, myms, x3)
            temp5 += 32 * einsum("xmkl,xk,xl->xm", x3, myomega, myrtc)
            temp5 += -16 * einsum("xmkln,xk,xln->xm", x4, myomega, mymsc)
            temp5 += -16 * einsum("xmnkl,xn,xkl->xm", x4, myrtc, myms)
            temp5 += tempx5_eAjsqAisqconj

            temp6 = 2 * einsum("x,xkl,xklmn->xmn", mycjs, mymsc, x4)
            temp6 += -16 * einsum("xk,xklmn,xl->xmn", myomega, x4, myrtc)
            temp6 += 8 * einsum("xk,xkmn->xmn", myomega, x5_eAisqconjmat)
            temp6 += 8 * einsum("xklmn,xkl->xmn", x5rhomat, myms)
            temp6 += tempAjsqx6_eAisqconj
            temp6 += -einsum("x,x,xkl->xkl", mycjs, mycis, x2)
            temp6 += -4 * einsum("x,xlmn,xl->xmn", mycjs, x3, myrtc)
            temp6 += -4 * einsum("x,xk,xkmn->xmn", mycis, myomega, x3)
            temp6 += 2 * einsum("x,xkl,xklmn->xmn", mycis, myms, x4)
            temp7 = -8 * einsum("x,xk,xk->x", mycjs, myrtc, x1)
            temp7 += 4 * einsum("x,xfg,xfg->x", mycjs, mymsc, x2)
            temp7 += -32 * einsum("xk,xf,xkf->x", myomega, myrtc, x2)
            temp7 += 16 * einsum("xk,xfg,xkfg->x", myomega, mymsc, x3)
            temp7 += 16 * einsum("xf,xkl,xfkl->x", myrtc, myms, x3)
            temp7 += -8 * einsum("xkl,xfg,xfgkl->x", mymsc, myms, x4)
            temp7 += -2 * einsum("x,x,x->x", mycis, mycjs, ovlp)
            temp7 += -8 * einsum("x,xk,xk->x", mycis, myomega, x1)
            temp7 += 4 * einsum("x,xkl,xkl->x", mycis, myms, x2)
            temp8 = 8 * einsum("x,xf,xfn->xn", mycjs, myrtc, x2)
            temp8 += -4 * einsum("x,xfg,xfgn->xn", mycjs, mymsc, x3)
            temp8 += 8 * einsum("x,xk,xkn->xn", mycis, myomega, x2)
            temp8 += -4 * einsum("x,xfg,xfgn->xn", mycis, myms, x3)
            temp8 += 2 * einsum("x,x,xn->xn", mycjs, mycis, x1)
            temp8 += -16 * einsum("xk,xfg,xknfg->xn", myomega, mymsc, x4)
            temp8 += 32 * einsum("xk,xg,xkgn->xn", myomega, myrtc, x3)
            temp8 += -16 * einsum("xf,xkl,xfnkl->xn", myrtc, myms, x4)
            temp8 += 8 * einsum("xkl,xnkl->xn", myms, x5_eAisqconjmat)
            temp9 = -4 * einsum("x,x->x", mycis, ovlp)
            temp9 += -16 * einsum("xk,xk->x", myrtc, x1)
            temp9 += 8 * einsum("xkl,xkl->x", x2, mymsc)
            temp10 = 4 * einsum("x,xk->xk", mycis, x1)
            temp10 += 16 * einsum("xl,xlk->xk", myrtc, x2)
            temp10 += -8 * einsum("xlm,xlmk->xk", mymsc, x3)

            return temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10

        def calculate_LK_derivs(
            indices,
            j_values,
            i_values,
            Asquaredderiv,
            cj_deriv,
            omega_deriv,
            mat,
            matmuj,
            const,
            temps,
        ):

            temp1, temp2, temp3, temp4, temp5, temp6 = temps[:6]
            retval = einsum("cx,x->xc", const[:, j_values], temp2)
            retval += einsum("cxnm,xnm->xc", Asquaredderiv[:, j_values], temp4)
            retval += einsum("x,cx->xc", temp1, cj_deriv[:, j_values])
            retval += einsum("cxk,xk->xc", omega_deriv[:, j_values], temp3)

            retval += einsum("cxm,xm->xc", matmuj[:, j_values], temp5)
            retval += einsum("xmn,cxnm->xc", temp6, mat[:, j_values])
            return retval

        def calculate_mu_derivs(indices, j_values, i_values, temps):
            temp7, temp8, temp9, temp10 = temps
            zs = einsum("xak,xk->xa", self.matrices[j_values], self.vectors[j_values])
            ck_derivs = einsum(
                "xal,xl->xa", self.matrices_squared[j_values], self.vectors[j_values]
            )
            vaAsquareds = self.matrices_squared[j_values]
            temp = einsum("x,xa->xa", temp7, zs)

            temp += einsum("xn,xan->xa", temp8, self.matrices[j_values])
            temp += einsum("x,xa->xa", temp9, ck_derivs)
            temp += einsum("xk,xak->xa", temp10, vaAsquareds)
            return temp

        D_tensor_alt = np.zeros(
            (ng, ng - start_index_WFnew, nd * (nd + 3)), dtype=np.complex128
        )
        mKm_list = []
        mLm_list = []
        self.ab_iter = 0
        for a in range(nd):
            for b in range(a + 1):
                self.ab_iter += 1
                mLm_list.append(self.mLm[:, a, b])
                mKm_list.append(self.mKm[:, a, b])
        mLm_list = np.array(mLm_list)
        mKm_list = np.array(mKm_list)

        Asquared_derivs_I_list = einsum("cjkx,jxl->cjkl", self.IabLj_list, self.matrices)
        Asquared_derivs_I_list += einsum("jkx,cjxl->cjkl", self.matrices, self.IabLj_list)
        Asquared_derivs_K_list = einsum("cjkx,jxl->cjkl", self.Kab_list, self.matrices)
        Asquared_derivs_K_list += einsum("jkx,cjxl->cjkl", self.matrices, self.Kab_list)
        cj_derivs_I_list = einsum("cjkk->cj", self.IabLj_list, dtype=np.complex128)
        cj_derivs_I_list += -2 * einsum(
            "jk,cjkl,jl->cj", self.vectors, Asquared_derivs_I_list, self.vectors
        )
        cj_derivs_K_list = einsum("cjkk->cj", self.Kab_list, dtype=np.complex128)
        cj_derivs_K_list += -2 * einsum(
            "jk,cjkl,jl->cj", self.vectors, Asquared_derivs_K_list, self.vectors
        )
        omega_derivs_K_list = einsum("jk,cjkl->cjl", self.vectors, Asquared_derivs_K_list)
        omega_derivs_I_list = einsum("jk,cjkl->cjl", self.vectors, Asquared_derivs_I_list)

        if ic.size > 0:
            conj = True
            temps = calculate_temp_mats(ic, jvc, ivc, conj=conj)
            D_tensor_alt[ivc, indices_j_vc, : self.ab_iter] = calculate_LK_derivs(
                ic,
                jvc,
                ivc,
                Asquared_derivs_I_list,
                cj_derivs_I_list,
                omega_derivs_I_list,
                self.IabLj_list,
                self.IabLjmuj_list,
                mLm_list,
                temps,
            )
            D_tensor_alt[ivc, indices_j_vc, self.ab_iter : 2 * self.ab_iter] = (
                1j
                * calculate_LK_derivs(
                    ic,
                    jvc,
                    ivc,
                    Asquared_derivs_K_list,
                    cj_derivs_K_list,
                    omega_derivs_K_list,
                    self.Kab_list,
                    self.Kabmuj_list,
                    mKm_list,
                    temps,
                )
            )
            D_tensor_alt[ivc, indices_j_vc, nd * (nd + 1) : nd * (nd + 2)] = (
                calculate_mu_derivs(ic, jvc, ivc, temps[6:])
            )

        if inc.size > 0:
            conj = False
            temps = calculate_temp_mats(inc, jvnc, ivnc, conj=conj)
            D_tensor_alt[ivnc, indices_j_vnc, : self.ab_iter] = calculate_LK_derivs(
                inc,
                jvnc,
                ivnc,
                Asquared_derivs_I_list,
                cj_derivs_I_list,
                omega_derivs_I_list,
                self.IabLj_list,
                self.IabLjmuj_list,
                mLm_list,
                temps,
            )
            D_tensor_alt[ivnc, indices_j_vnc, self.ab_iter : 2 * self.ab_iter] = (
                1j
                * calculate_LK_derivs(
                    inc,
                    jvnc,
                    ivnc,
                    Asquared_derivs_K_list,
                    cj_derivs_K_list,
                    omega_derivs_K_list,
                    self.Kab_list,
                    self.Kabmuj_list,
                    mKm_list,
                    temps,
                )
            )
            D_tensor_alt[ivnc, indices_j_vnc, nd * (nd + 1) : nd * (nd + 2)] = (
                calculate_mu_derivs(inc, jvnc, ivnc, temps[6:])
            )
        D_tensor_alt[:, :, nd * (nd + 2) : nd * (nd + 3)] = (
            1j * D_tensor_alt[:, :, nd * (nd + 1) : nd * (nd + 2)]
        )
        return D_tensor_alt

    def calculate_potential_times_kinetic_energy(self):
        self.potential_times_kinetic = np.zeros_like(self.overlap)
        potential_temp = (
            2
            / 3
            * self.lambda_
            * einsum(
                "xmmmkl,xkl->x",
                self.x5_temp[:, 1:, 1:, 1:, :, :],
                self.matrices_squared_of_interest,
            )
        )
        potential_temp += (
            -2
            * self.lambda_
            * einsum(
                "xmmmkl,xkl->x",
                self.x5_temp[:, :-1, :-1, 1:, :, :],
                self.matrices_squared_of_interest,
            )
        )
        potential_temp += (
            -4
            / 3
            * self.lambda_
            * einsum("xklll,xk->x", self.x4_temp[:, :, 1:, 1:, 1:], self.omega_T_of_interest)
        )
        potential_temp += (
            4
            * self.lambda_
            * einsum("xklll,xk->x", self.x4_temp[:, :, :-1, :-1, 1:], self.omega_T_of_interest)
        )
        potential_temp += -einsum(
            "xklmm,xkl->x", self.x4_temp, self.matrices_squared_of_interest
        )
        potential_temp += (
            -1
            / 3
            * self.lambda_
            * einsum("x,xkkk->x", self.cj_of_interest, self.x3_temp[:, 1:, 1:, 1:])
        )
        potential_temp += self.lambda_ * einsum(
            "x,xkkk->x", self.cj_of_interest, self.x3_temp[:, 0:-1, 0:-1, 1:]
        )
        potential_temp += 0.5 * einsum("x,xkk->x", self.cj_of_interest, self.x2_temp)
        potential_temp += 2 * einsum("xk,xkll->x", self.omega_T_of_interest, self.x3_temp)

        self.potential_times_kinetic[self.myrow_indices, self.mycol_indices] = potential_temp
        potential_temp = (
            2
            / 3
            * self.lambda_
            * einsum(
                "xmmmkl,xkl->x",
                self.x5_temp[:, 1:, 1:, 1:, :, :],
                self.matrices_i_squared_of_interest,
            )
        )
        potential_temp += (
            -2
            * self.lambda_
            * einsum(
                "xmmmkl,xkl->x",
                self.x5_temp[:, :-1, :-1, 1:, :, :],
                self.matrices_i_squared_of_interest,
            )
        )
        potential_temp += (
            -4
            / 3
            * self.lambda_
            * einsum(
                "xklll,xk->x", self.x4_temp[:, :, 1:, 1:, 1:], self.rho_T_conj_of_interest
            )
        )
        potential_temp += (
            4
            * self.lambda_
            * einsum(
                "xklll,xk->x", self.x4_temp[:, :, :-1, :-1, 1:], self.rho_T_conj_of_interest
            )
        )
        potential_temp += -einsum(
            "xklmm,xkl->x", self.x4_temp, self.matrices_i_squared_of_interest
        )
        potential_temp += (
            -1
            / 3
            * self.lambda_
            * einsum("x,xkkk->x", self.ci_of_interest, self.x3_temp[:, 1:, 1:, 1:])
        )
        potential_temp += self.lambda_ * einsum(
            "x,xkkk->x", self.ci_of_interest, self.x3_temp[:, 0:-1, 0:-1, 1:]
        )
        potential_temp += 0.5 * einsum("x,xkk->x", self.ci_of_interest, self.x2_temp)
        potential_temp += 2 * einsum("xk,xkll->x", self.rho_T_conj_of_interest, self.x3_temp)

        self.potential_times_kinetic[self.mycol_indices, self.myrow_indices] = np.conj(
            potential_temp
        )
        self.potential_times_kinetic += np.conj(self.potential_times_kinetic).T

        return self.potential_times_kinetic

    def calculate_full_potential_times_kinetic_deriv(self, start_index_WFnew):
        ng = self.num_gaussians
        nd = self.num_dimensions
        ngMs = start_index_WFnew
        if self.jvc is None:
            self.setupmyderivintermediates(start_index_WFnew)
        ic = self.ic
        inc = self.inc
        ivc = self.ivc
        ivnc = self.ivnc
        jvc = self.jvc
        jvnc = self.jvnc
        indices_j_vc = self.indices_j_vc
        indices_j_vnc = self.indices_j_vnc

        def calculate_temp_mats(indices, j_values, i_values, conj=True):
            x6 = np.conj(self.x6_temp[indices]) if conj else self.x6_temp[indices]
            x5 = np.conj(self.x5_temp[indices]) if conj else self.x5_temp[indices]
            x2 = np.conj(self.x2_temp[indices]) if conj else self.x2_temp[indices]
            x3 = np.conj(self.x3_temp[indices]) if conj else self.x3_temp[indices]
            x4 = np.conj(self.x4_temp[indices]) if conj else self.x4_temp[indices]
            x7_c3_temp = (
                np.conj(self.x7_c3_temp[indices]) if conj else self.x7_c3_temp[indices]
            )
            x7_c4_temp = (
                np.conj(self.x7_c4_temp[indices]) if conj else self.x7_c4_temp[indices]
            )
            mycis = self.cis[i_values]
            mycjs = self.cjs[j_values]
            mymsc = self.matrices_squared_conj[i_values, :, :]
            myms = self.matrices_squared[j_values, :, :]
            myrtc = self.rho_T_conjs[i_values, :]
            myomega = self.omega_T[j_values, :]
            x6e_c1_mat = einsum("xijkmmm->xijk", x6[:, :, :, :, :-1, :-1, 1:])
            x6_e_c2_mat = einsum("xijkmmm->xijk", x6[:, :, :, :, 1:, 1:, 1:])
            x6_e_c3_mat = einsum("xijklmm->xijkl", x6[:, :, :, :, :, :, :])
            x5e_c1_mat = einsum("xijmmm->xij", x5[:, :, :, :-1, :-1, 1:])
            x5e_c2_mat = einsum("xijmmm->xij", x5[:, :, :, 1:, 1:, 1:])
            x5_e_c3_mat = einsum("xiklmm->xikl", x5)
            x3e_c1_mat = einsum("xiii->x", x3[:, :-1, :-1, 1:])
            x3e_c2_mat = einsum("xiii->x", x3[:, 1:, 1:, 1:])
            x4e_cV1_mat = einsum("xijjj->xi", x4[:, :, :-1, :-1, 1:])
            x4e_cV2_mat = einsum("xijjj->xi", x4[:, :, 1:, 1:, 1:])
            x2e_c_mat = einsum("xii->x", x2)
            temp_Aisqconjx6e_mat = einsum("xij,xijkl->xkl", mymsc, x6_e_c3_mat)
            x4e_c1_mat = einsum("xi,xi->x", myrtc, x4e_cV1_mat)
            temp_Ajsqx6e = einsum("xij,xijkl->xkl", myms, x6_e_c3_mat)

            temp_contr6 = 4 / 3 * self.lambda_ * x6_e_c2_mat - 4 * self.lambda_ * x6e_c1_mat
            temp_contr5 = self.lambda_ * 1 / 3 * x5e_c2_mat - self.lambda_ * x5e_c1_mat
            temp_contr4 = +self.lambda_ * 2 * x4e_cV1_mat - self.lambda_ * 2 / 3 * x4e_cV2_mat
            temp_contr7 = +2 * self.lambda_ * x7_c3_temp - 2 / 3 * self.lambda_ * x7_c4_temp
            temp1 = 4 / 3 * self.lambda_ * einsum("xk,xk->x", myrtc, x4e_cV2_mat)
            temp1 += einsum("xkl,xkl->x", mymsc, -2 * temp_contr5)
            temp1 += -2 * einsum("xk,xkll->x", myrtc, x3)
            temp1 += einsum("xmn,xmnkk->x", mymsc, x4)
            temp1 += einsum("xk,xk->x", myomega, -2 * temp_contr4)
            temp1 += -2 * einsum("xkl,xkl->x", myms, temp_contr5)
            temp1 += -2 * einsum("xk,xkll->x", myomega, x3)
            temp1 += einsum("xmn,xmnkk->x", myms, x4)
            temp1 += einsum(
                "x,x->x",
                mycis,
                -self.lambda_ * x3e_c1_mat
                + 1 / 3 * self.lambda_ * x3e_c2_mat
                - 0.5 * x2e_c_mat,
            )
            temp1 += -einsum("x->x", 4 * self.lambda_ * x4e_c1_mat)
            temp1 += einsum(
                "x,x->x",
                mycjs,
                +self.lambda_ * 1 / 3 * x3e_c2_mat
                - 0.5 * x2e_c_mat
                - self.lambda_ * x3e_c1_mat,
            )
            temp2 = einsum("x,xm->xm", mycis, temp_contr4)
            temp2 += einsum("x,xmkk->xm", mycis, x3)
            temp2 += -8 * einsum("xl,xml->xm", myrtc, temp_contr5)
            temp2 += einsum("xlk,xmlk->xm", mymsc, temp_contr6)
            temp2 += 4 * einsum("xk,xmkll->xm", myrtc, x4)
            temp2 += -2 * einsum("xkl,xmkl->xm", mymsc, x5_e_c3_mat)
            temp2 += 4 * einsum("xk,xmkll->xm", myomega, x4)
            temp2 += -2 * einsum("xkl,xmkl->xm", myms, x5_e_c3_mat)
            temp2 += einsum("xl,xml->xm", myomega, -8 * temp_contr5)
            temp2 += einsum("xlk,xmlk->xm", myms, +temp_contr6)
            temp2 += einsum("x,xm->xm", mycjs, temp_contr4)
            temp2 += einsum("x,xmkk->xm", mycjs, x3)
            temp3 = einsum("x,xkl->xkl", mycis, temp_contr5)
            temp3 += -0.5 * einsum("x,xklzz->xkl", mycis, x4)
            temp3 += einsum("xm,xmkl->xkl", myrtc, temp_contr6)
            temp3 += einsum("xmn,xmnkl->xkl", mymsc, temp_contr7)
            temp3 += -2 * einsum("xm,xmkl->xkl", myrtc, x5_e_c3_mat)
            temp3 += temp_Aisqconjx6e_mat
            temp3 += einsum("x,xkl->xkl", mycjs, temp_contr5)
            temp3 += -0.5 * einsum("x,xklzz->xkl", mycjs, x4)
            temp3 += einsum("xm,xmkl->xkl", myomega, temp_contr6)
            temp3 += einsum("xmn,xmnkl->xkl", myms, temp_contr7)
            temp3 += -2 * einsum("xm,xmkl->xkl", myomega, x5_e_c3_mat)
            temp3 += temp_Ajsqx6e
            temp4 = 2 * temp_contr4
            temp4 += 2 * einsum("xkll->xk", x3)
            temp5 = 2 * temp_contr5
            temp5 += -einsum("xklmm->xkl", x4)
            temp6 = (
                self.lambda_ * x3e_c1_mat + 0.5 * x2e_c_mat - 1 / 3 * self.lambda_ * x3e_c2_mat
            )

            temp7 = einsum("x,xfgg->xf", mycis, x3)
            temp7 += einsum("x,xf->xf", mycis, temp_contr4)
            temp7 += einsum("xgk,xfgk->xf", mymsc, temp_contr6)
            temp7 += -2 * einsum("xkg,xfkg->xf", mymsc, x5_e_c3_mat)
            temp7 += einsum("xgk,xfgk->xf", myms, temp_contr6)
            temp7 += -2 * einsum("xkg,xfkg->xf", myms, x5_e_c3_mat)
            temp7 += einsum("x,xf->xf", mycjs, temp_contr4)
            temp7 += einsum("x,xfgg->xf", mycjs, x3)
            temp7 += 4 * einsum("xz,xzfkk->xf", myomega, x4)
            temp7 += -8 * einsum("xz,xzf->xf", myrtc, temp_contr5)
            temp7 += -8 * einsum("xz,xzf->xf", myomega, temp_contr5)
            temp7 += 4 * einsum("xz,xzfkk->xf", myrtc, x4)
            temp8 = -self.lambda_ * 2 * einsum("x,x->x", mycjs, x3e_c1_mat)
            temp8 += self.lambda_ * 2 / 3 * einsum("x,x->x", mycjs, x3e_c2_mat)
            temp8 += 2 * einsum("xfg,xfgkk->x", mymsc, x4)
            temp8 += -4 * einsum("xf,xfgg->x", myrtc, x3)
            temp8 += -2 * self.lambda_ * einsum("x,x->x", mycis, x3e_c1_mat)
            temp8 += 2 / 3 * self.lambda_ * einsum("x,x->x", mycis, x3e_c2_mat)
            temp8 += -8 * self.lambda_ * x4e_c1_mat
            temp8 += 8 / 3 * self.lambda_ * einsum("xf,xf->x", myrtc, x4e_cV2_mat)
            temp8 += -4 * einsum("xfg,xfg->x", mymsc, temp_contr5)
            temp8 += -einsum("x,x->x", mycis, x2e_c_mat)
            temp8 += -4 * einsum("xf,xf->x", myomega, temp_contr4)
            temp8 += -4 * einsum("xfg,xfg->x", myms, temp_contr5)
            temp8 += -einsum("x,x->x", mycjs, x2e_c_mat)
            temp8 += -4 * einsum("xf,xfgg->x", myomega, x3)
            temp8 += +2 * einsum("xfg,xfgkk->x", myms, x4)
            temp9 = (
                self.lambda_ * x3e_c1_mat - self.lambda_ * 1 / 3 * x3e_c2_mat + 0.5 * x2e_c_mat
            )
            return temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp_contr4

        def calculate_LK_derivs(
            indices,
            j_values,
            i_values,
            Asquaredderiv,
            cj_deriv,
            omega_deriv,
            mat,
            matmuj,
            const,
            temps,
        ):

            temp1, temp2, temp3, temp4, temp5, temp6 = temps[:6]

            retval = einsum("cx,x->xc", const[:, j_values], temp1)
            retval += einsum("cxm,xm->xc", matmuj[:, j_values], temp2)
            retval += einsum("cxkl,xkl->xc", mat[:, j_values], temp3)
            retval += einsum("cxk,xk->xc", omega_deriv[:, j_values], temp4)
            retval += einsum("cxkl,xkl->xc", Asquaredderiv[:, j_values], temp5)
            retval += einsum("cx,x->xc", cj_deriv[:, j_values], temp6)

            return retval

        def calculate_mu_derivs(indices, j_values, i_values, temps):
            temp7, temp8, temp9, temp_contr4 = temps
            x3 = np.conj(self.x3_temp[indices]) if conj else self.x3_temp[indices]
            mymat = self.matrices[j_values]
            myvec = self.vectors[j_values]
            zs = einsum("xak,xk->xa", mymat, myvec)
            omega_derivs = self.matrices_squared[j_values]
            cj_derivs = -4 * einsum("xak,xk->xa", omega_derivs, myvec)

            temp = einsum("xf,xaf->xa", temp7, mymat)
            temp += 2 * einsum("xaf,xf->xa", omega_derivs, temp_contr4)
            temp += 2 * einsum("xaf,xfgg->xa", omega_derivs, x3)
            temp += einsum("x,xa->xa", temp8, zs)
            temp += einsum("x,xa->xa", temp9, cj_derivs)

            return temp

        D_tensor_alt = np.zeros(
            (ng, ng - start_index_WFnew, nd * (nd + 3)), dtype=np.complex128
        )
        mKm_list = []
        mLm_list = []
        self.ab_iter = 0
        for a in range(nd):
            for b in range(a + 1):
                self.ab_iter += 1
                mLm_list.append(self.mLm[:, a, b])
                mKm_list.append(self.mKm[:, a, b])
        mLm_list = np.array(mLm_list)
        mKm_list = np.array(mKm_list)

        Asquared_derivs_I_list = einsum("cjkx,jxl->cjkl", self.IabLj_list, self.matrices)
        Asquared_derivs_I_list += einsum("jkx,cjxl->cjkl", self.matrices, self.IabLj_list)
        Asquared_derivs_K_list = einsum("cjkx,jxl->cjkl", self.Kab_list, self.matrices)
        Asquared_derivs_K_list += einsum("jkx,cjxl->cjkl", self.matrices, self.Kab_list)
        cj_derivs_I_list = einsum("cjkk->cj", self.IabLj_list, dtype=np.complex128)
        cj_derivs_I_list += -2 * einsum(
            "jk,cjkl,jl->cj", self.vectors, Asquared_derivs_I_list, self.vectors
        )
        cj_derivs_K_list = einsum("cjkk->cj", self.Kab_list, dtype=np.complex128)
        cj_derivs_K_list += -2 * einsum(
            "jk,cjkl,jl->cj", self.vectors, Asquared_derivs_K_list, self.vectors
        )
        omega_derivs_K_list = einsum("jk,cjkl->cjl", self.vectors, Asquared_derivs_K_list)
        omega_derivs_I_list = einsum("jk,cjkl->cjl", self.vectors, Asquared_derivs_I_list)

        if ic.size > 0:
            conj = True
            temps = calculate_temp_mats(ic, jvc, ivc, conj=conj)
            D_tensor_alt[ivc, indices_j_vc, : self.ab_iter] = calculate_LK_derivs(
                ic,
                jvc,
                ivc,
                Asquared_derivs_I_list,
                cj_derivs_I_list,
                omega_derivs_I_list,
                self.IabLj_list,
                self.IabLjmuj_list,
                mLm_list,
                temps,
            )
            D_tensor_alt[ivc, indices_j_vc, self.ab_iter : 2 * self.ab_iter] = (
                1j
                * calculate_LK_derivs(
                    ic,
                    jvc,
                    ivc,
                    Asquared_derivs_K_list,
                    cj_derivs_K_list,
                    omega_derivs_K_list,
                    self.Kab_list,
                    self.Kabmuj_list,
                    mKm_list,
                    temps,
                )
            )
            D_tensor_alt[ivc, indices_j_vc, nd * (nd + 1) : nd * (nd + 2)] = (
                calculate_mu_derivs(ic, jvc, ivc, temps[6:])
            )

        if inc.size > 0:
            conj = False
            temps = calculate_temp_mats(inc, jvnc, ivnc, conj=conj)
            D_tensor_alt[ivnc, indices_j_vnc, : self.ab_iter] = calculate_LK_derivs(
                inc,
                jvnc,
                ivnc,
                Asquared_derivs_I_list,
                cj_derivs_I_list,
                omega_derivs_I_list,
                self.IabLj_list,
                self.IabLjmuj_list,
                mLm_list,
                temps,
            )
            D_tensor_alt[ivnc, indices_j_vnc, self.ab_iter : 2 * self.ab_iter] = (
                1j
                * calculate_LK_derivs(
                    inc,
                    jvnc,
                    ivnc,
                    Asquared_derivs_K_list,
                    cj_derivs_K_list,
                    omega_derivs_K_list,
                    self.Kab_list,
                    self.Kabmuj_list,
                    mKm_list,
                    temps,
                )
            )
            D_tensor_alt[ivnc, indices_j_vnc, nd * (nd + 1) : nd * (nd + 2)] = (
                calculate_mu_derivs(inc, jvnc, ivnc, temps[6:])
            )
        D_tensor_alt[:, :, nd * (nd + 2) : nd * (nd + 3)] = (
            1j * D_tensor_alt[:, :, nd * (nd + 1) : nd * (nd + 2)]
        )

        return D_tensor_alt

    def calculate_potential_squared(self):
        potentialsq_temp = 0.25 * einsum("xiijj->x", self.x4_temp)
        potentialsq_temp += self.lambda_ * einsum(
            "xiijjj->x", self.x5_temp[:, :, :, :-1, :-1, 1:]
        )
        potentialsq_temp += (
            -self.lambda_ / 3 * einsum("xiijjj->x", self.x5_temp[:, :, :, 1:, 1:, 1:])
        )
        potentialsq_temp += (
            self.lambda_**2 / 9 * einsum("xiiijjj->x", self.x6_temp[:, 1:, 1:, 1:, 1:, 1:, 1:])
        )
        potentialsq_temp += self.lambda_**2 * einsum(
            "xiiijjj->x", self.x6_temp[:, :-1, :-1, 1:, :-1, :-1, 1:]
        )
        potentialsq_temp += (
            -2
            / 3
            * self.lambda_**2
            * einsum("xiiijjj->x", self.x6_temp[:, 1:, 1:, 1:, 0:-1, 0:-1, 1:])
        )

        self.potential_squared = np.zeros_like(self.overlap)
        self.potential_squared[self.myrow_indices, self.mycol_indices] = potentialsq_temp
        self.potential_squared[self.mycol_indices, self.myrow_indices] = np.conj(
            potentialsq_temp
        )
        return self.potential_squared

    def calculate_Hamiltonian(self):
        return self.calculate_potential() + self.calculate_kinetic_energy()

    def calculate_Hamiltonian_deriv(self, start_index_WFnew, include_kinetic=True):
        si = start_index_WFnew
        H_deriv = self.calculate_full_potential_deriv_matrix(start_index_WFnew)
        if include_kinetic:
            H_deriv += self.calculate_full_kinetic_energy_deriv(start_index_WFnew)
        if self.rank == 0:
            H_deriv_full = np.zeros_like(H_deriv)
        else:
            H_deriv_full = None
        self.comm.Reduce(
            [H_deriv, MPI.COMPLEX16], [H_deriv_full, MPI.COMPLEX16], op=MPI.SUM, root=0
        )
        if self.rank == 0:
            if include_kinetic:
                H_deriv_full += -0.5 * einsum(
                    "ij,jk->ijk",
                    self.potential[:, si:] + self.kinetic[:, si:],
                    self.fodmd[si:, :],
                )
            else:
                H_deriv_full += -0.5 * einsum(
                    "ij,jk->ijk", self.potential[:, si:], self.fodmd[si:, :]
                )
        return H_deriv_full

    def calculate_Hamiltonian_squared(self):
        return (
            self.calculate_kinetic_energy_squared()
            + self.calculate_potential_squared()
            + self.calculate_potential_times_kinetic_energy()
        )

    def calculate_Hamiltonian_squared_deriv(self, start_index_WFnew, include_kinetic=True):
        si = start_index_WFnew
        H2_deriv = self.calculate_full_potential_squared_deriv_matrix(start_index_WFnew)
        if include_kinetic:
            H2_deriv += self.calculate_full_kinetic_energy_squared_deriv(
                start_index_WFnew
            )  # Takes second most time
            H2_deriv += self.calculate_full_potential_times_kinetic_deriv(
                start_index_WFnew
            )  # Takes most time
        if self.rank == 0:
            H2_deriv_full = np.zeros_like(H2_deriv)
        else:
            H2_deriv_full = None
        self.comm.Reduce(
            [H2_deriv, MPI.COMPLEX16], [H2_deriv_full, MPI.COMPLEX16], op=MPI.SUM, root=0
        )
        if self.rank == 0:
            if include_kinetic:
                H2_deriv_full += -0.5 * einsum(
                    "ij,jk->ijk",
                    self.potential_squared[:, si:]
                    + self.kinetic_energy_squared[:, si:]
                    + self.potential_times_kinetic[:, si:],
                    self.fodmd[si:, :],
                )
            else:
                H2_deriv_full += -0.5 * einsum(
                    "ij,jk->ijk", self.potential_squared[:, si:], self.fodmd[si:, :]
                )
        return H2_deriv_full

    def calculate_S_mat(self, start_index_WFnew, h):
        if h == 0:
            self.S_mat = self.overlap[start_index_WFnew:, start_index_WFnew:]
        else:
            self.S_mat = (self.overlap + h**2 / 4 * self.Hsquared)[
                start_index_WFnew:, start_index_WFnew:
            ]
        return self.S_mat

    def calculate_rho_vec(self, start_index_WFnew, h):
        if h == 0:
            rho_mat = self.overlap[
                :start_index_WFnew, start_index_WFnew:
            ]  # Upper right corner
        else:
            rho_mat = (self.overlap - h**2 / 4 * self.Hsquared + 1j * h * self.H)[
                :start_index_WFnew, start_index_WFnew:
            ]  # Upper right corner
        self.rho_mat = rho_mat
        self.rho_vec = einsum("i,ij->j", self.coefficients[:start_index_WFnew], conj(rho_mat))
        return self.rho_vec

    def rothe_optimal_c(self, start_index_WFnew, h):
        rho = self.calculate_rho_vec(start_index_WFnew, h)
        S = self.calculate_S_mat(start_index_WFnew, h)
        self.Smat = S
        self.opt_c = solve(S, rho)
        self.rho = rho

    def rothe_optimal_c_normalization(self, start_index_WFnew, h):
        return self.rothe_optimal_c(start_index_WFnew, h)

    def euler_optimal_c(self, si, h):
        S_mat = (self.overlap)[si:, si:]
        rho_mat = (self.overlap + 1j * h * self.H)[:si, si:]  # Upper right corner
        rho_vec = einsum("i,ij->j", self.coefficients[:si], conj(rho_mat))
        return solve(S_mat, rho_vec)

    def calculate_euler_error(self, start_index_WFnew, h):
        si = start_index_WFnew
        self.rothe_optimal_c_normalization(si, h=h)
        new_lin_params = self.euler_optimal_c(si, h)
        old_lin_params = self.coefficients[:si]
        O = self.overlap
        H = self.calculate_Hamiltonian()
        H2 = self.calculate_Hamiltonian_squared()
        inner_ovlp = np.conj(new_lin_params).T @ O[si:, si:] @ new_lin_params
        old_old_term = (
            np.conj(old_lin_params).T @ (O[:si, :si] + h**2 * H2[:si, :si]) @ old_lin_params
        )
        cross_terms = -2 * (
            np.conj(new_lin_params).T @ (O[si:, :si] - h * 1j * H[si:, :si]) @ old_lin_params
        )
        return real(inner_ovlp + old_old_term + cross_terms)

    def calculate_euler_deriv(self, si, h):
        if self.rank == 0:
            opt_c = self.euler_optimal_c(si, h)
            self.setUpDerivs(si, Hsquared=False)
            S_deriv_vecs = self.ovlp_deriv[si:]
            rho_deriv_vecs = self.ovlp_deriv + 1j * self.H_deriv * h
            rho_indices = einsum(
                "abc,a->bc", rho_deriv_vecs[:si], conj(self.coefficients[:si])
            )
            opt_c_rep = np.repeat(
                opt_c[:, np.newaxis], self.num_dimensions * (self.num_dimensions + 3), axis=1
            )
            rho_optc_prods = real((rho_indices) * opt_c_rep + conj((rho_indices) * opt_c_rep))
            optc_smat_prod = einsum("a,aij,i->ij", conj(opt_c), S_deriv_vecs, opt_c) + einsum(
                "i,aij,a->ij", conj(opt_c), conj(S_deriv_vecs), opt_c
            )
            jacobian = real(optc_smat_prod - rho_optc_prods).flatten()
            return jacobian

    def rothe_jacobian(self, start_index_WFnew, h, include_kinetic=True):
        self.setUpDerivs(start_index_WFnew, include_kinetic=include_kinetic)
        if self.rank == 0:
            self.rothe_optimal_c_normalization(start_index_WFnew, h)
            S_deriv_vecs = self.calculate_all_S_deriv_vecs(start_index_WFnew, h)
            rho_indices = self.calculate_all_rho_derivs(start_index_WFnew, h)
            opt_c = self.opt_c
            opt_c_rep = np.repeat(
                opt_c[:, np.newaxis], self.num_dimensions * (self.num_dimensions + 3), axis=1
            )
            rho_optc_prods = real((rho_indices) * opt_c_rep + conj((rho_indices) * opt_c_rep))
            optc_smat_prod = einsum("a,aij,i->ij", conj(opt_c), S_deriv_vecs, opt_c) + einsum(
                "i,aij,a->ij", conj(opt_c), conj(S_deriv_vecs), opt_c
            )
            jacobian = real(optc_smat_prod - rho_optc_prods).flatten()
            return jacobian
        else:
            return None

    def setupMatrices(self):
        if self.h > 0:
            self.Hsquared = self.calculate_Hamiltonian_squared()
            self.H = self.calculate_Hamiltonian()
        else:
            self.Hsquared = np.zeros_like(self.overlap)
            self.H = np.zeros_like(self.overlap)
        for indices in self.ij_zero_indices:
            i, j = indices
            self.H[i, j] = self.Hsquared[i, j] = 0

    def setupMatrices_noKinetic(self):
        """Sets up the matrices when the kinetic energy is not included in the propagation"""
        """This function needs to be called after the WF is initialized"""
        self.H = self.potential
        self.myH = self.mypotential
        self.Hsquared = self.potential_squared
        self.myHsquared = self.mypotential_squared

    def setUpDerivs(self, start_index_WFnew, Hsquared=True, include_kinetic=True):
        self.ovlp_deriv = self.calculate_full_overlap_deriv_matrix(start_index_WFnew)
        if self.h > 0:
            if Hsquared:
                self.Hsquared_deriv = self.calculate_Hamiltonian_squared_deriv(
                    start_index_WFnew, include_kinetic
                )
            self.H_deriv = self.calculate_Hamiltonian_deriv(start_index_WFnew, include_kinetic)
        else:
            self.H_deriv = np.zeros_like(self.ovlp_deriv)
            self.Hsquared_deriv = np.zeros_like(self.ovlp_deriv)

    def calculate_all_S_deriv_vecs(self, start_index_WFnew, h):
        S_deriv_vecs = self.ovlp_deriv + self.Hsquared_deriv * h**2 / 4
        return S_deriv_vecs[start_index_WFnew:]

    def calculate_all_rho_derivs(self, start_index_WFnew, h):
        rho_deriv_vecs = (
            self.ovlp_deriv - h**2 / 4 * self.Hsquared_deriv + 1j * self.H_deriv * h
        )
        return einsum(
            "abc,a->bc",
            rho_deriv_vecs[:start_index_WFnew],
            conj(self.coefficients[:start_index_WFnew]),
        )

    def calculate_overlap_tildePhim(
        self, start_index_WFnew, h
    ):  # Calculate <\tilde \Phi_m|\tilde \Phi_m>
        S = self.overlap[
            :start_index_WFnew, :start_index_WFnew
        ].copy()  # Copy to avoid mutating the original overlap block during S update.
        contribution = self.Hsquared[:start_index_WFnew, :start_index_WFnew]
        S += 0.25 * h**2 * contribution
        return (
            conj(self.coefficients[:start_index_WFnew]).T
            @ S
            @ self.coefficients[:start_index_WFnew]
        )

    def rothe_error(self, start_index_WFnew, h):
        if self.rank == 0:
            overlap_term = self.calculate_overlap_tildePhim(start_index_WFnew, h)
            self.overlap_term = overlap_term
            self.rothe_optimal_c_normalization(start_index_WFnew, h)
            opt_c = self.opt_c
            projection_term = (
                2 * conj(self.rho).T @ opt_c - conj(self.opt_c).T @ self.Smat @ opt_c
            )
            difference = overlap_term - projection_term
            if np.isnan(difference):
                return 100
            else:
                return abs(real(difference))  # numerical noise makes this not exactly real
