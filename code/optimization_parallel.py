import os
import time

# import multiprocessing
import h5py
import numpy as np
import scipy
from mpi4py import MPI
from numpy import arctanh, conj, cosh, log, real, sqrt, tanh
from numpy.linalg import inv
from scipy.optimize import minimize
from utils_heller import propagate_kinetic_analytical
from WF_HenonHeiles_parallel import WF


class ConvergedError(Exception):
    pass


class TimeEvolution:
    def __init__(
        self,
        wave_function,
        h,
        T,
        epsilon,
        t0,
        error_t0,
        filename,
        dimension,
        max_num_Gaussians=20,
        norm=1.0,
        energy=0.0,
    ):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.mask_lincoeff = [
            0.8865587802866912,
            -13.91445103929891,
            78.58424070320325,
            -105.23506996256583,
            -5.7783020569168,
            93.33308512709664,
            -22.677314700584247,
            -71.12872075341511,
            53.47552955091851,
            25.68122441131635,
            -65.63471326554463,
            42.15941658394149,
            8.056986128893868,
            -45.845105790207526,
            56.248886712777676,
            -45.14073539380479,
            26.356604035710916,
            -11.018158140453579,
            2.985107837486339,
            -0.395107357239624,
        ]
        self.mask_nonlincoeff = [
            0.0067379471029344745,
            0.008766285770907432,
            0.011405219508509397,
            0.014838556721848828,
            0.01930543876899837,
            0.025116996050942624,
            0.032678018901526475,
            0.04251515254382658,
            0.05531357941370039,
            0.07196474390737227,
            0.09362844384567327,
            0.12181361360477637,
            0.15848342544522337,
            0.20619202963199804,
            0.26826245756591804,
            0.34901807946651336,
            0.4540837384217124,
            0.5907775357109573,
            0.7686205562022081,
            1.0000000374130311,
        ]
        self.max_num_Gaussians = max_num_Gaussians
        self.wf = wave_function
        self.lambda_ = wave_function.lambda_
        self.filename = filename
        self.h = h
        self.direction = None
        self.T = T
        self.epsilon = epsilon
        self.maxerr = epsilon / T * h
        self.t0 = t0
        self.t = t0
        self.error_t0 = error_t0
        self.accumulated_error = error_t0
        self.params_old_nonlin = self.wf.nonlin_params
        self.params_old_lin = self.wf.lin_params
        self.ng_initial = len(self.params_old_lin)
        self.params_oldold_nonlin = np.zeros_like(self.params_old_nonlin)
        self.dimension = dimension
        if self.dimension == 2:
            self.energy_t0 = 5.596282666666666
        elif self.dimension == 4:
            self.energy_t0 = 11.78884800
        elif self.dimension == 3:
            self.energy_t0 = 8.692565333333334
        elif self.dimension == 6:
            self.energy_t0 = 17.981413333333343
        elif self.dimension == 5:
            self.energy_t0 = 14.885130666666672

        else:
            self.energy_t0 = 1
        if self.t <= 0:
            self.energy = self.energy_t0
            self.normalization = 1
        else:
            self.energy = energy
            self.normalization = norm
        if t0 < 0.5:
            self.removed_at_prev_timestep = 0
        else:
            self.removed_at_prev_timestep = 2
        self.start_index = len(self.params_old_lin)
        self.grad0_eps = 1e-15
        self.prev_hess = None
        if self.rank == 0:
            if not os.path.isfile(filename):
                with h5py.File(filename, "w") as data_file:
                    data_file.create_dataset(
                        "parameters_t=%.3f" % self.t, data=self.wf.nonlin_params
                    )
                    data_file.create_dataset(
                        "coefficients_t=%.3f" % self.t, data=self.wf.nonlin_params
                    )
                    new_times = [t0]
                    accumulated_error = [0]
                    normalizations = [1]
                    energies = [self.energy_t0]
                    data_file.create_dataset("times", data=new_times)
                    data_file.create_dataset("rothe_error", data=accumulated_error)
                    data_file.create_dataset("energies", data=energies)
                    data_file.create_dataset("normalizations", data=normalizations)
            else:
                with h5py.File(filename, "r+") as data_file:
                    times_read = np.array(data_file["times"])
                    index_to_delete_after = np.argmin(abs(times_read - t0))
                    if t0 > (times_read[-1] + 1e-5):
                        print(
                            "You cannot start your run from this time, as no reference data exists"
                        )
                        raise IndexError
                    else:
                        times_new = np.array(data_file["times"])[: index_to_delete_after + 1]
                        re_new = np.array(data_file["rothe_error"])[
                            : index_to_delete_after + 1
                        ]
                        norm_new = np.array(data_file["normalizations"])[
                            : index_to_delete_after + 1
                        ]
                        energies_new = np.array(data_file["energies"])[
                            : index_to_delete_after + 1
                        ]
                        del data_file["rothe_error"]
                        del data_file["times"]
                        del data_file["energies"]
                        del data_file["normalizations"]
                        data_file.create_dataset("times", data=times_new)
                        data_file.create_dataset("rothe_error", data=re_new)
                        data_file.create_dataset("energies", data=energies_new)
                        data_file.create_dataset("normalizations", data=norm_new)
                        if self.t0 >= self.h - 1e-10:
                            self.params_oldold_nonlin = np.array(
                                data_file["parameters_t=%.3f" % (self.t0 - self.h)]
                            )

    def rothe_gradient(self, nonlin_params):
        start_index = len(self.params_old_lin)
        start = time.time()
        deriv = self.WF_opt.rothe_jacobian(
            start_index_WFnew=start_index, h=self.h, include_kinetic=True
        )
        end = time.time()
        deriv = self.comm.bcast(deriv, root=0)
        return deriv

    def rothe_gradient_no_kinetic(self, nonlin_params):
        start_index = len(self.params_old_lin)
        start = time.time()
        deriv = self.WF_opt.rothe_jacobian(
            start_index_WFnew=start_index, h=self.h, include_kinetic=False
        )
        end = time.time()
        deriv = self.comm.bcast(deriv, root=0)
        return deriv

    def rothe_error(self, nonlin_params, calculate_Gradient=True):
        n = self.dimension
        num_basis_functions = len(nonlin_params) // (n * (n + 3))
        start_index = len(self.params_old_lin)
        full_nonlin = np.concatenate(
            (np.array(self.params_old_nonlin).flatten(), nonlin_params)
        )
        full_nonlin = np.reshape(full_nonlin, (-1, n * (n + 3)))
        full_lin = np.concatenate((self.params_old_lin, np.zeros(num_basis_functions)))

        optimization_WF = WF(full_nonlin, full_lin, lambda_=self.lambda_)
        self.WF_opt = optimization_WF
        RE = optimization_WF.rothe_error(start_index_WFnew=start_index, h=self.h)
        """Send Rothe error to all processes.
        The reason for this is that all processes need to know the RE in order for the optimization to work.
        """

        RE = self.comm.bcast(RE, root=0)
        return RE

    def rothe_error_no_kinetic(self, nonlin_params, calculate_Gradient=True, debug=False):
        n = self.dimension
        num_basis_functions = len(nonlin_params) // (n * (n + 3))
        start_index = len(self.params_old_lin)
        full_nonlin = np.concatenate(
            (np.array(self.params_old_nonlin).flatten(), nonlin_params)
        )
        full_nonlin = np.reshape(full_nonlin, (-1, n * (n + 3)))
        full_lin = np.concatenate((self.params_old_lin, np.zeros(num_basis_functions)))
        if debug:
            print(full_nonlin)
            print(full_lin)
        optimization_WF = WF(full_nonlin, full_lin, lambda_=self.lambda_)
        optimization_WF.setupMatrices_noKinetic()
        self.WF_opt = optimization_WF
        RE = optimization_WF.rothe_error(start_index_WFnew=start_index, h=self.h)
        """Send Rothe error to all processes.
        The reason for this is that all processes need to know the RE in order for the optimization to work.
        """

        RE = self.comm.bcast(RE, root=0)
        return RE

    def overlap_refit_error(self, nonlin_params):
        n = self.dimension
        num_basis_functions = len(nonlin_params) // (n * (n + 3))

        start_index = len(self.params_old_lin)
        full_nonlin = np.concatenate(
            (np.array(self.params_old_nonlin).flatten(), nonlin_params.flatten())
        )
        full_nonlin = np.reshape(full_nonlin, (-1, n * (n + 3)))
        full_lin = np.concatenate((self.params_old_lin, np.zeros(num_basis_functions)))
        optimization_WF = WF(
            full_nonlin, full_lin, lambda_=self.lambda_, calculate_Gradient=False, h=0
        )
        self.WF_opt = optimization_WF
        RE = optimization_WF.rothe_error(start_index_WFnew=start_index, h=0)

        return RE

    def overlap_refit_gradient(self, nonlin_params):
        start_index = len(self.params_old_lin)
        deriv = self.WF_opt.rothe_jacobian(start_index_WFnew=start_index, h=0)
        del self.WF_opt
        return deriv

    def calculate_mask_coefficients(self):
        n = self.dimension
        n = self.dimension
        num_basis_functions = len(self.params_old_nonlin.flatten()) // (n * (n + 3))
        mask_nonlin = []
        mask_lin = []
        for i in range(num_basis_functions):
            params_i = np.reshape(self.params_old_nonlin, (-1, n * (n + 3)))[i]
            L_coefs = params_i[: int(1 / 2 * n * (n + 1))]
            K_coefs = params_i[int(1 / 2 * n * (n + 1)) : int(n * (n + 1))]
            L = np.zeros((n, n))
            K = np.zeros((n, n))
            L[np.tril_indices(n)] = L_coefs
            K[np.tril_indices(n)] = K_coefs
            oldmat = L @ L.T + 1j * (K + K.T)
            old_shift = params_i[n * (n + 1) : n * (n + 1) + n] + 1j * (
                params_i[int(n * (n + 1)) + n :]
            )
            sum_matrix = oldmat + np.conj(oldmat)
            invert = np.linalg.inv(sum_matrix)
            bvec = old_shift.T @ oldmat + conj(old_shift.T @ oldmat)
            vmip = np.einsum("a,ab,b->", old_shift, oldmat, old_shift)
            diagonal_exponent = bvec.T @ invert @ bvec - vmip - conj(vmip)
            eigs = np.prod(sqrt(np.linalg.eig(sum_matrix)[0]))
            normalization_old = 1 / eigs / 2
            # normalization_old_exp=*diagonal_exponent
            old_at_old = oldmat @ old_shift
            for j in range(len(self.mask_nonlincoeff)):
                new_params = params_i.copy()
                newmat = oldmat.copy() + np.eye(n) * self.mask_nonlincoeff[j]
                newL = np.linalg.cholesky(np.real(newmat))
                updated_nonlin_L = newL[np.tril_indices(n)]
                newmat_inv = inv(newmat)

                updated_nonlin_shift = newmat_inv @ old_at_old
                new_params[: int(1 / 2 * n * (n + 1))] = updated_nonlin_L
                new_params[n * (n + 1) : n * (n + 1) + n] = np.real(updated_nonlin_shift)
                new_params[n * (n + 1) + n :] = np.imag(updated_nonlin_shift)
                sum_matrixN = newmat + np.conj(newmat)
                invertN = np.linalg.inv(sum_matrixN)
                bvecN = updated_nonlin_shift.T @ newmat + conj(updated_nonlin_shift.T @ newmat)
                vmipN = np.einsum(
                    "a,ab,b->", updated_nonlin_shift, newmat, updated_nonlin_shift
                )
                diagonal_exponentN = bvecN.T @ invertN @ bvecN - vmipN - conj(vmipN)
                eigsN = np.prod(sqrt(np.linalg.eig(sum_matrixN)[0]))
                normalization_new = 1 / eigsN / 2
                change_in_c = np.exp(
                    -old_shift.T @ old_at_old
                    + old_at_old.T @ newmat_inv @ old_at_old
                    - 0.5 * diagonal_exponent
                    + 0.5 * diagonal_exponentN
                    + 0.5 * log(normalization_new)
                    - 0.5 * log(normalization_old)
                )
                mask_nonlin.append(new_params)
                mask_lin.append(change_in_c * self.mask_lincoeff[j] * self.params_old_lin[i])
        self.mask_nonlin = np.array(mask_nonlin)
        self.mask_lin = np.array(mask_lin)

    def make_mask_function(self, nonlin_params):
        n = self.dimension
        num_basis_functions = len(nonlin_params) // (n * (n + 3))
        mask_nonlin_params = self.mask_nonlin
        full_nonlin = np.concatenate((mask_nonlin_params.flatten(), nonlin_params))
        full_nonlin = np.reshape(full_nonlin, (-1, n * (n + 3)))
        mask_lin_params = self.mask_lin
        full_lin = np.concatenate((mask_lin_params, np.zeros(num_basis_functions)))
        optimization_WF = WF(
            full_nonlin,
            full_lin,
            lambda_=self.lambda_,
            calculate_Gradient=False,
            h=0.00,
            onlyX1X2=True,
        )
        error_calculate_c = np.concatenate(
            (mask_lin_params.copy(), -self.params_old_lin.copy())
        )
        errorx = abs(
            np.conj(error_calculate_c.T) @ optimization_WF.overlap @ error_calculate_c
        )
        return optimization_WF, errorx

    def rothe_error_mask(self, nonlin_params):
        self.WF_opt, nonopt_err = self.make_mask_function(nonlin_params)
        start_index = len(self.mask_lin)
        RE = self.WF_opt.rothe_error(start_index_WFnew=start_index, h=0.00)

        self.WF_opt.rothe_optimal_c_normalization(start_index, h=0)
        RE = self.comm.bcast(RE, root=0)
        return RE

    def rothe_error_mask_update(self, nonlin_params, onlyX1X2=True):
        start_index = len(self.mask_lin)
        num_gauss_total = self.WF_opt.num_gaussians
        num_gauss_optimization = len(nonlin_params) // (self.dimension * (self.dimension + 3))
        indices = np.arange(num_gauss_total - num_gauss_optimization, num_gauss_total)
        nonlin_reshaped = np.reshape(
            nonlin_params, (-1, self.dimension * (self.dimension + 3))
        )
        self.WF_opt.update_overlap_and_overlap_derivs(indices, nonlin_reshaped)
        RE = self.WF_opt.rothe_error(start_index_WFnew=start_index, h=0)
        self.WF_opt.rothe_optimal_c_normalization(start_index, h=0)
        RE = self.comm.bcast(RE, root=0)
        return RE

    def rothe_gradient_mask(self, nonlin_params):
        start_index = len(self.mask_lin)
        deriv = self.WF_opt.rothe_jacobian(start_index_WFnew=start_index, h=0.00)
        deriv = self.comm.bcast(deriv, root=0)
        return deriv

    def get_start_parameters(self):
        old = np.array(self.params_old_nonlin.copy()).flatten()
        oldold = np.array(self.params_oldold_nonlin.copy()).flatten()
        old = self.comm.bcast(old, root=0)
        oldold = self.comm.bcast(oldold, root=0)
        if old.shape != oldold.shape:
            return old.flatten()
        direction = old.flatten() - oldold.flatten()

        error_set = []
        alphas = np.linspace(0, 1, 6)
        for alpha in alphas:
            val = self.rothe_error(old + alpha * direction, calculate_Gradient=False)
            if val > 0:
                error_set.append(sqrt(val))
            else:
                error_set.append(1e100)
        error_set = np.nan_to_num(error_set, nan=1e15, posinf=1e15, neginf=1e15)
        i = np.argmin(error_set)
        return old + direction * alphas[i]

    def get_start_parameters_noKinetic(self, startGuess, direction=None):
        if direction is None:
            return startGuess

        error_set = []
        alphas = np.linspace(0, 1, 6)
        for alpha in alphas:
            val = self.rothe_error(startGuess + alpha * direction, calculate_Gradient=False)
            if val > 0:
                error_set.append(sqrt(val))
            else:
                error_set.append(1e100)
        error_set = np.nan_to_num(error_set, nan=1e15, posinf=1e15, neginf=1e15)
        i = np.argmin(error_set)
        return startGuess + direction * alphas[i]

    def minimize_transformed_bonds(
        self,
        error_function,
        gradient,
        maxiter,
        start_params,
        args,
        gtol=0,
        multi_bonds=5e-1,
        printx=False,
    ):
        """
        Minimizes with min_max bonds as described in https://lmfit.github.io/lmfit-py/bounds.html
        """

        def transform_params(untransformed_params):
            return arctanh(2 * (untransformed_params - mins) / (maxs - mins) - 1)
            # return arcsin(2*(untransformed_params-mins)/(maxs-mins)-1)

        def untransform_params(transformed_params):
            return mins + (maxs - mins) / 2 * (1 + tanh(transformed_params))
            # return mins+(maxs-mins)/2*(1+sin(transformed_params))

        def chainrule_params(transformed_params):
            returnval = 0.5 * (maxs - mins) / (cosh(transformed_params) ** 2)
            return returnval
            # return 0.5*(maxs-mins)*cos(transformed_params)

        def transformed_error(transformed_params):
            error = error_function(untransform_params(transformed_params))
            return error

        def transformed_gradient(transformed_params):
            orig_grad = gradient(untransform_params(transformed_params))
            chainrule_grad = chainrule_params(transformed_params)
            grad = orig_grad * chainrule_grad
            if np.isnan(sum(grad)):
                """
                print("gradient has nan")
                print(grad)
                print(transformed_error(transformed_params))
                if np.isnan(np.sum(orig_grad)):
                    print("Original gradient has nan...")
                """
                return np.nan_to_num(grad)
            return grad

        if printx and self.rank == 0:
            print(start_params)
        dp = multi_bonds * np.ones(
            len(start_params)
        )  # Percentage (times 100) how much the parameters are alowed to change compared to previous time step
        range_notMu = [0.5] * (self.dimension * (self.dimension + 1))
        range_mu = [0.5] * (self.dimension * 2)
        rangex = range_notMu + range_mu
        num_WF = len(start_params) // len(rangex)
        rangex = np.array(rangex * num_WF)
        mins = start_params - rangex - dp * abs(start_params)
        maxs = start_params + rangex + dp * abs(start_params)
        transformed_params = transform_params(start_params)
        transformed_params = np.real(np.nan_to_num(transformed_params))

        start = time.time()
        grad0 = transformed_gradient(transformed_params)
        dg = self.grad0_eps * np.ones(len(grad0))
        hess_inv0 = np.diag(1 / abs(grad0 + dg))
        self.numiter = 0
        f_storage = []

        def callback_func(intermediate_result: scipy.optimize.OptimizeResult):
            xval = intermediate_result.x
            fun = intermediate_result.fun
            self.transformed_sol = xval
            self.minval = fun
            f_storage.append(sqrt(fun))
            compareto = 20
            if self.numiter >= 30:  # At least 50 iterations
                if (
                    f_storage[-1] / f_storage[-compareto - 1] > 0.9995
                    and f_storage[-1] / f_storage[-compareto - 1] < 1
                ):
                    if self.rank == 0:
                        print("The function is not decreasing anymore.")
                    self.transformed_sol = xval

                    raise ConvergedError
            self.numiter += 1

        try:
            solver = minimize(
                transformed_error,
                transformed_params,
                method="BFGS",
                jac=transformed_gradient,
                options={
                    "maxiter": maxiter,
                    "gtol": gtol,
                    "hess_inv0": hess_inv0,
                    "c1": 1e-4,
                    "c2": 0.9,
                },
                callback=callback_func,
            )
            self.transformed_sol = solver.x
            self.minval = solver.fun  # Not really fun, I am lying here
            self.prev_hess = solver.hess_inv
        except ConvergedError:
            """
            This means that the callback function tells me that the function is not decreasing anymore.
            The important values are updated in the callback function (self.transformed_sol and self.minval)
            so that this "error" can safely be ignored.
            """
            self.prev_hess = None
            pass
        end = time.time()
        if self.rank == 0:
            print(
                "  REG: Time to optimize: %.3f seconds, niter : %d"
                % (end - start, self.numiter)
            )
        return untransform_params(self.transformed_sol), self.numiter, self.minval

    def time_evolve(self):
        self.removed_at_prev_timestep += 1
        if self.rank == 0:
            print("Start time t0=%.3f" % self.t)
        nonlin_params_initial = np.array(self.params_old_nonlin.copy()).flatten()
        if self.removed_at_prev_timestep >= 0:
            nonlin_params_initial = self.get_start_parameters()

        RE = sqrt(self.rothe_error(nonlin_params_initial))
        grad0 = self.rothe_gradient(nonlin_params_initial)
        if self.rank == 0:
            print("Initial Rothe error: %f" % RE)
            # print("Initial Rothe gradient:",np.linalg.norm(grad0))
        maxiter = 300

        opt_func = self.rothe_error

        dg = self.grad0_eps * np.ones(len(grad0))
        hess_inv0 = np.diag(1 / abs(grad0 + dg))
        self.niterino = 0
        self.counterino = 0
        gtol = 1e-9
        if self.t < 0.1 or self.removed_at_prev_timestep < 1:
            start = time.time()
            solver = minimize(
                opt_func,
                nonlin_params_initial,
                method="BFGS",
                jac=self.rothe_gradient,
                options={"maxiter": maxiter, "gtol": gtol, "hess_inv0": hess_inv0},
            )
            best_nonlin_params = solver.x
            niter = solver.nit
            end = time.time()
            if self.rank == 0:
                print(
                    "UNREG: Time to optimize: %f; number of iterations: %d"
                    % (end - start, niter)
                )
            rotheerror = solver.fun

        else:

            best_nonlin_params, niter, rotheerror = self.minimize_transformed_bonds(
                opt_func,
                self.rothe_gradient,
                maxiter=maxiter,
                start_params=nonlin_params_initial,
                args=None,
                gtol=gtol,
                multi_bonds=3,
            )

        end = time.time()
        if self.rank == 0:
            print("Final Rothe error: %f/%f" % ((sqrt(rotheerror), self.maxerr)))

        self.comm.bcast(
            best_nonlin_params, root=0
        )  # Just to make sure that each thread has the same parameters?

        self.new_params = best_nonlin_params
        n = self.dimension
        divisible_by_5_timestep = int(np.round(self.t / self.h)) % 5 == 0

        self.accumulated_error += np.sqrt(rotheerror)
        num_basis_functions = len(best_nonlin_params) // (n * (n + 3))
        full_nonlin = np.concatenate(
            (np.array(self.params_old_nonlin).flatten(), best_nonlin_params)
        )
        start_index = len(self.params_old_lin)
        full_nonlin = np.reshape(full_nonlin, (-1, n * (n + 3)))
        full_lin = np.concatenate((self.params_old_lin, np.zeros(num_basis_functions)))
        optimization_WF = WF(full_nonlin, full_lin, lambda_=self.lambda_)
        optimization_WF.rothe_optimal_c_normalization(self.start_index, h=self.h)
        optimization_WF.rothe_error(start_index_WFnew=start_index, h=self.h)
        best_lin_params = optimization_WF.opt_c
        rv = rho_vec = optimization_WF.rho
        Stilde = Stil = optimization_WF.Smat
        S = optimization_WF.overlap[len(rv) :, len(rv) :]
        H = optimization_WF.H[len(rv) :, len(rv) :]
        if self.rank == 0:
            print("Entering conservation")
            ovlp_t = optimization_WF.overlap_term

            def optimization_in_c(cfull):
                rec, conj_c = cfull[: len(cfull) // 2], cfull[len(cfull) // 2 :]
                c = rec + 1j * conj_c
                return real(ovlp_t - (2 * np.conj(rv).T @ c - np.conj(c).T @ Stil @ c))

            def deriv_in_c(cfull):
                halflen = len(cfull) // 2
                rec, conj_c = cfull[:halflen], cfull[halflen:]
                c = rec + 1j * conj_c
                derivs = np.zeros_like(cfull, dtype=complex)

                derivs[:halflen] = (
                    -2 * np.conj(rv[:halflen])
                    + 2 * np.real(Stilde[:halflen]).T @ rec
                    + 1j * (Stilde[:halflen] - Stilde[:halflen].T) @ conj_c
                )

                # For the second half of derivs
                derivs[halflen:] = (
                    -2j * np.conj(rv[:halflen])
                    + 2 * np.real(Stilde[:halflen]) @ conj_c
                    + 1j * (Stilde[:halflen].T - Stilde[:halflen]) @ rec
                )
                return np.real(derivs)

            def constraint1(cfull):
                rec, conj_c = cfull[: len(cfull) // 2], cfull[len(cfull) // 2 :]
                c = rec + 1j * conj_c
                return real(np.conj(c) @ S @ c) - self.normalization

            c_init = np.concatenate((np.real(best_lin_params), np.imag(best_lin_params)))

            def constraint2(cfull):
                rec, conj_c = cfull[: len(cfull) // 2], cfull[len(cfull) // 2 :]
                c = rec + 1j * conj_c
                return real(np.conj(c) @ H @ c) - self.energy

            print(sqrt(optimization_in_c(c_init)))
            solver = minimize(
                optimization_in_c,
                c_init,
                jac=deriv_in_c,
                constraints=[
                    {"type": "eq", "fun": constraint1},
                    {"type": "eq", "fun": constraint2},
                ],
                options={"maxiter": 200, "ftol": 1e-10},
            )
            c_new = solver.x[: len(solver.x) // 2] + 1j * solver.x[len(solver.x) // 2 :]
            best_lin_params = c_new
        self.comm.bcast(best_lin_params, root=0)  # Share with other nodes

        self.params_oldold_nonlin = self.params_old_nonlin
        self.params_old_lin = best_lin_params
        self.params_old_nonlin = best_nonlin_params
        if self.rank == 0:
            print(
                "Cumulative Rothe error: %f. Estimated (cum): %f, Estimated (init): %f"
                % (
                    self.accumulated_error,
                    self.accumulated_error / (self.t + self.h) * 100,
                    np.sqrt(rotheerror) / self.h * 100,
                )
            )
            print(best_nonlin_params.shape, best_lin_params.shape)
        new_WF = WF(
            np.reshape(best_nonlin_params, (-1, n * (n + 3))),
            best_lin_params,
            lambda_=self.lambda_,
        )

        normalization = abs(
            np.conj(best_lin_params).T @ new_WF.overlap @ np.array(best_lin_params)
        )
        energy = abs(np.conj(best_lin_params).T @ new_WF.H @ np.array(best_lin_params))
        dfo = np.array(
            new_WF.calculate_gaussian_distances_from_origo()
        )  # Distances from origo

        if self.rank == 0:
            print("Normalization: %.10f, Energy: %.10f" % ((normalization), (energy)))
            # print("Ovlerpa matrix:")
            # print(np.array2string(abs(new_WF.overlap), precision=1, suppress_small=False,formatter={'float_kind':lambda x: f"{x:.1e}"}))
            # print("Number of small coefficients: %d"%(np.sum(abs(new_WF.overlap) < -1)//2))
        mask_err = 0
        mask_start = 0
        if divisible_by_5_timestep and self.t >= mask_start:
            self.calculate_mask_coefficients()
            # trash,mask_err=self.make_mask_function(best_nonlin_params)
            mask_err = self.rothe_error_mask(best_nonlin_params)
            # self.rothe_error_mask_update(best_nonlin_params)
            # sys.exit(0)
        if sqrt(mask_err) > 1e-3:

            grad0 = self.rothe_gradient_mask(best_nonlin_params)
            dg = self.grad0_eps * np.ones(len(grad0))
            hess_inv0 = np.diag(1 / abs(grad0 + dg))
            start = time.time()
            func = self.rothe_error_mask_update
            start = time.time()
            solver = minimize(
                func,
                best_nonlin_params,
                method="BFGS",
                jac=self.rothe_gradient_mask,
                options={"maxiter": 100, "gtol": 5e-7},
            )
            end = time.time()
            best_nonlin_params = solver.x
            RE = sqrt(func(best_nonlin_params))
            if self.rank == 0:
                print(best_nonlin_params)
                print(RE)
                print("Time to optimize mask: %.3f" % (end - start))
            self.new_params = best_nonlin_params
            n = self.dimension

            self.WF_opt, err = self.make_mask_function(best_nonlin_params)
            start_index = len(self.mask_lin)
            RE = self.WF_opt.rothe_error(start_index_WFnew=start_index, h=0)
            self.WF_opt.rothe_optimal_c_normalization(start_index, h=0)
            best_lin_params = self.WF_opt.opt_c
            self.params_old_lin = best_lin_params
            self.params_old_nonlin = best_nonlin_params
            self.removed_at_prev_timestep = 0
            new_WF = WF(
                np.reshape(best_nonlin_params, (-1, n * (n + 3))),
                best_lin_params,
                lambda_=self.lambda_,
            )
            normalization = abs(
                np.conj(best_lin_params).T @ new_WF.overlap @ np.array(best_lin_params)
            )
            energy = abs(np.conj(best_lin_params).T @ new_WF.H @ np.array(best_lin_params))
            self.normalization = normalization
            self.energy = energy
            if self.rank == 0:
                print("Normalization: %.10f, Energy: %.10f" % ((normalization), (energy)))

        remove_pos = 50
        if self.t > mask_start:
            abs_blp = np.abs(best_lin_params)
            left = np.where(abs_blp / np.max(abs_blp) < 1e-6)[
                0
            ]  # Those that do not contribute (e.g. 1e-6 of the maximum coefficient)
        else:
            left = []

        while len(left) > 0:
            if self.rank == 0:
                print(left)
            num_bas = len(dfo)
            self.removed_at_prev_timestep = 0
            # print(new_WF.lin_params)
            if self.rank == 0:
                print("Removing a Gaussian as it's beyond %d" % remove_pos)
            elem = left[0]
            nonlin_params_removed_indices = list(range(elem)) + list(range(elem + 1, num_bas))
            nonlin_params_removed = np.reshape(best_nonlin_params, (-1, n * (n + 3)))[
                nonlin_params_removed_indices, :
            ]
            new_lin_params = list(best_lin_params[nonlin_params_removed_indices].copy())
            new_nonlin_params = list(nonlin_params_removed.copy())
            new_nonlin_params = np.array(new_nonlin_params)
            new_lin_params = np.array(new_lin_params)
            full_nonlin = np.concatenate(
                (np.array(self.params_old_nonlin).flatten(), new_nonlin_params.flatten())
            )
            full_nonlin = np.reshape(full_nonlin, (-1, n * (n + 3)))
            full_lin = np.concatenate((self.params_old_lin, np.zeros(num_basis_functions)))
            self.params_old_lin = new_lin_params
            self.params_old_nonlin = new_nonlin_params
            best_nonlin_params = new_nonlin_params
            new_WF = WF(
                np.reshape(new_nonlin_params, (-1, n * (n + 3))),
                new_lin_params,
                lambda_=self.lambda_,
            )
            normalization = abs(
                np.conj(new_lin_params).T @ new_WF.overlap @ np.array(new_lin_params)
            )
            energy = abs(
                np.conj(new_lin_params).T
                @ new_WF.calculate_Hamiltonian()
                @ np.array(new_lin_params)
            )
            dfo = np.array(new_WF.calculate_gaussian_distances_from_origo())

            if self.rank == 0:
                print(
                    "Normalization: %.10f, Energy: %.10f" % (abs(normalization), abs(energy))
                )
            left = np.where(np.abs(new_lin_params) < 1e-20)[0]
        self.start_index = len(self.params_old_lin)

    def time_evolve_noKinetic(self):
        self.removed_at_prev_timestep += 1
        old_params_lin = self.params_old_lin
        old_params_nonlin = self.params_old_nonlin
        if self.rank == 0:
            print("Start time t0=%.3f" % self.t)
        nonlin_params_initial = np.array(self.params_old_nonlin.copy()).flatten()
        if self.removed_at_prev_timestep >= 0:
            nonlin_params_initial = self.get_start_parameters_noKinetic(
                nonlin_params_initial, self.direction
            )
        # Now, we do half a time step with the kinetic energy
        nonlin_propagated, c_propagated = propagate_kinetic_analytical(
            self.params_old_nonlin.reshape(-1, self.dimension * (self.dimension + 3)),
            self.params_old_lin,
            self.h / 2,
        )

        nonlin_params_initial = np.asarray(nonlin_propagated).flatten()
        self.params_old_lin = np.asarray(c_propagated)
        self.params_old_nonlin = np.asarray(nonlin_propagated).flatten()

        # Now I have propagated the linear and the nonlinear coefficients
        # Take a snapshot of the potential energy expectation value, which is to be conserved.
        WF_temp = WF(nonlin_propagated, c_propagated, lambda_=self.lambda_)
        potential = WF_temp.potential
        energy_temp = abs(np.conj(c_propagated).T @ potential @ c_propagated)
        opt_func = self.rothe_error_no_kinetic
        opt_grad = self.rothe_gradient_no_kinetic

        RE = sqrt(opt_func(nonlin_params_initial))
        grad0 = opt_grad(nonlin_params_initial)
        if self.rank == 0:
            print("Initial Rothe error: %f" % RE)
            # print("Initial Rothe gradient:",np.linalg.norm(grad0))
        maxiter = 300

        dg = self.grad0_eps * np.ones(len(grad0))
        hess_inv0 = np.diag(1 / abs(grad0 + dg))
        self.niterino = 0
        self.counterino = 0
        gtol = 1e-10
        if self.t < 0.1 or self.removed_at_prev_timestep < 1:
            start = time.time()
            solver = minimize(
                opt_func,
                nonlin_params_initial,
                method="BFGS",
                jac=opt_grad,
                options={"maxiter": maxiter, "gtol": gtol, "hess_inv0": hess_inv0},
            )
            best_nonlin_params = solver.x
            niter = solver.nit
            end = time.time()
            if self.rank == 0:
                print(
                    "UNREG: Time to optimize: %f; number of iterations: %d"
                    % (end - start, niter)
                )
            rotheerror = solver.fun

        else:

            best_nonlin_params, niter, rotheerror = self.minimize_transformed_bonds(
                opt_func,
                opt_grad,
                maxiter=maxiter,
                start_params=nonlin_params_initial,
                args=None,
                gtol=gtol,
                multi_bonds=3,
            )

        end = time.time()
        if self.rank == 0:
            print("Finallo Rothe error: %f/%f" % ((sqrt(rotheerror), self.maxerr)))

        self.comm.bcast(
            best_nonlin_params, root=0
        )  # Just to make sure that each thread has the same parameters?
        self.new_params = best_nonlin_params
        n = self.dimension
        divisible_by_5_timestep = int(np.round(self.t / self.h)) % 5 == 0

        self.accumulated_error += np.sqrt(rotheerror)
        num_basis_functions = len(best_nonlin_params) // (n * (n + 3))
        full_nonlin = np.concatenate(
            (np.array(self.params_old_nonlin).flatten(), best_nonlin_params)
        )
        start_index = len(self.params_old_lin)
        full_nonlin = np.reshape(full_nonlin, (-1, n * (n + 3)))
        full_lin = np.concatenate((self.params_old_lin, np.zeros(num_basis_functions)))
        optimization_WF = WF(full_nonlin, full_lin, lambda_=self.lambda_)
        optimization_WF.setupMatrices_noKinetic()
        optimization_WF.rothe_optimal_c_normalization(self.start_index, h=self.h)
        best_lin_params = optimization_WF.opt_c
        self.comm.bcast(best_lin_params, root=0)  # Share with other nodes
        self.comm.bcast(best_nonlin_params, root=0)  # Share with other nodes
        self.direction = best_nonlin_params - nonlin_params_initial
        # Conservation of energy and normalization is faulty
        # The idea is: "kinetic" propagation exactly preserves the kinetic energy and the norm
        # The "non-kinetic" propagation should preserve the norm and the expectation value of the potential energy
        # Thus, we do not conserve the energy, but the expectation value of the potential energy
        full_nonlin = np.concatenate(
            (np.array(self.params_old_nonlin).flatten(), best_nonlin_params)
        )
        start_index = len(old_params_lin)
        full_nonlin = np.reshape(full_nonlin, (-1, n * (n + 3)))
        full_lin = np.concatenate((self.params_old_lin, np.zeros(num_basis_functions)))
        optimization_WF = WF(full_nonlin, full_lin, lambda_=self.lambda_)
        optimization_WF.setupMatrices_noKinetic()
        optimization_WF.rothe_optimal_c_normalization(self.start_index, h=self.h)
        re = optimization_WF.rothe_error(start_index_WFnew=start_index, h=self.h)
        rv = rho_vec = optimization_WF.rho
        Stilde = optimization_WF.Smat
        S = optimization_WF.overlap[len(rv) :, len(rv) :]
        H = optimization_WF.potential[len(rv) :, len(rv) :]
        energy_new = abs(np.conj(best_lin_params).T @ H @ best_lin_params)
        if self.rank == 0:
            print("Entering conservation of norm: %.14f" % self.normalization)
            ovlp_t = optimization_WF.overlap_term

            def optimization_in_c(cfull):
                rec, conj_c = cfull[: len(cfull) // 2], cfull[len(cfull) // 2 :]
                c = rec + 1j * conj_c
                val = ovlp_t - (2 * np.conj(rv).T @ c - np.conj(c).T @ Stilde @ c)
                return real(val)

            def deriv_in_c(cfull):
                halflen = len(cfull) // 2
                rec, conj_c = cfull[:halflen], cfull[halflen:]
                c = rec + 1j * conj_c
                derivs = np.zeros_like(cfull, dtype=complex)

                derivs[:halflen] = (
                    -2 * np.conj(rv[:halflen])
                    + 2 * np.real(Stilde[:halflen]).T @ rec
                    + 1j * (Stilde[:halflen] - Stilde[:halflen].T) @ conj_c
                )

                # For the second half of derivs
                derivs[halflen:] = (
                    -2j * np.conj(rv[:halflen])
                    + 2 * np.real(Stilde[:halflen]) @ conj_c
                    + 1j * (Stilde[:halflen].T - Stilde[:halflen]) @ rec
                )
                return np.real(derivs)

            def constraint1(cfull):
                rec, conj_c = cfull[: len(cfull) // 2], cfull[len(cfull) // 2 :]
                c = rec + 1j * conj_c
                return real(np.conj(c) @ S @ c) - self.normalization

            c_init = np.concatenate((np.real(best_lin_params), np.imag(best_lin_params)))

            def constraint2(cfull):
                rec, conj_c = cfull[: len(cfull) // 2], cfull[len(cfull) // 2 :]
                c = rec + 1j * conj_c
                return real(np.conj(c) @ H @ c) - energy_temp

            solver = minimize(
                optimization_in_c,
                c_init,
                method="SLSQP",
                jac=deriv_in_c,
                constraints=[
                    {"type": "eq", "fun": constraint1},
                    {"type": "eq", "fun": constraint2},
                ],
                options={"maxiter": 200, "ftol": 1e-10},
            )
            c_new = solver.x[: len(solver.x) // 2] + 1j * solver.x[len(solver.x) // 2 :]
            best_lin_params = c_new
            print("Energy deviation: %.6f percent" % (100 * constraint2(solver.x)))
        self.comm.bcast(best_lin_params, root=0)  # Share with other nodes
        nonlin_propagated, c_propagated = propagate_kinetic_analytical(
            best_nonlin_params.reshape(-1, n * (n + 3)), best_lin_params, self.h / 2
        )
        best_lin_params = c_propagated
        best_nonlin_params = nonlin_propagated.flatten()
        self.comm.bcast(best_lin_params, root=0)  # Share with other nodes
        self.comm.bcast(best_nonlin_params, root=0)  # Share with other nodes

        self.params_oldold_nonlin = old_params_nonlin  # .flatten()
        self.params_old_lin = c_propagated
        self.params_old_nonlin = nonlin_propagated  # .flatten()
        if self.rank == 0:
            print(
                "Cumulative Rothe error: %f. Estimated (cum): %f, Estimated (init): %f"
                % (
                    self.accumulated_error,
                    self.accumulated_error / (self.t + self.h) * 100,
                    np.sqrt(rotheerror) / self.h * 100,
                )
            )
            print(best_nonlin_params.shape, best_lin_params.shape)
        new_WF = WF(
            np.reshape(best_nonlin_params, (-1, n * (n + 3))),
            best_lin_params,
            lambda_=self.lambda_,
        )

        normalization = abs(
            np.conj(best_lin_params).T @ new_WF.overlap @ np.array(best_lin_params)
        )
        energy = abs(np.conj(best_lin_params).T @ new_WF.H @ np.array(best_lin_params))
        dfo = np.array(
            new_WF.calculate_gaussian_distances_from_origo()
        )  # Distances from origo

        if self.rank == 0:
            print("Normalization: %.10f, Energy: %.10f" % ((normalization), (energy)))
            # print("Ovlerpa matrix:")
            # print(np.array2string(abs(new_WF.overlap), precision=1, suppress_small=False,formatter={'float_kind':lambda x: f"{x:.1e}"}))
            # print("Number of small coefficients: %d"%(np.sum(abs(new_WF.overlap) < -1)//2))
        mask_err = 0
        mask_start = 0
        if divisible_by_5_timestep and self.t >= mask_start:
            self.calculate_mask_coefficients()
            # trash,mask_err=self.make_mask_function(best_nonlin_params)
            mask_err = self.rothe_error_mask(best_nonlin_params)
            # self.rothe_error_mask_update(best_nonlin_params)
            # sys.exit(0)
        if sqrt(mask_err) > 1e-3:

            grad0 = self.rothe_gradient_mask(best_nonlin_params)
            dg = self.grad0_eps * np.ones(len(grad0))
            hess_inv0 = np.diag(1 / abs(grad0 + dg))
            start = time.time()
            func = self.rothe_error_mask_update
            start = time.time()
            solver = minimize(
                func,
                best_nonlin_params,
                method="BFGS",
                jac=self.rothe_gradient_mask,
                options={"maxiter": 300, "gtol": 5e-7},
            )
            end = time.time()
            best_nonlin_params = solver.x
            RE = sqrt(func(best_nonlin_params))
            if self.rank == 0:
                print(best_nonlin_params)
                print(RE)
                print("Time to optimize mask: %.3f" % (end - start))
            self.new_params = best_nonlin_params
            n = self.dimension

            self.WF_opt, err = self.make_mask_function(best_nonlin_params)
            start_index = len(self.mask_lin)
            RE = self.WF_opt.rothe_error(start_index_WFnew=start_index, h=0)
            self.WF_opt.rothe_optimal_c_normalization(start_index, h=0)
            best_lin_params = self.WF_opt.opt_c
            self.params_old_lin = best_lin_params
            self.params_old_nonlin = best_nonlin_params
            self.removed_at_prev_timestep = 0
            new_WF = WF(
                np.reshape(best_nonlin_params, (-1, n * (n + 3))),
                best_lin_params,
                lambda_=self.lambda_,
            )
            normalization = abs(
                np.conj(best_lin_params).T @ new_WF.overlap @ np.array(best_lin_params)
            )
            energy = abs(np.conj(best_lin_params).T @ new_WF.H @ np.array(best_lin_params))
            self.normalization = normalization
            self.energy = energy
            if self.rank == 0:
                print("Normalization: %.10f, Energy: %.10f" % ((normalization), (energy)))

        remove_pos = 50
        if self.t > mask_start:
            abs_blp = np.abs(best_lin_params)
            left = np.where(abs_blp / np.max(abs_blp) < 1e-6)[
                0
            ]  # Those that do not contribute (e.g. 1e-6 of the maximum coefficient)
        else:
            left = []
        if self.rank == 0:
            print(left)
        while len(left) > 0:
            num_bas = len(dfo)
            self.removed_at_prev_timestep = 0
            # print(new_WF.lin_params)
            if self.rank == 0:
                print("Removing a Gaussian as it's beyond %d" % remove_pos)
            elem = left[0]
            nonlin_params_removed_indices = list(range(elem)) + list(range(elem + 1, num_bas))
            nonlin_params_removed = np.reshape(best_nonlin_params, (-1, n * (n + 3)))[
                nonlin_params_removed_indices, :
            ]
            new_lin_params = list(best_lin_params[nonlin_params_removed_indices].copy())
            new_nonlin_params = list(nonlin_params_removed.copy())
            new_nonlin_params = np.array(new_nonlin_params)
            new_lin_params = np.array(new_lin_params)
            full_nonlin = np.concatenate(
                (np.array(self.params_old_nonlin).flatten(), new_nonlin_params.flatten())
            )
            full_nonlin = np.reshape(full_nonlin, (-1, n * (n + 3)))
            full_lin = np.concatenate((self.params_old_lin, np.zeros(num_basis_functions)))
            self.params_old_lin = new_lin_params
            self.params_old_nonlin = new_nonlin_params
            best_nonlin_params = new_nonlin_params
            new_WF = WF(
                np.reshape(new_nonlin_params, (-1, n * (n + 3))),
                new_lin_params,
                lambda_=self.lambda_,
            )
            normalization = abs(
                np.conj(new_lin_params).T @ new_WF.overlap @ np.array(new_lin_params)
            )
            energy = abs(
                np.conj(new_lin_params).T
                @ new_WF.calculate_Hamiltonian()
                @ np.array(new_lin_params)
            )
            dfo = np.array(new_WF.calculate_gaussian_distances_from_origo())

            if self.rank == 0:
                print(
                    "Normalization: %.10f, Energy: %.10f" % (abs(normalization), abs(energy))
                )
            left = np.where(np.abs(new_lin_params) < 1e-20)[0]
        self.start_index = len(self.params_old_lin)

    def time_evolver(self, type="full"):
        while self.t <= self.T:
            if type == "full":
                self.time_evolve()
            elif type == "noKinetic" or type == "nokinetic":
                self.time_evolve_noKinetic()
            self.t += self.h
            n = self.dimension
            if self.rank == 0:
                with h5py.File(self.filename, "r+") as data_file:
                    try:
                        data_file.create_dataset(
                            "parameters_t=%.3f" % self.t,
                            data=np.reshape(self.params_old_nonlin, (-1, n * (n + 3))),
                        )
                        data_file.create_dataset(
                            "coefficients_t=%.3f" % self.t, data=self.params_old_lin
                        )
                    except ValueError:
                        del data_file["parameters_t=%.3f" % self.t]
                        del data_file["coefficients_t=%.3f" % self.t]
                        data_file.create_dataset(
                            "parameters_t=%.3f" % self.t,
                            data=np.reshape(self.params_old_nonlin, (-1, n * (n + 3))),
                        )
                        data_file.create_dataset(
                            "coefficients_t=%.3f" % self.t, data=self.params_old_lin
                        )
                    old_times = data_file["times"]
                    new_times = list(old_times)
                    new_times.append(self.t)
                    del data_file["times"]
                    data_file.create_dataset("times", data=new_times)
                    old_energies = data_file["energies"]
                    new_energies = list(old_energies)
                    new_energies.append(self.energy)
                    del data_file["energies"]
                    data_file.create_dataset("energies", data=new_energies)
                    old_norms = data_file["normalizations"]
                    new_norms = list(old_norms)
                    new_norms.append(self.normalization)
                    del data_file["normalizations"]
                    data_file.create_dataset("normalizations", data=new_norms)
                    old_re = data_file["rothe_error"]
                    new_re = list(old_re)
                    new_re.append(self.accumulated_error)
                    del data_file["rothe_error"]
                    data_file.create_dataset("rothe_error", data=new_re)
