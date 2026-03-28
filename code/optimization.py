import time
import warnings

# import multiprocessing
import h5py
import scipy
from numpy import arcsin, arctanh, cos, cosh, sin, tanh
from scipy.optimize import fmin_bfgs, minimize
from WF_HenonHeiles import *

numiter = 0


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
    ):

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
        self.params_oldold_nonlin = np.zeros_like(self.params_old_lin)
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
        if t0 < 0.5:
            self.removed_at_prev_timestep = 0
        else:
            self.removed_at_prev_timestep = 2
        self.start_index = len(self.params_old_lin)
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
                index_to_delete_after = np.argmin(abs(times_read - t0)) + 1
                if abs(times_read[index_to_delete_after - 1] - t0) > 1e-5:
                    print(
                        "You cannot start your run from this time, as no reference data exists"
                    )
                    raise IndexError
                else:
                    times_new = times_read[:index_to_delete_after]
                    re_new = np.array(data_file["rothe_error"])[:index_to_delete_after]
                    del data_file["rothe_error"]
                    del data_file["times"]
                    data_file.create_dataset("times", data=times_new)
                    data_file.create_dataset("rothe_error", data=re_new)
                    if self.t0 >= self.h - 1e-10:
                        self.params_oldold_nonlin = np.array(
                            data_file["parameters_t=%.3f" % (self.t0 - self.h)]
                        )

    def rothe_gradient(self, nonlin_params):
        start_index = len(self.params_old_lin)
        start = time.time()
        deriv = self.WF_opt.rothe_jacobian(start_index_WFnew=start_index, h=self.h)
        end = time.time()
        # print("Time to calculate RE dderiv: %f"%(end-start))
        # del self.WF_opt
        return deriv

    def euler_gradient(self, nonlin_params):
        start_index = len(self.params_old_lin)
        deriv = self.WF_opt.calculate_euler_deriv(si=start_index, h=self.h)
        del self.WF_opt
        return deriv

    def rothe_gradient_numerical(self, nonlin_params):
        n = self.dimension
        num_basis_functions = len(nonlin_params) // (n * (n + 3))
        full_nonlin = np.concatenate(
            (np.array(self.params_old_nonlin).flatten(), nonlin_params)
        )
        full_nonlin = np.reshape(full_nonlin, (-1, n * (n + 3)))
        full_lin = np.concatenate((self.params_old_lin, np.zeros(num_basis_functions)))
        optimization_WF = WF(full_nonlin, full_lin, lambda_=self.lambda_)
        self.WF_opt = optimization_WF
        start_index = len(self.params_old_lin)
        RE = optimization_WF.rothe_error(start_index_WFnew=start_index, h=self.h)
        jac = np.zeros_like(nonlin_params)

        for i in range(len(nonlin_params)):
            new_nonlin = nonlin_params.copy()
            new_nonlin[i] += 1e-7
            full_nonlin = np.concatenate(
                (np.array(self.params_old_nonlin).flatten(), new_nonlin)
            )
            full_nonlin = np.reshape(full_nonlin, (-1, n * (n + 3)))
            full_lin = np.concatenate((self.params_old_lin, np.zeros(num_basis_functions)))
            optimization_WF = WF(full_nonlin, full_lin, lambda_=self.lambda_)
            optimization_WF
            REh = optimization_WF.rothe_error(start_index_WFnew=start_index, h=self.h)
            jac[i] = 1 / 1e-7 * (REh - RE)
        return jac

    def rothe_error(self, nonlin_params, calculate_Gradient=True):
        n = self.dimension
        num_basis_functions = len(nonlin_params) // (n * (n + 3))
        start_index = len(self.params_old_lin)
        full_nonlin = np.concatenate(
            (np.array(self.params_old_nonlin).flatten(), nonlin_params)
        )
        full_nonlin = np.reshape(full_nonlin, (-1, n * (n + 3)))
        full_lin = np.concatenate((self.params_old_lin, np.zeros(num_basis_functions)))

        optimization_WF = WF(
            full_nonlin, full_lin, lambda_=self.lambda_, calculate_Gradient=calculate_Gradient
        )
        self.WF_opt = optimization_WF
        RE = optimization_WF.rothe_error(start_index_WFnew=start_index, h=self.h)
        return RE

    def rothe_error_update(self, nonlin_params):
        num_gauss_total = self.WF_opt.num_gaussians
        num_gauss_optimization = len(nonlin_params) // (self.dimension * (self.dimension + 3))
        indices = np.arange(num_gauss_total - num_gauss_optimization, num_gauss_total)
        start_index = len(self.params_old_lin)
        self.WF_opt.update_params_overlapFitting(
            indices, np.reshape(nonlin_params, (-1, self.dimension * (self.dimension + 3)))
        )
        RE = self.WF_opt.rothe_error(start_index_WFnew=start_index, h=self.h)
        self.WF_opt.setUpDerivs(start_index)
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

        num_basis_functions = len(self.params_old_nonlin) // (n * (n + 3))
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
            h=0,
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
        RE = self.WF_opt.rothe_error(start_index_WFnew=start_index, h=0)
        print("%e" % sqrt(RE))
        self.WF_opt.rothe_optimal_c_normalization(start_index, h=0)
        return RE

    def rothe_error_mask_update(self, nonlin_params, onlyX1X2=True):
        start_index = len(self.mask_lin)
        num_gauss_total = self.WF_opt.num_gaussians
        num_gauss_optimization = len(nonlin_params) // (self.dimension * (self.dimension + 3))
        indices = np.arange(num_gauss_total - num_gauss_optimization, num_gauss_total)
        # print(indices)

        self.WF_opt.update_params_overlapFitting(
            indices,
            np.reshape(nonlin_params, (-1, self.dimension * (self.dimension + 3))),
            onlyX1X2=onlyX1X2,
        )
        RE = self.WF_opt.rothe_error(start_index_WFnew=start_index, h=0)
        print("%e" % sqrt(RE))
        self.WF_opt.rothe_optimal_c_normalization(start_index, h=0)

        return RE

    def rothe_gradient_mask(self, nonlin_params):
        start_index = len(self.mask_lin)
        deriv = self.WF_opt.rothe_jacobian(start_index_WFnew=start_index, h=0)
        # del self.WF_opt
        return deriv

    def euler_error(self, nonlin_params, calculate_Gradient=False):
        n = self.dimension
        num_basis_functions = len(nonlin_params) // (n * (n + 3))
        start_index = len(self.params_old_lin)
        full_nonlin = np.concatenate(
            (np.array(self.params_old_nonlin).flatten(), nonlin_params)
        )
        full_nonlin = np.reshape(full_nonlin, (-1, n * (n + 3)))
        full_lin = np.concatenate((self.params_old_lin, np.zeros(num_basis_functions)))
        optimization_WF = WF(
            full_nonlin, full_lin, lambda_=self.lambda_, calculate_Gradient=calculate_Gradient
        )
        self.WF_opt = optimization_WF
        EE = optimization_WF.calculate_euler_error(start_index_WFnew=start_index, h=self.h)
        return EE

    def euler_gradient_numerical(self, nonlin_params):
        n = self.dimension
        num_basis_functions = len(nonlin_params) // (n * (n + 3))
        full_nonlin = np.concatenate(
            (np.array(self.params_old_nonlin).flatten(), nonlin_params)
        )
        full_nonlin = np.reshape(full_nonlin, (-1, n * (n + 3)))
        full_lin = np.concatenate((self.params_old_lin, np.zeros(num_basis_functions)))
        optimization_WF = WF(full_nonlin, full_lin, lambda_=self.lambda_)
        self.WF_opt = optimization_WF
        start_index = len(self.params_old_lin)
        RE = optimization_WF.calculate_euler_error(start_index_WFnew=start_index, h=self.h)
        jac = np.zeros_like(nonlin_params)

        for i in range(len(nonlin_params)):
            new_nonlin = nonlin_params.copy()
            new_nonlin[i] += 1e-7
            full_nonlin = np.concatenate(
                (np.array(self.params_old_nonlin).flatten(), new_nonlin)
            )
            full_nonlin = np.reshape(full_nonlin, (-1, n * (n + 3)))
            full_lin = np.concatenate((self.params_old_lin, np.zeros(num_basis_functions)))
            optimization_WF = WF(full_nonlin, full_lin, lambda_=self.lambda_)
            optimization_WF
            REh = optimization_WF.calculate_euler_error(
                start_index_WFnew=start_index, h=self.h, calculate_Gradient=False
            )
            jac[i] = 1 / 1e-7 * (REh - RE)
        return jac

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

                print("gradient has nan")
                print(grad)
                print(transformed_error(transformed_params))
                if np.isnan(np.sum(orig_grad)):
                    print("Original gradient has nan...")
                return np.nan_to_num(grad)
            return grad

        if printx:
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
        hess_inv0 = np.diag(1 / abs(grad0 + 1e-15 * np.array(len(grad0))))
        self.numiter = 0
        self.f_storage = []

        def callback_func(xval):
            if self.numiter % 10 == 0:
                self.f_storage.append(transformed_error(xval))
                if self.numiter >= 20:
                    if (
                        sqrt(self.f_storage[-1] / self.f_storage[-2]) > 0.999
                        and self.f_storage[-1] / self.f_storage[-2] < 1
                    ):
                        print("The function is not decreasing anymore.")
                        self.transformed_sol = xval

                        raise ConvergedError
            self.numiter += 1

        try:
            self.transformed_sol, minval, gopt, bopt, nit, grad_calls, warnflag = fmin_bfgs(
                f=transformed_error,
                x0=transformed_params,
                fprime=transformed_gradient,
                maxiter=200,
                gtol=1e-8,
                hess_inv0=hess_inv0,
                c1=1e-5,
                c2=3e-1,
                callback=callback_func,
                full_output=True,
            )
        except ConvergedError:
            pass
        end = time.time()
        print(
            "  REG: Time to optimize: %.3f seconds, niter : %d" % (end - start, self.numiter)
        )
        return untransform_params(self.transformed_sol), self.numiter

    def get_start_parameters(self):
        old = np.array(self.params_old_nonlin.copy()).flatten()
        oldold = np.array(self.params_oldold_nonlin.copy()).flatten()
        try:
            direction = old.flatten() - oldold.flatten()
        except:
            return old.flatten()

        error_set = []
        alphas = np.linspace(0, 1.5, 16)
        for alpha in alphas:
            val = self.euler_error(old + alpha * direction, calculate_Gradient=False)
            if val > 0:
                error_set.append(sqrt(val))
            else:
                error_set.append(1e100)
        error_set = np.nan_to_num(error_set, nan=1e15, posinf=1e15, neginf=1e15)
        i = np.argmin(error_set)
        print("alpha: %.2f" % alphas[i])
        return old + direction * alphas[i]

    def calculate_importance_remove(self, nonlin_coefficients):

        n = self.dimension
        orig_RE = sqrt(
            self.rothe_error(nonlin_coefficients.flatten(), calculate_Gradient=False)
        )
        nonlin_params_orig = np.array(nonlin_coefficients.copy()).flatten()
        num_basis_functions_orig = len(nonlin_params_orig) // (n * (n + 3))
        if num_basis_functions_orig == 1:
            return nonlin_coefficients.flatten()
        r_errors = np.zeros(num_basis_functions_orig)
        for i in range(num_basis_functions_orig):

            nonlin_params_removed_indices = list(range(i)) + list(
                range(i + 1, num_basis_functions_orig)
            )
            nonlin_params_removed = nonlin_coefficients.copy()[
                nonlin_params_removed_indices, :
            ]
            r_errors[i] = self.rothe_error(
                nonlin_params_removed.flatten(), calculate_Gradient=False
            )
        for i in range(num_basis_functions_orig):
            print("Rothe error upon removal of basis function %d: %f" % (i, sqrt(r_errors[i])))
        best = np.argmin(r_errors)
        if sqrt(r_errors[best]) > 10 * orig_RE:
            print("No basis function is removed as the error would be to big.")
            return nonlin_coefficients.flatten()
        nonlin_params_removed_indices = list(range(best)) + list(
            range(best + 1, num_basis_functions_orig)
        )
        nonlin_coefficients_new = nonlin_coefficients.copy()
        # nonlin_coefficients_new[best,:]=[sqrt(0.5),0.,sqrt(0.5),0,0,0,0,0,0,0]
        nonlin_coefficients_new = np.array(
            nonlin_coefficients[nonlin_params_removed_indices, :]
        )
        startparams = np.array(
            nonlin_coefficients_new
        ).flatten()  # +(0.5-np.random.rand(len(nonlin_coefficients_new.flatten())))*1e-5
        print(sqrt(self.rothe_error(startparams)))

        grad0 = self.rothe_gradient(startparams)
        hess_inv0 = np.diag(1 / abs(grad0 + 1e-8 * np.array(len(grad0))))
        solver = minimize(
            self.rothe_error,
            startparams,
            method="BFGS",
            jac=self.rothe_gradient,
            options={"maxiter": 2000, "gtol": 1e-16, "hess_inv0": hess_inv0},
        )
        best_nonlin_params = solver.x
        new_RE = sqrt(self.rothe_error(best_nonlin_params))
        print("Rothe error after reoptimization: %f" % new_RE)
        if new_RE * 1.2 < orig_RE or new_RE < 3e-4:
            print("One gaussian was removed")
            added = True
            return best_nonlin_params
        else:
            print("No Gaussian was removed")
            return nonlin_coefficients.flatten()

    def get_Guess_distribution(self, vals, n):
        x = np.linspace(np.min(vals), np.max(vals), 10000)
        returnvals = np.zeros(len(x))
        for a in vals:
            returnvals += (
                np.exp(-((x - a) ** 2) / (2 * (a + 1e-10) ** 2))
                * 1
                / np.sqrt(2 * np.pi * (a + 1e-10) ** 2)
            )
        p = returnvals / np.sum(returnvals)
        return np.random.choice(x, p=p, size=n)

    def gaussianAddingStrategy_ludwik(self, nonlin_coefficients):
        n = 1000
        reshaped = np.reshape(nonlin_coefficients, (-1, self.dimension * (self.dimension + 3)))
        samples = np.zeros((n, len(reshaped[0])))
        for i in range(len(reshaped[0])):
            samples[:, i] = self.get_Guess_distribution(reshaped[:, i], n)
        # samples[:,6:]=0 #Added Gaussian NEEDS to be in the origin
        errors = np.zeros(n)
        for i in range(n):
            params_to_add = samples[i, :]
            nonlin_coefficients_new = list(reshaped.flatten()) + list(params_to_add)
            startparams = np.array(nonlin_coefficients_new).flatten()
            error = self.rothe_error(startparams, calculate_Gradient=False)
            errors[i] = error
        errors = np.nan_to_num(errors, nan=1e100, posinf=1e100, neginf=1e100)
        i = np.argmin(errors)
        return samples[i, :]

    def add_gaussian(self, nonlin_coefficients):
        optimal_params = self.gaussianAddingStrategy_ludwik(nonlin_coefficients)
        # optimal_params=[sqrt(1/2),0,sqrt(1/2),0,0,0,0,0,0,0]
        nonlin_coefficients_new = np.array(
            list(nonlin_coefficients.flatten()) + list(optimal_params)
        )
        startparams = (
            np.array(nonlin_coefficients_new).flatten()
            + (np.random.rand(len(nonlin_coefficients_new.flatten())) - 0.5) * 1e-4
        )
        print("Error before optimization: %f" % sqrt(self.rothe_error(startparams)))

        grad0 = self.rothe_gradient(startparams)
        hess_inv0 = np.diag(1 / abs(grad0 + 1e-8 * np.array(len(grad0))))
        solver = minimize(
            self.rothe_error,
            startparams,
            method="BFGS",
            jac=self.rothe_gradient,
            options={"maxiter": 10, "gtol": 1e-7, "hess_inv0": hess_inv0},
        )
        new_params = solver.x
        new_RE = self.rothe_error(new_params)
        print("Rothe error after reoptimization: %f" % (sqrt(new_RE)))
        return new_params, new_RE

    def time_evolve(self):
        self.removed_at_prev_timestep += 1
        print("Start time t0=%.3f" % self.t)
        nonlin_params_initial = np.array(self.params_old_nonlin.copy()).flatten()
        if self.removed_at_prev_timestep >= 0:
            nonlin_params_initial = self.get_start_parameters()
        run_euler = False
        pre_euler_params = nonlin_params_initial.copy()
        pre_euler_error = sqrt(self.rothe_error(nonlin_params_initial))
        print("Initial Rothe error: %f" % pre_euler_error)
        if run_euler:
            grad0 = self.euler_gradient(nonlin_params_initial)
            hess_inv0 = np.diag(1 / abs(grad0 + 1e-14 * np.array(len(grad0))))
            start = time.time()
            solver = minimize(
                self.euler_error,
                nonlin_params_initial,
                method="BFGS",
                jac=self.euler_gradient,
                options={"maxiter": 5, "gtol": 1e-4, "hess_inv0": hess_inv0},
            )
            euler_params = solver.x

            end = time.time()
            print("Euler time: %f" % (end - start))
            post_euler_error = sqrt(self.rothe_error(euler_params))
            print("Post Euler Rothe error: %f" % post_euler_error)
            if post_euler_error < pre_euler_error:
                nonlin_params_initial = euler_params
        maxiter = 200
        gtol = 1e-8
        if self.dimension <= 6:
            opt_func = self.rothe_error

        if self.removed_at_prev_timestep >= 2 and self.t >= 0.2:
            best_nonlin_params, niter = self.minimize_transformed_bonds(
                opt_func,
                self.rothe_gradient,
                maxiter=maxiter,
                start_params=nonlin_params_initial,
                args=None,
                gtol=gtol,
                multi_bonds=2e0,
            )
        else:
            grad0 = self.rothe_gradient(nonlin_params_initial)
            hess_inv0 = np.diag(1 / abs(grad0 + 1e-15 * np.array(len(grad0))))
            start = time.time()
            solver = minimize(
                opt_func,
                nonlin_params_initial,
                method="BFGS",
                jac=self.rothe_gradient,
                options={"maxiter": maxiter, "gtol": gtol, "hess_inv0": hess_inv0},
            )
            best_nonlin_params = solver.x
            end = time.time()
            niter = solver.nit
            print(
                "UNREG: Time to optimize: %f; number of iterations: %d" % (end - start, niter)
            )
        if niter <= 3:
            print("(Almost) No iterations were made, do a new optimization")
            start = time.time()
            solver = minimize(
                opt_func,
                pre_euler_params,
                method="BFGS",
                jac=self.rothe_gradient,
                options={"maxiter": maxiter, "gtol": 1e-14},
            )
            best_nonlin_params = solver.x
            end = time.time()
            niter = solver.nit
            print("Time to optimize: %f; number of iterations: %d" % (end - start, niter))
        self.new_params = best_nonlin_params
        n = self.dimension
        rotheerror = self.rothe_error(best_nonlin_params)

        print("Finallo Rothe error: %f/%f" % ((sqrt(rotheerror), self.maxerr)))
        divisible_by_10_timestep = int(np.round(self.t / self.h)) % 10 == 0
        divisible_by_50_timestep = int(np.round(self.t / self.h)) % 50 == 0

        # if divisible_by_10_timestep and (len(self.params_old_lin)>self.ng_initial or self.t>1) and False:
        #    best_nonlin_params=self.calculate_importance_remove(np.reshape(best_nonlin_params,(-1,n*(n+3))))
        added = False
        """
        if (sqrt(rotheerror)>self.maxerr and len(self.params_old_lin)<self.max_num_Gaussians) or len(best_nonlin_params)/10<self.ng_initial:
            print("Adding a Gaussian.")
            best_nonlin_params,rotheerror=self.add_gaussian(np.reshape(best_nonlin_params,(-1,n*(n+3))))
            self.removed_at_prev_timestep=0
            added=True
        """
        self.accumulated_error += np.sqrt(rotheerror)
        num_basis_functions = len(best_nonlin_params) // (n * (n + 3))
        full_nonlin = np.concatenate(
            (np.array(self.params_old_nonlin).flatten(), best_nonlin_params)
        )
        full_nonlin = np.reshape(full_nonlin, (-1, n * (n + 3)))
        full_lin = np.concatenate((self.params_old_lin, np.zeros(num_basis_functions)))
        optimization_WF = WF(full_nonlin, full_lin, lambda_=self.lambda_)
        optimization_WF.rothe_optimal_c_normalization(self.start_index, h=self.h)

        best_lin_params = optimization_WF.opt_c
        self.params_oldold_nonlin = self.params_old_nonlin
        self.params_old_lin = best_lin_params
        self.params_old_nonlin = best_nonlin_params
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
        # new_WF.overlap[abs(new_WF.overlap)<1e-4]=0

        normalization = abs(
            np.conj(best_lin_params).T @ new_WF.overlap @ np.array(best_lin_params)
        )
        # print("Overlap matrix:")
        # print(abs(new_WF.overlap))
        energy = abs(
            np.conj(best_lin_params).T
            @ new_WF.calculate_Hamiltonian()
            @ np.array(best_lin_params)
        )
        self.normalization = normalization
        self.energy = energy
        print("Normalization: %.10f, Energy: %.10f" % ((normalization), (energy)))
        print("Updated lin parameters (absolute squared):")
        print(np.abs(best_lin_params) ** 2)
        print("Distances from origo:")
        dfo = np.array(new_WF.calculate_gaussian_distances_from_origo())
        print(dfo)
        eigvals = np.linalg.eigh(new_WF.overlap)[0]
        print("Smallest three eigenvalues of overlap matrix: ", eigvals[:3])
        mask_err = 0
        if divisible_by_10_timestep:
            self.calculate_mask_coefficients()
            # trash,mask_err=self.make_mask_function(best_nonlin_params)
            mask_err = self.rothe_error_mask(best_nonlin_params)
            print("Error of the mask: %e" % sqrt(mask_err))
        if sqrt(mask_err) > 1e-3:
            grad0 = self.rothe_gradient_mask(best_nonlin_params)
            hess_inv0 = np.diag(1 / abs(grad0 + 1e-15 * np.array(len(grad0))))
            start = time.time()
            solver = minimize(
                self.rothe_error_mask_update,
                best_nonlin_params,
                method="BFGS",
                jac=self.rothe_gradient_mask,
                options={"maxiter": 200, "gtol": 1e-6},
            )
            best_nonlin_params = solver.x
            mask_err_new = self.rothe_error_mask_update(best_nonlin_params)
            print("Error of the mask after %e" % sqrt(mask_err_new))
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
            energy = abs(
                np.conj(best_lin_params).T
                @ new_WF.calculate_Hamiltonian()
                @ np.array(best_lin_params)
            )
            self.normalization = normalization
            self.energy = energy
            print("Normalization: %.10f, Energy: %.10f" % ((normalization), (energy)))

        refit = False
        if (
            refit
            and divisible_by_10_timestep
            and not added
            and self.t > 0.5
            and self.ng_initial > 1
        ):
            least_important_gaussian = np.argmin(
                abs(self.params_old_lin)
            )  # A heuristic to pic the least important Gaussian
            print(
                "Least important Gaussian is %d with absolute linear coefficient %f"
                % (
                    least_important_gaussian,
                    abs(self.params_old_lin[least_important_gaussian]),
                )
            )
            new_nonlin = np.reshape(self.params_old_nonlin.copy(), (-1, n * (n + 3)))
            new_nonlin = np.delete(new_nonlin, least_important_gaussian, axis=0).flatten()
            ore = self.overlap_refit_error(new_nonlin)
            new_nonlin += new_nonlin * (0.5 - np.random.rand(len(new_nonlin.flatten()))) * 1e-2
            print("(Naive) Overlap refit error: %f" % sqrt(ore))
            grad0 = self.overlap_refit_gradient(new_nonlin)
            hess_inv0 = np.diag(1 / abs(grad0 + 1e-8 * np.array(len(grad0))))
            solver = minimize(
                self.overlap_refit_error,
                new_nonlin,
                method="BFGS",
                jac=self.overlap_refit_gradient,
                options={"maxiter": 2000, "gtol": 1e-16, "hess_inv0": hess_inv0},
            )
            newest_nonlin = solver.x
            new_ore = self.overlap_refit_error(newest_nonlin)
            print("New Overlap refit error: %f" % sqrt(new_ore))
            if sqrt(new_ore) < 1e-3 and False:
                print("We are removing a Gaussian, yay!")
                full_nonlin = np.concatenate(
                    (np.array(self.params_old_nonlin).flatten(), newest_nonlin)
                )
                full_nonlin = np.reshape(full_nonlin, (-1, n * (n + 3)))
                full_lin = np.concatenate(
                    (self.params_old_lin, np.zeros(len(self.params_old_lin) - 1))
                )
                print(full_nonlin.shape)
                print(full_lin.shape)
                optimization_WF = WF(full_nonlin, full_lin, lambda_=self.lambda_)
                optimization_WF.rothe_optimal_c_normalization(self.start_index, h=0)

                best_lin_params = optimization_WF.opt_c
                self.params_old_lin = best_lin_params
                self.params_old_nonlin = newest_nonlin
                self.removed_at_prev_timestep = 0
        # if divisible_by_10_timestep and (len(self.params_old_lin)>self.ng_initial or self.t>1):
        #    best_nonlin_params=self.calculate_importance_remove(np.reshape(best_nonlin_params,(-1,n*(n+3))))
        # H=new_WF.H
        # H2=new_WF.Hsquared
        # S=new_WF.overlap
        # eigvals,eigvecs=scipy.linalg.eigh(H,S,eigvals_only=False)
        # print("H  Eigenvalues at this time")
        # print(eigvals)
        # H_decomposition=np.conj(eigvecs).T@S@best_lin_params
        # print(abs(H_decomposition))

        # eigvalsH2,eigvecsH2=scipy.linalg.eigh(H2,S,eigvals_only=False)
        # print("H2 Eigenvalues at this time")
        # print(eigvalsH2)
        # H2_decomposition=np.conj(eigvecsH2).T@S@best_lin_params
        # print(abs(H2_decomposition))

        # assert abs(normalization-(np.conj(H_decomposition).T@H_decomposition))<1e-8 #Make sure that the decomposition is correct
        # assert sum(abs(best_lin_params - H_decomposition.T@eigvecs.T))<1e-8*len(best_lin_params)
        E_cutoff = 20
        d = 0.1
        # H_decomposition[eigvals>E_cutoff]*=np.exp(-sqrt(2*H_decomposition[eigvals>E_cutoff])/2/d)**self.h #Shrink high energy eigenvectors
        # self.params_old_lin=H_decomposition.T@eigvecs.T
        # normalization=abs(np.conj(self.params_old_lin).T@S@np.array(self.params_old_lin))
        # print("New normalization: %f"%normalization)
        remove_pos = 50
        if self.t > 1:
            left = np.where(np.abs(best_lin_params) < 1e-6)[0]  # Those that do not contribute
        else:
            left = []
        print(left)
        while len(left) > 0:
            num_bas = len(dfo)
            self.removed_at_prev_timestep = 0
            # print(new_WF.lin_params)
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
            # params_oldold_nonlin=np.reshape(self.params_oldold_nonlin,(-1,n*(n+3)))
            # params_oldold_nonlin[elem,:]=np.array([sqrt(0.5),0.,sqrt(0.5),0,0,0,0,0,0,0])
            # self.params_oldold_nonlin=params_oldold_nonlin.flatten()
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
            print("Normalization: %.10f, Energy: %.10f" % (abs(normalization), abs(energy)))
            print("Distances from origo:")
            dfo = np.array(new_WF.calculate_gaussian_distances_from_origo())
            print(dfo)
            left = np.where(np.abs(new_lin_params) < 1e-20)[0]
        self.start_index = len(self.params_old_lin)

    def time_evolver(self):
        while self.t <= self.T:
            self.time_evolve()
            self.t += self.h
            n = self.dimension
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
