import numpy as np
from WF_HenonHeiles_parallel import WF


def test_derivs():
    par_ind = 0
    dim = 3
    lambda_ = 0.1

    np.random.seed(42)
    params = np.random.rand(2, 2 * dim + dim * (dim + 1)) * 1e-1
    k = -1
    for i in range(1, dim + 1):
        k += i
        params[:, k] = np.sqrt(1 / 2) + 0.1 * (np.random.rand() - 0.5)

    wf = WF(params, [1, 0], lambda_=lambda_)
    all_derivs = wf.calculate_Hamiltonian_deriv(1)
    assert all_derivs is not None

    h = 1e-8
    dparams = params.copy()
    dparams[1, par_ind] += h
    dwf = WF(dparams, [1, 0], lambda_=lambda_)
    numerical = (dwf.calculate_Hamiltonian()[0, 1] - wf.calculate_Hamiltonian()[0, 1]) / h
    absolute_error = abs(all_derivs[0, 0, par_ind] - numerical)

    assert absolute_error < 1e-5
