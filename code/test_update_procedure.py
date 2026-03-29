import numpy as np
from numpy import sqrt
from WF_HenonHeiles_parallel import WF


def test_update_procedure():
    dim = 3
    num_gauss = 3
    m = 2

    np.random.seed(42)
    cvals = np.random.rand(num_gauss * m) - 0.5 + 1j * (np.random.rand(num_gauss * m) - 0.5)
    params = np.random.rand(num_gauss * m, dim * (dim + 3)) * 1e-1

    k = -1
    for i in range(1, dim + 1):
        k += i
        params[:, k] = sqrt(1 / 2) + 0.1 * (np.random.rand() - 0.5)

    wf_old = WF(params, cvals)

    params_updated = params.copy()
    params_new = (np.random.rand(num_gauss, dim * (dim + 3)) - 0.5) * 1e-1
    params_updated[(m - 1) * num_gauss :] = params_new

    wf_fresh = WF(params_updated, cvals)

    indices = np.arange((m - 1) * num_gauss, m * num_gauss)
    wf_old.update_overlap_and_overlap_derivs(indices, params_new, new_lin_params=None)

    re_diff = abs(
        wf_old.rothe_error((m - 1) * num_gauss, 1e-2)
        - wf_fresh.rothe_error((m - 1) * num_gauss, 1e-2)
    )
    jac_diff = float(
        np.sum(
            abs(
                wf_old.rothe_jacobian((m - 1) * num_gauss, 1e-2)
                - wf_fresh.rothe_jacobian((m - 1) * num_gauss, 1e-2)
            )
        )
    )

    assert np.isfinite(re_diff)
    assert np.isfinite(jac_diff)
