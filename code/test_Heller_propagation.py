import numpy as np
import pytest
from numpy import sqrt
from WF_HenonHeiles_parallel import WF, propagate_kinetic_analytical


@pytest.mark.parametrize("dimension,len_initial", [(2, 2), (2, 4), (3, 2), (3, 4)])
def test_heller_propagation_norm_conservation(dimension, len_initial):
    """Kinetic-step propagation should preserve norm."""
    lambda_ = 0.111803
    timestep = 1
    start_pos = 2
    np.random.seed(1000 + 10 * dimension + len_initial)

    nonlin_params = np.zeros((len_initial, dimension * (dimension + 3)))
    k = -1
    for i in range(1, dimension + 1):
        k += i
        nonlin_params[:, k] = sqrt(1 / 2)
    startparam = dimension * (dimension + 1)
    random_spread = 5
    for i in range(dimension):
        nonlin_params[0, startparam + i] = start_pos
        nonlin_params[1:, startparam + i] = (
            random_spread * (0.5 - np.random.rand(len_initial - 1)) + start_pos
        )

    lin_params = np.random.rand(len_initial) - 0.5 + 1j * (np.random.rand(len_initial) - 0.5)
    nonlin_params += np.random.rand(nonlin_params.shape[0], nonlin_params.shape[1]) * 1 - 0.5

    wave_function = WF(nonlin_params, lin_params, lambda_=lambda_)
    old_norm = np.conj(lin_params).T @ wave_function.overlap @ lin_params
    nonlin_new, lin_new = propagate_kinetic_analytical(nonlin_params, lin_params, timestep)
    wave_function = WF(nonlin_new, lin_new, lambda_=lambda_)
    lin_new = np.array(lin_new)
    new_norm = np.conj(lin_new).T @ wave_function.overlap @ lin_new

    assert np.isclose(new_norm, old_norm, rtol=1e-6, atol=1e-8)
