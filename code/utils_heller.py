import numpy as np
import scipy
import scipy.linalg as la
from numpy import einsum, exp, log, pi, sqrt
from scipy.linalg import logm


def transform_ECG_heller(ECG_A, ECG_B, ECG_mu, ECG_p, ECG_c):
    """
    This function transforms the unnormalized ECG to the Heller wavepacket.
    """
    d = len(ECG_p)
    D = -(ECG_A + 1j * ECG_B)
    heller_C = -2j * D
    Dinv = np.linalg.inv(D)
    invA = np.linalg.inv(ECG_A)
    heller_q = (-ECG_p.T @ ECG_B + ECG_mu.T @ ECG_A) @ np.linalg.inv(ECG_A)
    b = D @ (ECG_mu + 1j * ECG_p)
    heller_p = 2j * (b - D @ heller_q)
    ECG_normalization_log = (
        (ECG_p.T @ ECG_B @ invA @ ECG_B @ ECG_p + ECG_p.T @ ECG_A @ ECG_p)
        + d / 4 * log(pi / 2)
        - 0.25 * log(np.linalg.det(ECG_A))
    )
    heller_zeta = -1j * (
        np.log(ECG_c)
        - ECG_normalization_log
        + 1j * heller_p.T @ heller_q
        - heller_q.T @ D @ heller_q
        + b @ Dinv @ b
    )
    return heller_C, heller_q, heller_p, heller_zeta
    "..."


def transform_heller_ECG(heller_C, heller_q, heller_p, heller_zeta):
    """
    This function transforms the Heller wavepacket to the unnormalized ECG.
    """
    D = 1j * heller_C / 2
    D_inv = np.linalg.inv(D)
    ECG_A = -np.real(D)
    invA = np.linalg.inv(ECG_A)
    ECG_B = -np.imag(D)
    b = D @ heller_q - 0.5 * 1j * heller_p
    d = len(heller_q)
    mu = D_inv @ b
    ECG_mu = mu.real
    ECG_p = mu.imag
    ECG_normalization_log = (
        (ECG_p.T @ ECG_B @ invA @ ECG_B @ ECG_p + ECG_p.T @ ECG_A @ ECG_p)
        + d / 4 * log(pi / 2)
        - 0.25 * log(np.linalg.det(ECG_A))
    )
    ECG_c_unnormalized = np.exp(
        1j * heller_zeta
        - 1j * heller_p.T @ heller_q
        + heller_q.T @ D @ heller_q
        - b @ D_inv @ b
    )
    ECG_c = ECG_c_unnormalized  # *np.exp(ECG_normalization_log)
    return ECG_A, ECG_B, ECG_mu, ECG_p, ECG_c


def transform_ECG_heller_fullyUnnormalized(ECG_A, ECG_B, ECG_mu, ECG_p, ECG_c):
    """
    This function transforms the unnormalized ECG to the Heller wavepacket.
    """
    d = len(ECG_p)
    D = -(ECG_A + 1j * ECG_B)
    heller_C = -2j * D
    Dinv = np.linalg.inv(D)
    invA = np.linalg.inv(ECG_A)
    heller_q = (-ECG_p.T @ ECG_B + ECG_mu.T @ ECG_A) @ np.linalg.inv(ECG_A)
    b = D @ (ECG_mu + 1j * ECG_p)
    heller_p = 2j * (b - D @ heller_q)
    ECG_normalization_log = (
        (ECG_p.T @ ECG_B @ invA @ ECG_B @ ECG_p + ECG_p.T @ ECG_A @ ECG_p)
        + d / 4 * log(pi / 2)
        - 0.25 * log(np.linalg.det(ECG_A))
    )
    heller_zeta = -1j * (
        np.log(ECG_c) + 1j * heller_p.T @ heller_q - heller_q.T @ D @ heller_q + b @ Dinv @ b
    )
    return heller_C, heller_q, heller_p, heller_zeta


def eval_heller(r, C, q, p, zeta):
    """
    This function evaluates the Heller wavepacket at the position r.
    """
    d = len(r)
    return exp(1j * (0.5 * (r - q).T @ C @ (r - q) + p.T @ (r - q) + zeta))


def eval_ECG(r, ECG_A, ECG_B, ECG_mu, ECG_p, ECG_c):
    """
    This function evaluates the ECG at the position r.
    """
    d = len(r)
    shift = ECG_mu + 1j * ECG_p
    invA = np.linalg.inv(ECG_A)
    return ECG_c * exp(
        -(r - shift).T @ (ECG_A + 1j * ECG_B) @ (r - shift)
    )  # /normalization_inv


def propagate_kinetic_analytical(nonlin_params, lin_params, h):
    """
    Uses eq. (mogo mogo) in (mogo mogo) to analytically propagate the Gaussians using the kinetic energy operator with time step h
    """

    def propagate_kinetic(C, q, p, zeta, dt):
        q_full = q + dt * p
        tempmat = np.eye(len(q)) + dt * C
        C_full_min = C @ np.linalg.inv(tempmat)
        zeta_full_min = zeta + 0.5 * dt * np.inner(p, p) + 1j / 2 * np.trace(logm(tempmat))
        return C_full_min, q_full, p, zeta_full_min

    new_params = np.zeros_like(nonlin_params)
    new_lincoeff = np.zeros(len(lin_params), dtype=complex)
    nd = dim = np.rint(0.5 * (-3 + sqrt(4 * len(nonlin_params[0]) + 9))).astype(int)
    ng = len(lin_params)
    mytril_indices = np.tril_indices(nd)
    L_matrices = np.zeros((ng, nd, nd))
    K_matrices = np.zeros((ng, nd, nd))
    mytril_indices = np.tril_indices(nd)
    for i in range(ng):
        L_matrices[i, :, :][mytril_indices] = nonlin_params[i][: int(0.5 * nd * (nd + 1))]
        K_matrices[i, :, :][mytril_indices] = nonlin_params[i][
            int(0.5 * nd * (nd + 1)) : (nd * (nd + 1))
        ]
    vectors = (
        nonlin_params[:, (nd * (nd + 1)) : (nd * (nd + 1)) + nd]
        + 1j * nonlin_params[:, nd * (nd + 1) + nd :]
    )
    matrices = einsum("ijk,ilk->ijl", L_matrices, L_matrices) - 1j
    matrices += 1j
    matrices += 1j * (K_matrices + einsum("ijk->ikj", K_matrices))

    for i in range(ng):
        c = lin_params[i]
        if abs(c) < 1e-16:
            new_params[i] = nonlin_params[i]
            new_lincoeff[i] = lin_params[i]
            continue
        C = matrices[i]
        A = np.real(C)
        B = np.imag(C)
        mu_ECG = np.real(vectors[i])
        p_ECG = np.imag(vectors[i])
        heller_C, heller_q, heller_p, heller_zeta = transform_ECG_heller(
            A, B, mu_ECG, p_ECG, c
        )
        heller_C_new, heller_q_new, heller_p_new, heller_zeta_new = propagate_kinetic(
            heller_C, heller_q, heller_p, heller_zeta, h
        )
        A_new, B_new, mu_ECG_new, p_ECG_new, c_new = transform_heller_ECG(
            heller_C_new, heller_q_new, heller_p_new, heller_zeta_new
        )
        params_L = np.linalg.cholesky(A_new)[np.tril_indices(dim)]
        Bcopy = np.copy(B_new)
        np.fill_diagonal(Bcopy, np.diag(Bcopy) / 2)
        params_K = Bcopy[np.tril_indices(dim)]
        params = np.concatenate((params_L, params_K, mu_ECG_new, p_ECG_new))
        new_params[i] = params
        new_lincoeff[i] = (
            c_new / abs(c_new) * abs(c)
        )  # Only the phase is updated - the Gaussians are normalized!
    return new_params, new_lincoeff
