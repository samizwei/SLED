"""Discrete 4-point Gauss-Hermite Hubbard-Stratonovich transformation.

    exp(dtau W (O + alpha)^2) = 1/4 sum_l gamma_l exp(g eta_l (O + alpha)) + O(dtau^4),
    g = sqrt(dtau W),  l in {-2, -1, +1, +2}  (stored as field index 0..3).

For every vertex type we tabulate, for each field value, the p x p matrices
exp(g eta_l A), their inverses and Delta(l -> l') = exp(g (eta_l' - eta_l) A) - 1,
plus the scalar log-weight log(gamma_l) + g eta_l alpha.
"""
import numpy as np

_s6 = np.sqrt(6.0)
GAMMA = np.array([1 - _s6 / 3, 1 + _s6 / 3, 1 + _s6 / 3, 1 - _s6 / 3])
ETA = np.array([-np.sqrt(2 * (3 + _s6)), -np.sqrt(2 * (3 - _s6)),
                np.sqrt(2 * (3 - _s6)), np.sqrt(2 * (3 + _s6))])
N_FIELD = 4


def expm_herm(A, x):
    """exp(x A) for Hermitian A and real or complex scalar x."""
    lam, U = np.linalg.eigh(A)
    return (U * np.exp(x * lam)) @ U.conj().T


def vertex_tables(model, dtau, dtype):
    nt, p = model.A.shape[0], model.A.shape[1]
    expA = np.empty((nt, N_FIELD, p, p), dtype=dtype)
    expAinv = np.empty_like(expA)
    delta = np.empty((nt, N_FIELD, N_FIELD, p, p), dtype=dtype)
    logw = np.empty((nt, N_FIELD))
    for it in range(nt):
        A = model.A[it]
        g = np.sqrt(dtau * model.W[it])
        for l in range(N_FIELD):
            expA[it, l] = _cast(expm_herm(A, g * ETA[l]), dtype)
            expAinv[it, l] = _cast(expm_herm(A, -g * ETA[l]), dtype)
            logw[it, l] = np.log(GAMMA[l]) + g * ETA[l] * model.alpha[it]
            for l2 in range(N_FIELD):
                delta[it, l, l2] = _cast(expm_herm(A, g * (ETA[l2] - ETA[l])) - np.eye(p), dtype)
    return expA, expAinv, delta, logw


def _cast(M, dtype):
    if dtype == np.float64:
        assert np.abs(np.imag(M)).max() < 1e-12
        return np.real(M)
    return M
