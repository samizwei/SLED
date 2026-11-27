"""Numerical stabilization: UDT decompositions and stable Green's functions.

Right products  B(tau,0) R0 = U D T   (U: isometric columns, D > 0, T well conditioned)
Left products   L0 B(L,tau) = T D U   (U: isometric rows)

R0 = 1, L0 = 1 for finite T;  R0 = P, L0 = P^dag for the projector (P: N x Np trial state).
The decompositions use column-pivoted QR (scipy.linalg.qr, LAPACK geqp3).
"""
import numpy as np
import scipy.linalg as sla


class UDT:
    """Container for (U, D, T). For a left product the meaning is T D U."""
    __slots__ = ("U", "D", "T")

    def __init__(self, U, D, T):
        self.U, self.D, self.T = U, D, T

    def copy(self):
        return UDT(self.U.copy(), self.D.copy(), self.T.copy())


def qr_udt(M):
    """M = U diag(D) T with column-pivoted QR (economic). M is (n, m), n >= m."""
    Q, R, piv = sla.qr(M, mode="economic", pivoting=True)
    D = np.abs(np.diag(R))
    D[D == 0] = 1e-300
    T = np.empty_like(R)
    T[:, piv] = R / D[:, None]          # M[:, piv] = Q R  =>  M = Q R Pi^T
    return Q, D, T


def right_identity(n, dtype):
    return UDT(np.eye(n, dtype=dtype), np.ones(n), np.eye(n, dtype=dtype))


def right_from(M):
    return UDT(*qr_udt(M))


def left_from(Mrow):
    """Mrow (m x n) = T D U  via QR of Mrow^dag."""
    Q, D, T = qr_udt(Mrow.conj().T)
    return UDT(Q.conj().T, D, T.conj().T)


def right_mult(udt, B):
    """B @ (U D T)  ->  new UDT."""
    Q, D, T = qr_udt((B @ udt.U) * udt.D[None, :])
    return UDT(Q, D, T @ udt.T)


def left_mult(udt, B):
    """(T D U) @ B  ->  new UDT (left form)."""
    X = udt.D[:, None] * (udt.U @ B)
    Q, D, T = qr_udt(X.conj().T)
    return UDT(Q.conj().T, D, udt.T @ T.conj().T)



def green_finite_T(R, L):
    """G = [1 + U_R D_R T_R T_L D_L U_L]^{-1} and the phase of det(1 + B(L,0)).

    Scale separation D = D_max D_min with D_max = max(D,1), D_min = min(D,1):
        1 + ... = U_R D_Rmax M D_Lmax U_L,
        M = D_Rmax^{-1} U_R^dag U_L^dag D_Lmax^{-1} + D_Rmin (T_R T_L) D_Lmin.
    """
    DRmax, DRmin = np.maximum(R.D, 1.0), np.minimum(R.D, 1.0)
    DLmax, DLmin = np.maximum(L.D, 1.0), np.minimum(L.D, 1.0)
    URd, ULd = R.U.conj().T, L.U.conj().T
    M = (URd @ ULd) / DRmax[:, None] / DLmax[None, :] \
        + DRmin[:, None] * (R.T @ L.T) * DLmin[None, :]
    lu = sla.lu_factor(M)
    X = sla.lu_solve(lu, URd / DRmax[:, None])
    G = ULd @ (X / DLmax[:, None])
    ph = _sgn(R.U) * _lu_phase(lu) * _sgn(L.U)
    return G, ph


def green_projector(R, L):
    """G = 1 - U_R (U_L U_R)^{-1} U_L and the phase of det(P^dag B(L,0) P)."""
    O = L.U @ R.U
    N = R.U.shape[0]
    G = np.eye(N, dtype=O.dtype) - R.U @ np.linalg.solve(O, L.U)
    ph = _sgn(L.T) * _sgn(O) * _sgn(R.T)
    return G, ph


def _lu_phase(lu):
    d = np.diag(lu[0])
    return np.prod(d / np.abs(d)) * (-1) ** np.count_nonzero(lu[1] != np.arange(len(lu[1])))


def _sgn(M):
    """Phase of det(M), overflow-safe. (Avoids numpy's complex det, which raises
    spurious floating-point warnings with this OpenBLAS build.)"""
    return _lu_phase(sla.lu_factor(M, check_finite=False))


def _split(d):
    return np.maximum(d, 1.0), np.minimum(d, 1.0)


def green_tau_finite_T(R, L):
    """Time-displaced G(tau,0) = <c(tau) c^dag(0)> and G(0,tau) = -<c^dag(tau) c(0)>.

    With B(tau,0) = U_R D_R T_R and B(beta,tau) = T_L D_L U_L:
        G(tau,0) =  [B(tau,0)^{-1} + B(beta,tau)]^{-1}
                 =  U_L^dag X^{-1} T_R,     X = D_R^{-1} Q + T D_L
        G(0,tau) = -[B(tau,0) + B(beta,tau)^{-1}]^{-1}
                 = -T_L Y^{-1} U_R^dag,     Y = D_R T + Q D_L^{-1}
    with Q = U_R^dag U_L^dag, T = T_R T_L. For d1 X d2-type sums d1 A + B d2 we use
        d1 A + B d2 = d1max [d1min A d2max^{-1} + d1max^{-1} B d2min] d2max,
    so every inverse is of a well-conditioned matrix.
    """
    Q = R.U.conj().T @ L.U.conj().T
    T = R.T @ L.T
    # G(tau,0): d1 = 1/D_R (rows), d2 = D_L (cols), X = d1 Q + T d2
    d1max, d1min = _split(1.0 / R.D)
    d2max, d2min = _split(L.D)
    M = d1min[:, None] * Q / d2max[None, :] + (T / d1max[:, None]) * d2min[None, :]
    Gt0 = L.U.conj().T @ (np.linalg.solve(M, R.T / d1max[:, None]) / d2max[:, None])
    # G(0,tau): d1 = D_R, d2 = 1/D_L, Y = d1 T + Q d2
    d1max, d1min = _split(R.D)
    d2max, d2min = _split(1.0 / L.D)
    M = d1min[:, None] * T / d2max[None, :] + (Q / d1max[:, None]) * d2min[None, :]
    G0t = -L.T @ (np.linalg.solve(M, R.U.conj().T / d1max[:, None]) / d2max[:, None])
    return Gt0, G0t
