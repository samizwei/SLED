"""Small independent ED (full Fock space, Jordan-Wigner) for test references.

Only the single-particle matrix h0 is taken from SLED (it is checked against the
analytic dispersion in test_core.py). The interaction is built directly from fermion
operators, independent of SLED's vertex/HS representation.
"""
import numpy as np
import scipy.sparse as sp


class Fock:
    def __init__(self, M):
        self.M = M
        D = 2 ** M
        st = np.arange(D)
        self.c = []
        for j in range(M):
            occ = (st >> j) & 1
            below = np.zeros(D, dtype=np.int64)
            for k in range(j):
                below += (st >> k) & 1
            sgn = np.where(below % 2 == 0, 1.0, -1.0)
            src = st[occ == 1]
            self.c.append(sp.csr_matrix((sgn[occ == 1], (src ^ (1 << j), src)), shape=(D, D)))
        self.cd = [o.T.tocsr() for o in self.c]
        self.ntot = sum(((st >> j) & 1) for j in range(M))

    def n(self, j):
        return self.cd[j] @ self.c[j]

    def bilinear(self, h):
        D = 2 ** self.M
        H = sp.csr_matrix((D, D), dtype=complex)
        for i in range(self.M):
            for j in range(self.M):
                if abs(h[i, j]) > 1e-14:
                    H = H + h[i, j] * (self.cd[i] @ self.c[j])
        return H


def hubbard_int(F, n_sites, U):
    return sum(U * (F.n(2 * s) - 0.5 * sp.identity(2 ** F.M)) @ (F.n(2 * s + 1) - 0.5 * sp.identity(2 ** F.M))
               for s in range(n_sites))


def twoband_int(F, n_sites, U, comps):
    sig = {"x": np.array([[0, 1], [1, 0]], complex), "y": np.array([[0, -1j], [1j, 0]]),
           "z": np.array([[1, 0], [0, -1]], complex)}
    H = 0
    for s in range(n_sites):
        for a in comps:
            S = 0
            for al in range(2):
                for be in range(2):
                    v = sig[a][al, be]
                    if v != 0:
                        S = S + v * (F.cd[4 * s + al] @ F.c[4 * s + 2 + be])
            S = S + S.conj().T
            H = H - U * (S @ S)
    return H


def thermal(H0, Hint, F, beta):
    H = (H0 + Hint).toarray()
    E, V = np.linalg.eigh(H)
    w = np.exp(-beta * (E - E.min()))
    Z = w.sum()

    def ev(O):
        X = O @ V                                   # sparse x dense
        return np.real((np.sum(V.conj() * X, axis=0) * w).sum()) / Z

    Nop = sp.diags(F.ntot.astype(float))
    return {"E_kin": ev(H0), "E_int": ev(Hint), "E_tot": ev(H0 + Hint), "N": ev(Nop)}


def ground(H0, Hint, F, n_part):
    idx = np.where(F.ntot == n_part)[0]
    H = (H0 + Hint).toarray()[np.ix_(idx, idx)]
    E, V = np.linalg.eigh(H)
    v = V[:, 0]
    gap = E[1] - E[0]

    def ev(O):
        O = O.toarray()[np.ix_(idx, idx)]
        return np.real(v.conj() @ O @ v)

    return {"E_kin": ev(H0), "E_int": ev(Hint), "E_tot": E[0], "gap": gap}


def site_op(F, n_orb, site, A):
    """O_site = sum_ab A_ab c^dag_{site,a} c_{site,b} (sparse)."""
    O = 0
    for a in range(n_orb):
        for b in range(n_orb):
            if abs(A[a, b]) > 1e-14:
                O = O + A[a, b] * (F.cd[n_orb * site + a] @ F.c[n_orb * site + b])
    return O


def tau_corr_thermal(H, F, beta, Oi, Oj, taus):
    """<O_i(tau) O_j(0)> = Tr[e^{-(beta-tau)H} O_i e^{-tau H} O_j] / Z."""
    E, V = np.linalg.eigh(H.toarray())
    E = E - E.min()
    Oi_ = V.conj().T @ (Oi @ V)
    Oj_ = V.conj().T @ (Oj @ V)
    Z = np.exp(-beta * E).sum()
    out = []
    for t in taus:
        w1 = np.exp(-(beta - t) * E)
        w2 = np.exp(-t * E)
        out.append(np.real(np.einsum("n,nm,m,mn->", w1, Oi_, w2, Oj_)) / Z)
    return np.array(out)


def tau_corr_ground(H, F, n_part, Oi, Oj, taus):
    """<0| O_i e^{-tau (H - E0)} O_j |0> in the n_part sector (O conserves N)."""
    idx = np.where(F.ntot == n_part)[0]
    Hs = H.toarray()[np.ix_(idx, idx)]
    E, V = np.linalg.eigh(Hs)
    Oi_ = V.conj().T @ Oi.toarray()[np.ix_(idx, idx)] @ V
    Oj_ = V.conj().T @ Oj.toarray()[np.ix_(idx, idx)] @ V
    return np.array([np.real(np.sum(Oi_[0, :] * np.exp(-t * (E - E[0])) * Oj_[:, 0])) for t in taus])
