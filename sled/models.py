"""Models: single-particle matrix h0 and the interaction as a sum of perfect squares.

Single-particle index: ``i = site * n_orb + orb`` (same as ed_twoband.py).
One fermion flavour that contains the spin explicitly (no N_SUN reduction).

Every interaction is written as

    H_int = sum_v [ -W_v (psi^dag A_v psi + alpha_v)^2 + e_v ],

with A_v a Hermitian p x p matrix acting on the orbitals P_v of one site and W_v > 0.
Each vertex v gets its own discrete HS field (see hs.py). ``e_v`` is a constant
that only enters the measured energy.

Models
------
hubbard  : H = -t sum_<ij>,s c^dag c - chem N + U sum_i (n_up - 1/2)(n_dn - 1/2)
           U > 0: spin channel   -U/2 (n_up - n_dn)^2 + U/4
           U < 0: charge channel -|U|/2 (n_up + n_dn - 1)^2 + |U|/4
twoband  : Christensen et al. H_0 (c, d orbitals) - chem N
           - U sum_i sum_{a in coupling} (S^a_i)^2,
           S^a_i = sum c^dag_a sigma^a_ab d_b + h.c.;  coupling in Ising / XY / Heisenberg.
"""
from dataclasses import dataclass, field

import numpy as np

from .lattice import Lattice

SIGMA = {
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
}
COUPLINGS = {"Ising": ("z",), "XY": ("x", "y"), "Heisenberg": ("x", "y", "z")}


@dataclass
class Model:
    name: str
    lattice: Lattice
    n_orb: int
    orb_labels: list
    h0: np.ndarray                 # (Ndim, Ndim) Hermitian, includes -chem
    # vertices
    P: np.ndarray                  # (n_v, p) int, orbital indices of vertex v
    vtype: np.ndarray              # (n_v,) int, index into A, W, alpha, e_const
    A: np.ndarray                  # (n_types, p, p)
    W: np.ndarray                  # (n_types,)
    alpha: np.ndarray              # (n_types,)
    e_const: np.ndarray            # (n_types,)
    vlabels: list                  # name of each vertex type
    # measurements: on-site bilinears O = psi^dag A psi and pair operators
    # Delta = sum_ab B_ab c_a c_b, both as n_orb x n_orb matrices
    bilinears: dict = field(default_factory=dict)
    pairs: dict = field(default_factory=dict)
    orbital_groups: dict = field(default_factory=dict)   # name -> list of orbitals

    @property
    def n_dim(self):
        return self.h0.shape[0]

    @property
    def n_v(self):
        return len(self.vtype)

    @property
    def dtype(self):
        cplx = np.iscomplexobj(self.h0) or np.iscomplexobj(self.A)
        return np.complex128 if cplx else np.float64


def _real_if_possible(M):
    M = np.asarray(M)
    if np.iscomplexobj(M) and np.abs(M.imag).max(initial=0.0) < 1e-14:
        return M.real.copy()
    return M


def hopping_matrix(lat, n_orb, t_of):
    """h[i,j] from bonds. t_of(label, orb) -> hopping t (enters as -t c^dag c + h.c.)."""
    N = lat.n_sites * n_orb
    h = np.zeros((N, N))
    for (i, j, label) in lat.bonds():
        for o in range(n_orb):
            t = t_of(label, o)
            if t != 0.0:
                h[i * n_orb + o, j * n_orb + o] += -t
                h[j * n_orb + o, i * n_orb + o] += -t
    return h


def hubbard(lat, t=1.0, U=4.0, chem=0.0):
    n_orb = 2
    h0 = hopping_matrix(lat, n_orb, lambda label, o: t)
    h0 -= chem * np.eye(h0.shape[0])
    Ns = lat.n_sites
    P = np.array([[n_orb * s, n_orb * s + 1] for s in range(Ns)], dtype=np.int64)
    if U >= 0:
        A = np.diag([1.0, -1.0])[None]
        W, alpha = np.array([U / 2]), np.array([0.0])
        lab = ["n_up-n_dn"]
    else:
        A = np.eye(2)[None]
        W, alpha = np.array([-U / 2]), np.array([-1.0])
        lab = ["n-1"]
    bil = {"n": np.eye(2), "Sz": np.diag([1.0, -1.0]), "Sx": SIGMA["x"].real.copy()}
    pairs = {"s": np.array([[0, 0.5], [-0.5, 0]])}     # Delta = c_up c_dn
    return Model("hubbard", lat, n_orb, ["up", "dn"], h0, P, np.zeros(Ns, np.int64),
                 A, W, alpha, np.array([abs(U) / 4]), lab, bil, pairs,
                 {"N_up": [0], "N_dn": [1]})


def twoband(lat, coupling="Ising", t=1.0, delta=0.4, mu=-2.0, U=1.0, chem=0.0):
    """Two-band model of Christensen et al. PRL 125, 247001 (2020).

    Orbitals per site: 0 = c_up, 1 = c_dn, 2 = d_up, 3 = d_dn.
    c hops t+delta along x and t-delta along y; d the other way round.
    On a chain only the x bonds exist.
    """
    if coupling not in COUPLINGS:
        raise ValueError(f"coupling must be one of {list(COUPLINGS)}")
    n_orb = 4
    thop = {("x", "c"): t + delta, ("y", "c"): t - delta,
            ("x", "d"): t - delta, ("y", "d"): t + delta}
    h0 = hopping_matrix(lat, n_orb, lambda label, o: thop[(label, "c" if o < 2 else "d")])
    onsite = np.array([-mu, -mu, mu, mu]) - chem        # -mu (n_c - n_d) - chem n
    h0 += np.diag(np.tile(onsite, lat.n_sites))

    comps = COUPLINGS[coupling]
    Smat = {}
    for a in "xyz":
        S = np.zeros((4, 4), dtype=complex)
        S[0:2, 2:4] = SIGMA[a]
        S[2:4, 0:2] = SIGMA[a].conj().T
        Smat[a] = _real_if_possible(S)
    A = np.array([Smat[a] for a in comps])
    A = _real_if_possible(A)
    Ns = lat.n_sites
    P, vtype = [], []
    for s in range(Ns):
        for ia in range(len(comps)):
            P.append(list(range(4 * s, 4 * s + 4)))
            vtype.append(ia)
    pairs = np.zeros((4, 4))
    pairs[0, 1], pairs[1, 0], pairs[2, 3], pairs[3, 2] = 1, -1, -1, 1   # 2(c_up c_dn - d_up d_dn)
    bil = {"n": np.eye(4), **{f"S{a}": Smat[a] for a in "xyz"}}
    return Model(f"twoband_{coupling}", lat, n_orb, ["c_up", "c_dn", "d_up", "d_dn"], h0,
                 np.array(P, np.int64), np.array(vtype, np.int64), A,
                 np.full(len(comps), float(U)), np.zeros(len(comps)), np.zeros(len(comps)),
                 [f"S{a}^2" for a in comps], bil, {"P": pairs},
                 {"N_c": [0, 1], "N_d": [2, 3]})


def make_model(p, lat):
    """Build a model from a Params object."""
    if p.model == "hubbard":
        return hubbard(lat, t=p.t, U=p.U, chem=p.chem)
    if p.model == "twoband":
        return twoband(lat, coupling=p.coupling, t=p.t, delta=p.delta, mu=p.mu, U=p.U,
                       chem=p.chem)
    raise ValueError(f"unknown model {p.model!r}")
