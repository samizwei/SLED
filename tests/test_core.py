"""Unit tests of the building blocks (fast)."""
import numpy as np
import pytest
import scipy.linalg as sla

import sled
from sled import linalg as la
from sled.hs import ETA, GAMMA, vertex_tables
from sled.kernels import vertex_sweep, vertex_sweep_numpy
from sled.lattice import chain, square
from sled.models import hubbard, twoband


# ------------------------------------------------------------------ HS
def test_ghs_identity_dtau4():
    """1/4 sum_l gamma_l e^{g eta_l x} = e^{g^2 x^2} + O(g^8) for the integer
    eigenvalues x of O (here up to |x| = 2)."""
    errs = []
    for g in (0.2, 0.1):
        x = np.arange(-2, 3)
        lhs = 0.25 * (GAMMA[None] * np.exp(g * np.outer(x, ETA))).sum(1)
        errs.append(np.abs(lhs - np.exp(g ** 2 * x ** 2)).max())
    assert errs[0] < 1e-4
    assert errs[0] / errs[1] > 2 ** 8 * 0.8          # error ~ g^8 = dtau^4


# -------------------------------------------------------------- lattice / h0
@pytest.mark.parametrize("L1,L2", [(4, 4), (2, 2), (4, 2)])
def test_twoband_h0_spectrum(L1, L2):
    t, d, mu = 1.0, 0.4, -2.0
    m = twoband(square(L1, L2), t=t, delta=d, mu=mu, U=0.0)
    ks = m.lattice.k_points()
    ec = -2 * (t + d) * np.cos(ks[:, 0]) - 2 * (t - d) * np.cos(ks[:, 1]) - mu
    ed = -2 * (t - d) * np.cos(ks[:, 0]) - 2 * (t + d) * np.cos(ks[:, 1]) + mu
    expect = np.sort(np.concatenate([ec, ec, ed, ed]))
    assert np.allclose(np.linalg.eigvalsh(m.h0), expect)


def test_chain_hubbard_h0_spectrum():
    m = hubbard(chain(6), t=1.0, U=0.0, chem=0.3)
    k = 2 * np.pi * np.arange(6) / 6
    e = -2 * np.cos(k) - 0.3
    assert np.allclose(np.linalg.eigvalsh(m.h0), np.sort(np.concatenate([e, e])))


# ---------------------------------------------------------- stabilization
def _random_B(N, rng, dtype, scale):
    X = rng.normal(size=(N, N))
    if dtype == np.complex128:
        X = X + 1j * rng.normal(size=(N, N))
    return sla.expm(scale * (X + X.conj().T) / 2)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_green_finite_T_vs_direct(dtype):
    rng = np.random.default_rng(0)
    N = 8
    Bs = [_random_B(N, rng, dtype, 0.3) for _ in range(6)]
    R = la.right_identity(N, dtype)
    for B in Bs[:3]:
        R = la.right_mult(R, B)
    L = la.right_identity(N, dtype)
    for B in Bs[:2:-1]:          # left product B6 B5 B4 built from the left boundary
        L = la.left_mult(L, B)
    full = np.eye(N, dtype=dtype)
    for B in Bs:
        full = B @ full
    # G(tau=3) = [1 + B3 B2 B1 B6 B5 B4]^{-1}
    prod = Bs[2] @ Bs[1] @ Bs[0] @ Bs[5] @ Bs[4] @ Bs[3]
    G, ph = la.green_finite_T(R, L)
    assert np.allclose(G, np.linalg.inv(np.eye(N) + prod), atol=1e-10)
    d = np.linalg.det(np.eye(N) + prod)
    assert np.isclose(ph, d / abs(d))


def test_green_projector_vs_direct():
    rng = np.random.default_rng(1)
    N, Np = 8, 3
    Bs = [_random_B(N, rng, np.float64, 0.3) for _ in range(4)]
    P = np.linalg.qr(rng.normal(size=(N, Np)))[0]
    R = la.right_from(P)
    for B in Bs[:2]:
        R = la.right_mult(R, B)
    L = la.left_from(P.T)
    for B in Bs[:1:-1]:
        L = la.left_mult(L, B)
    right = Bs[1] @ Bs[0] @ P
    left = P.T @ Bs[3] @ Bs[2]
    G_ex = np.eye(N) - right @ np.linalg.solve(left @ right, left)
    G, ph = la.green_projector(R, L)
    assert np.allclose(G, G_ex, atol=1e-10)
    assert np.isclose(ph, np.sign(np.linalg.det(left @ right)))


# ------------------------------------------------------------- kernel
@pytest.mark.parametrize("coupling", ["Ising", "XY", "Heisenberg"])
@pytest.mark.parametrize("up", [True, False])
def test_kernel_numba_equals_numpy_and_recompute(coupling, up):
    """One slice of local updates: Numba == NumPy, and the updated G equals the
    Green's function recomputed from scratch with the new fields."""
    p = sled.Params(model="twoband", coupling=coupling, L1=2, L2=2, U=2.0, beta=1.0,
                    dtau=0.1, n_wrap=5, seed=7)
    sim = sled.DQMC(p)
    m = sim.model
    tau = 3

    def g_stable(t):
        # cond(B(L,0)) ~ 1e17 here: the reference must be stabilized slice by slice
        R = la.right_identity(sim.N, sim.dtype)
        for s in range(1, t + 1):
            R = la.right_mult(R, sim.B_slice(s))
        L = la.right_identity(sim.N, sim.dtype)
        for s in range(sim.L, t, -1):
            L = la.left_mult(L, sim.B_slice(s))
        return la.green_finite_T(R, L)[0]

    G0 = g_stable(tau - 1 if up else tau)
    if not up:
        G0 = sim.expKinv @ G0 @ sim.expK
    rng = np.random.default_rng(3)
    shift = rng.integers(1, 4, size=m.n_v).astype(np.int64)
    u = rng.random(m.n_v) * 0.5
    res = []
    for kern in (vertex_sweep_numpy, vertex_sweep):
        G = G0.copy()
        f = sim.fields[tau - 1].copy()
        n_acc, ph = kern(G, m.P, m.vtype, f, sim.expA, sim.expAinv, sim.delta, sim.logw,
                         shift, u, up)
        res.append((G, f, n_acc, ph))
    assert np.allclose(res[0][0], res[1][0], atol=1e-12)
    assert np.array_equal(res[0][1], res[1][1]) and res[0][2] == res[1][2]
    assert res[0][2] > 0
    # recompute with the new fields
    G, f = res[1][0], res[1][1]
    sim.fields[tau - 1] = f
    if up:
        G = sim.expK @ G @ sim.expKinv
        t_at = tau
    else:
        t_at = tau - 1
    assert np.allclose(G, g_stable(t_at), atol=1e-11)


# ------------------------------------------------------------- U = 0 exact
def _free_finite_T(h0, beta):
    e, U = np.linalg.eigh(h0)
    f = 1 / (np.exp(beta * e) + 1)
    Gc = (U * f) @ U.conj().T           # <c_i^dag c_j> transposed: Gc[i,j] = <c_i^dag c_j>
    return Gc.T, (e * f).sum(), f.sum()


@pytest.mark.parametrize("model,lattice,L1,L2", [("twoband", "square", 4, 4),
                                                ("hubbard", "chain", 8, 1)])
def test_U0_finite_T_exact(model, lattice, L1, L2):
    p = sled.Params(model=model, lattice=lattice, L1=L1, L2=L2, U=0.0, chem=0.3, beta=10.0,
                    dtau=0.1, n_wrap=10, n_warmup=0, n_bins=1, n_sweeps=1, measure_corr=False)
    res = sled.DQMC(p).run()
    s = res.summary()
    m = sled.DQMC(p).model
    Gc, E, N = _free_finite_T(m.h0, p.beta)
    assert np.isclose(s.loc["E_kin", "mean"], E, atol=1e-8)
    assert np.isclose(s.loc["N", "mean"], N, atol=1e-8)
    assert res.bins.stab_err.max() < 1e-8


def test_U0_projector_exact():
    p = sled.Params(model="hubbard", lattice="chain", L1=6, U=0.0, algorithm="projector",
                    theta=5.0, beta=1.0, dtau=0.1, trial_eps=0.0, n_part=6,
                    n_warmup=0, n_bins=1, n_sweeps=1, measure_corr=False)
    res = sled.DQMC(p).run()
    m = hubbard(chain(6), U=0.0)
    e = np.linalg.eigvalsh(m.h0)
    assert np.isclose(res.summary().loc["E_kin", "mean"], e[:6].sum(), atol=1e-8)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_green_tau_finite_T_vs_direct(dtype):
    rng = np.random.default_rng(4)
    N = 8
    Bs = [_random_B(N, rng, dtype, 0.3) for _ in range(6)]
    R = la.right_identity(N, dtype)
    for B in Bs[:3]:
        R = la.right_mult(R, B)
    L = la.right_identity(N, dtype)
    for B in Bs[:2:-1]:
        L = la.left_mult(L, B)
    Bt0 = Bs[2] @ Bs[1] @ Bs[0]
    Bbt = Bs[5] @ Bs[4] @ Bs[3]
    G0 = np.linalg.inv(np.eye(N) + Bbt @ Bt0)
    Gt0, G0t = la.green_tau_finite_T(R, L)
    assert np.allclose(Gt0, Bt0 @ G0, atol=1e-10)                       # G(tau,0) = B(tau,0) G(0)
    assert np.allclose(G0t, -(np.eye(N) - G0) @ np.linalg.inv(Bt0), atol=1e-10)
