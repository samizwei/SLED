"""Time-displaced Green's functions and correlations."""
import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(__file__))
import sled
from ed_small import Fock, hubbard_int, twoband_int, site_op, tau_corr_thermal, tau_corr_ground
from sled.models import COUPLINGS


def _green_R(model, M):
    """1/Ncells sum_i sum_o M[(i+R,o),(i,o)] for a Bravais lattice."""
    lat, no = model.lattice, model.n_orb
    T = lat.translation_table()
    Ns = lat.n_sites
    Ms = np.einsum("iojo->ij", M.reshape(Ns, no, Ns, no))
    return Ms[T, np.arange(Ns)[:, None]].mean(0)


@pytest.mark.parametrize("algorithm", ["finite_T", "projector"])
def test_U0_green_tau_exact(algorithm):
    kw = dict(model="twoband", coupling="Heisenberg", lattice="square", L1=4, L2=4, U=0.0,
              chem=0.3, beta=8.0, theta=6.0, dtau=0.1, n_wrap=10, algorithm=algorithm,
              trial_eps=0.0, n_warmup=0, n_bins=1, n_sweeps=1, measure_corr=False,
              measure_tau=True)
    m = sled.make_model(sled.Params(**kw), sled.make_lattice("square", 4, 4))
    e, U = np.linalg.eigh(m.h0)
    n_part = min(n for n in range(20, 45) if e[n] - e[n - 1] > 1e-3)     # closed shell
    p = sled.Params(**kw, n_part=n_part)
    res = sled.DQMC(p).run()
    if algorithm == "finite_T":
        w = 1 / (1 + np.exp(-p.beta * e))           # 1 - f, without cancellation
    else:
        w = (np.arange(len(e)) >= n_part).astype(float)
    df = res.tau_mean("Green")
    for tau in (0.0, 1.0, 3.0, 8.0):
        exact = _green_R(m, (U * (np.exp(-tau * e) * w)) @ U.conj().T)
        got = df[np.isclose(df.tau, tau)].sort_values(["ry", "rx"])["mean"].to_numpy()
        assert np.allclose(got, exact.real, atol=1e-8), tau


@pytest.mark.parametrize("algorithm", ["finite_T", "projector"])
def test_tau0_equals_equal_time(algorithm):
    """On one configuration the tau = 0 Wick formulas must reproduce the equal-time ones."""
    p = sled.Params(model="twoband", coupling="Heisenberg", lattice="square", L1=2, L2=2,
                    U=1.0, beta=1.0, theta=1.0, dtau=0.05, algorithm=algorithm, measure_tau=True)
    sim = sled.DQMC(p)
    for _ in range(3):
        sim.sweep()
    sim._reset_bin()
    t0 = sim.window[0] if sim.proj else 0
    sim.ntau = 0
    G = sim.G.copy()
    for tau in range(1, t0 + 1):                       # bring G to tau0 (projector)
        G = sim.B_slice(tau) @ G @ sim.B_slice_inv(tau)
    sim.tau_pass()
    Gs = sim.expKhinv @ G @ sim.expKh
    eq = sim.meas.correlations(Gs)
    for name, C in eq.items():
        assert np.allclose(sim.acc_tau[name][0] / sim.phase, C, atol=1e-8), name


CASES = [
    dict(model="twoband", coupling="Heisenberg", lattice="chain", L1=2, U=1.0, chem=0.3,
         beta=2.0, dtau=0.025, obs="Sz"),
    dict(model="hubbard", lattice="chain", L1=4, U=4.0, chem=0.0, beta=2.0, dtau=0.05, obs="Sz"),
    dict(model="hubbard", lattice="chain", L1=6, U=2.0, algorithm="projector", theta=4.0,
         beta=2.0, dtau=0.05, n_part=6, obs="Sz"),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['model']}-{c.get('algorithm','finite_T')}")
def test_tau_vs_ed(case):
    case = dict(case)
    obs = case.pop("obs")
    p = sled.Params(**case, n_warmup=20, n_bins=12, n_sweeps=40, seed=4, measure_corr=False,
                    measure_tau=True)
    sim = sled.DQMC(p)
    m = sim.model
    F = Fock(m.n_dim)
    H = F.bilinear(m.h0) + (hubbard_int(F, m.lattice.n_sites, p.U) if p.model == "hubbard"
                            else twoband_int(F, m.lattice.n_sites, p.U, COUPLINGS[p.coupling]))
    A = m.bilinears[obs]
    O0, O1 = site_op(F, m.n_orb, 0, A), site_op(F, m.n_orb, 1, A)
    taus = np.array([0.0, 0.5, 1.0])
    if p.algorithm == "projector":
        ref0 = tau_corr_ground(H, F, p.n_part, O0, O0, taus)
        ref1 = tau_corr_ground(H, F, p.n_part, O1, O0, taus)
    else:
        ref0 = tau_corr_thermal(H, F, p.beta, O0, O0, taus)
        ref1 = tau_corr_thermal(H, F, p.beta, O1, O0, taus)
    df = sim.run().tau_mean(obs, n_discard=2)
    for t, r0, r1 in zip(taus, ref0, ref1):
        for rx, ref in ((0, r0), (1, r1)):
            row = df[np.isclose(df.tau, t) & (df.rx == rx) & (df.ry == 0)].iloc[0]
            assert abs(row["mean"] - ref) < 4 * row["err"] + 0.01 * abs(ref) + 1e-3, (t, rx, row["mean"], ref)
