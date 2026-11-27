"""Interacting checks against the small independent ED (a few seconds each)."""
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(__file__))
import sled
from ed_small import Fock, hubbard_int, twoband_int, thermal, ground
from sled.models import COUPLINGS

CASES = [
    dict(model="hubbard", lattice="chain", L1=4, U=4.0, chem=0.3, beta=2.0, dtau=0.05),
    dict(model="hubbard", lattice="chain", L1=4, U=-4.0, chem=0.3, beta=2.0, dtau=0.05),
    dict(model="twoband", coupling="Heisenberg", lattice="chain", L1=2, U=1.0, chem=0.3,
         beta=2.0, dtau=0.025),
    dict(model="hubbard", lattice="chain", L1=6, U=2.0, algorithm="projector", theta=4.0,
         beta=0.5, dtau=0.05, n_part=6),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['model']}-{c.get('coupling','')}-{c.get('algorithm','finite_T')}-U{c['U']}")
def test_vs_ed(case):
    p = sled.Params(**case, n_warmup=20, n_bins=12, n_sweeps=40, seed=21, measure_corr=False)
    sim = sled.DQMC(p)
    m = sim.model
    F = Fock(m.n_dim)
    H0 = F.bilinear(m.h0)
    Hint = (hubbard_int(F, m.lattice.n_sites, p.U) if p.model == "hubbard"
            else twoband_int(F, m.lattice.n_sites, p.U, COUPLINGS[p.coupling]))
    ref = ground(H0, Hint, F, p.n_part) if p.algorithm == "projector" else thermal(H0, Hint, F, p.beta)
    s = sim.run().summary(n_discard=2)
    assert abs(s.loc["sign", "mean"] - 1) < 1e-8
    for k in ("E_kin", "E_int", "E_tot"):
        assert abs(s.loc[k, "mean"] - ref[k]) < 4 * s.loc[k, "err"] + 0.01 * abs(ref[k]), k
