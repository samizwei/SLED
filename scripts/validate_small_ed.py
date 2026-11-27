"""QMC vs small independent ED on chains (finite T and projector).

Usage: python scripts/validate_small_ed.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import numpy as np
import sled
from ed_small import Fock, hubbard_int, twoband_int, thermal, ground
from sled.models import COUPLINGS

CASES = [
    dict(model="hubbard", lattice="chain", L1=4, U=4.0, chem=0.3, beta=2.0, dtau=0.05),
    dict(model="hubbard", lattice="chain", L1=4, U=-4.0, chem=0.3, beta=2.0, dtau=0.05),
    dict(model="twoband", coupling="Ising", lattice="chain", L1=3, U=1.0, chem=0.3, beta=2.0, dtau=0.05),
    dict(model="twoband", coupling="XY", lattice="chain", L1=3, U=1.0, chem=0.3, beta=2.0, dtau=0.025),
    dict(model="twoband", coupling="Heisenberg", lattice="chain", L1=3, U=1.0, chem=0.3, beta=2.0, dtau=0.025),
    dict(model="hubbard", lattice="chain", L1=6, U=2.0, algorithm="projector", theta=4.0, beta=0.5, dtau=0.05, n_part=6),
    dict(model="twoband", coupling="Heisenberg", lattice="chain", L1=3, U=1.0, algorithm="projector",
         theta=4.0, beta=0.5, dtau=0.025, n_part=6),
]


def ed_reference(p, m):
    Ns = m.lattice.n_sites
    F = Fock(m.n_dim)
    H0 = F.bilinear(m.h0)
    Hint = hubbard_int(F, Ns, p.U) if p.model == "hubbard" else twoband_int(F, Ns, p.U, COUPLINGS[p.coupling])
    if p.algorithm == "projector":
        return ground(H0, Hint, F, p.n_part)
    return thermal(H0, Hint, F, p.beta)


def main(n_bins=20, n_sweeps=50):
    for c in CASES:
        p = sled.Params(**c, n_warmup=20, n_bins=n_bins, n_sweeps=n_sweeps, seed=11, measure_corr=False)
        sim = sled.DQMC(p)
        ref = ed_reference(p, sim.model)
        t = time.time()
        res = sim.run()
        s = res.summary(n_discard=2)
        tag = f"{p.model} {p.coupling if p.model == 'twoband' else ''} {p.algorithm} U={p.U}"
        print(f"\n{tag}  ({time.time() - t:.1f}s, sign={s.loc['sign','mean']:.4f}, "
              f"acc={res.bins.acc_rate.mean():.3f}, stab={res.bins.stab_err.max():.1e})")
        for k in ("E_kin", "E_int", "E_tot", "N"):
            if k in ref:
                m_, e_ = s.loc[k, "mean"], s.loc[k, "err"]
                print(f"  {k:6s} QMC {m_: .5f} +- {e_:.5f}   ED {ref[k]: .5f}   dev {(m_ - ref[k]) / e_: .1f} sigma")


if __name__ == "__main__":
    main()
