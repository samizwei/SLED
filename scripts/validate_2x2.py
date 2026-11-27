"""QMC on the 2x2 two-band model vs ../../Hubbard_Model_Check/ed_twoband.py (beta=2, chem=0.3).

Usage: python scripts/validate_2x2.py COUPLING DTAU [n_bins]
Prints the quantities in the same naming as the ED script.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import sled

coupling, dtau = sys.argv[1], float(sys.argv[2])
n_bins = int(sys.argv[3]) if len(sys.argv) > 3 else 40
p = sled.Params(model="twoband", coupling=coupling, lattice="square", L1=2, L2=2, U=1.0,
                chem=0.3, beta=2.0, dtau=dtau, n_warmup=50, n_bins=n_bins, n_sweeps=50, seed=5)
res = sled.DQMC(p).run()
s = res.summary(n_discard=2)
names = {"E_kin": "Kin", "E_int": "Pot", "E_tot": "Ener", "N": "Part", "N_c": "N_c", "N_d": "N_d"}
print(f"QMC {coupling} dtau={dtau}  sign={s.loc['sign','mean']:.4f}  stab={res.bins.stab_err.max():.1e}")
for k, n in names.items():
    print(f"{n:14s} {s.loc[k,'mean']: .5f} +- {s.loc[k,'err']:.5f}")
for a in "xyz":
    for q, lab in (((np.pi, np.pi), "(pi,pi)"), ((0, 0), "(0,0)")):
        r = res.structure_factor(f"S{a}", [q], n_discard=2).iloc[0]
        print(f"S{a}{a}{lab:9s}  {r['mean']: .5f} +- {r['err']:.5f}")
r = res.structure_factor("P", [(0, 0)], n_discard=2).iloc[0]
print(f"P(0,0)         {r['mean']: .5f} +- {r['err']:.5f}")
