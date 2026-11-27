"""Compare SLED and ALF series (same model, same time steps) and their dtau -> 0 limits.

Usage:
  python scripts/compare_alf.py --sled runs/sled_4x4_Heisenberg --alf runs/alf_4x4_Heisenberg \
         [--n-discard 2] [--out comparison]

Both directories contain dtau_<dtau>/ subdirectories (scripts/dtau_extrapolate.py and
scripts/run_alf_series.sh). ALF output is read with ALF_SZ/ALF/Infos_SZ/alf_to_pandas.py
(ana.out results, n_skip from the ALF parameter file).
Compared: Kin, Pot, Ener, Part, N_c, N_d and S^aa(q), P(q) at q = (0,0), (pi,pi).
ALF subtracts the background <S^a>^2 from the spin correlators; it vanishes by symmetry here.
Writes <out>_per_dtau.csv and <out>_extrapolated.csv into the SLED directory.
"""
import argparse, glob, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "ALF_SZ", "ALF", "Infos_SZ"))
import numpy as np
import pandas as pd
import alf_to_pandas as ap
import sled
from sled.analysis import observable_table, fit_dtau2

SCAL = {"Kin": "E_kin", "Pot": "E_int", "Ener": "E_tot", "Part": "N", "N_c": "N_c", "N_d": "N_d"}
EQ = {"SpinX": "Sx", "SpinY": "Sy", "SpinZ": "Sz", "Pair": "P"}


def alf_table(run):
    d = ap.load_run(run, raw=False)
    rows = [(SCAL[r.label], r["mean"], r["error"]) for _, r in d.scal.iterrows() if r.label in SCAL]
    for _, r in d.eq.iterrows():
        if r.obs in EQ:
            for lab, q in (("(0,0)", (0, 0)), ("(pi,pi)", (np.pi, np.pi))):
                if np.allclose([r.kx, r.ky], q, atol=1e-6):
                    rows.append((EQ[r.obs] + lab, r.re, r.re_error))
    return pd.DataFrame(rows, columns=["obs", "mean", "err"])


def series(base):
    out = {}
    for d in glob.glob(os.path.join(base, "dtau_[0-9]*")):
        out[float(re.sub(r".*dtau_", "", d))] = d
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--sled", required=True)
    a.add_argument("--alf", required=True)
    a.add_argument("--n-discard", type=int, default=2)
    a.add_argument("--out", default="comparison")
    a = a.parse_args()
    S, A = series(a.sled), series(a.alf)
    parts = []
    for dt in sorted(set(S) & set(A), reverse=True):
        ts = observable_table(sled.Results.load(S[dt]), a.n_discard)
        ta = alf_table(A[dt])
        m = ts.merge(ta, on="obs", suffixes=("_sled", "_alf"))
        m.insert(1, "dtau", dt)
        parts.append(m)
    per = pd.concat(parts, ignore_index=True)
    per["dev_sigma"] = (per.mean_sled - per.mean_alf) / np.hypot(per.err_sled, per.err_alf)
    fs = fit_dtau2(per.rename(columns={"mean_sled": "mean", "err_sled": "err"})[["obs", "dtau", "mean", "err"]])
    fa = fit_dtau2(per.rename(columns={"mean_alf": "mean", "err_alf": "err"})[["obs", "dtau", "mean", "err"]])
    ext = fs.merge(fa, on="obs", suffixes=("_sled", "_alf"))
    ext["dev_sigma"] = (ext.O0_sled - ext.O0_alf) / np.hypot(ext.O0_err_sled, ext.O0_err_alf)
    per.to_csv(os.path.join(a.sled, f"{a.out}_per_dtau.csv"), index=False)
    ext.to_csv(os.path.join(a.sled, f"{a.out}_extrapolated.csv"), index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.5f}".format)
    print(per.pivot(index="obs", columns="dtau", values="dev_sigma").round(2).to_string())
    print()
    print(ext[["obs", "O0_sled", "O0_err_sled", "O0_alf", "O0_err_alf", "dev_sigma"]].to_string(index=False))


if __name__ == "__main__":
    main()
