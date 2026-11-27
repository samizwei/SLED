"""Run (or load) a series of time steps and extrapolate O(dtau) = O0 + b dtau^2.

Usage:
  python scripts/dtau_extrapolate.py PARAMS.toml --dtaus 0.05 0.025 0.0125 --out runs/series \
         [--jobs 3] [--n-discard 2] [--meas-dt 0.05] [--plot] [key=value ...]

Each time step runs in OUT/dtau_<dtau>/ (skipped if bins.parquet exists there).
meas_every is set to round(meas_dt/dtau), so all runs measure at the same density in tau.
Writes OUT/dtau_table.csv (obs, dtau, mean, err) and OUT/dtau_fit.csv (obs, O0, O0_err, b, ...),
and OUT/dtau_extrapolation.png with --plot.
"""
import argparse, ast, os, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
import sled
from sled.analysis import observable_table, fit_dtau2


def run_dir(out, dtau):
    return os.path.join(out, f"dtau_{dtau:g}")


def _run_one(args):
    params, dtau, out, meas_dt, over = args
    d = run_dir(out, dtau)
    if os.path.exists(os.path.join(d, "bins.parquet")):
        return d
    meas_every = max(1, int(round(meas_dt / dtau)))
    sled.run(params, dtau=dtau, meas_every=meas_every, out_dir=d, **over)
    return d


def series_table(dirs_dtaus, n_discard):
    parts = []
    for d, dtau in dirs_dtaus:
        t = observable_table(sled.Results.load(d), n_discard)
        t.insert(1, "dtau", dtau)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def plot(table, fit, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    obs = [o for o in fit.obs if o != "sign"]
    nc = 4
    nr = (len(obs) + nc - 1) // nc
    fig, axs = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.5 * nr), squeeze=False)
    for ax, o in zip(axs.flat, obs):
        g = table[table.obs == o]
        ax.errorbar(g.dtau ** 2, g["mean"], g["err"], fmt="o", ms=4, capsize=2)
        f = fit[fit.obs == o].iloc[0]
        x = np.linspace(0, (g.dtau ** 2).max() * 1.05, 20)
        ax.plot(x, f.O0 + f.b * x, "-", lw=1)
        ax.errorbar([0], [f.O0], [f.O0_err], fmt="s", ms=4, capsize=2)
        ax.set_title(o, fontsize=9)
        ax.set_xlabel(r"$\Delta\tau^2$", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in list(axs.flat)[len(obs):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params")
    ap.add_argument("--dtaus", type=float, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--n-discard", type=int, default=2)
    ap.add_argument("--meas-dt", type=float, default=0.05)
    ap.add_argument("--plot", action="store_true")
    a, rest = ap.parse_known_args()
    over = {}
    for kv in rest:
        k, v = kv.split("=", 1)
        try:
            over[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            over[k] = v
    os.makedirs(a.out, exist_ok=True)
    jobs = [(a.params, dt, a.out, a.meas_dt, over) for dt in a.dtaus]
    with ProcessPoolExecutor(a.jobs) as ex:
        dirs = list(ex.map(_run_one, jobs))
    table = series_table(zip(dirs, a.dtaus), a.n_discard)
    fit = fit_dtau2(table)
    table.to_csv(os.path.join(a.out, "dtau_table.csv"), index=False)
    fit.to_csv(os.path.join(a.out, "dtau_fit.csv"), index=False)
    pd.set_option("display.width", 160)
    print(table.pivot(index="obs", columns="dtau", values="mean").to_string())
    print()
    print(fit.to_string(index=False))
    if a.plot:
        plot(table, fit, os.path.join(a.out, "dtau_extrapolation.png"))


if __name__ == "__main__":
    main()
