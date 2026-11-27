"""Results as pandas DataFrames, Parquet I/O, jackknife analysis.

``Results.bins``  one row per bin. Observable columns hold the sign-weighted bin
                  average <s O>_bin; ``sign`` holds <s>_bin. Use ``summary()`` for
                  <O> = <s O>/<s> with jackknife errors.
``Results.corr``  long format: bin, obs, rx, ry, b1, b2, value (sign-weighted),
                  C(R; b1, b2) = 1/N_cells sum_i <O_{i+R,b1} O_{i,b2}>.
``Results.corr_tau`` long format: bin, obs, tau, rx, ry, b1, b2, value (sign-weighted,
                  sign in ``bins.sign_tau``); obs = Green and all correlators of ``corr``.
``Results.params`` dict of run parameters.
"""
import json
import os

import numpy as np
import pandas as pd

DIAG_COLS = ["bin", "sign", "sign_tau", "acc_rate", "stab_err", "phase_err", "n_meas", "t_bin"]


def jackknife_ratio(num, den):
    """Jackknife mean and error of sum(num)/sum(den); num: (n_bins, ...)."""
    num, den = np.asarray(num), np.asarray(den)
    n = len(den)
    full = num.sum(0) / den.sum()
    if n < 2:
        return full, np.full_like(full, np.nan, dtype=float)
    jk = (num.sum(0)[None] - num) / (den.sum() - den).reshape((n,) + (1,) * (num.ndim - 1))
    err = np.sqrt((n - 1) / n * ((jk - jk.mean(0)) ** 2).sum(0))
    return full, err


class Results:
    def __init__(self, params, bins, corr, wall=None, corr_tau=None):
        self.params = params
        self.bins = bins
        self.corr = corr
        self.corr_tau = corr_tau if corr_tau is not None else pd.DataFrame()
        self.wall = wall

    @classmethod
    def from_run(cls, p, bin_rows, corr_rows, tau_rows=(), wall=None):
        bins = pd.DataFrame(bin_rows)
        corr = pd.DataFrame(corr_rows, columns=["bin", "obs", "rx", "ry", "b1", "b2", "value"])
        ctau = pd.DataFrame(list(tau_rows),
                            columns=["bin", "obs", "tau", "rx", "ry", "b1", "b2", "value"])
        return cls(p.as_dict(), bins, corr, wall, ctau)

    # ------------------------------------------------------------------ I/O
    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.bins.to_parquet(os.path.join(out_dir, "bins.parquet"))
        if len(self.corr):
            self.corr.to_parquet(os.path.join(out_dir, "corr.parquet"))
        if len(self.corr_tau):
            self.corr_tau.to_parquet(os.path.join(out_dir, "corr_tau.parquet"))
        with open(os.path.join(out_dir, "params.json"), "w") as f:
            json.dump({**self.params, "wall_time": self.wall}, f, indent=2)

    @classmethod
    def load(cls, out_dir):
        bins = pd.read_parquet(os.path.join(out_dir, "bins.parquet"))
        cp = os.path.join(out_dir, "corr.parquet")
        corr = pd.read_parquet(cp) if os.path.exists(cp) else pd.DataFrame()
        tp = os.path.join(out_dir, "corr_tau.parquet")
        ctau = pd.read_parquet(tp) if os.path.exists(tp) else pd.DataFrame()
        with open(os.path.join(out_dir, "params.json")) as f:
            params = json.load(f)
        return cls(params, bins, corr, params.pop("wall_time", None), ctau)

    # ------------------------------------------------------------- analysis
    def observables(self):
        return [c for c in self.bins.columns if c not in DIAG_COLS]

    def summary(self, n_discard=0):
        """DataFrame indexed by observable with columns mean, err (jackknife)."""
        b = self.bins.iloc[n_discard:]
        rows = []
        for c in self.observables():
            m, e = jackknife_ratio(b[c].to_numpy(), b["sign"].to_numpy())
            rows.append((c, float(m), float(e)))
        m, e = jackknife_ratio(b["sign"].to_numpy(), np.ones(len(b)))
        rows.append(("sign", float(m), float(e)))
        df = pd.DataFrame(rows, columns=["obs", "mean", "err"]).set_index("obs")
        return df

    def diagnostics(self):
        b = self.bins
        return pd.Series({"acc_rate": b["acc_rate"].mean(), "stab_err_max": b["stab_err"].max(),
                          "phase_err_max": b["phase_err"].max(), "n_bins": len(b),
                          "wall_time": self.wall})

    def structure_factor(self, obs, qs, n_discard=0, b1=0, b2=0):
        """S(q) = sum_R e^{i q R} C(R) for each q (Cartesian), jackknife over bins.
        Returns DataFrame with qx, qy, mean, err."""
        lat_a1 = np.array([1.0, 0.0])
        lat_a2 = np.array([0.0, 1.0])
        if self.params.get("lattice") not in ("square", "chain"):
            raise NotImplementedError("structure_factor: only Bravais lattices with unit vectors")
        c = self.corr[(self.corr.obs == obs) & (self.corr.b1 == b1) & (self.corr.b2 == b2)]
        c = c[c.bin >= n_discard]
        piv = c.pivot_table(index="bin", columns=["rx", "ry"], values="value")
        R = np.array([rx * lat_a1 + ry * lat_a2 for rx, ry in piv.columns])
        sign = self.bins.set_index("bin").loc[piv.index, "sign"].to_numpy()
        rows = []
        for q in np.atleast_2d(qs):
            phase = np.cos(R @ q)              # C(R) real and C(R) = C(-R) for Hermitian O
            Sq_bins = piv.to_numpy() @ phase
            m, e = jackknife_ratio(Sq_bins, sign)
            rows.append((q[0], q[1], float(m), float(e)))
        return pd.DataFrame(rows, columns=["qx", "qy", "mean", "err"])

    def corr_mean(self, obs, n_discard=0):
        """Real-space correlation <O_{i+R} O_i> with jackknife errors (long DataFrame)."""
        c = self.corr[(self.corr.obs == obs) & (self.corr.bin >= n_discard)]
        piv = c.pivot_table(index="bin", columns=["rx", "ry", "b1", "b2"], values="value")
        sign = self.bins.set_index("bin").loc[piv.index, "sign"].to_numpy()
        m, e = jackknife_ratio(piv.to_numpy(), sign)
        out = pd.DataFrame(list(piv.columns), columns=["rx", "ry", "b1", "b2"])
        out["mean"], out["err"] = m, e
        return out

    # ------------------------------------------------------- time-displaced
    def _tau_pivot(self, obs, n_discard, b1, b2):
        c = self.corr_tau
        c = c[(c.obs == obs) & (c.b1 == b1) & (c.b2 == b2) & (c.bin >= n_discard)]
        piv = c.pivot_table(index="bin", columns=["tau", "rx", "ry"], values="value")
        sign = self.bins.set_index("bin").loc[piv.index, "sign_tau"].to_numpy()
        return piv, sign

    def tau_mean(self, obs, n_discard=0, b1=0, b2=0):
        """C(R, tau) with jackknife errors: DataFrame tau, rx, ry, mean, err."""
        piv, sign = self._tau_pivot(obs, n_discard, b1, b2)
        m, e = jackknife_ratio(piv.to_numpy(), sign)
        out = pd.DataFrame(list(piv.columns), columns=["tau", "rx", "ry"])
        out["mean"], out["err"] = m, e
        return out

    def tau_q(self, obs, q, n_discard=0, b1=0, b2=0):
        """S(q, tau) = sum_R cos(q.R) C(R, tau): DataFrame tau, mean, err."""
        piv, sign = self._tau_pivot(obs, n_discard, b1, b2)
        cols = np.array(list(piv.columns))
        taus = np.unique(cols[:, 0])
        ph = np.cos(cols[:, 1] * q[0] + cols[:, 2] * q[1])
        X = piv.to_numpy() * ph[None]
        Sq = np.stack([X[:, cols[:, 0] == t].sum(1) for t in taus], axis=1)
        m, e = jackknife_ratio(Sq, sign)
        return pd.DataFrame({"tau": taus, "mean": m, "err": e})

    def susceptibility(self, obs, q, n_discard=0, b1=0, b2=0):
        """chi(q) = int_0^tau_max dtau S(q, tau) (trapezoid; finite T with tau_max = beta
        gives the static susceptibility). Returns (mean, err)."""
        piv, sign = self._tau_pivot(obs, n_discard, b1, b2)
        cols = np.array(list(piv.columns))
        taus = np.unique(cols[:, 0])
        w = np.full(len(taus), taus[1] - taus[0] if len(taus) > 1 else 0.0)
        w[0] = w[-1] = w[0] / 2
        wt = dict(zip(taus, w))
        ph = np.cos(cols[:, 1] * q[0] + cols[:, 2] * q[1]) * np.array([wt[t] for t in cols[:, 0]])
        m, e = jackknife_ratio(piv.to_numpy() @ ph, sign)
        return float(m), float(e)


# ------------------------------------------------------------ tables / fits
SKIP_IN_TABLE = ("n_c_up", "n_c_dn", "n_d_up", "n_d_dn", "n_up", "n_dn")


def observable_table(res, n_discard=0):
    """Scalars plus S(q) at q = 0 and the zone corner for every correlator.
    Returns DataFrame (obs, mean, err); structure factors are named e.g. 'Sz(pi,pi)'."""
    s = res.summary(n_discard)
    rows = [(k, r["mean"], r["err"]) for k, r in s.iterrows() if k not in SKIP_IN_TABLE]
    if len(res.corr):
        chain = res.params.get("lattice") == "chain"
        qs = {"(0,0)": (0.0, 0.0), "(pi,pi)": (np.pi, 0.0 if chain else np.pi)}
        for obs in res.corr.obs.unique():
            for lab, q in qs.items():
                r = res.structure_factor(obs, [q], n_discard).iloc[0]
                rows.append((f"{obs}{lab}", r["mean"], r["err"]))
    return pd.DataFrame(rows, columns=["obs", "mean", "err"])


def fit_dtau2(df):
    """Weighted fit O(dtau) = O0 + b dtau^2 for each obs.
    df: columns obs, dtau, mean, err. Returns DataFrame obs, O0, O0_err, b, b_err, chi2_dof."""
    out = []
    for obs, g in df.groupby("obs", sort=False):
        x, y, e = g.dtau.to_numpy() ** 2, g["mean"].to_numpy(), g["err"].to_numpy()
        if len(g) < 2 or np.any(~np.isfinite(e)) or np.any(e <= 0):
            out.append((obs, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        w = 1 / e ** 2
        A = np.stack([np.ones_like(x), x], 1)
        cov = np.linalg.inv(A.T @ (A * w[:, None]))
        a, b = cov @ (A.T @ (w * y))
        chi2 = np.sum(w * (y - a - b * x) ** 2)
        dof = len(g) - 2
        out.append((obs, a, np.sqrt(cov[0, 0]), b, np.sqrt(cov[1, 1]),
                    chi2 / dof if dof > 0 else np.nan))
    return pd.DataFrame(out, columns=["obs", "O0", "O0_err", "b", "b_err", "chi2_dof"])
