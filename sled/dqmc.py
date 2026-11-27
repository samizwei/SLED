"""DQMC driver: finite-temperature (BSS) and projector algorithms.

Time slices tau = 1..L, B_tau = exp(-dtau K) V_tau (see kernels.py).
Finite T:   G(tau) = [1 + B(tau,0) B(L,tau)]^{-1},  L dtau = beta.
Projector:  G(tau) = 1 - B(tau,0) P [P^dag B(L,0) P]^{-1} P^dag B(L,tau),
            L dtau = 2 theta + beta, measurements for tau in [theta, theta + beta].

Stabilization points tau_s = 0, n_wrap, 2 n_wrap, ..., L. ``self.R[s]`` holds
B(tau_s,0) R0 (filled during the upward sweep), ``self.Lst[s]`` holds L0 B(L,tau_s)
(filled during the downward sweep). One sweep = upward + downward pass; every field
is visited twice. Equal-time measurements are taken in the upward pass with the
symmetrized G~ = exp(dtau K/2) G exp(-dtau K/2) (Trotter error O(dtau^2)).

Time-displaced measurements (``measure_tau``): after each sweep one extra upward pass
without updates (fixed configuration) from tau0 (0 for finite T, theta for the
projector) to tau0 + tau_max. Between stabilization points the functions are wrapped,
G(tau+1,tau0) = B G(tau,tau0), G(tau0,tau+1) = G(tau0,tau) B^{-1}; at the points:
  finite T:  stable UDT formula (linalg.green_tau_finite_T),
  projector: G(tau,tau0) = G(tau,tau_s) G(tau_s,tau0) with G(tau,tau_s) = B(tau,tau_s) G(tau_s)
             (G is idempotent at T = 0), and G(tau0,tau) = G(tau0,tau_s)(1 - G(tau_s)) B(tau,tau_s)^{-1}.
"""
import time

import numpy as np

from . import linalg as la
from .hs import vertex_tables
from .kernels import vertex_sweep, vertex_sweep_numpy
from .lattice import make_lattice
from .measure import Measurer
from .models import make_model


def _expm_herm(K, x):
    lam, U = np.linalg.eigh(K)
    return (U * np.exp(x * lam)) @ U.conj().T


class DQMC:
    def __init__(self, p, model=None):
        self.p = p
        self.model = model if model is not None else make_model(p, make_lattice(p.lattice, p.L1, p.L2))
        m = self.model
        self.dtype = m.dtype
        self.rng = np.random.default_rng(p.seed)
        self.N = m.n_dim
        self.proj = p.algorithm == "projector"
        if p.algorithm not in ("finite_T", "projector"):
            raise ValueError("algorithm must be finite_T or projector")

        # time slices and measurement window
        nb = int(round(p.beta / p.dtau))
        if self.proj:
            nt = int(round(p.theta / p.dtau))
            self.L = 2 * nt + nb
            self.window = (nt, nt + nb)
        else:
            if nb < 1:
                raise ValueError("beta/dtau must be >= 1")
            self.L = nb
            self.window = (1, self.L)
        self.stab = list(range(0, self.L, p.n_wrap)) + [self.L]
        if p.measure_tau:
            tmax = p.tau_max if p.tau_max >= 0 else (self.window[1] - self.window[0]) * p.dtau \
                if self.proj else p.beta
            self.ntau = int(round(tmax / p.dtau))
            t0 = self.window[0] if self.proj else 0
            if t0 + self.ntau > self.L:
                raise ValueError("tau_max too large for the time axis")

        # kinetic propagators
        K = m.h0.astype(self.dtype)
        self.expK = self._cast(_expm_herm(K, -p.dtau))
        self.expKinv = self._cast(_expm_herm(K, p.dtau))
        self.expKh = self._cast(_expm_herm(K, -p.dtau / 2))
        self.expKhinv = self._cast(_expm_herm(K, p.dtau / 2))

        # HS fields and vertex tables
        self.expA, self.expAinv, self.delta, self.logw = vertex_tables(m, p.dtau, self.dtype)
        self.fields = self.rng.integers(0, 4, size=(self.L, m.n_v)).astype(np.int64)
        self.kernel = vertex_sweep if p.use_numba else vertex_sweep_numpy

        # boundary states
        if self.proj:
            self.Ptrial = self._trial_state()
            self.R0 = la.right_from(self.Ptrial)
            self.L0 = la.left_from(self.Ptrial.conj().T)
            self.green = la.green_projector
        else:
            self.R0 = la.right_identity(self.N, self.dtype)
            self.L0 = la.right_identity(self.N, self.dtype)
            self.green = la.green_finite_T

        self.meas = Measurer(m)
        self.n_acc = 0
        self.n_prop = 0
        self.stab_err = 0.0
        self.phase_err = 0.0
        self._init_stacks()

    # ------------------------------------------------------------------ setup
    def _cast(self, M):
        if self.dtype == np.float64:
            return np.ascontiguousarray(np.real(M))
        return np.ascontiguousarray(M.astype(np.complex128))

    def _trial_state(self):
        """Lowest n_part eigenvectors of h0 + eps * X (X real, orbital diagonal,
        spin independent, random on bonds) -> respects the Kramers symmetry of the
        two-band models and lifts accidental degeneracies."""
        m, p = self.model, self.p
        n_part = p.n_part if p.n_part > 0 else self.N // 2
        r = np.random.default_rng(12345)
        X = np.zeros((m.lattice.n_sites, m.lattice.n_sites))
        for (i, j, _) in m.lattice.bonds():
            v = r.uniform(-1, 1)
            X[i, j] += v
            X[j, i] += v
        h = m.h0 + p.trial_eps * np.kron(X, np.eye(m.n_orb))
        e, U = np.linalg.eigh(h)
        gap = e[n_part] - e[n_part - 1] if n_part < self.N else np.inf
        if gap < 1e-8:
            raise ValueError(f"trial state not closed shell (gap {gap:.2e}); change n_part or trial_eps")
        self.trial_gap = gap
        return self._cast(U[:, :n_part])

    def B_slice(self, tau):
        """B_tau (dense) for slice tau = 1..L."""
        B = self.expK.copy()
        m = self.model
        f = self.fields[tau - 1]
        for k in range(m.n_v - 1, -1, -1):
            idx = m.P[k]
            B[:, idx] = B[:, idx] @ self.expA[m.vtype[k], f[k]]
        return B

    def B_slice_inv(self, tau):
        """B_tau^{-1} = V_0^{-1} ... V_{n-1}^{-1} exp(+dtau K)."""
        M = self.expKinv.copy()
        m = self.model
        f = self.fields[tau - 1]
        for k in range(m.n_v - 1, -1, -1):
            idx = m.P[k]
            M[idx, :] = self.expAinv[m.vtype[k], f[k]] @ M[idx, :]
        return M

    def B_chunk(self, t0, t1):
        """B(t1, t0) = B_t1 ... B_{t0+1}."""
        M = np.eye(self.N, dtype=self.dtype)
        for tau in range(t0 + 1, t1 + 1):
            M = self.B_slice(tau) @ M
        return M

    def _init_stacks(self):
        ns = len(self.stab)
        self.R = [None] * ns
        self.Lst = [None] * ns
        self.R[0] = self.R0.copy()
        self.Lst[ns - 1] = self.L0.copy()
        for s in range(ns - 2, -1, -1):
            self.Lst[s] = la.left_mult(self.Lst[s + 1], self.B_chunk(self.stab[s], self.stab[s + 1]))
        self.G, self.phase = self.green(self.R[0], self.Lst[0])

    # ----------------------------------------------------------------- sweeps
    def _kernel(self, tau, up):
        m = self.model
        nv = m.n_v
        shift = self.rng.integers(1, 4, size=nv).astype(np.int64)
        u = self.rng.random(nv)
        f = self.fields[tau - 1]
        n_acc, ph = self.kernel(self.G, m.P, m.vtype, f, self.expA, self.expAinv,
                                self.delta, self.logw, shift, u, up)
        self.n_acc += n_acc
        self.n_prop += nv
        self.phase *= ph

    def _restabilize(self, s):
        Gnew, ph = self.green(self.R[s], self.Lst[s])
        self.stab_err = max(self.stab_err, np.abs(Gnew - self.G).max())
        self.phase_err = max(self.phase_err, abs(ph - self.phase))
        self.G, self.phase = Gnew, ph

    def sweep_up(self, measure=False):
        s = 0
        for tau in range(1, self.L + 1):
            self._kernel(tau, up=True)
            self.G = self.expK @ self.G @ self.expKinv
            if tau == self.stab[s + 1]:
                s += 1
                self.R[s] = la.right_mult(self.R[s - 1], self.B_chunk(self.stab[s - 1], tau))
                self._restabilize(s)
            if measure and self.window[0] <= tau <= self.window[1] \
                    and (tau - self.window[0]) % self.p.meas_every == 0:
                self._measure()

    def sweep_down(self):
        s = len(self.stab) - 1
        for tau in range(self.L, 0, -1):
            self.G = self.expKinv @ self.G @ self.expK
            self._kernel(tau, up=False)
            if tau - 1 == self.stab[s - 1]:
                s -= 1
                self.Lst[s] = la.left_mult(self.Lst[s + 1], self.B_chunk(tau - 1, self.stab[s + 1]))
                self._restabilize(s)

    def sweep(self, measure=False):
        self.sweep_up(measure)
        self.sweep_down()
        if measure and self.p.measure_tau:
            self.tau_pass()

    def tau_pass(self):
        """Time-displaced measurement on the current (fixed) configuration.
        Requires G = G(0) and a valid left stack (true after sweep_down)."""
        I = np.eye(self.N, dtype=self.dtype)
        t0 = self.window[0] if self.proj else 0
        G = self.G.copy()
        s = 0
        track = False
        Gt0 = G0t = G00 = None
        for tau in range(0, t0 + self.ntau + 1):
            if tau > 0:
                B, Bi = self.B_slice(tau), self.B_slice_inv(tau)
                G = B @ G @ Bi
                if track:
                    if self.proj:
                        Bacc, Bacc_inv = B @ Bacc, Bacc_inv @ Bi
                    else:
                        Gt0, G0t = B @ Gt0, G0t @ Bi
                if tau == self.stab[s + 1]:
                    s += 1
                    self.R[s] = la.right_mult(self.R[s - 1], self.B_chunk(self.stab[s - 1], tau))
                    G, _ = self.green(self.R[s], self.Lst[s])
                    if track and not self.proj:
                        Gt0, G0t = la.green_tau_finite_T(self.R[s], self.Lst[s])
                    if track and self.proj:          # re-anchor on the fresh equal-time G
                        Ta0, Sa0 = Bacc @ Ga @ Ta0, Sa0 @ (I - Ga) @ Bacc_inv
                        Ga, Bacc, Bacc_inv = G, I.copy(), I.copy()
            if tau == t0:
                track = True
                G00 = G.copy()
                Gt0, G0t = G.copy(), G - I
                if self.proj:
                    Ga, Ta0, Sa0, Bacc, Bacc_inv = G, G.copy(), G - I, I.copy(), I.copy()
            if track:
                if self.proj:
                    Gt0, G0t = Bacc @ Ga @ Ta0, Sa0 @ (I - Ga) @ Bacc_inv
                self._measure_tau(tau - t0, G, G00, Gt0, G0t)

    # ----------------------------------------------------------- measurements
    def _reset_bin(self):
        self.acc = {"sign": 0.0, "n": 0}
        for k in self.meas.scalar_names:
            self.acc[k] = 0.0
        self.acc_corr = {k: 0.0 for k in self.meas.corr_names}
        self.acc_tau = {}
        self.acc_tau_sign = 0.0

    def _measure_tau(self, k, Gtt, G00, Gt0, G0t):
        sym = lambda X: self.expKhinv @ X @ self.expKh
        ph = self.phase
        if k == 0:
            self.acc_tau_sign += ph
        for name, C in self.meas.tau_correlations(sym(Gtt), sym(G00), sym(Gt0), sym(G0t)).items():
            if name not in self.acc_tau:
                self.acc_tau[name] = np.zeros((self.ntau + 1,) + C.shape, dtype=complex)
            self.acc_tau[name][k] += ph * C

    def _measure(self):
        Gs = self.expKhinv @ self.G @ self.expKh
        ph = self.phase
        self.acc["sign"] += ph
        self.acc["n"] += 1
        for k, v in self.meas.scalars(Gs).items():
            self.acc[k] += ph * v
        if self.p.measure_corr:
            for k, v in self.meas.correlations(Gs).items():
                self.acc_corr[k] = self.acc_corr[k] + ph * v

    # -------------------------------------------------------------------- run
    def run(self, progress=False):
        from .analysis import Results
        p = self.p
        t0 = time.time()
        for _ in range(p.n_warmup):
            self.sweep()
        bins, corr_rows, tau_rows = [], [], []
        lat = self.model.lattice
        for b in range(p.n_bins):
            self._reset_bin()
            self.n_acc = self.n_prop = 0
            self.stab_err = self.phase_err = 0.0
            tb = time.time()
            for _ in range(p.n_sweeps):
                self.sweep(measure=True)
            n = self.acc["n"]
            row = {"bin": b, "sign": (self.acc["sign"] / n).real}
            for k in self.meas.scalar_names:
                row[k] = (self.acc[k] / n).real
            row.update(acc_rate=self.n_acc / max(self.n_prop, 1), stab_err=self.stab_err,
                       phase_err=self.phase_err, n_meas=n, t_bin=time.time() - tb)
            bins.append(row)
            if p.measure_corr:
                for name, C in self.acc_corr.items():
                    C = C / n
                    for R in range(C.shape[0]):
                        rx, ry = R % lat.L1, R // lat.L1
                        for b1 in range(C.shape[1]):
                            for b2 in range(C.shape[2]):
                                corr_rows.append((b, name, rx, ry, b1, b2, C[R, b1, b2].real))
            if p.measure_tau:
                ns = p.n_sweeps
                row["sign_tau"] = (self.acc_tau_sign / ns).real
                for name, C in self.acc_tau.items():
                    C = C / ns
                    for k in range(C.shape[0]):
                        for R in range(C.shape[1]):
                            rx, ry = R % lat.L1, R // lat.L1
                            for b1 in range(C.shape[2]):
                                for b2 in range(C.shape[3]):
                                    tau_rows.append((b, name, k * p.dtau, rx, ry, b1, b2,
                                                     C[k, R, b1, b2].real))
            if progress:
                print(f"bin {b + 1}/{p.n_bins}  sign={row['sign']:.4f}  E={row['E_tot'] / row['sign']:.5f}"
                      f"  acc={row['acc_rate']:.3f}  stab={self.stab_err:.1e}  {row['t_bin']:.1f}s",
                      flush=True)
        res = Results.from_run(p, bins, corr_rows, tau_rows, wall=time.time() - t0)
        if p.out_dir:
            res.save(p.out_dir)
        return res
