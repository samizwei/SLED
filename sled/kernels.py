"""Hot loop: local HS-field updates on one time slice.

Slice convention: B_tau = exp(-dtau K) V_{n_v-1} ... V_1 V_0,  V_k = exp(g eta_{l_k} A_k).

Upward  (G(tau-1) -> G(tau)):  for k = 0..n_v-1:  G <- V_k G V_k^{-1}, then update field k.
                               (the caller then applies exp(-dtau K)).
Downward (G(tau) -> G(tau-1)): caller first removes exp(-dtau K); then for
                               k = n_v-1..0: update field k, then G <- V_k^{-1} G V_k.

Update of field k with G "just after V_k" (G = (1 + V_k X)^{-1}):
    delta = exp(g (eta' - eta) A) - 1    (p x p, on orbitals P_k)
    S     = (1 - G)_{PP}
    ratio = det(1 + S delta) * gamma_l'/gamma_l * exp(g alpha (eta' - eta))
    G'    = G - G_{:,P} delta (1 + S delta)^{-1} (1 - G)_{P,:}
Metropolis with |ratio|; the phase ratio/|ratio| is returned for sign tracking.

``vertex_sweep_numpy`` is the plain reference, ``vertex_sweep`` the Numba version.
Random numbers are drawn outside (``shift`` in 1..3 proposes l' = (l + shift) % 4,
``u`` uniform in [0,1)), so both versions give identical trajectories.
"""
import numpy as np
import numba


def vertex_sweep_numpy(G, P, vtype, fields, expA, expAinv, delta, logw, shift, u, up):
    nv, p = P.shape
    n_acc = 0
    phase = 1.0 + 0.0 * G[0, 0]
    order = range(nv) if up else range(nv - 1, -1, -1)
    for k in order:
        idx = P[k]
        t = vtype[k]
        if up:
            l = fields[k]
            G[idx, :] = expA[t, l] @ G[idx, :]
            G[:, idx] = G[:, idx] @ expAinv[t, l]
        l = fields[k]
        lnew = (l + shift[k]) % 4
        d = delta[t, l, lnew]
        S = np.eye(p) - G[np.ix_(idx, idx)]
        M = np.eye(p) + S @ d
        R = np.linalg.det(M) * np.exp(logw[t, lnew] - logw[t, l])
        if u[k] < abs(R):
            X = G[:, idx] @ (d @ np.linalg.inv(M))
            Y = -G[idx, :]
            Y[:, idx] += np.eye(p)
            G -= X @ Y
            fields[k] = lnew
            n_acc += 1
            phase *= R / abs(R)
        if not up:
            l = fields[k]
            G[idx, :] = expAinv[t, l] @ G[idx, :]
            G[:, idx] = G[:, idx] @ expA[t, l]
    return n_acc, phase


@numba.njit(cache=True)
def _rows_left(G, idx, V):
    """G[idx, :] = V @ G[idx, :]"""
    p = idx.shape[0]
    N = G.shape[1]
    tmp = np.empty((p, N), dtype=G.dtype)
    for a in range(p):
        tmp[a, :] = G[idx[a], :]
    out = V @ tmp
    for a in range(p):
        G[idx[a], :] = out[a, :]


@numba.njit(cache=True)
def _cols_right(G, idx, V):
    """G[:, idx] = G[:, idx] @ V"""
    p = idx.shape[0]
    N = G.shape[0]
    tmp = np.empty((N, p), dtype=G.dtype)
    for a in range(p):
        tmp[:, a] = G[:, idx[a]]
    out = tmp @ V
    for a in range(p):
        G[:, idx[a]] = out[:, a]


@numba.njit(cache=True)
def vertex_sweep(G, P, vtype, fields, expA, expAinv, delta, logw, shift, u, up):
    nv, p = P.shape
    N = G.shape[0]
    n_acc = 0
    phase = G[0, 0] * 0.0 + 1.0
    Ip = np.eye(p).astype(G.dtype)
    S = np.empty((p, p), dtype=G.dtype)
    Xc = np.empty((N, p), dtype=G.dtype)
    Y = np.empty((p, N), dtype=G.dtype)
    for kk in range(nv):
        k = kk if up else nv - 1 - kk
        idx = P[k]
        t = vtype[k]
        if up:
            l = fields[k]
            _rows_left(G, idx, expA[t, l])
            _cols_right(G, idx, expAinv[t, l])
        l = fields[k]
        lnew = (l + shift[k]) % 4
        d = np.ascontiguousarray(delta[t, l, lnew])
        for a in range(p):
            for b in range(p):
                S[a, b] = Ip[a, b] - G[idx[a], idx[b]]
        M = Ip + S @ d
        R = np.linalg.det(M) * np.exp(logw[t, lnew] - logw[t, l])
        aR = abs(R)
        if u[k] < aR:
            C = d @ np.linalg.inv(M)
            for a in range(p):
                Xc[:, a] = G[:, idx[a]]
                Y[a, :] = -G[idx[a], :]
            for a in range(p):
                Y[a, idx[a]] += 1.0
            G -= (Xc @ C) @ Y
            fields[k] = lnew
            n_acc += 1
            phase *= R / aR
        if not up:
            l = fields[k]
            _rows_left(G, idx, expAinv[t, l])
            _cols_right(G, idx, expA[t, l])
    return n_acc, phase
