"""Equal-time measurements from a single-particle Green's function G_ij = <c_i c_j^dag>.

Wick (no anomalous terms):  Gc = <c_a^dag c_b> = delta_ab - G_ba
    <c_a^dag c_b c_c^dag c_d> = Gc_ab Gc_cd + Gc_ad G_bc
    <c_b^dag c_a^dag c_c c_d> = Gc_bd Gc_ac - Gc_bc Gc_ad

Correlations are returned translation averaged:
    C_O(R; b1, b2) = 1/N_cells sum_i <O_{i+R, b1} O_{i, b2}>              (bilinears)
    C_D(R; b1, b2) = 1/N_cells sum_i <Delta^dag_{i+R, b1} Delta_{i, b2}>  (pairs)
so that S(q) = 1/N_sites sum_ij e^{iq(r_i-r_j)} <O_i O_j> = sum_R e^{iqR} C(R) for one basis site.
"""
import numpy as np


class Measurer:
    def __init__(self, model):
        self.m = model
        lat = model.lattice
        self.no = model.n_orb
        self.Ns = lat.n_sites
        self.nb = lat.n_basis
        self.T = lat.translation_table()          # (Ns, Ncells)
        # cell of site i and basis index
        self.cell = np.arange(self.Ns) // self.nb
        self.basis = np.arange(self.Ns) % self.nb
        self.scalar_names = (["E_kin", "E_int", "E_tot", "N"] + list(model.orbital_groups)
                             + [f"n_{o}" for o in model.orb_labels])
        self.corr_names = list(model.bilinears) + list(model.pairs)

    def scalars(self, G):
        m = self.m
        N = G.shape[0]
        Gc = np.eye(N) - G.T
        out = {}
        out["E_kin"] = np.sum(m.h0 * Gc).real
        # interaction energy, vertex by vertex
        P = m.P
        Gc_v = Gc[P[:, :, None], P[:, None, :]]           # (n_v, p, p)
        G_v = G[P[:, :, None], P[:, None, :]]
        A = m.A[m.vtype]                                   # (n_v, p, p)
        O1 = np.einsum("vab,vab->v", A, Gc_v)
        O2 = O1 ** 2 + np.einsum("vab,vcd,vad,vbc->v", A, A, Gc_v, G_v)
        al = m.alpha[m.vtype]
        e = -m.W[m.vtype] * (O2 + 2 * al * O1 + al ** 2) + m.e_const[m.vtype]
        out["E_int"] = np.sum(e).real
        out["E_tot"] = out["E_kin"] + out["E_int"]
        dens = np.diag(Gc).real.reshape(self.Ns, self.no)
        out["N"] = dens.sum()
        for name, orbs in m.orbital_groups.items():
            out[name] = dens[:, orbs].sum()
        for o, lab in enumerate(m.orb_labels):
            out[f"n_{lab}"] = dens[:, o].mean()
        return out

    def correlations(self, G):
        """dict name -> array (Ncells, nb, nb), complex (take .real for Hermitian O)."""
        no, Ns = self.no, self.Ns
        N = G.shape[0]
        Gc = np.eye(N) - G.T
        G4 = G.reshape(Ns, no, Ns, no)
        Gc4 = Gc.reshape(Ns, no, Ns, no)
        out = {}
        for name, A in self.m.bilinears.items():
            mvec = np.einsum("ab,iaib->i", A, Gc4)
            C = np.outer(mvec, mvec) + np.einsum("ab,cd,iajd,ibjc->ij", A, A, Gc4, G4,
                                                 optimize=True)
            out[name] = self._translate(C)
        for name, B in self.m.pairs.items():
            Bc = B.conj()
            C = (np.einsum("ab,cd,ibjd,iajc->ij", Bc, B, Gc4, Gc4, optimize=True)
                 - np.einsum("ab,cd,ibjc,iajd->ij", Bc, B, Gc4, Gc4, optimize=True))
            out[name] = self._translate(C)
        return out

    def tau_correlations(self, Gtt, G00, Gt0, G0t):
        """Time-displaced correlations for one tau.

        Gtt = G(tau,tau), G00 = G(0,0), Gt0 = <c(tau) c^dag(0)>, G0t = -<c^dag(tau) c(0)>.
            Green(R)  = 1/Ncells sum_i sum_o <c_{i+R,o}(tau) c^dag_{i,o}(0)>
            O (bil.)  : <O_i(tau) O_j(0)> = m_i(tau) m_j(0) - sum A_ab A_cd G0t[jd,ia] Gt0[ib,jc]
            Pair      : <D^dag_i(tau) D_j(0)> = sum B*_ab B_cd (G0t[jd,ib] G0t[jc,ia] - G0t[jc,ib] G0t[jd,ia])
        (no background subtracted). Same array layout as ``correlations``.
        """
        no, Ns = self.no, self.Ns
        N = Gtt.shape[0]
        Gt04 = Gt0.reshape(Ns, no, Ns, no)
        G0t4 = G0t.reshape(Ns, no, Ns, no)
        Gct = (np.eye(N) - Gtt.T).reshape(Ns, no, Ns, no)
        Gc0 = (np.eye(N) - G00.T).reshape(Ns, no, Ns, no)
        out = {"Green": self._translate(np.einsum("iojo->ij", Gt04))}
        for name, A in self.m.bilinears.items():
            mt = np.einsum("ab,iaib->i", A, Gct)
            m0 = np.einsum("ab,iaib->i", A, Gc0)
            C = np.outer(mt, m0) - np.einsum("ab,cd,jdia,ibjc->ij", A, A, G0t4, Gt04, optimize=True)
            out[name] = self._translate(C)
        for name, B in self.m.pairs.items():
            Bc = B.conj()
            C = (np.einsum("ab,cd,jdib,jcia->ij", Bc, B, G0t4, G0t4, optimize=True)
                 - np.einsum("ab,cd,jcib,jdia->ij", Bc, B, G0t4, G0t4, optimize=True))
            out[name] = self._translate(C)
        return out

    def _translate(self, C):
        """C[i, j] -> Cr[R, b1, b2] = 1/Ncells sum_{j} C[j+R (basis b1), j (basis b2)]."""
        nc = self.Ns // self.nb
        if self.nb == 1:
            return C[self.T, np.arange(self.Ns)[:, None]].mean(axis=0)[:, None, None]
        Cr = np.zeros((nc, self.nb, self.nb), dtype=complex)
        for j in range(self.Ns):
            b2 = self.basis[j]
            for R in range(nc):
                i0 = self.T[j, R]                    # same basis as j, shifted by R
                cell_i = i0 // self.nb
                for b1 in range(self.nb):
                    Cr[R, b1, b2] += C[cell_i * self.nb + b1, j]
        return Cr / nc
