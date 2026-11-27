"""Lattices: Bravais lattice + basis, periodic boundary conditions.

Site index convention
---------------------
    site = (x + L1 * y) * n_basis + b,     0 <= x < L1, 0 <= y < L2, 0 <= b < n_basis

Bonds are given per unit cell as (b_from, b_to, (dx, dy), label). ``label`` is the
bond direction ("x", "y", ...); models use it to choose direction-dependent
hoppings (e.g. t+delta along x for the c orbital). New lattices (kagome, ...) only
need a new constructor that fills these fields.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Lattice:
    name: str
    L1: int
    L2: int
    a1: np.ndarray                      # primitive vectors (2D embedding)
    a2: np.ndarray
    basis: np.ndarray                   # (n_basis, 2) positions inside the unit cell
    bond_types: list = field(default_factory=list)  # [(b_from, b_to, (dx, dy), label)]

    @property
    def n_basis(self):
        return len(self.basis)

    @property
    def n_cells(self):
        return self.L1 * self.L2

    @property
    def n_sites(self):
        return self.n_cells * self.n_basis

    def site(self, x, y, b=0):
        return ((x % self.L1) + self.L1 * (y % self.L2)) * self.n_basis + b

    def cell_xy(self, site):
        """(x, y, b) of a site index."""
        cell, b = divmod(site, self.n_basis)
        return cell % self.L1, cell // self.L1, b

    def positions(self):
        """Real-space positions, shape (n_sites, 2)."""
        pos = np.empty((self.n_sites, 2))
        for s in range(self.n_sites):
            x, y, b = self.cell_xy(s)
            pos[s] = x * self.a1 + y * self.a2 + self.basis[b]
        return pos

    def bonds(self):
        """All bonds with PBC as a list of (i, j, label), one entry per bond i -> j.

        For L = 2 along a direction the bonds i -> i+a and (i+a) -> i+2a = i are both
        kept, so the hopping between the two sites is doubled. This is the correct
        periodic result and matches the dispersion -2t cos(k) on the k-grid.
        """
        out = []
        for y in range(self.L2):
            for x in range(self.L1):
                for (b0, b1, (dx, dy), label) in self.bond_types:
                    out.append((self.site(x, y, b0), self.site(x + dx, y + dy, b1), label))
        return out

    def translation_table(self):
        """T[i, R] = site reached from site i by cell translation R = rx + L1*ry.

        Basis index is kept. Used to average correlations over translations.
        """
        T = np.empty((self.n_sites, self.n_cells), dtype=np.int64)
        for s in range(self.n_sites):
            x, y, b = self.cell_xy(s)
            for ry in range(self.L2):
                for rx in range(self.L1):
                    T[s, rx + self.L1 * ry] = self.site(x + rx, y + ry, b)
        return T

    def k_points(self):
        """Allowed momenta (n_cells, 2) in Cartesian coordinates."""
        A = np.array([self.a1, self.a2])            # rows a1, a2
        Bm = 2 * np.pi * np.linalg.inv(A).T          # rows b1, b2 with a_i.b_j = 2 pi delta_ij
        ks = [(m / self.L1) * Bm[0] + (n / self.L2) * Bm[1]
              for n in range(self.L2) for m in range(self.L1)]
        return np.array(ks)


def chain(L):
    """1D chain of L sites (PBC), embedded in 2D with L2 = 1."""
    return Lattice("chain", L, 1, np.array([1.0, 0.0]), np.array([0.0, 1.0]),
                   np.zeros((1, 2)), [(0, 0, (1, 0), "x")])


def square(L1, L2):
    """Square lattice L1 x L2 (PBC), nearest-neighbour bonds along x and y."""
    if L2 < 2:
        raise ValueError("square lattice needs L2 >= 2; use chain(L) for 1D")
    return Lattice("square", L1, L2, np.array([1.0, 0.0]), np.array([0.0, 1.0]),
                   np.zeros((1, 2)), [(0, 0, (1, 0), "x"), (0, 0, (0, 1), "y")])


def kagome(L1, L2):
    """Kagome lattice: placeholder until the two-band hopping pattern is decided."""
    raise NotImplementedError("kagome: deferred (see CLAUDE.md design decisions)")


def make_lattice(name, L1, L2=1):
    if name == "chain":
        return chain(L1)
    if name == "square":
        return square(L1, L2)
    if name == "kagome":
        return kagome(L1, L2)
    raise ValueError(f"unknown lattice {name!r}")
