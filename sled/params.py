"""Run parameters (dataclass) and TOML loading."""
import tomllib
from dataclasses import dataclass, asdict, fields


@dataclass
class Params:
    # model
    model: str = "twoband"            # "hubbard" | "twoband"
    coupling: str = "Ising"           # twoband: "Ising" | "XY" | "Heisenberg"
    lattice: str = "square"           # "square" | "chain"
    L1: int = 2
    L2: int = 2
    t: float = 1.0
    delta: float = 0.4                # twoband anisotropy
    mu: float = -2.0                  # twoband orbital splitting  -mu (n_c - n_d)
    chem: float = 0.0                 # chemical potential  -chem N
    U: float = 1.0
    # algorithm
    algorithm: str = "finite_T"       # "finite_T" | "projector"
    beta: float = 2.0                 # finite_T: 1/T;  projector: measurement window
    theta: float = 5.0                # projector: projection length on each side
    dtau: float = 0.05
    n_wrap: int = 10                  # slices between stabilizations
    n_part: int = -1                  # projector: particle number (-1: n_dim/2)
    trial_eps: float = 1e-2           # projector: symmetry-respecting perturbation of h0
    # Monte Carlo
    n_warmup: int = 20                # sweeps (1 sweep = up + down)
    n_bins: int = 20
    n_sweeps: int = 20                # sweeps per bin
    meas_every: int = 1               # measure every n-th slice in the window
    measure_corr: bool = True
    measure_tau: bool = False         # time-displaced correlations (one pass per sweep)
    tau_max: float = -1.0             # largest tau; -1: beta (finite T) / window (projector)
    seed: int = 1
    use_numba: bool = True
    out_dir: str = ""                 # "" -> do not write files

    def as_dict(self):
        return asdict(self)


def load_params(path, **overrides):
    with open(path, "rb") as f:
        d = tomllib.load(f)
    flat = {}
    for k, v in d.items():
        if isinstance(v, dict):          # allow [model] / [mc] sections
            flat.update(v)
        else:
            flat[k] = v
    flat.update(overrides)
    known = {f.name for f in fields(Params)}
    unknown = set(flat) - known
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)}")
    return Params(**flat)
