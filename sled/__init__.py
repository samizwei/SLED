"""SLED - Simulations of Lattice Electrons with Determinants (DQMC in Python + Numba)."""
from .params import Params, load_params
from .lattice import make_lattice, chain, square
from .models import make_model, hubbard, twoband
from .dqmc import DQMC
from .analysis import Results


def run(params=None, progress=False, **kw):
    """Run a simulation. ``params``: Params, path to a TOML file, or None (defaults + kw)."""
    if params is None:
        params = Params(**kw)
    elif isinstance(params, str):
        params = load_params(params, **kw)
    elif kw:
        params = Params(**{**params.as_dict(), **kw})
    return DQMC(params).run(progress=progress)
