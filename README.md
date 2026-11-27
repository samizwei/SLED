# SLED
SLED = Simulations of Lattice Electrons with Determinants

Determinant QMC (finite-T BSS and projector) in Python + NumPy/SciPy, with Numba for
the local-update kernel. Target: Hubbard and multiband Hubbard models on small to
moderate lattices, as a test bed next to ALF and ED.

## How to cite
If you use SLED, or results obtained with it, in a publication, thesis or talk, please cite:

> Sami Zweidler, *SLED: Simulations of Lattice Electrons with Determinants*,
> sami.zweidler@nbi.ku.dk

and contact the author for the current reference.

## Models
| `model` | Hamiltonian | HS channel |
|---|---|---|
| `hubbard` | −t Σ c†c − chem N + U Σ (n↑−½)(n↓−½) | U>0: −U/2 (n↑−n↓)², U<0: −\|U\|/2 (n−1)² |
| `twoband` | H₀ of Christensen *et al.* PRL 125, 247001 − chem N − U Σᵢ Σₐ (Sᵃᵢ)² | one field per site and component |

`coupling` for `twoband`: `Ising` (a = z), `XY` (x, y), `Heisenberg` (x, y, z), with
Sᵃᵢ = Σ c†σᵃd + h.c. Lattices: `square`, `chain` (kagome: lattice layer ready, model deferred).

## Usage
```bash
python scripts/run.py params/twoband_2x2_ising.toml                       # TOML file
python scripts/run.py params/twoband_2x2_ising.toml coupling=Heisenberg dtau=0.0125
```
```python
import sled
res = sled.run(model="twoband", coupling="XY", L1=4, L2=4, U=1.0, beta=4.0, dtau=0.025)
res.bins                                   # pandas: one row per bin (sign-weighted)
res.summary(n_discard=2)                   # mean, jackknife error of <sO>/<s>
res.structure_factor("Sz", [(np.pi, np.pi)])
res.corr_mean("P")                         # real-space pair correlation
res.save("runs/xy"); sled.Results.load("runs/xy")   # bins.parquet, corr.parquet, params.json
```

### Time-displaced correlations
Set `measure_tau = true` (and optionally `tau_max`). After every measured sweep one extra
pass on the fixed configuration measures, for 0 ≤ τ ≤ tau_max (default β, or the
projector window), Green(R,τ) = Σₒ⟨c_{i+R,o}(τ)c†_{i,o}(0)⟩ and ⟨O_{i+R}(τ)O_i(0)⟩ for all
bilinears and pair operators (no background subtracted).
```python
res.corr_tau                               # long format: bin, obs, tau, rx, ry, b1, b2, value
res.tau_q("Sz", (np.pi, np.pi))            # S(q, tau) with errors
res.susceptibility("Sz", (np.pi, np.pi))   # chi(q) = int_0^beta S(q, tau) dtau
```

### Δτ extrapolation and comparison with ALF
```bash
# SLED series, O(dtau) = O0 + b dtau^2 fit, plot
python scripts/dtau_extrapolate.py params/twoband_4x4.toml --dtaus 0.05 0.025 0.0125 \
       --out runs/sled_4x4_Heisenberg --jobs 3 --plot coupling=Heisenberg
# same parameters with ALF (../ALF_SZ/ALF, TwoBand_Spin)
scripts/run_alf_series.sh runs/alf_4x4_Heisenberg Heisenberg "0.05 0.025 0.0125" 4 4 4.0 0.3 1.0 25 40
# per-dtau deviations and comparison of the extrapolated values
python scripts/compare_alf.py --sled runs/sled_4x4_Heisenberg --alf runs/alf_4x4_Heisenberg
```

## Layout
- `sled/lattice.py`: Bravais lattice + basis, bonds with direction labels, PBC
- `sled/models.py`: h₀ and the interaction as vertices −W(ψ†Aψ+α)²
- `sled/hs.py`: 4-point Gauss–Hermite HS tables
- `sled/linalg.py`: UDT (pivoted QR) stacks, stable G (finite T / projector)
- `sled/kernels.py`: local updates on one slice (NumPy reference + Numba)
- `sled/dqmc.py`: sweeps, stabilization, measurements
- `sled/measure.py`: equal-time scalars and correlations (Wick)
- `sled/analysis.py`: `Results`, Parquet I/O, jackknife
- `tests/`: unit tests, U = 0 exact, ED (`ed_small.py`)
- `scripts/validate_small_ed.py`, `scripts/validate_2x2.py`: validation against ED
- `scripts/dtau_extrapolate.py`, `scripts/run_alf_series.sh`, `scripts/compare_alf.py`: Δτ series and ALF comparison

Tests: `python -m pytest -q tests` (~1 min).
