"""Run SLED from a TOML parameter file.

Usage: python scripts/run.py params/file.toml [key=value ...]
e.g.   python scripts/run.py params/twoband_2x2_ising.toml coupling=XY dtau=0.025 out_dir=runs/xy
"""
import ast, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sled

over = {}
for a in sys.argv[2:]:
    k, v = a.split("=", 1)
    try:
        over[k] = ast.literal_eval(v)
    except (ValueError, SyntaxError):
        over[k] = v
res = sled.run(sys.argv[1], progress=True, **over)
print(res.summary(n_discard=2).to_string())
print(res.diagnostics().to_string())
