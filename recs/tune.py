import json, math, random, numpy as np
from pathlib import Path

def sample_simplex(n=3, num=100, kind="dirichlet", seed=42):
    random.seed(seed); np.random.seed(seed)
    if kind == "grid":
        # coarse grid (step 0.1) – small but deterministic
        step = 0.1
        vals = np.arange(0.0, 1.0 + 1e-9, step)
        out = []
        for a in vals:
            for b in vals:
                c = 1.0 - a - b
                if c < -1e-9: continue
                out.append((float(a), float(b), float(max(0.0, c))))
        return out
    # default: random Dirichlet (denser coverage quickly)
    out = np.random.dirichlet(alpha=np.ones(n), size=num)
    return [tuple(map(float, row)) for row in out]

def save_weights(path: Path, mapping: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2))

def load_weights(path: Path, fallback: dict):
    if Path(path).exists():
        try:
            return json.loads(Path(path).read_text())
        except Exception:
            return fallback
    return fallback
