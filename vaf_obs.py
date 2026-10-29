"""Expected observed VAF spectra for selected scenarios, for vafobs.html. Uses the engine in clone_rhizome.py.

Run: python3 vaf_obs.py -> vafobs.js
"""
import json
from math import exp, factorial

import clone_rhizome as cr

FPR = 0.17
FP_ALT_MEAN = 3.0
SCHEMES = {"equal": (1 / 3, 1 / 3, 1 / 3), "typical": (0.15, 0.35, 0.50)}
CONFIGS = [
    dict(k=1), dict(k=3), dict(k=12),
    dict(k=3, loh=1e-4), dict(k=3, loh=2.5e-4), dict(k=3, loh=4e-4),
    dict(k=1, layer_keep=0.99), dict(k=1, layer_keep=1.0), dict(k=2, layer_keep=1.0),
]


def fp_spectrum():
    pa = [exp(-FP_ALT_MEAN) * FP_ALT_MEAN ** a / factorial(a) for a in range(40)]
    z = sum(pa[cr.MIN_ALT:])
    out = [0.0] * (cr.NBIN + 1)
    for d, pd in cr.DEPTHS:
        for a in range(cr.MIN_ALT, min(40, d + 1)):
            out[min(cr.NBIN, int(a / d / cr.BIN))] += pd * pa[a] / z
    return out


def main():
    nodes, root, _ = cr.tree_for(5e5)
    out = {"bin": cr.BIN, "fpr": FPR, "schemes": SCHEMES, "runs": [], "fp": fp_spectrum()}
    for cfg in CONFIGS:
        res = {"k": cfg["k"], "loh": cfg.get("loh", 0.0), "layer_keep": cfg.get("layer_keep", 0.0), "hist": {}, "detected": {}}
        for name, w in SCHEMES.items():
            _, tips = cr.simulate(cr.Params(weights=w, **cfg), nodes, root)
            h = [sum(t["hist"][b] for t in tips) / cr.SAMPLE for b in range(cr.NBIN + 1)]
            res["hist"][name], res["detected"][name] = h, sum(h)
        out["runs"].append(res)
        print(cfg, {k: round(v) for k, v in res["detected"].items()}, flush=True)
    with open("vafobs.js", "w") as f:
        f.write("const VAFOBS = " + json.dumps(out) + ";\n")


if __name__ == "__main__":
    main()
