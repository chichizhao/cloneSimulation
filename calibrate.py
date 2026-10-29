"""Grid search of unknown parameters of the combined model against the manuscript's per-sample numbers.

Targets (per sample): 1,434-2,434 somatic calls and 160-290 sample-specific sites with VAF >= 0.95 (after removing the
209 sites shared by all samples). A combination is "consistent" when the model's means fall inside both ranges.
Biology parameters need a new simulation; false-call parameters are applied afterwards (finalize).
Run: python3 calibrate.py -> calib.js (viewed by calib.html and model.html)
"""
import itertools
import json
import time

import clone_rhizome as cr

TARGET = {"calls": (1434, 2434, 1934), "hom95": (160, 290, 225)}
WEIGHTS = (0.15, 0.35, 0.50)
SIM_GRID = {"k": [1, 3], "layer_keep": [0.0, 0.97, 0.99, 1.0], "loh": [0.0, 1e-4, 2e-4, 3e-4, 4e-4]}
FP_GRID = {"fpr": [0.10, 0.17, 0.25], "fp_hom": [0.0, 0.1, 0.2]}
TOP_WITH_SOURCES = 40


def score(sm):
    a = (sm["calls"] - TARGET["calls"][2]) / 250
    b = (sm["hom95"] - TARGET["hom95"][2]) / 32
    return a * a + b * b


def main():
    nodes, root, _ = cr.tree_for(5e5)
    rows, t0 = [], time.time()
    sims = list(itertools.product(*SIM_GRID.values()))
    for n, (k, keep, loh) in enumerate(sims):
        base = cr.Params(k=k, layer_keep=keep, loh=loh, weights=WEIGHTS)
        _, tips = cr.simulate(base, nodes, root)
        for fpr, fp_hom in itertools.product(*FP_GRID.values()):
            p = cr.Params(k=k, layer_keep=keep, loh=loh, weights=WEIGHTS, fpr=fpr, fp_hom=fp_hom)
            fin = cr.finalize(tips, p)
            sm = cr.summarize(fin)
            ok = all(TARGET[key][0] <= sm[key] <= TARGET[key][1] for key in TARGET)
            src = cr.source_vaf(fin)
            rows.append({"k": k, "layer_keep": keep, "loh": loh, "fpr": fpr, "fp_hom": fp_hom,
                         **{key: round(v, 4) for key, v in sm.items()}, "score": round(score(sm), 3), "ok": ok,
                         "vaf": [round(sum(src[s][b] for s in src), 3) for b in range(cr.NBIN + 1)],
                         "src": {s: [round(x, 3) for x in v] for s, v in src.items()}})
        print(f"{n + 1}/{len(sims)} {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda r: r["score"])
    for i, r in enumerate(rows):
        if i >= TOP_WITH_SOURCES:
            del r["src"]
    grid = {**SIM_GRID, **FP_GRID}
    fixed = {k: v for k, v in cr.asdict(cr.Params(weights=WEIGHTS)).items() if k not in grid}
    with open("calib.js", "w") as f:
        f.write("const CALIB = " + json.dumps({"target": TARGET, "grid": grid, "fixed": fixed, "rows": rows}) + ";\n")
    ok = [r for r in rows if r["ok"]]
    print(f"consistent: {len(ok)}/{len(rows)}")
    for key in grid:
        print(key, {v: sum(1 for r in ok if r[key] == v) for v in grid[key]})
    for r in rows[:6]:
        print({k: r[k] for k in ("k", "layer_keep", "loh", "fpr", "fp_hom", "calls", "calls_min", "calls_max",
                                 "hom95", "low", "te_share", "age_paper_method", "score")})


if __name__ == "__main__":
    main()
