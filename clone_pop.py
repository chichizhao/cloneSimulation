"""One founder plant -> a clonal population over ~1800 years (Typha latifolia model).

Each ramet carries 12 meristem initials (3 layers x 4). Every year: mutation in each initial, within-layer drift
(optional cell-level selection against deleterious mutations), and ramet founding with a bottleneck of k founder
cells per layer. Mutations are nodes in a tree (parent pointers) so cell genotypes are cheap to copy.
Parameters are assumptions anchored on Zhao et al. 2026 (mu = 5.52e-9/site/yr x 217 Mb = 1.2/genome/yr).
Run: python3 clone_pop.py  -> clone.js (viewed by clone.html)
"""
import json
import math
import random
import sys
from array import array

YEARS = 1800
N_RAMETS = 300          # alive ramets simulated (real clone is far larger)
SAMPLE = 72
MU = 5.52e-9 * 216.96e6  # new mutations per initial per year
LAYERS, INIT = 3, 4
BRANCH_P = 0.3          # chance a ramet founds a daughter each year
STEP_M = 8.0            # mean rhizome step (m)
TE_FRAC, FUNC_OF_NONTE, B = 0.34, 0.15, 4.4
LAYER_SWAP = 0.002      # per cell-year chance a cell is replaced by a neighbour-layer cell
DETECT = 4              # cells (of 12) that must carry a mutation for it to be called (~VAF >= 0.17)
SEL = 0.02              # cell-level fitness cost per deleterious mutation

WEIGHTS = (TE_FRAC * B, (1 - TE_FRAC) * FUNC_OF_NONTE, (1 - TE_FRAC) * (1 - FUNC_OF_NONTE))
KINDS = (1, 2, 0)       # 1 = TE, 2 = deleterious functional, 0 = neutral non-TE

SCENARIOS = [
    {"name": "narrow", "label": "窄瓶颈:1 个创始细胞", "k": 1},
    {"name": "medium", "label": "中等瓶颈:3 个创始细胞", "k": 3},
    {"name": "none", "label": "无瓶颈:整个分生组织原样复制", "k": 12},
]


def run(k, seed):
    rng = random.Random(seed)
    parent, kind, dele = array("i", [-1]), array("b", [0]), array("i", [0])
    cdf, acc = [], 0.0
    for i in range(12):
        acc += math.exp(-MU) * MU ** i / math.factorial(i)
        cdf.append(acc)

    def n_mut():
        u = rng.random()
        for i, c in enumerate(cdf):
            if u <= c:
                return i
        return len(cdf)

    ramets = [{"cells": [0] * (LAYERS * INIT), "x": 0.0, "y": 0.0}]
    for _ in range(YEARS):
        for r in ramets:
            cells = r["cells"]
            for ci in range(len(cells)):
                for _ in range(n_mut()):
                    kd = rng.choices(KINDS, WEIGHTS)[0]
                    parent.append(cells[ci])
                    kind.append(kd)
                    dele.append(dele[cells[ci]] + (1 if kd == 2 else 0))
                    cells[ci] = len(parent) - 1
            new = []
            for l in range(LAYERS):
                block = cells[l * INIT:(l + 1) * INIT]
                w = [(1 - SEL) ** dele[c] for c in block]
                new.extend(rng.choices(block, w, k=INIT))
            for ci in range(len(new)):
                if rng.random() < LAYER_SWAP:
                    l = ci // INIT
                    nl = min(LAYERS - 1, max(0, l + rng.choice((-1, 1))))
                    new[ci] = new[nl * INIT + rng.randrange(INIT)]
            r["cells"] = new
        births = []
        for r in ramets:
            if rng.random() < BRANCH_P:
                if k >= LAYERS * INIT:
                    dc = list(r["cells"])
                else:
                    dc = rng.choices(rng.sample(r["cells"], k), k=LAYERS * INIT)
                a, d = rng.uniform(0, 2 * math.pi), rng.expovariate(1 / STEP_M)
                births.append({"cells": dc, "x": r["x"] + d * math.cos(a), "y": r["y"] + d * math.sin(a)})
        ramets += births
        if len(ramets) > N_RAMETS:
            rng.shuffle(ramets)
            del ramets[N_RAMETS:]

    def path(n):
        out = []
        while n > 0:
            out.append(n)
            n = parent[n]
        return out

    sample = rng.sample(ramets, min(SAMPLE, len(ramets)))
    rows, fixed_sets, vaf = [], [], []
    for r in sample:
        paths = [set(path(c)) for c in r["cells"]]
        union, fixed = set().union(*paths), set.intersection(*paths)
        counts = {}
        for p in paths:
            for m in p - fixed:
                counts[m] = counts.get(m, 0) + 1
        vaf.extend(v / len(paths) / 2 for v in counts.values())
        det = set(fixed) | {m for m, v in counts.items() if v >= DETECT}
        te = lambda s: sum(kind[m] == 1 for m in s) / len(s) if s else 0
        rows.append({"x": round(r["x"], 1), "y": round(r["y"], 1), "total": len(union), "fixed": len(fixed),
                     "detected": len(det), "chimeric": len(det) - len(fixed), "te_fixed": te(fixed),
                     "te_all": te(det), "func": sum(kind[m] == 2 for m in det)})
        fixed_sets.append(fixed)
    pairs = []
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            geo = math.hypot(sample[i]["x"] - sample[j]["x"], sample[i]["y"] - sample[j]["y"])
            pairs.append([round(geo, 1), len(fixed_sets[i] ^ fixed_sets[j])])
    hist = [0] * 25
    for v in vaf:
        hist[min(24, int(v / 0.5 * 25))] += 1
    xs = [r["x"] for r in sample]
    ys = [r["y"] for r in sample]
    return {"ramets": rows, "pairs": pairs, "vaf_hist": hist, "nodes": len(parent),
            "extent_m": [round(max(xs) - min(xs)), round(max(ys) - min(ys))]}


def main():
    out = {"years": YEARS, "n_ramets": N_RAMETS, "mu": MU, "scenarios": []}
    for i, sc in enumerate(SCENARIOS):
        res = run(sc["k"], 100 + i)
        out["scenarios"].append({**sc, **res})
        rows = res["ramets"]
        m = lambda key: sum(r[key] for r in rows) / len(rows)
        print(f'{sc["name"]}: detected/ramet {m("detected"):.0f}, fixed {m("fixed"):.0f}, chimeric {m("chimeric"):.0f}, '
              f'TE(detected) {m("te_all"):.2f}, deleterious/ramet {m("func"):.1f}, extent {res["extent_m"]} m', flush=True)
    with open("clone.js", "w") as f:
        f.write("const CLONE = " + json.dumps(out) + ";\n")


if __name__ == "__main__":
    main()
