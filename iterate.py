"""Score clone_real runs against the paper's observations and log every run.

Usage: python3 iterate.py <label> '<json list of parameter overrides>'
Appends one line per run to iterations.jsonl and prints the runs sorted by score.
"""
import json
import math
import sys
from dataclasses import replace
from multiprocessing import Pool

import clone_real as R

# observations: (value, tolerance); score = sum of squared (sim - value) / tolerance
TARGETS = {
    "calls": (1934, 250),          # 1,434-2,434 per sample
    "calls_lo": (1434, 300),
    "calls_hi": (2434, 300),
    "hom95": (225, 35),            # 160-290 per sample after removing the 209 shared sites
    "hom95_lo": (160, 40),
    "hom95_hi": (290, 40),
    "shared": (209, 60),
    "te_share": (0.71, 0.03),
    "nonsyn": (13.7, 4),
    "age": (1781, 200),            # paper-method age; radiocarbon ~1,900 +- 100
    "germ72": (1.0, 0.15),         # founder het sites still het in all 72 (peak at 72 in Fig. 3B)
    "k5": (1.0, 0.35),             # share of samples whose BIC-best GMM has 5 components
}
CORE = ("calls", "calls_lo", "calls_hi", "hom95", "hom95_lo", "hom95_hi", "shared", "age", "germ72")


def metrics(res):
    s, per = res["summary"], res["per_sample"]
    c = sorted(x["calls"] for x in per)
    h = sorted(x["hom95"] for x in per)
    g = res["germline"]
    q = lambda v, f: v[int(f * (len(v) - 1))]
    return {"calls": s["calls"], "calls_lo": q(c, 0.02), "calls_hi": q(c, 0.98), "hom95": s["hom95"],
            "hom95_lo": q(h, 0.02), "hom95_hi": q(h, 0.98), "shared": s["shared_removed"], "te_share": s["te_share"],
            "nonsyn": s["nonsyn"], "age": s["age"], "germ72": g[R.NREF] / max(1, sum(g)),
            "k5": sum(p["best_k"] == 5 for p in res["peaks"]) / len(res["peaks"]),
            "peaks5": [round(c["mu"], 2) for c in res["peaks"][0]["comp5"]],
            "rev95": res["hom95_src"]["reversed"], "loh95": res["hom95_src"]["loh_full"] + res["hom95_src"]["loh_part"],
            "fp_share": s["fp_share"]}


def score(m, keys=TARGETS):
    return sum(((m[k] - TARGETS[k][0]) / TARGETS[k][1]) ** 2 for k in keys if k in TARGETS)


def one(kw):
    kw = dict(kw)
    if "weights" in kw:
        kw["weights"] = tuple(kw["weights"])
    p = replace(R.Params(), **kw)
    res = R.run(p)
    m = metrics(res)
    return {"params": kw, "metrics": m, "score": score(m), "core": score(m, CORE)}


def main():
    label, sets = sys.argv[1], json.loads(sys.argv[2])
    with Pool(2) as pool:
        out = pool.map(one, sets)
    with open("iterations.jsonl", "a") as f:
        for o in out:
            f.write(json.dumps({"iter": label, **o}) + "\n")
    for o in sorted(out, key=lambda o: o["score"]):
        m = o["metrics"]
        print(f"{o['score']:7.1f} core {o['core']:6.1f} | calls {m['calls']:.0f} [{m['calls_lo']}-{m['calls_hi']}] "
              f"hom95 {m['hom95']:.0f} [{m['hom95_lo']}-{m['hom95_hi']}] rev {m['rev95']:.0f} loh {m['loh95']:.0f} "
              f"shared {m['shared']} age {m['age']:.0f} germ {m['germ72']:.2f} nonsyn {m['nonsyn']:.0f} "
              f"k5 {m['k5']:.2f} pk {m['peaks5']} | {o['params']}")


if __name__ == "__main__":
    main()
