"""Clonal propagation simulation for a higher plant: genetic/epigenetic change over generations.

Pure standard library. Run: python3 simulate.py  -> writes results.json/results.js (viewed by index.html).
Parameters are illustrative, not calibrated to a species.
"""
import json
import math
import random
from dataclasses import dataclass, asdict

LAYERS = 3          # L1, L2, L3 of the shoot apical meristem
INITIALS = 4        # stem-cell initials per layer
CPG_SITES = 1000    # tracked methylation sites


@dataclass
class Scenario:
    name: str
    label: str
    snp_per_division: float     # new somatic mutations per genome per cell division
    divisions_per_gen: int      # meristem cell divisions between propagation events
    meth_flip_prob: float       # per-site methylation flip probability per generation
    te_burst_rate: float        # expected new TE insertions per generation
    ploidy_double_prob: float   # chance of whole-genome doubling per generation
    regen_single_cell: bool     # propagule regenerated from one cell (callus/adventitious)


SCENARIOS = [
    Scenario("cutting", "Stem cuttings / axillary buds",
             snp_per_division=0.05, divisions_per_gen=10, meth_flip_prob=0.0005,
             te_burst_rate=0.02, ploidy_double_prob=0.0005, regen_single_cell=False),
    Scenario("culture", "Tissue culture (callus, single-cell regeneration)",
             snp_per_division=0.15, divisions_per_gen=25, meth_flip_prob=0.004,
             te_burst_rate=0.4, ploidy_double_prob=0.01, regen_single_cell=True),
]


def poisson(rng, lam):
    if lam <= 0:
        return 0
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


class Plant:
    def __init__(self, base_meth):
        self.cells = [frozenset() for _ in range(LAYERS * INITIALS)]  # mutation ids per initial
        self.meth = base_meth
        self.te = 0
        self.te_pos = []
        self.ploidy = 2

    def drift_step(self, rng):
        # Within each layer, initials are resampled (sorting-out of chimeras).
        new = []
        for layer in range(LAYERS):
            block = self.cells[layer * INITIALS:(layer + 1) * INITIALS]
            new.extend(rng.choice(block) for _ in range(INITIALS))
        self.cells = new

    def mutate(self, rng, sc, pos):
        for _ in range(sc.divisions_per_gen):
            for _ in range(poisson(rng, sc.snp_per_division)):
                i = rng.randrange(len(self.cells))
                mid = len(pos)
                pos[mid] = round(rng.random(), 5)
                self.cells[i] = self.cells[i] | {mid}
            self.drift_step(rng)

    def propagate(self, rng, sc):
        if sc.regen_single_cell:
            src = rng.choice(self.cells)
            self.cells = [src] * (LAYERS * INITIALS)
        # cuttings keep the layered structure (periclinal chimeras persist)
        flips = poisson(rng, sc.meth_flip_prob * CPG_SITES)
        for _ in range(flips):
            self.meth ^= 1 << rng.randrange(CPG_SITES)
        for _ in range(poisson(rng, sc.te_burst_rate)):
            self.te += 1
            self.te_pos.append(round(rng.random(), 4))
        if self.ploidy == 2 and rng.random() < sc.ploidy_double_prob:
            self.ploidy = 4

    def stats(self, base_meth):
        counts = {}
        for c in self.cells:
            for m in c:
                counts[m] = counts.get(m, 0) + 1
        n = len(self.cells)
        fixed = sum(1 for v in counts.values() if v == n)
        chimeric = len(counts) - fixed
        return fixed, chimeric, bin(self.meth ^ base_meth).count("1") / CPG_SITES, self.te, self.ploidy


def snapshot(plant, base_meth, pos):
    counts = {}
    for c in plant.cells:
        for m in c:
            counts[m] = counts.get(m, 0) + 1
    n = len(plant.cells)
    diff = plant.meth ^ base_meth
    return {
        "fixed": [pos[m] for m, v in counts.items() if v == n],
        "chimeric": [[pos[m], v / n] for m, v in counts.items() if v < n],
        "meth": [i for i in range(CPG_SITES) if diff >> i & 1],
        "te": plant.te_pos,
        "ploidy": plant.ploidy,
    }


def mean_sd(xs):
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def run(sc, generations, lineages, seed):
    rng = random.Random(seed)
    base_meth = rng.getrandbits(CPG_SITES)
    plants = [Plant(base_meth) for _ in range(lineages)]
    pos = {}
    snaps = []
    series = {k: [] for k in ("fixed", "chimeric", "methylation", "te", "tetraploid")}
    for g in range(generations + 1):
        if g > 0:
            for p in plants:
                p.mutate(rng, sc, pos)
                p.propagate(rng, sc)
        snaps.append(snapshot(plants[0], base_meth, pos))
        st = [p.stats(base_meth) for p in plants]
        for key, idx in (("fixed", 0), ("chimeric", 1), ("methylation", 2), ("te", 3)):
            m, s = mean_sd([x[idx] for x in st])
            series[key].append({"mean": m, "sd": s})
        series["tetraploid"].append({"mean": sum(x[4] == 4 for x in st) / len(st), "sd": 0})
    return series, snaps


def main(generations=40, lineages=300, seed=1):
    out = {"generations": generations, "lineages": lineages, "scenarios": []}
    for sc in SCENARIOS:
        series, snaps = run(sc, generations, lineages, seed)
        out["scenarios"].append({**asdict(sc), "series": series, "dna": snaps})
    with open("results.json", "w") as f:
        json.dump(out, f)
    with open("results.js", "w") as f:
        f.write("const DATA = " + json.dumps(out) + ";\n")
    for s in out["scenarios"]:
        last = {k: round(v[-1]["mean"], 3) for k, v in s["series"].items()}
        print(s["name"], last)


if __name__ == "__main__":
    main()
