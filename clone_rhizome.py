"""Rhizomatous clone of Typha latifolia (1e5-1e6 ramets, 72 sampled): genealogy + somatic mutation fates + sequencing.

 1. Genealogy of the 72 sampled ramets: structured coalescent along a ~2 km rhizome transect colonised outward from
    one founder. Only lineages that keep being re-established (rate beta) contribute.
 2. Mutations (SNP / insertion / deletion; TE-biased; some in functional DNA) arise in the 3x4 meristem initials.
    Their fate follows an exact Markov chain (mean-field over mutations): within-layer drift with cell-level selection,
    layer exchange, LOH of fixed sites, and bud bottlenecks that keep layer identity with probability layer_keep.
 3. Each sampled shoot is founded from a rhizome bud and grows `shoot_years` before sampling.
 4. Observation: leaf tissue mixes layers (weights), reads are sampled at ~45x (negative binomial), 5% of sites sit in
    3-copy regions, and a site is called with >=4 alt reads. False calls (low-VAF noise, collapsed paralogs, apparent
    homozygotes; TE-enriched) are added at rate fpr, and each sample covers a random callable fraction of the genome.
Only MU and the genome numbers come from the paper; everything else is an assumption to be explored (see calibrate.py).
Run: python3 clone_rhizome.py -> rhizome.js (viewed by rhizome.html)
"""
import itertools
import json
import math
import random
from dataclasses import dataclass, asdict
from math import comb, exp, lgamma, log

T = 1800
MU = 5.52e-9 * 216.96e6            # new mutations per meristem cell per year (paper's rate x genome size)
CELLS, LAYERS, INIT = 12, 3, 4
PATCHES, PATCH_M = 200, 10.0
FILL_FRAC, TAU_FILL, MIGRATE, F_EFF = 0.6, 50.0, 0.05, 0.1
SAMPLE = 72
TE_FRAC, FUNC_OF_NONTE, B_TE = 0.34, 0.15, 4.4
P_TYPE = {"SNP": 0.70, "INS": 0.13, "DEL": 0.17}      # SNP share from manuscript het counts / het fractions
S_FUNC = {"SNP": 0.02, "INS": 0.06, "DEL": 0.06}
W_KIND = {"TE": TE_FRAC * B_TE, "FUNC": (1 - TE_FRAC) * FUNC_OF_NONTE, "NEUT": (1 - TE_FRAC) * (1 - FUNC_OF_NONTE)}
TIME_GRID = 5
DEPTH_R, MIN_DEPTH, MAX_DEPTH, MIN_ALT = 45, 15, 130, 4   # NB(mean 45, var 90); call needs >=4 alt reads
CNV_SHARE = 0.05
BIN = 0.02
NBIN = int(round(1 / BIN))


@dataclass(frozen=True)
class Params:
    k: int = 3                      # founder cells of a bud (per layer when layers are kept)
    layer_keep: float = 0.0         # probability a bud keeps L1/L2/L3 identity of its founders
    swap: float = 0.002             # per cell-year chance of replacement by a cell from another layer
    loh: float = 0.0                # per site per cell-year LOH rate of fixed heterozygous sites
    beta: float = 0.5               # per-year rate a lineage passes through a bud bottleneck
    weights: tuple = (1 / 3, 1 / 3, 1 / 3)   # share of leaf cells derived from L1, L2, L3
    shoot_years: int = 1
    n_total: float = 5e5
    fpr: float = 0.17               # share of calls that are false (manuscript FPR; definition uncertain)
    fp_te: float = 0.85             # share of false calls inside TEs (mapping artefacts in repeats)
    fp_mid: float = 0.3             # share of false calls from collapsed paralogs (VAF 0.2-0.5)
    fp_hom: float = 0.0             # share of false calls that look homozygous (VAF ~1)
    callable_min: float = 0.6       # per-sample callable fraction ~ U(callable_min, 1)
    shared_hom: int = 209           # VAF=1 sites shared by all samples (founder/reference), reported separately


STATES = [(a, b, c) for a in range(5) for b in range(a, 5) for c in range(b, 5)]
IDX = {s: i for i, s in enumerate(STATES)}
NS = len(STATES) + 1
HOM = NS - 1
FIX, LOST, E0 = IDX[(4, 4, 4)], IDX[(0, 0, 0)], IDX[(0, 0, 1)]
GROUPS = sorted({0.0, *S_FUNC.values()})
CLASSES = [(t, kd) for t in P_TYPE for kd in W_KIND]
WSUM = sum(W_KIND.values())
RATE = {c: CELLS * MU * P_TYPE[c[0]] * W_KIND[c[1]] / WSUM for c in CLASSES}
GROUP_OF = {c: (S_FUNC[c[0]] if c[1] == "FUNC" else 0.0) for c in CLASSES}


def idx(t):
    return IDX[tuple(sorted(t))]


def binom(n, p, x):
    return comb(n, x) * p ** x * (1 - p) ** (n - x)


def identity():
    return [[1.0 if i == j else 0.0 for j in range(NS)] for i in range(NS)]


def matmul(A, Bm):
    Bt = list(zip(*Bm))
    return [[sum(x * y for x, y in zip(row, col)) for col in Bt] for row in A]


def madd(A, Bm, a=1.0, b=1.0):
    return [[a * x + b * y for x, y in zip(r1, r2)] for r1, r2 in zip(A, Bm)]


def vecmat(v, M):
    out = [0.0] * NS
    for i, vi in enumerate(v):
        if vi:
            row = M[i]
            for j in range(NS):
                out[j] += vi * row[j]
    return out


def from_rows(rowfun):
    M = identity()
    for i, rep in enumerate(STATES):
        M[i][i] = 0.0
        for j, pr in rowfun(rep).items():
            M[i][j] += pr
    return M


def per_layer_product(rep, pm_of):
    out = {}
    pms = [pm_of(c) for c in rep]
    for x in range(5):
        for y in range(5):
            for z in range(5):
                pr = pms[0][x] * pms[1][y] * pms[2][z]
                if pr:
                    j = idx((x, y, z))
                    out[j] = out.get(j, 0.0) + pr
    return out


def drift_matrix(s):
    w = 1 - s
    return from_rows(lambda rep: per_layer_product(rep, lambda c: [binom(4, c * w / (c * w + 4 - c), x) for x in range(5)]))


def swap_matrix(rate):
    q = min(1.0, CELLS * rate)

    def row(rep):
        out = {IDX[rep]: 1 - q}
        for l in range(3):
            for m in range(3):
                if m == l:
                    continue
                pt, ps = rep[l] / 4, rep[m] / 4
                for tm, ptm in ((1, pt), (0, 1 - pt)):
                    for sm, psm in ((1, ps), (0, 1 - ps)):
                        pr = q / 6 * ptm * psm
                        if pr:
                            new = list(rep)
                            new[l] += sm - tm
                            j = idx(new)
                            out[j] = out.get(j, 0.0) + pr
        return out
    return from_rows(row)


def mixed_bottleneck(k):
    if k >= CELLS:
        return identity()
    byM = {}
    for Mt in range(CELLS + 1):
        px = [0.0] * (CELLS + 1)
        for j in range(0, min(k, Mt) + 1):
            if k - j <= CELLS - Mt:
                h = comb(Mt, j) * comb(CELLS - Mt, k - j) / comb(CELLS, k)
                for x in range(CELLS + 1):
                    px[x] += h * binom(CELLS, j / k, x)
        out = {}
        for x in range(CELLS + 1):
            if px[x]:
                for d1 in range(5):
                    for d2 in range(5):
                        d3 = x - d1 - d2
                        if 0 <= d3 <= 4:
                            j = idx((d1, d2, d3))
                            out[j] = out.get(j, 0.0) + px[x] * comb(4, d1) * comb(4, d2) * comb(4, d3) / comb(CELLS, x)
        byM[Mt] = out
    return from_rows(lambda rep: byM[sum(rep)])


def layered_bottleneck(k):
    k = min(k, INIT)
    if k >= INIT:
        return identity()

    def pm(c):
        out = [0.0] * 5
        for j in range(0, min(k, c) + 1):
            if k - j <= 4 - c:
                h = comb(c, j) * comb(4 - c, k - j) / comb(4, k)
                for x in range(5):
                    out[x] += h * binom(4, j / k, x)
        return out
    return from_rows(lambda rep: per_layer_product(rep, pm))


def loh_matrix(lam):
    M = identity()
    M[FIX][FIX] = 1 - lam
    M[FIX][HOM] = lam / 2
    M[FIX][LOST] += lam / 2
    return M


class Dynamics:
    """Yearly transition P per selection group, cached as P^(2^i) and S_(2^i) = sum_{j=1..2^i} P^j."""
    LEVELS = 12

    def __init__(self, p: Params):
        self.p = p
        self.B = madd(layered_bottleneck(p.k), mixed_bottleneck(p.k), p.layer_keep, 1 - p.layer_keep)
        blend = madd(identity(), self.B, 1 - p.beta, p.beta)
        W, L = swap_matrix(p.swap), loh_matrix(p.loh)
        self.P, self.Pshoot, self.pw, self.sw = {}, {}, {}, {}
        for s in GROUPS:
            core = matmul(matmul(W, drift_matrix(s)), L)
            P = matmul(blend, core)
            self.P[s], self.Pshoot[s] = P, core
            pw, sw = [P], [P]
            for _ in range(self.LEVELS):
                A, S = pw[-1], sw[-1]
                sw.append(madd(S, matmul(A, S)))
                pw.append(matmul(A, A))
            self.pw[s], self.sw[s] = pw, sw

    def propagate(self, s, v, d):
        """Row vector after d years starting from v, with one new mutation per year entering state E0."""
        a, b = list(v), [0.0] * NS
        c = [0.0] * NS
        c[E0] = 1.0
        i = 0
        while d:
            if d & 1:
                b = [x + y for x, y in zip(b, vecmat(c, self.sw[s][i]))]
                a = vecmat(a, self.pw[s][i])
                c = vecmat(c, self.pw[s][i])
            d >>= 1
            i += 1
        return [x + y for x, y in zip(a, b)]

    def shoot(self, s, v):
        for _ in range(self.p.shoot_years):
            v = vecmat(v, self.Pshoot[s])
            v[E0] += 1.0
        return v


# ---------- observation model ----------
def _nb(d):
    return exp(lgamma(d + DEPTH_R) - lgamma(DEPTH_R) - lgamma(d + 1) + (DEPTH_R + d) * log(0.5))


DEPTHS = [(d, _nb(d)) for d in range(MIN_DEPTH, MAX_DEPTH + 1)]
_DN = sum(p for _, p in DEPTHS)
DEPTHS = [(d, p / _DN) for d, p in DEPTHS]
_SPEC = {}


def spectrum(f):
    """(VAF histogram, P(VAF>=0.95)) of calls for a site with true alt fraction f; mass = P(called)."""
    key = round(f, 4)
    if key not in _SPEC:
        h, hi = [0.0] * (NBIN + 1), 0.0
        if key > 0:
            for d, pd in DEPTHS:
                lf, lq = (log(key), log(1 - key)) if key < 1 else (0.0, None)
                for a in range(MIN_ALT, d + 1):
                    if key >= 1:
                        pr = pd if a == d else 0.0
                    else:
                        pr = pd * exp(lgamma(d + 1) - lgamma(a + 1) - lgamma(d - a + 1) + a * lf + (d - a) * lq)
                    if pr > 1e-12:
                        h[min(NBIN, int(a / d / BIN))] += pr
                        if a / d >= 0.95:
                            hi += pr
        _SPEC[key] = (h, hi)
    return _SPEC[key]


_ARR = {}


def arrangements(j):
    if j not in _ARR:
        _ARR[j] = sorted(set(itertools.permutations(STATES[j])))
    return _ARR[j]


TRUE_SOURCES = ("fixed", "partial", "loh", "cnv")
FP_SOURCES = ("fp_low", "fp_mid", "fp_hom")
SOURCES = TRUE_SOURCES + FP_SOURCES


def observe(u, weights):
    """Expected calls from a unit-rate state vector u, split by source: {source: (VAF histogram, count VAF>=0.95)}."""
    out = {s: [[0.0] * (NBIN + 1), 0.0] for s in TRUE_SOURCES}
    for j in range(NS):
        m = u[j]
        if j == LOST or m < 1e-9:
            continue
        if j == HOM:
            parts = [("loh", m, 1.0)]
        else:
            base = "fixed" if j == FIX else "partial"
            arr = arrangements(j)
            parts = []
            for a in arr:
                lay = sum(w * c / 4 for w, c in zip(weights, a))
                mm = m / len(arr)
                parts += [(base, mm * (1 - CNV_SHARE), 0.5 * lay), ("cnv", mm * CNV_SHARE / 2, lay / 3),
                          ("cnv", mm * CNV_SHARE / 2, 2 * lay / 3)]
        for src, mm, f in parts:
            sh, shi = spectrum(f)
            h = out[src][0]
            for b in range(NBIN + 1):
                h[b] += mm * sh[b]
            out[src][1] += mm * shi
    return out


_FP = {}


def fp_spectra():
    """Unit-mass VAF spectra (histogram, share >= 0.95) of the three kinds of false calls."""
    if not _FP:
        pa = [exp(-3.0) * 3.0 ** a / math.factorial(a) for a in range(40)]
        z = sum(pa[MIN_ALT:])
        low = [0.0] * (NBIN + 1)
        for d, pd in DEPTHS:
            for a in range(MIN_ALT, min(40, d + 1)):
                low[min(NBIN, int(a / d / BIN))] += pd * pa[a] / z
        mids = [spectrum(f) for f in (0.2, 0.25, 1 / 3, 0.5)]
        zm = sum(sum(h) for h, _ in mids)
        mid = [sum(h[b] for h, _ in mids) / zm for b in range(NBIN + 1)]
        hh, hhi = spectrum(1.0)
        zh = sum(hh)
        _FP.update(fp_low=(low, 0.0), fp_mid=(mid, sum(hi for _, hi in mids) / zm), fp_hom=([x / zh for x in hh], hhi / zh))
    return _FP


# ---------- genealogy ----------
def poisson(rng, lam):
    if lam <= 0:
        return 0
    if lam > 40:
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def sample_tree(rng, n_total, beta=0.5):
    mid = PATCHES // 2
    v = (PATCHES / 2 * PATCH_M) / (FILL_FRAC * T)
    tc = [abs(i - mid) * PATCH_M / v for i in range(PATCHES)]
    units_full = n_total / PATCHES * F_EFF
    tips = rng.sample(range(PATCHES), SAMPLE)
    nodes = [{"t": T, "ch": [], "patch": p} for p in tips]
    active = [[i, p] for i, p in enumerate(tips)]
    dt, s = 0.5, float(T)
    while len(active) > 1 and s > 0:
        s -= dt
        for a in active:
            p = a[1]
            if s < tc[p] and p != mid:
                a[1] = p + (1 if mid > p else -1)
            elif rng.random() < MIGRATE * dt:
                q = p + rng.choice((-1, 1))
                if 0 <= q < PATCHES and tc[q] <= s:
                    a[1] = q
        groups = {}
        for a in active:
            groups.setdefault(a[1], []).append(a)
        merged, new = set(), []
        for p, g in groups.items():
            if len(g) < 2:
                continue
            nu = max(5.0, units_full * min(1.0, (s - tc[p]) / TAU_FILL)) if s >= tc[p] else 5.0
            rate = 2 * beta * dt / nu
            for i in range(len(g)):
                if id(g[i]) in merged:
                    continue
                for j in range(i + 1, len(g)):
                    if id(g[j]) not in merged and rng.random() < rate:
                        nodes.append({"t": s, "ch": [g[i][0], g[j][0]], "patch": p})
                        new.append([len(nodes) - 1, p])
                        merged.update((id(g[i]), id(g[j])))
                        break
        if merged:
            active = [a for a in active if id(a) not in merged] + new
    while len(active) > 1:
        a, b = active.pop(), active.pop()
        nodes.append({"t": 0.0, "ch": [a[0], b[0]], "patch": mid})
        active.append([len(nodes) - 1, mid])
    root = active[0][0]
    for nd in nodes:
        if nd["ch"]:
            nd["t"] = round(nd["t"] / TIME_GRID) * TIME_GRID
    for nd in nodes:
        for c in nd["ch"]:
            nodes[c]["t"] = max(nodes[c]["t"], nd["t"])
    return nodes, root


# ---------- simulation ----------
def simulate(p: Params, nodes, root, rng=None):
    """Unit-rate state vectors at every node and per-tip expected observations."""
    dyn = Dynamics(p)
    zero = [0.0] * NS
    state = {}
    stack = [(root, {s: dyn.propagate(s, zero, int(nodes[root]["t"])) for s in GROUPS})]
    while stack:
        n, u = stack.pop()
        state[n] = u
        for pos, c in enumerate(nodes[n]["ch"]):
            start = {s: vecmat(u[s], dyn.B) for s in GROUPS} if pos == 1 else u
            d = int(nodes[c]["t"] - nodes[n]["t"])
            stack.append((c, {s: dyn.propagate(s, start[s], d) for s in GROUPS}))
    tips = []
    for i in range(SAMPLE):
        u = {s: dyn.shoot(s, vecmat(state[i][s], dyn.B)) for s in GROUPS}
        obs = {s: observe(u[s], p.weights) for s in GROUPS}
        r = {"patch": nodes[i]["patch"], "src": {k: [0.0] * (NBIN + 1) for k in TRUE_SOURCES}, "calls": 0.0,
             "hom95": 0.0, "SNP": 0.0, "INS": 0.0, "DEL": 0.0, "te": 0.0, "func": 0.0, "fixed": 0.0, "hom": 0.0}
        for c in CLASSES:
            s, rc = GROUP_OF[c], RATE[c]
            calls = 0.0
            for src, (h, hi) in obs[s].items():
                calls += rc * sum(h)
                r["hom95"] += rc * hi
                r["src"][src] = [x + rc * y for x, y in zip(r["src"][src], h)]
            r["calls"] += calls
            r[c[0]] += calls
            r["te"] += calls if c[1] == "TE" else 0.0
            r["func"] += calls if c[1] == "FUNC" else 0.0
            r["fixed"] += rc * u[s][FIX]
            r["hom"] += rc * u[s][HOM]
        r["hist"] = [sum(r["src"][k][b] for k in TRUE_SOURCES) for b in range(NBIN + 1)]
        tips.append(r)
    return state, tips


def finalize(tips, p: Params, seed=7):
    """Add false calls and per-sample callable fraction to the true-call expectations of each tip."""
    rng = random.Random(seed)
    fps = fp_spectra()
    share = {"fp_low": 1 - p.fp_mid - p.fp_hom, "fp_mid": p.fp_mid, "fp_hom": p.fp_hom}
    out = []
    for t in tips:
        cf = rng.uniform(p.callable_min, 1.0)
        n_fp = t["calls"] * p.fpr / (1 - p.fpr)
        src = {k: [x * cf for x in t["src"][k]] for k in TRUE_SOURCES}
        hom95 = t["hom95"]
        for k in FP_SOURCES:
            h, hi = fps[k]
            src[k] = [x * n_fp * share[k] * cf for x in h]
            hom95 += n_fp * share[k] * hi
        r = {"patch": t["patch"], "callable": cf, "src": src,
             "hist": [sum(src[k][b] for k in SOURCES) for b in range(NBIN + 1)],
             "calls": (t["calls"] + n_fp) * cf, "fp": n_fp * cf, "hom95": hom95 * cf,
             "te": (t["te"] + n_fp * p.fp_te) * cf, "func": t["func"] * cf, "fixed": t["fixed"], "hom": t["hom"]}
        for ty in P_TYPE:
            r[ty] = (t[ty] + n_fp * P_TYPE[ty]) * cf
        out.append(r)
    return out


def summarize(tips):
    m = lambda k: sum(t[k] for t in tips) / len(tips)
    calls = [t["calls"] for t in tips]
    cf = [t.get("callable", 1.0) for t in tips]
    return {"calls": m("calls"), "calls_min": min(calls), "calls_max": max(calls), "hom95": m("hom95"),
            "snp_share": m("SNP") / m("calls"), "te_share": m("te") / m("calls"), "fixed": m("fixed"),
            "hom": m("hom"), "fp": m("fp") if "fp" in tips[0] else 0.0,
            "low": sum(sum(t["hist"][:10]) for t in tips) / len(tips) / m("calls"),
            "age_paper_method": sum(c / f for c, f in zip(calls, cf)) / len(tips) / MU}


def source_vaf(tips):
    keys = tips[0]["src"].keys()
    return {k: [sum(t["src"][k][b] for t in tips) / len(tips) for b in range(NBIN + 1)] for k in keys}


def noisy_rows(tips, rng):
    rows = []
    for t in tips:
        r = {"patch": t["patch"]}
        for k in ("calls", "hom95", "SNP", "INS", "DEL", "te", "func", "fixed", "hom"):
            r[k] = poisson(rng, t[k])
        rows.append(r)
    return rows


def pair_distances(nodes, state, rng):
    parent = {c: n for n, nd in enumerate(nodes) for c in nd["ch"]}
    F = {n: sum(RATE[c] * (state[n][GROUP_OF[c]][FIX] + state[n][GROUP_OF[c]][HOM]) for c in CLASSES) for n in state}

    def anc(n):
        out = [n]
        while n in parent:
            n = parent[n]
            out.append(n)
        return out

    pairs = []
    for _ in range(900):
        i, j = rng.sample(range(SAMPLE), 2)
        ai = set(anc(i))
        m = next(x for x in anc(j) if x in ai)
        geo = abs(nodes[i]["patch"] - nodes[j]["patch"]) * PATCH_M
        pairs.append([geo, poisson(rng, max(0.0, F[i] - F[m])) + poisson(rng, max(0.0, F[j] - F[m]))])
    return pairs


def nested(nodes, n):
    nd = nodes[n]
    if not nd["ch"]:
        return {"t": nd["t"], "i": n, "x": nd["patch"] * PATCH_M}
    return {"t": nd["t"], "c": [nested(nodes, c) for c in nd["ch"]]}


def tree_for(n_total):
    rng = random.Random(int(n_total) % 9973)
    nodes, root = sample_tree(rng, n_total)
    return nodes, root, rng


def main():
    out = {"T": T, "mu": MU, "runs": []}
    configs = []
    for n in (1e5, 5e5, 1e6):
        configs += [(n, Params(k=k, n_total=n), lab) for k, lab in
                    ((1, "窄瓶颈:1 个创始细胞"), (3, "中等瓶颈:3 个创始细胞"), (12, "无瓶颈:12 个细胞原样复制"))]
    configs += [(5e5, Params(k=3, loh=lam), "中等瓶颈:3 个创始细胞") for lam in (1e-4, 2.5e-4, 4e-4)]
    configs += [(5e5, Params(k=1, layer_keep=lk), f"每层 1 个创始细胞,保留层结构概率 {lk}") for lk in (0.5, 0.9, 1.0)]
    trees = {}
    for n, p, label in configs:
        if n not in trees:
            trees[n] = tree_for(n)
        nodes, root, rng = trees[n]
        state, tips = simulate(p, nodes, root)
        tips = finalize(tips, p)
        sm = summarize(tips)
        print(f"N={int(n):>7} k={p.k:>2} keep={p.layer_keep} loh={p.loh:g}: calls {sm['calls']:.0f} "
              f"({sm['calls_min']:.0f}-{sm['calls_max']:.0f}) VAF>=0.95 {sm['hom95']:.0f} SNP {sm['snp_share']:.2f} "
              f"TE {sm['te_share']:.2f} low(<0.2) {sm['low']:.2f} age(paper method) {sm['age_paper_method']:.0f}", flush=True)
        vaf = [sum(t["hist"][b] for t in tips) / SAMPLE for b in range(NBIN + 1)]
        out["runs"].append({"n_total": n, "label": label, "params": asdict(p), "k": p.k, "loh": p.loh,
                            "layer_keep": p.layer_keep, "summary": sm, "ramets": noisy_rows(tips, rng), "vaf": vaf,
                            "pairs": pair_distances(nodes, state, rng), "tree": nested(nodes, root)})
    with open("rhizome.js", "w") as f:
        f.write("const RHIZOME = " + json.dumps(out) + ";\n")


if __name__ == "__main__":
    main()
