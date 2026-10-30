"""Site-level stochastic simulation of the Ganhaizi Typha latifolia clone, built to mirror the real study end to end.

Biology
 1. Ramet genealogy: 72 sampled ramets + 1 reference ramet, from the structured rhizome coalescent of clone_rhizome.
 2. Every meristem has 3 layers (L1/L2/L3) x 4 initials. Each year the initials of a layer are resampled from that
    layer (drift), cells are displaced by the adjacent outer layer (L1->L2, L2->L3; rarely L2->L1), and with rate beta
    the lineage passes through an axillary rhizome bud founded by k cells per layer (layers kept with prob layer_keep,
    otherwise the bud comes from k cells of the whole meristem).
 3. Individual cells are tracked forward, so every sampled cell has an exact cell genealogy.
 4. Mutations (SNP/INS/DEL; TE-biased) fall on that genealogy at MU per cell-year. LOH (mitotic recombination /
    gene conversion) hits each heterozygous site in a cell lineage at rate loh per cell-year and makes the lineage
    homozygous mutant or wild type with equal chance. Founder (germline) heterozygous sites get the same LOH.
 5. The sampled leaf of each ramet is built from leaf_founders cells per layer, mixed with layer weights.

Reference and sequencing
 6. The haploid reference is assembled from the reference ramet: at each of its mutations the assembly keeps the
    mutant allele when the haplotype it follows is mostly mutant. Such sites have reversed polarity in every sample.
 7. A share of sites sits in collapsed 3-copy regions (more in TEs): VAF denominators of 3 and 1.5x depth.
 8. Per sample: callable fraction ~U(callable_min,1) applied to a per-site mappability; depth NB(45, var 90);
    reads binomial with base error; call needs depth>=15 and >=4 alt reads.
 9. False calls are generated, not added as a rate: collapsed-paralog sites (mis-mapped reads, every sample),
    error hotspots (site-specific high error, sporadic), and assembly errors (VAF=1 everywhere).
10. Pipeline as in the paper: drop sites with VAF>=0.95 in all 72 samples, then count calls, sample-specific VAF>=0.95,
    VAF spectrum (bin 0.02), GMM peaks, and age = calls / (mu x callable sites).
Run: python3 clone_real.py [k layer_keep loh ...] -> real.js (viewed by real.html)
"""
import json
import math
import random
import sys
from array import array
from dataclasses import dataclass, asdict, replace

import clone_rhizome as C

T, MU, G, MU_SITE = C.T, C.MU, 216.96e6, 5.52e-9
NREF = C.SAMPLE                       # tip index of the reference ramet; tips 0..71 are the samples
TE_FRAC, B_TE, FUNC_OF_NONTE = C.TE_FRAC, C.B_TE, C.FUNC_OF_NONTE
P_TYPE = C.P_TYPE
BIN = 0.02
NBIN = int(round(1 / BIN))
MIN_DEPTH, MIN_ALT, DEPTH_MEAN, DEPTH_R = 15, 4, 45.0, 45.0
SOURCES = ("fixed", "mosaic", "loh_full", "loh_part", "cnv", "reversed", "fp_paralog", "fp_hotspot", "fp_asm")


@dataclass(frozen=True)
class Params:
    k: int = 1                     # bud founder cells per layer
    layer_keep: float = 0.97       # chance a bud keeps layer identity (else k cells from the whole meristem)
    swap: float = 0.002            # per cell-year displacement by the adjacent outer layer
    turnover: float = 1.0          # per initial-year chance of replacement within its layer (1 = Wright-Fisher)
    beta: float = 0.5              # bud passages per lineage-year
    loh: float = 2e-6              # per het site per cell-year LOH rate (founder het sharing caps it near 1e-5)
    shoot_years: int = 1
    leaf_founders: int = 2         # leaf cells per layer contributing to the sampled leaf
    weights: tuple = (1 / 3, 1 / 3, 1 / 3)
    cnv_te: float = 0.25           # share of TE sites in collapsed 3-copy regions
    cnv_other: float = 0.02
    n_paralog: int = 150           # collapsed-paralog artefact sites (85% in TEs)
    n_hotspot: int = 2500          # site-specific error hotspots
    hot_err: float = 0.02          # mean error rate at hotspots
    n_asm: int = 0                 # assembly errors (VAF=1 in all samples)
    base_err: float = 0.001
    callable_min: float = 0.82
    germline_het: int = 2000       # founder heterozygous sites (for the sharing histogram)
    ref_split: float = 0.0        # >=0: reference lineage splits from the samples' ancestry at this year (else a random ramet)
    call_a50: float = 0.0          # caller sensitivity: P(call | alt reads a) = 1/(1+exp(-(a-call_a50))), on top of >=4 alt
    dropout: float = 0.007           # independent per site x sample loss (local low coverage, filters)
    mu_scale: float = 0.7          # effective somatic rate relative to the paper's species rate
    lib_sd: float = 0.4            # per-sample library quality: log-sd of artefact (hotspot/paralog) intensity
    indel_bias: float = 0.15        # share of indel-carrying reads lost in mapping (lowers indel VAF)
    od: float = 0.01                # allelic overdispersion (beta-binomial rho)
    func_keep: float = 0.25         # purifying selection: share of FUNC mutations that escape elimination
    shared_rule: str = "all"       # "all": VAF>=0.95 in all 72; "covered": in every sample with depth>=15
    n_total: float = 5e5
    seed: int = 1


# ---------- genealogy of ramets ----------
def ramet_tree(p):
    C.SAMPLE = NREF + 1
    try:
        rng = random.Random(p.seed * 7919 + int(p.n_total) % 9973)
        nodes, root = C.sample_tree(rng, p.n_total)
    finally:
        C.SAMPLE = NREF
    if p.ref_split >= 0:
        nodes, root = move_ref(nodes, root, p.ref_split)
    return nodes, root


def move_ref(nodes, root, t_split):
    """Detach the reference tip and re-attach it to the ancestral lineage at year t_split (above the samples' MRCA)."""
    parent = {c: n for n, nd in enumerate(nodes) for c in nd["ch"]}
    q = parent[NREF]
    sib = [c for c in nodes[q]["ch"] if c != NREF][0]
    if q == root:
        root = sib
    else:
        g = parent[q]
        nodes[g]["ch"] = [sib if c == q else c for c in nodes[g]["ch"]]
    nodes[q]["ch"] = []
    t_split = min(t_split, nodes[root]["t"])
    nodes.append({"t": t_split, "ch": [root, NREF], "patch": nodes[NREF]["patch"]})
    return nodes, len(nodes) - 1


# ---------- forward cell simulation ----------
class Cells:
    """Append-only cell records: parent record and creation year."""

    def __init__(self):
        self.par, self.tim = array("i", [-1]), array("i", [0])   # record 0 = zygote

    def new(self, parent, t):
        self.par.append(parent)
        self.tim.append(t)
        return len(self.par) - 1


def step_year(cells, mer, t, p, rng):
    """One year of growth for meristem mer (3 lists of 4 record ids); returns the new meristem."""
    out = []
    for l in range(3):
        lay = mer[l]
        nl = []
        for i in range(4):
            src = lay[i]
            if rng.random() < p.turnover:
                src = lay[rng.randrange(4)]
            if l > 0 and rng.random() < p.swap:
                src = mer[l - 1][rng.randrange(4)]
            elif l == 0 and rng.random() < p.swap / 10:
                src = mer[1][rng.randrange(4)]
            nl.append(src)
        out.append(nl)
    if rng.random() < p.beta:
        out = bud(out, p, rng)
    return [[cells.new(c, t) for c in lay] for lay in out]


def bud(mer, p, rng):
    if rng.random() < p.layer_keep:
        res = []
        for lay in mer:
            f = rng.sample(lay, min(p.k, 4))
            res.append(f + [rng.choice(f) for _ in range(4 - len(f))])
        return res
    pool = [c for lay in mer for c in lay]
    f = rng.sample(pool, min(p.k, 12))
    return [[rng.choice(f) for _ in range(4)] for _ in range(3)]


def grow(p, nodes, root, rng):
    """Forward simulation along the ramet tree; returns cells and, per tip, the sampled leaf cells by layer."""
    cells = Cells()
    mer = [[cells.new(0, 0) for _ in range(4)] for _ in range(3)]
    t0 = int(nodes[root]["t"])
    for t in range(1, t0 + 1):
        mer = step_year(cells, mer, t, p, rng)
    leaves = {}
    stack = [(root, mer)]
    while stack:
        n, m = stack.pop()
        nd = nodes[n]
        if not nd["ch"]:
            m = bud(m, p, rng)
            for y in range(p.shoot_years):
                m = step_year(cells, m, T + y + 1, p, rng)
            leaves[n] = [[cells.new(c, T + p.shoot_years) for c in rng.sample(lay, p.leaf_founders)] for lay in m]
            continue
        for pos, c in enumerate(nd["ch"]):
            mm = bud(m, p, rng) if pos == 1 else m
            mm = [[cells.new(x, int(nd["t"])) for x in lay] for lay in mm]
            for t in range(int(nd["t"]) + 1, int(nodes[c]["t"]) + 1):
                mm = step_year(cells, mm, t, p, rng)
            stack.append((c, mm))
    return cells, leaves


def genealogy(cells, leaves):
    """Compressed genealogy of sampled cells: kept nodes, parent, branch length, descendant bitmask, layer/tip of leaves."""
    leaf_rec, leaf_info = [], []
    for tip in sorted(leaves):
        for l, lay in enumerate(leaves[tip]):
            for r in lay:
                leaf_rec.append(r)
                leaf_info.append((tip, l))
    nchild = {}
    for r in leaf_rec:
        x = r
        while True:
            q = cells.par[x]
            if q < 0:
                break
            first = q not in nchild
            nchild[q] = nchild.get(q, 0) + 1
            if not first:
                break
            x = q
    keep = set(leaf_rec) | {q for q, c in nchild.items() if c > 1} | {0}
    kpar, desc = {}, {}
    for i, r in enumerate(leaf_rec):
        desc[r] = desc.get(r, 0) | (1 << i)
    order = sorted(keep, reverse=True)          # children always have larger record ids
    for r in keep:
        if r == 0:
            continue
        q = cells.par[r]
        while q not in keep:
            q = cells.par[q]
        kpar[r] = q
    for r in order:
        if r in kpar:
            desc[kpar[r]] = desc.get(kpar[r], 0) | desc.get(r, 0)
    children = {}
    for r, q in kpar.items():
        children.setdefault(q, []).append(r)
    blen = {r: cells.tim[r] - cells.tim[q] for r, q in kpar.items()}
    return {"leaf_info": leaf_info, "kpar": kpar, "desc": desc, "children": children, "blen": blen}


# ---------- mutations and LOH on the cell genealogy ----------
def loh_masks(gen, start, rem, lam, rng):
    """LOH below a point `rem` years above node `start`: (homozygous-mutant mask, lost mask)."""
    hom = lost = 0
    stack = [(start, rem)]
    while stack:
        u, L = stack.pop()
        if lam > 0 and rng.random() < 1 - math.exp(-lam * L):
            if rng.random() < 0.5:
                hom |= gen["desc"][u]
            else:
                lost |= gen["desc"][u]
            continue
        for c in gen["children"].get(u, ()):
            stack.append((c, gen["blen"][c]))
    return hom, lost


KINDS = [("TE", TE_FRAC * B_TE), ("FUNC", (1 - TE_FRAC) * FUNC_OF_NONTE), ("NEUT", (1 - TE_FRAC) * (1 - FUNC_OF_NONTE))]


def pick(rng, items):
    z = sum(w for _, w in items)
    x = rng.random() * z
    for k, w in items:
        x -= w
        if x <= 0:
            return k
    return items[-1][0]


def mutations(gen, p, rng):
    muts = []
    for u, L in gen["blen"].items():
        if L <= 0:
            continue
        for _ in range(C.poisson(rng, MU * p.mu_scale * L)):
            hom, lost = loh_masks(gen, u, rng.random() * L, p.loh, rng)
            kind = pick(rng, KINDS)
            if kind == "FUNC" and rng.random() > p.func_keep:
                continue
            cnv = rng.random() < (p.cnv_te if kind == "TE" else p.cnv_other)
            muts.append({"carry": gen["desc"][u], "hom": hom, "lost": lost, "kind": kind,
                         "type": pick(rng, list(P_TYPE.items())), "cnv": cnv, "map": rng.random()})
    return muts


def tip_bits(gen):
    """Per tip: list of (bit, layer)."""
    out = {}
    for i, (tip, l) in enumerate(gen["leaf_info"]):
        out.setdefault(tip, []).append((1 << i, l))
    return out


def fractions(m, bits, w, nf):
    """(mutant allele fraction, carrier fraction, homozygous fraction) in a leaf; carrier/hom weighted by layer."""
    car = hom = 0.0
    for b, l in bits:
        if m["carry"] & b and not m["lost"] & b:
            if m["hom"] & b:
                hom += w[l] / nf
            else:
                car += w[l] / nf
    den = 3 if m["cnv"] else 2
    return (car + 2 * hom) / den, car + hom, hom


# ---------- sequencing ----------
def nb_depth(rng, mean):
    return C.poisson(rng, rng.gammavariate(DEPTH_R, mean / DEPTH_R))


def binom(rng, n, q):
    if q <= 0:
        return 0
    if q >= 1:
        return n
    if n * min(q, 1 - q) > 25:
        return max(0, min(n, int(round(rng.gauss(n * q, math.sqrt(n * q * (1 - q)))))))
    return sum(rng.random() < q for _ in range(n))


def read_site(rng, a, mean, err, od=0.0):
    d = nb_depth(rng, mean)
    if d < MIN_DEPTH:
        return None
    if od > 0 and 0 < a < 1:
        n = 1 / od - 1
        a = rng.betavariate(a * n, (1 - a) * n)
    return binom(rng, d, a * (1 - err) + (1 - a) * err / 3), d


def run(p: Params, gmm_samples=6):
    rng = random.Random(p.seed)
    nodes, root = ramet_tree(p)
    cells, leaves = grow(p, nodes, root, rng)
    gen = genealogy(cells, leaves)
    bits = tip_bits(gen)
    nf = p.leaf_founders
    w = p.weights
    muts = mutations(gen, p, rng)

    # reference assembly from the reference ramet
    sites = []                                   # every candidate site: dict with per-sample true alt fraction
    for m in muts:
        f_ref, c_ref, h_ref = fractions(m, bits[NREF], w, nf)
        rev = False
        if not m["cnv"] and c_ref > 0:
            pr = 0.5 * (c_ref > 0.5) + 0.5 * (h_ref > 0.5)
            rev = rng.random() < pr
        m["rev"] = rev
        carriers = {}
        mm = m["carry"] & ~m["lost"]
        while mm:
            b = mm & -mm
            mm ^= b
            tip = gen["leaf_info"][b.bit_length() - 1][0]
            if tip != NREF:
                carriers[tip] = None
        alt = {}
        for tip in carriers:
            f, c, h = fractions(m, bits[tip], w, nf)
            if c > 0:
                src = "cnv" if m["cnv"] else ("loh_full" if h >= 0.999 else "loh_part" if h > 0 else
                                              "fixed" if c >= 0.999 else "mosaic")
                alt[tip] = (f, src)
        if rev:
            alt = {s: ((1 - alt[s][0]) if s in alt else 1.0, "reversed") for s in range(NREF)}
        if alt:
            sites.append({"alt": alt, "kind": m["kind"], "type": m["type"], "cnv": m["cnv"], "map": m["map"],
                          "true": True})
    for i in range(p.n_paralog):
        te = rng.random() < 0.85
        sites.append({"para": rng.uniform(0.08, 0.45), "kind": "TE" if te else "NEUT", "type": pick(rng, list(P_TYPE.items())),
                      "cnv": False, "map": rng.random(), "true": False, "src": "fp_paralog"})
    for i in range(p.n_hotspot):
        te = rng.random() < 0.85
        sites.append({"hot": rng.expovariate(1 / p.hot_err), "kind": "TE" if te else "NEUT",
                      "type": pick(rng, list(P_TYPE.items())), "cnv": False, "map": rng.random(), "true": False,
                      "src": "fp_hotspot"})
    for i in range(p.n_asm):
        sites.append({"asm": True, "kind": "TE" if rng.random() < 0.5 else "NEUT", "type": pick(rng, list(P_TYPE.items())),
                      "cnv": False, "map": rng.random(), "true": False, "src": "fp_asm"})

    callable_ = [rng.uniform(p.callable_min, 1.0) for _ in range(NREF)]
    lib = [math.exp(rng.gauss(0, p.lib_sd)) for _ in range(NREF)]
    calls = [[] for _ in range(NREF)]            # per sample: (site index, vaf, source)
    n_cov = [0] * len(sites)
    for si, s in enumerate(sites):
        for smp in range(NREF):
            if s["map"] > callable_[smp] or (p.dropout and rng.random() < p.dropout):
                continue
            mean = DEPTH_MEAN * (1.5 if s["cnv"] else 1.0)
            if "alt" in s:
                if smp not in s["alt"]:
                    continue
                a, src = s["alt"][smp]
                err = p.base_err
            elif "para" in s:
                a, src = min(0.95, s["para"] * lib[smp] ** 0.5 * rng.gammavariate(11, 1 / 11)), "fp_paralog"
                mean /= 1 - a
                err = p.base_err
            elif "hot" in s:
                a, src, err = 0.0, "fp_hotspot", min(0.5, 3 * s["hot"] * lib[smp] * rng.gammavariate(4, 0.25))
            else:
                a, src, err = 1.0, "fp_asm", p.base_err
            if p.indel_bias and s["type"] != "SNP" and 0 < a < 1:
                a = a * (1 - p.indel_bias) / (a * (1 - p.indel_bias) + 1 - a)
            r = read_site(rng, a, mean, err, p.od)
            if r:
                n_cov[si] += 1
                if r[0] >= MIN_ALT and (p.call_a50 <= 0 or rng.random() < 1 / (1 + math.exp(p.call_a50 - r[0]))):
                    calls[smp].append((si, r[0] / r[1], src))

    # paper pipeline: remove sites with VAF>=0.95 in all 72 samples
    n_hi = {}
    for smp in range(NREF):
        for si, v, _ in calls[smp]:
            if v >= 0.95:
                n_hi[si] = n_hi.get(si, 0) + 1
    need = (lambda si: NREF) if p.shared_rule == "all" else (lambda si: max(2, n_cov[si]))
    shared = {si for si, c in n_hi.items() if c >= need(si)}
    shared_src = {}
    for si in shared:
        s = sites[si]
        k = "fp_asm" if "asm" in s else ("reversed" if s.get("true") else s.get("src", "other"))
        shared_src[k] = shared_src.get(k, 0) + 1
    calls = [[c for c in cs if c[0] not in shared] for cs in calls]
    return summarize(p, sites, calls, callable_, shared, shared_src, gen, nodes, rng, gmm_samples)


# ---------- germline heterozygous sites ----------
def germline_sharing(gen, p, rng):
    """Histogram over founder het sites of the number of the 72 samples still heterozygous."""
    bits = tip_bits(gen)
    hist = [0] * (NREF + 1)
    root_kids = gen["children"].get(0, [])
    for _ in range(p.germline_het):
        hom = lost = 0
        for c in root_kids:
            h, l = loh_masks(gen, c, gen["blen"][c], p.loh, rng)
            hom |= h
            lost |= l
        n = 0
        for tip in range(NREF):
            # a sample still looks heterozygous if at least a third of its leaf is not LOH
            share = sum(p.weights[l] / p.leaf_founders for b, l in bits[tip] if not (hom | lost) & b)
            n += share > 0.3
        hist[n] += 1
    return hist


# ---------- 1-D Gaussian mixture (EM, BIC) ----------
def gmm(x, K, iters=200):
    xs = sorted(x)
    n = len(xs)
    mu = [xs[int((i + 0.5) * n / K)] for i in range(K)]
    sd = [0.08] * K
    pi = [1 / K] * K
    ll = 0.0
    for _ in range(iters):
        R = []
        ll = 0.0
        for v in xs:
            ps = [pi[k] / (sd[k] * 2.5066) * math.exp(-0.5 * ((v - mu[k]) / sd[k]) ** 2) for k in range(K)]
            z = sum(ps) or 1e-300
            ll += math.log(z)
            R.append([q / z for q in ps])
        for k in range(K):
            nk = sum(r[k] for r in R) + 1e-9
            mu[k] = sum(r[k] * v for r, v in zip(R, xs)) / nk
            sd[k] = max(0.01, math.sqrt(sum(r[k] * (v - mu[k]) ** 2 for r, v in zip(R, xs)) / nk))
            pi[k] = nk / n
    bic = -2 * ll + (3 * K - 1) * math.log(n)
    comp = sorted(zip(mu, sd, pi))
    return bic, [{"mu": a, "sd": b, "w": c} for a, b, c in comp]


def summarize(p, sites, calls, callable_, shared, shared_src, gen, nodes, rng, gmm_samples):
    per = []
    hist_src = {s: [0.0] * (NBIN + 1) for s in SOURCES}
    hom95_src = {s: 0.0 for s in SOURCES}
    share_n = {}
    for smp, cs in enumerate(calls):
        r = {"calls": len(cs), "hom95": 0, "te": 0, "SNP": 0, "nonsyn": 0, "fp": 0, "low": 0, "callable": callable_[smp],
             "patch": nodes[smp]["patch"], "dref": abs(nodes[smp]["patch"] - nodes[NREF]["patch"]) * C.PATCH_M,
             "rev95": 0, "loh95": 0}
        for si, v, src in cs:
            s = sites[si]
            share_n[si] = share_n.get(si, 0) + 1
            r["te"] += s["kind"] == "TE"
            r["SNP"] += s["type"] == "SNP"
            r["nonsyn"] += s["kind"] == "FUNC" and s["type"] == "SNP" and rng.random() < 0.7
            r["fp"] += not s["true"]
            r["low"] += v < 0.2
            if v >= 0.95:
                r["hom95"] += 1
                r["rev95"] += src == "reversed"
                r["loh95"] += src in ("loh_full", "loh_part")
                hom95_src[src] += 1 / NREF
            hist_src[src][min(NBIN, int(v / BIN))] += 1 / NREF
        r["age"] = r["calls"] / (MU_SITE * G * callable_[smp])
        per.append(r)
    m = lambda k: sum(x[k] for x in per) / len(per)
    sharing = [0] * (NREF + 1)
    sharing_true = [0] * (NREF + 1)
    for si, n in share_n.items():
        sharing[n] += 1
        if sites[si]["true"]:
            sharing_true[n] += 1
    peaks = []
    for smp in range(min(gmm_samples, NREF)):
        x = [v for _, v, _ in calls[smp]]
        fits = {K: gmm(x, K) for K in (4, 5, 6)}
        best = min(fits, key=lambda K: fits[K][0])
        peaks.append({"sample": smp, "best_k": best, "bic": {K: round(f[0], 1) for K, f in fits.items()},
                      "comp": fits[best][1], "comp5": fits[5][1]})
    calls_list = [x["calls"] for x in per]
    return {
        "params": asdict(p),
        "summary": {"calls": m("calls"), "calls_min": min(calls_list), "calls_max": max(calls_list),
                    "hom95": m("hom95"), "hom95_min": min(x["hom95"] for x in per),
                    "hom95_max": max(x["hom95"] for x in per), "te_share": m("te") / m("calls"),
                    "snp_share": m("SNP") / m("calls"), "fp_share": m("fp") / m("calls"), "low": m("low") / m("calls"),
                    "nonsyn": m("nonsyn"), "age": m("age"), "age_sd": stdev([x["age"] for x in per]),
                    "shared_removed": len(shared), "shared_src": shared_src,
                    "germline_all72": None},
        "per_sample": per, "hist_src": hist_src, "hom95_src": hom95_src,
        "sharing": sharing, "sharing_true": sharing_true, "peaks": peaks,
        "germline": germline_sharing(gen, p, rng),
    }


def stdev(x):
    mu = sum(x) / len(x)
    return math.sqrt(sum((v - mu) ** 2 for v in x) / (len(x) - 1))


def main():
    kw, out = {}, "real.js"
    for a in sys.argv[1:]:
        k, v = a.split("=")
        if k == "out":
            out = v
            continue
        cur = getattr(Params(), k)
        kw[k] = tuple(map(float, v.split(","))) if k == "weights" else v if isinstance(cur, str) else type(cur)(float(v))
    p = replace(Params(), **kw)
    res = run(p, gmm_samples=24)
    g = res["germline"]
    res["summary"]["germline_all72"] = g[NREF] / max(1, sum(g))
    with open(out, "w") as f:
        f.write("const REAL = " + json.dumps(res, separators=(",", ":")) + ";\n")
    print(json.dumps(res["summary"], indent=1))
    print("peaks", [(q["best_k"], [round(c["mu"], 2) for c in q["comp"]]) for q in res["peaks"]])


if __name__ == "__main__":
    main()
