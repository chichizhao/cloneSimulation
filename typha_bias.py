"""Does a TE-biased somatic mutation spectrum need an intrinsic 'preference'? (Typha latifolia model)

Observed (Zhao et al. 2026): 71% of somatic mutations fall in transposable elements (TE), which are 34% of the genome.
Null: mutations land proportional to length -> 34% in TE. We sweep
  b = relative per-bp mutability of TE vs non-TE sequence
  s = strength of purifying selection against mutations in functional non-TE sequence
and find which (b, s) reproduce 71%. Writes typha.js for typha.html.
"""
import json
import random

TE_FRAC = 0.34
FUNCTIONAL_OF_NONTE = 0.15   # share of non-TE sequence where a mutation is deleterious (assumption)
OBSERVED = 0.71
N_MUT = 2000                 # mutations drawn per replicate
REPS = 200
BS = [1 + 0.25 * i for i in range(0, 25)]   # 1 .. 7
SELS = [0.0, 0.5, 0.9]


def expected_te_fraction(b, s):
    te = TE_FRAC * b
    nonte = (1 - TE_FRAC) * (1 - FUNCTIONAL_OF_NONTE * s)
    return te / (te + nonte)


def simulate(b, s, rng):
    p = expected_te_fraction(b, s)
    fracs = []
    for _ in range(REPS):
        k = sum(rng.random() < p for _ in range(N_MUT))
        fracs.append(k / N_MUT)
    fracs.sort()
    return sum(fracs) / REPS, fracs[int(0.025 * REPS)], fracs[int(0.975 * REPS)]


def required_b(s):
    nonte = (1 - TE_FRAC) * (1 - FUNCTIONAL_OF_NONTE * s)
    return OBSERVED * nonte / (TE_FRAC * (1 - OBSERVED))


def main():
    rng = random.Random(7)
    out = {"te_frac": TE_FRAC, "observed": OBSERVED, "bs": BS, "curves": []}
    for s in SELS:
        pts = [simulate(b, s, rng) for b in BS]
        out["curves"].append({"s": s, "mean": [p[0] for p in pts], "lo": [p[1] for p in pts],
                              "hi": [p[2] for p in pts], "required_b": required_b(s)})
        print(f"s={s}: need TE mutability x{required_b(s):.2f} to reach {OBSERVED:.0%} "
              f"(null b=1 gives {expected_te_fraction(1, s):.1%})")
    with open("typha.js", "w") as f:
        f.write("const TYPHA = " + json.dumps(out) + ";\n")


if __name__ == "__main__":
    main()
