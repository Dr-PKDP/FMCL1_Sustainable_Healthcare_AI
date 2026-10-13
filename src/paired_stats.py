"""
paired_stats.py -- paired comparison of rounds-to-target between two result
directories written by run_matched_target_study.py, matched by seed.

Used for the Table 5 policy comparison (energy_aware vs random, both run
with --paired_availability) and for the stress-test comparison against the
baseline. Seeds are matched by file name; a seed that did not reach the
target in either directory is dropped and reported.

    python src/paired_stats.py --a_dir results/paired_energy_aware \
        --b_dir results/paired_random

Reports the mean paired difference (a - b), a paired t-test, a Wilcoxon
signed-rank test, and a percentile bootstrap 95% CI of the mean difference
(10,000 resamples, fixed seed). The CI is the quantity to read; with ten
seeds the tests have limited power.
"""
import argparse
import glob
import json
import os

import numpy as np


def load(dir_path, model):
    out = {}
    for fp in sorted(glob.glob(os.path.join(dir_path, f"mitbih_{model}_seed*.json"))):
        with open(fp) as f:
            r = json.load(f)
        out[os.path.basename(fp)] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_dir", required=True)
    ap.add_argument("--b_dir", required=True)
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    A, B = load(args.a_dir, "resnet"), load(args.b_dir, "resnet")
    names = sorted(set(A) & set(B))
    pairs = [(A[n]["rounds_to_target"], B[n]["rounds_to_target"]) for n in names
             if A[n]["reached_target"] and B[n]["reached_target"]]
    dropped = len(names) - len(pairs)
    if len(pairs) < 2:
        raise SystemExit("fewer than two seeds reached the target in both directories")
    a = np.array([p[0] for p in pairs], float)
    b = np.array([p[1] for p in pairs], float)
    d = a - b
    print(f"matched seeds: {len(names)}; used: {len(pairs)}; dropped (target not reached): {dropped}")
    print(f"mean rounds  A: {a.mean():.1f} (range {a.min():.0f}-{a.max():.0f}, median {np.median(a):.1f})")
    print(f"mean rounds  B: {b.mean():.1f} (range {b.min():.0f}-{b.max():.0f}, median {np.median(b):.1f})")
    print(f"mean paired difference A-B: {d.mean():+.2f}  (A fewer in {(d < 0).sum()}, ties {(d == 0).sum()}, A more in {(d > 0).sum()})")
    rng = np.random.default_rng(0)
    boots = rng.choice(d, size=(args.n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"bootstrap 95% CI of the mean difference: [{lo:+.1f}, {hi:+.1f}]")
    try:
        from scipy import stats
        t = stats.ttest_rel(a, b)
        print(f"paired t-test p = {t.pvalue:.3f}")
        try:
            w = stats.wilcoxon(a, b)
            print(f"Wilcoxon signed-rank p = {w.pvalue:.3f}")
        except ValueError as e:
            print(f"Wilcoxon not computed: {e}")
    except ImportError:
        print("scipy not installed: p-values skipped (pip install scipy)")


if __name__ == "__main__":
    main()
