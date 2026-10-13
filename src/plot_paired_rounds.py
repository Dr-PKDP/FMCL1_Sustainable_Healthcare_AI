"""
plot_paired_rounds.py -- rounds to target by seed under two selection policies,
drawn as paired lines (one line per seed), from the per-seed JSON files written
by run_matched_target_study.py. Several conditions can be drawn as separate
panels.

Usage:
    python plot_paired_rounds.py --out paired_rounds.png \
        --panel "Run B" results/rerun_energy_aware results/rerun_random \
        --panel "Held-out" results/heldout_energy_aware results/heldout_random
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rounds(dir_path):
    out = {}
    for fp in sorted(glob.glob(os.path.join(dir_path, "mitbih_resnet_seed*.json"))):
        with open(fp) as f:
            r = json.load(f)
        out[r["config"]["seed"]] = r["rounds_to_target"] if r["reached_target"] else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", nargs=3, action="append", required=True,
                    metavar=("TITLE", "ENERGY_AWARE_DIR", "RANDOM_DIR"))
    ap.add_argument("--out", default="paired_rounds.png")
    args = ap.parse_args()
    n = len(args.panel)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 2.6), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (title, da, db) in zip(axes, args.panel):
        a, b = rounds(da), rounds(db)
        seeds = [s for s in sorted(a) if s in b and a[s] is not None and b[s] is not None]
        for s in seeds:
            ax.plot([0, 1], [a[s], b[s]], color="0.55", lw=0.8, marker="o",
                    ms=3.5, mfc="white", mec="0.2", mew=0.8)
        ma = sum(a[s] for s in seeds) / len(seeds)
        mb = sum(b[s] for s in seeds) / len(seeds)
        ax.plot([0, 1], [ma, mb], color="black", lw=2.0, marker="s", ms=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Energy-aware", "Random"], fontsize=8)
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylabel("Rounds to target", fontsize=8)
        ax.set_title(f"{title} (n = {len(seeds)} seeds)", fontsize=8.5)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300)


if __name__ == "__main__":
    main()
