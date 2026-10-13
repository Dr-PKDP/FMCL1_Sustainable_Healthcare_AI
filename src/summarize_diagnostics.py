"""
summarize_diagnostics.py -- per-class and selection diagnostics from the
per-seed JSON files written by run_matched_target_study.py.

For each results directory it reports, at the first round of the qualifying
streak, which is the round counted as rounds to target (or at the last round
for seeds that did not reach the target):
    - per-class recall (mean over seeds, classes absent from the evaluation
      set excluded) and macro F1;
    - the confusion matrix summed over seeds, with row-normalized recall;
    - the fraction of rounds in which selection was binding (more candidates
      than slots) and the mean relative energy multiplier of the selected
      clients (1.00 = high tier, 1.10 = medium, 1.70 = low).

With --dir_b it also reports, for rounds in which selection was binding in
both runs of the same seed, the mean Jaccard overlap of the selected sets.
That comparison is meaningful only when both runs used --paired_availability.

The balanced accuracy stored as balanced_acc_at_target is the lowest value
within the five-evaluation qualifying streak, so it can be below the mean
per-class recall at the first round of the streak. --per_seed prints both for
every seed.

Usage:
    python summarize_diagnostics.py --dir results/paired_energy_aware \\
        --dir_b results/paired_random [--per_seed]
"""
import argparse
import glob
import json
import os

CLASS_NAMES = ["N", "S", "V", "F", "Q"]


def load(dir_path):
    out = []
    for fp in sorted(glob.glob(os.path.join(dir_path, "mitbih_resnet_seed*.json"))):
        with open(fp) as f:
            out.append(json.load(f))
    return out


def snapshot(run):
    """History entry at the credited round, else the last round."""
    h = run["history"]
    if run["reached_target"]:
        idx = run["rounds_to_target"] - 1
    else:
        idx = len(h) - 1
    return h[idx]


def per_seed_table(label, runs, window=5):
    """One row per seed: credited round, balanced accuracy at the snapshot and
    over the qualifying streak, and the per-class recalls at the snapshot."""
    print(f"\n--- {label}: per-seed values ---")
    print("  seed  rounds  BA@first  BA_min(streak)  BA_mean(streak)  mean recall   "
          + "  ".join(f"{n:>5}" for n in CLASS_NAMES))
    for r in runs:
        h = r["history"]
        idx = r["rounds_to_target"] - 1 if r["reached_target"] else len(h) - 1
        streak = [e["balanced_acc"] for e in h[idx:idx + window]]
        rec = h[idx]["per_class_recall"]
        present = [v for v in rec if v is not None]
        print(f"  {r['config']['seed']:>4}  {idx + 1:>6}  {h[idx]['balanced_acc']:>8.3f}"
              f"  {min(streak):>14.3f}  {sum(streak) / len(streak):>15.3f}"
              f"  {sum(present) / len(present):>11.3f}   "
              + "  ".join(f"{v:>5.2f}" if v is not None else "  n/a" for v in rec))


def summarize(label, runs):
    print(f"\n=== {label}: {len(runs)} seeds, "
          f"{sum(r['reached_target'] for r in runs)} reached the target ===")
    if not runs:
        return
    if "per_class_recall" not in runs[0]["history"][0]:
        raise SystemExit(f"{label}: these files were written before per-class "
                         "outputs were recorded; rerun with the current "
                         "run_matched_target_study.py.")
    print(f"  evaluation protocol: {runs[0].get('eval_protocol', 'not recorded')}")
    n_cls = len(CLASS_NAMES)
    sums = [0.0] * n_cls
    cnts = [0] * n_cls
    conf = [[0] * n_cls for _ in range(n_cls)]
    f1s = []
    for r in runs:
        s = snapshot(r)
        for c, v in enumerate(s["per_class_recall"]):
            if v is not None:
                sums[c] += v
                cnts[c] += 1
        for i in range(n_cls):
            for j in range(n_cls):
                conf[i][j] += s["confusion"][i][j]
        f1s.append(s["macro_f1"])
    print("  mean per-class recall: " + "  ".join(
        f"{CLASS_NAMES[c]}={sums[c] / cnts[c]:.3f}" if cnts[c] else f"{CLASS_NAMES[c]}=n/a"
        for c in range(n_cls)))
    print(f"  mean macro F1: {sum(f1s) / len(f1s):.3f}")
    print("  confusion matrix summed over seeds (rows true, columns predicted):")
    print("        " + "".join(f"{n:>9}" for n in CLASS_NAMES))
    for i in range(n_cls):
        print(f"    {CLASS_NAMES[i]:>3} " + "".join(f"{conf[i][j]:>9d}" for j in range(n_cls)))
    bf = [r["selection_binding_fraction"] for r in runs]
    em = [r["mean_selected_energy_mult"] for r in runs]
    print(f"  selection binding in {sum(bf) / len(bf):.3f} of rounds (mean over seeds)")
    print(f"  mean energy multiplier of selected clients: {sum(em) / len(em):.3f}")


def overlap(runs_a, runs_b):
    by_seed_b = {r["config"]["seed"]: r for r in runs_b}
    vals = []
    for ra in runs_a:
        rb = by_seed_b.get(ra["config"]["seed"])
        if rb is None:
            continue
        for ha, hb in zip(ra["history"], rb["history"]):
            K = ra["config"]["K"]
            if ha["n_candidates"] > K and hb["n_candidates"] > K:
                a, b = set(ha["selected_ids"]), set(hb["selected_ids"])
                if a | b:
                    vals.append(len(a & b) / len(a | b))
    if vals:
        print(f"\n  selected-set Jaccard overlap in binding rounds: "
              f"{sum(vals) / len(vals):.3f} over {len(vals)} rounds")
    else:
        print("\n  no rounds in which selection was binding in both runs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--dir_b", default=None)
    ap.add_argument("--per_seed", action="store_true",
                    help="also print one row per seed")
    args = ap.parse_args()
    a = load(args.dir)
    summarize(args.dir, a)
    if args.per_seed:
        per_seed_table(args.dir, a)
    if args.dir_b:
        b = load(args.dir_b)
        summarize(args.dir_b, b)
        if args.per_seed:
            per_seed_table(args.dir_b, b)
        overlap(a, b)
