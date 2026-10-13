"""
make_table.py -- builds Table 5 (Section 5.2.4) from the per-seed JSON files
written by run_matched_target_study.py.

Usage:
    python make_table.py --energy_aware_dir results/paired_energy_aware \\
                         --random_dir results/paired_random

Expects files named mitbih_resnet_seed{00..NN}.json in each directory.
"""
import argparse
import glob
import json
import os
import statistics


def load_seeds(dir_path, model_name):
    files = sorted(glob.glob(os.path.join(dir_path, f"mitbih_{model_name}_seed*.json")))
    out = []
    for fp in files:
        with open(fp) as f:
            out.append(json.load(f))
    return out


def summarize(results):
    reached = [r for r in results if r["reached_target"]]
    n_total = len(results)
    n_reached = len(reached)
    row = {"n_seeds": n_total, "n_reached": n_reached}
    if reached:
        rounds = [r["rounds_to_target"] for r in reached]
        bal_acc = [r["balanced_acc_at_target"] for r in reached]
        carbon = [r["lifecycle_to_target_kg_CO2e"]["FMCL"] for r in reached]
        row.update({
            "rounds_mean": statistics.mean(rounds),
            "rounds_min": min(rounds),
            "rounds_max": max(rounds),
            "balanced_acc_mean": statistics.mean(bal_acc),
            "carbon_mean_kg": statistics.mean(carbon),
            "carbon_min_kg": min(carbon),
            "carbon_max_kg": max(carbon),
        })
    else:
        row.update({k: None for k in (
            "rounds_mean", "rounds_min", "rounds_max",
            "balanced_acc_mean", "carbon_mean_kg", "carbon_min_kg", "carbon_max_kg")})
    return row


def fmt_row(label, row):
    if row["n_reached"] == 0:
        return (f"{label:<22} reached target in 0/{row['n_seeds']} seeds "
                f"(not included in the comparison)")
    return (f"{label:<22} "
            f"{row['rounds_mean']:5.1f} ({row['rounds_min']:.0f}-{row['rounds_max']:.0f})  "
            f"{row['balanced_acc_mean']:.3f}              "
            f"{row['carbon_mean_kg']:.3f} ({row['carbon_min_kg']:.3f}-{row['carbon_max_kg']:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy_aware_dir", required=True)
    parser.add_argument("--random_dir", required=True)
    args = parser.parse_args()

    print("=== ResNet1D ===")
    ea_resnet = summarize(load_seeds(args.energy_aware_dir, "resnet"))
    rand_resnet = summarize(load_seeds(args.random_dir, "resnet"))
    print(f"{'Policy':<22} {'Rounds (mean,range)':<22} {'Bal.acc':<18} "
          f"{'Carbon kg CO2e (mean,range)'}")
    print(fmt_row("Energy-aware (FMCL)", ea_resnet))
    print(fmt_row("Random", rand_resnet))
