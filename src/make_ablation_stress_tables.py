"""
make_ablation_stress_tables.py -- builds the weight-grid ablation table and the
stress-test comparison from the per-seed JSON files written by
run_matched_target_study.py with --alpha/--beta/--gamma and --stress_test.
Both tables use ResNet1D results.

USAGE
-----
Weight-grid ablation (one directory per (alpha, beta, gamma) triple; the
default triple 0.4/0.3/0.3 is read from the Table 5 energy-aware directory,
restricted to its first --n_seeds seeds):

    python make_ablation_stress_tables.py ablation \\
        --baseline_dir results/paired_energy_aware --n_seeds 3 \\
        --grid 0.4,0.3,0.3,results/paired_energy_aware \\
        --grid 0.6,0.2,0.2,results/paired_abl_a06_b02_g02 \\
        --grid 0.2,0.2,0.6,results/paired_abl_a02_b02_g06 \\
        --grid 0.2,0.6,0.2,results/paired_abl_a02_b06_g02 \\
        --grid 0.33,0.33,0.34,results/paired_abl_a033_b033_g034

Stress test (compared with the Table 5 energy-aware directory):

    python make_ablation_stress_tables.py stress \\
        --baseline_dir results/paired_energy_aware \\
        --stress_dir results/paired_stress
"""
import argparse
import sys

from make_table import load_seeds, summarize, fmt_row


def cmd_ablation(args):
    print("=== alpha/beta/gamma weight-grid ablation (ResNet1D, energy-aware policy) ===")
    print(f"{'(alpha, beta, gamma)':<24} {'Rounds (mean,range)':<22} {'Bal.acc':<18} "
          f"{'Carbon kg CO2e (mean,range)'}")
    for spec in args.grid:
        a, b, g, d = spec.split(",", 3)
        label = f"({float(a):.2f}, {float(b):.2f}, {float(g):.2f})"
        seeds = load_seeds(d, "resnet")
        if args.n_seeds:
            seeds = seeds[:args.n_seeds]
        print(fmt_row(label, summarize(seeds)))


def cmd_stress(args):
    print("=== Heterogeneity/availability stress test (ResNet1D, energy-aware policy) ===")
    print(f"{'Condition':<24} {'Rounds (mean,range)':<22} {'Bal.acc':<18} "
          f"{'Carbon kg CO2e (mean,range)'}")
    baseline = summarize(load_seeds(args.baseline_dir, "resnet"))
    stress = summarize(load_seeds(args.stress_dir, "resnet"))
    print(fmt_row("Baseline (Table 5)", baseline))
    print(fmt_row("Stress condition", stress))
    if stress["n_reached"] < stress["n_seeds"]:
        print(f"\nThe stress condition reached the target in "
              f"{stress['n_reached']}/{stress['n_seeds']} seeds; the means "
              f"above are computed over those seeds only.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_abl = sub.add_parser("ablation")
    p_abl.add_argument("--baseline_dir", required=True,
                        help="Unused directly (kept for symmetry/documentation); "
                             "the 0.4,0.3,0.3 row in --grid should point here.")
    p_abl.add_argument("--grid", action="append", required=True,
                        help="alpha,beta,gamma,result_dir -- repeatable.")
    p_abl.add_argument("--n_seeds", type=int, default=None,
                        help="Use only the first N seeds of every directory (the baseline "
                             "directory has more seeds than the ablation runs).")

    p_str = sub.add_parser("stress")
    p_str.add_argument("--baseline_dir", required=True,
                        help="Table 5's existing ResNet1D/energy_aware result directory.")
    p_str.add_argument("--stress_dir", required=True,
                        help="Output directory from a --stress_test run.")

    args = parser.parse_args()
    if args.cmd == "ablation":
        cmd_ablation(args)
    else:
        cmd_stress(args)
