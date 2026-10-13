#!/usr/bin/env bash
# Runs seed 0 of the energy-aware policy twice with identical settings and
# compares the two trajectories. Run from the repository root with DATA set to
# the MIT-BIH directory. Each run takes a few minutes on a GPU.
set -euo pipefail
: "${DATA:?set DATA to the directory holding the MIT-BIH files}"
mkdir -p results logs
for K in a b; do
  python src/run_matched_target_study.py --data_dir "$DATA" --policy energy_aware \
      --paired_availability --n_seeds 1 --out_dir results/determinism_$K \
      2>&1 | tee -a logs/determinism_$K.log
done
python - <<'PY'
import json
a = json.load(open("results/determinism_a/mitbih_resnet_seed00.json"))
b = json.load(open("results/determinism_b/mitbih_resnet_seed00.json"))
print("rounds_to_target:", a["rounds_to_target"], b["rounds_to_target"])
ha, hb = a["history"], b["history"]
same_sel = all(x["selected_ids"] == y["selected_ids"] for x, y in zip(ha, hb))
diff = [abs(x["balanced_acc"] - y["balanced_acc"]) for x, y in zip(ha, hb)]
print("selected clients identical in every round:", same_sel)
print("rounds compared:", len(diff), " max |BA difference|:", max(diff))
print("bitwise identical trajectories:", max(diff) == 0.0 and same_sel)
PY
