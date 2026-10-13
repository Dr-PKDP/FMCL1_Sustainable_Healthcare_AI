#!/usr/bin/env bash
# Additional runs for Section 5.2.4, in stages. Run from the repository root.
#
#   DATA=/path/to/mitdb bash scripts/run_gpu_followups.sh baseline
#   DATA=/path/to/mitdb bash scripts/run_gpu_followups.sh p03
#   DATA=/path/to/mitdb bash scripts/run_gpu_followups.sh heldout_probe
#   DATA=/path/to/mitdb TARGET=0.40 bash scripts/run_gpu_followups.sh heldout
#   bash scripts/run_gpu_followups.sh report
#
# Stages write to results/rerun_*, results/p03_* and results/heldout_*, and do
# not touch results/paired_*. A seed whose JSON already exists is skipped, so an
# interrupted stage can be restarted with the same command.
set -euo pipefail
STAGE="${1:?stage: baseline | p03 | heldout_probe | heldout | report}"
SRC=src
RUN="python $SRC/run_matched_target_study.py"
mkdir -p results logs

need_data() { : "${DATA:?set DATA to the directory holding the MIT-BIH .hea/.dat/.atr files}"; }

case "$STAGE" in
  baseline)   # Table 5 protocol with per-class output recorded
    need_data
    for P in energy_aware random; do
      $RUN --data_dir "$DATA" --policy $P --paired_availability --n_seeds 10 \
           --out_dir results/rerun_$P 2>&1 | tee -a logs/rerun_$P.log
    done ;;
  p03)        # higher availability: about 13 candidates for K = 6
    need_data
    for P in energy_aware random; do
      $RUN --data_dir "$DATA" --policy $P --paired_availability --participation_p 0.3 \
           --n_seeds 10 --out_dir results/p03_$P 2>&1 | tee -a logs/p03_$P.log
    done ;;
  heldout_probe)  # balanced accuracy that 9 held-out recordings allow
    need_data
    $RUN --data_dir "$DATA" --policy energy_aware --paired_availability \
         --heldout_records 9 --probe_only --n_seeds 3 \
         --out_dir results/heldout_probe 2>&1 | tee -a logs/heldout_probe.log ;;
  heldout)    # paired policies on held-out recordings, target taken from the probe
    need_data
    : "${TARGET:?set TARGET from the held-out probe, e.g. TARGET=0.30}"
    for P in energy_aware random; do
      $RUN --data_dir "$DATA" --policy $P --paired_availability \
           --heldout_records 9 --target "$TARGET" --n_seeds 10 \
           --out_dir results/heldout_$P 2>&1 | tee -a logs/heldout_$P.log
    done ;;
  report)     # summaries for whichever stages have finished
    for PAIR in "rerun_energy_aware rerun_random" "p03_energy_aware p03_random" \
                "heldout_energy_aware heldout_random"; do
      set -- $PAIR
      if [ -d results/$1 ] && [ -d results/$2 ]; then
        echo; echo "################ $1 vs $2 ################"
        python $SRC/make_table.py --energy_aware_dir results/$1 --random_dir results/$2 || true
        python $SRC/paired_stats.py --a_dir results/$1 --b_dir results/$2 || true
        python $SRC/summarize_diagnostics.py --dir results/$1 --dir_b results/$2 || true
      fi
    done
    if [ -d results/heldout_probe ]; then
      echo; echo "################ heldout_probe ################"
      python $SRC/summarize_diagnostics.py --dir results/heldout_probe || true
    fi ;;
  *) echo "unknown stage: $STAGE"; exit 1 ;;
esac
