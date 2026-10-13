# Reproducibility package: lifecycle-carbon accounting and client selection for FMCL

Code for the computed results of "Toward Sustainable Healthcare AI Infrastructure: A Lifecycle-Aware Federated Mobile Crowd Learning Framework".

| Paper element | Script |
|---|---|
| Section 5.2.4, Table 5: energy-aware versus random client selection on MIT-BIH arrhythmia classification | `run_matched_target_study.py`, `make_table.py`, `paired_stats.py` |
| Section 5.2.4, Table 7 (weight-grid ablation) and the stress test | `run_matched_target_study.py`, `make_ablation_stress_tables.py` |
| Section 5.2.4, Table 6 (per-class recall), higher availability, held-out recordings | `run_matched_target_study.py`, `summarize_diagnostics.py`, `scripts/run_gpu_followups.sh` |
| Section 5.2.4, Fig. 6 (rounds to target by seed) | `plot_paired_rounds.py` |
| Section 5.2.6, Table 8: four-architecture lifecycle comparison | `table3_accounting.py` |
| Section 5.3, Table 9 and the joint Monte Carlo analyses | `sensitivity_analysis.py` |

## Layout

```
src/
  models.py                       ResNet1D
  fl_harness.py                   FLOP and byte counting, selection policies,
                                  FedProx local training, FedAvg aggregation
  engine.py                       lifecycle carbon accounting used for Table 5
  data_mitbih.py                  MIT-BIH loader (one client per recording)
  run_matched_target_study.py     experiment driver
  make_table.py                   Table 5 from the per-seed JSON files
  make_ablation_stress_tables.py  ablation and stress-test comparisons
  paired_stats.py                 paired comparison of rounds to target
  summarize_diagnostics.py        per-class recall, confusion matrix, selection diagnostics
  plot_paired_rounds.py           Fig. 6, paired rounds to target by seed
  table3_accounting.py            Table 8
  sensitivity_analysis.py         Table 9 and the Monte Carlo analyses
scripts/
  run_gpu_followups.sh            staged runs: replication, higher availability, held-out
  check_determinism.sh            two identical runs of seed 0, compared
results/                          per-seed JSON files, one directory per condition
requirements.txt
```

## Table 8 and Table 9

Neither script needs data or a GPU, and both run in seconds.

```
python src/table3_accounting.py       # Table 8
python src/sensitivity_analysis.py    # Table 9, joint Monte Carlo, asymmetric-grid and pre-existing-edge analyses
```

`table3_accounting.py` reproduces the arithmetic of Table 4 and Figure 4 (200 rounds, 400 g CO2e/kWh, a 50-server edge fleet at 350 kg CO2e per server over 5 years). Per-device embodied carbon for the calendar-time and shared-infrastructure conventions is 35.98 kg CO2e, the production-phase GWP100 of the Fairphone 2 life-cycle assessment (Fraunhofer IZM, 2016; reference [73] of the paper). The module docstring lists every assumption.

`sensitivity_analysis.py` also repeats the joint Monte Carlo with participant, edge and cloud grid intensities drawn independently, and with the edge fleet treated as already existing, in which case the campaign is charged only the power it adds above idle during active rounds and no embodied carbon.

## Table 5 experiment

### Design

Each of the 44 MIT-BIH recordings (48 recordings less four paced-rhythm recordings, under the AAMI convention) is one client, and no beat appears in more than one client. Recordings 201 and 202 come from the same subject, so the 44 clients represent 43 subjects.

Each beat is a 300-sample window (130 samples before and 170 after the R peak, both leads) with an AAMI EC57 class label. The signal is not rescaled. By default there is no train, validation and test partition. Every beat of a recording belongs to its client's local training set, and the global model is scored after every round on one class-stratified sample of 2,000 beats (up to 200 per class) drawn once per run from the pooled client data. Balanced accuracy then measures how quickly the federation fits the pooled data, not performance on unseen recordings. `--heldout_records` changes this (see below).

Training uses K = 6 clients per round, FedProx with mu = 0.01, two local epochs, learning rate 0.001, top-k gradient sparsification (0.3), per-round client availability 0.1 and at most 150 rounds. A run stops when balanced accuracy (mean per-class recall) has stayed at or above the target, 0.30 by default, for five consecutive evaluations. Chance for five classes is 0.20. Balanced accuracy is used because the majority class alone gives 63.5% raw accuracy. The round counted as rounds to target is the first round of the qualifying streak, and `balanced_acc_at_target` is the lowest of the five values in it. The diagnostics report per-class recall at the first round of the streak, so their mean can exceed `balanced_acc_at_target`.

Seed `i` starts from the same model initialization under both policies. With `--paired_availability`, the per-round availability draws come from a separate random stream that is identical across policies for a given seed, so a difference between policies can arise only from the selection rule. Each history entry records `n_candidates`, the number of available clients in that round, and selection changes the participant set only when `n_candidates` exceeds K.

`lifecycle_to_target_kg_CO2e` is an estimate. The counted FLOPs and bytes per client-round are converted to energy by the device and communication models in `engine.py`, priced at 0.156 kg CO2e/kWh, and combined with a wear-derived embodied term for 1,000 devices. Six clients are charged per round whichever devices were selected, so carbon to target is proportional to rounds to target and does not model per-round tier energy. Carbon is charged through the first round of the qualifying streak, so the four later rounds that confirm the stopping criterion and the evaluation computation are not included. A run that does not reach the target is written with `reached_target: false` and `lifecycle_to_target_kg_CO2e: null`.

### Running it

Download the MIT-BIH Arrhythmia Database from PhysioNet (record set `mitdb`):

```
wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/
pip install -r requirements.txt
```

`PATH` below is the directory that holds the `.hea`, `.dat` and `.atr` files. Ten seeds per policy:

```
python src/run_matched_target_study.py --data_dir PATH \
    --policy energy_aware --paired_availability --n_seeds 10 \
    --out_dir results/paired_energy_aware

python src/run_matched_target_study.py --data_dir PATH \
    --policy random --paired_availability --n_seeds 10 \
    --out_dir results/paired_random

python src/make_table.py --energy_aware_dir results/paired_energy_aware \
    --random_dir results/paired_random
python src/paired_stats.py --a_dir results/paired_energy_aware \
    --b_dir results/paired_random
```

A run skips any seed whose output JSON already exists, so an interrupted run restarts with the same command. `paired_stats.py` reports the mean paired difference in rounds to target, a paired t-test, a Wilcoxon signed-rank test and a bootstrap 95% confidence interval of the mean difference.

## Fig. 6

`plot_paired_rounds.py` draws rounds to target by seed for each policy, with a line joining the two policies within a seed and a black line for the means. Each `--panel` takes a title, the energy-aware results directory and the random results directory. It needs matplotlib.

```
python src/plot_paired_rounds.py \
    --panel "Run B (n = 10 seeds)" results/rerun_energy_aware results/rerun_random \
    --panel "Held-out (n = 10 seeds)" results/heldout_energy_aware results/heldout_random \
    --out paired_rounds.png
```

## Weight-grid ablation and stress test

Both use `--paired_availability`. The default weights (0.4, 0.3, 0.3) are the first three seeds of `results/paired_energy_aware`. The four other triples are run with three seeds each:

```
for W in "0.6 0.2 0.2 a06_b02_g02" "0.2 0.2 0.6 a02_b02_g06" \
         "0.2 0.6 0.2 a02_b06_g02" "0.33 0.33 0.34 a033_b033_g034"; do
  set -- $W
  python src/run_matched_target_study.py --data_dir PATH \
      --policy energy_aware --paired_availability \
      --alpha $1 --beta $2 --gamma $3 --n_seeds 3 \
      --out_dir results/paired_abl_$4
done

python src/make_ablation_stress_tables.py ablation \
    --baseline_dir results/paired_energy_aware --n_seeds 3 \
    --grid 0.4,0.3,0.3,results/paired_energy_aware \
    --grid 0.6,0.2,0.2,results/paired_abl_a06_b02_g02 \
    --grid 0.2,0.2,0.6,results/paired_abl_a02_b02_g06 \
    --grid 0.2,0.6,0.2,results/paired_abl_a02_b06_g02 \
    --grid 0.33,0.33,0.34,results/paired_abl_a033_b033_g034
```

The weights change the participant set only in rounds where more than K clients are available, about 17% of rounds at availability 0.1.

The stress test uses the energy-aware policy and ten seeds with `--stress_test`:

```
python src/run_matched_target_study.py --data_dir PATH \
    --policy energy_aware --paired_availability --stress_test \
    --n_seeds 10 --out_dir results/paired_stress

python src/make_ablation_stress_tables.py stress \
    --baseline_dir results/paired_energy_aware \
    --stress_dir results/paired_stress

python src/paired_stats.py --a_dir results/paired_stress \
    --b_dir results/paired_energy_aware
```

`--stress_test` applies three conditions together: per-round availability of 0.06, 0.10 and 0.14 for low-, medium- and high-tier devices (mean 0.10); correlated dropout through a two-state Markov chain with lag-1 autocorrelation `--dropout_rho` (default 0.5) and the same stationary availability; and a restriction of each client to its `--keep_top_n_classes` (default 3) most frequent classes. The per-recording split is already strongly label-skewed (mean dominant-class share about 0.91, 26 of 44 clients above 0.95), and the class restriction changes 88 of 100,688 beats.

## Diagnostics and additional conditions

Every run records, for each round, the per-class recall, confusion matrix, macro F1, identifiers of the selected clients and the mean relative energy multiplier of the selected clients. `summarize_diagnostics.py` reads one results directory, or two for a policy comparison, and prints mean per-class recall, macro F1, the confusion matrix summed over seeds, the fraction of rounds in which selection was binding, the energy multiplier and the selected-set overlap between policies:

```
python src/summarize_diagnostics.py --dir results/paired_energy_aware \
    --dir_b results/paired_random
```

`--per_seed` adds one row per seed with the counted round, the balanced accuracy at the first round of the streak and its minimum and mean over the streak, and the per-class recalls with their arithmetic mean.

Result files written by earlier versions of the driver lack these fields and have to be regenerated.

`scripts/run_gpu_followups.sh` runs the remaining conditions in stages. From the repository root, with `DATA` set to the MIT-BIH directory:

```
bash scripts/run_gpu_followups.sh baseline         # replication of the Table 5 configuration with per-class output (results/rerun_*)
bash scripts/run_gpu_followups.sh p03              # availability 0.3 (results/p03_*)
bash scripts/run_gpu_followups.sh heldout_probe    # held-out recordings, no early stopping
TARGET=0.40 bash scripts/run_gpu_followups.sh heldout
bash scripts/run_gpu_followups.sh report
```

### Higher availability

`--participation_p 0.3` gives about 13 candidates per round for K = 6, so selection decides the participant set in most rounds.

### Held-out recordings

`--heldout_records 9` withholds nine recordings (201 and 202 are withheld together), trains on the rest and draws the evaluation sample from the withheld recordings only. `--split_seed` fixes which recordings are withheld, identically for every seed and policy. The held-out recordings contain no class-Q beats, so balanced accuracy is taken over four classes and chance is 0.25. `--probe_only` runs without early stopping so that a target can be chosen from the observed balanced accuracy; the paper uses 0.40.

## Output files

Each seed writes `mitbih_resnet_seed{NN}.json` with:

- `history`: per-round trajectory (round, balanced accuracy, raw accuracy, number of classes seen, number of clients selected, `n_candidates`, per-class recall, confusion matrix, macro F1, selected client identifiers, mean energy multiplier of the selected clients).
- `eval_protocol` (`pooled_training_sample` or `heldout_recordings`), `heldout_ids`, `selection_binding_fraction`, `mean_selected_energy_mult`.
- `reached_target`, `rounds_to_target`, `balanced_acc_at_target`.
- `lifecycle_to_target_kg_CO2e`: lifecycle GWP100 (kg CO2e) for the FMCL, EDGE and CLOUD architectures at the round the target was reached, or `null`. Table 5 reports the FMCL entry.
- `final_balanced_acc`, `final_raw_acc`: the last values of the trajectory.
- `counted`: FLOPs per client-round, upload size and total client-rounds used by the carbon estimate.
- `config`: every setting the run was launched with.

## Reproducibility

Trajectories are not bit-identical across runs or machines (GPU nondeterminism, PyTorch and BLAS versions). The first set of Table 5 results (run A) was produced by an earlier version of the driver that did not record per-class output. Two runs of the Table 5 configuration gave 25.2 and 24.0 mean rounds for energy-aware selection and 26.6 and 27.3 for random selection; the per-seed `config` is recorded in every result file. `scripts/check_determinism.sh` runs seed 0 twice with identical settings and compares the two trajectories. In the reference run both executions selected the same clients in every round and credited round 21, and balanced accuracy differed by at most 0.002 at an evaluation. Differences of this size can shift the round at which the stopping threshold is first sustained.
