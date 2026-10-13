"""
run_matched_target_study.py -- computational comparison of client-selection
policies at a fixed balanced-accuracy target (Section 5.2.4, Table 5):
ResNet1D on MIT-BIH arrhythmia classification under FMCL's energy-aware
selection score (Section 4.3, Proposition 1) and under uniform random
selection.

Each run trains to a common balanced-accuracy target and stops once the target
has been held for --stability_window consecutive evaluations. Carbon to target
is estimated through the round at which the target is credited (engine.py), not
at a fixed round count.

Balanced accuracy (mean per-class recall) is the target metric because the
MIT-BIH majority class alone yields 63.5% raw accuracy while distinguishing no
arrhythmia. A run that does not hold the target within --max_rounds is written
to its output JSON with reached_target=False and no lifecycle-carbon figure.

Availability. With --paired_availability the per-round client-availability
draws come from their own random stream, identical across policies for a given
seed, so that differences between policies arise only from the selection rule.
Without it, availability and any randomness inside the selection policy share
one stream. Each history entry records n_candidates, the number of available
clients in that round; selection changes the participant set only when
n_candidates exceeds K.

USAGE
-----
    # Table 5: ten seeds per policy, availability paired across policies
    python run_matched_target_study.py --data_dir PATH \\
        --policy energy_aware --paired_availability --target 0.30 \\
        --max_rounds 150 --n_seeds 10 --out_dir results/paired_energy_aware
    python run_matched_target_study.py --data_dir PATH \\
        --policy random --paired_availability --target 0.30 \\
        --max_rounds 150 --n_seeds 10 --out_dir results/paired_random

    # Balanced accuracy reached within a long round budget, no early stopping
    python run_matched_target_study.py --data_dir PATH \\
        --probe_only --max_rounds 400 --n_seeds 1

Depends on models.py, fl_harness.py, engine.py and data_mitbih.py, a local
copy of the MIT-BIH Arrhythmia Database (see data_mitbih.py), and the packages
in requirements.txt.

WEIGHT-GRID ABLATION (--alpha/--beta/--gamma)
----------------------------------------------
Overrides the weights of the energy-aware score
s_i = alpha*U_data + beta*U_sys - gamma*E_marginal (default 0.4/0.3/0.3) for
one run; only meaningful with --policy energy_aware. Each (alpha, beta, gamma)
triple is run into its own --out_dir under otherwise identical settings.

STRESS TEST (--stress_test)
---------------------------
Applies three conditions on top of the baseline protocol:
  1. Class restriction: each client keeps only its --keep_top_n_classes most
     frequent local AAMI classes (default 3).
  2. Tier-dependent availability: low/medium/high-tier devices are available
     with per-round probability 0.06/0.10/0.14, averaging the baseline 0.10.
  3. Correlated dropout: availability follows a two-state Markov chain with
     lag-1 autocorrelation --dropout_rho (default 0.5); each tier's long-run
     availability equals its stated rate.
The three conditions are applied together as one configuration.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from fl_harness import (
    ClientState, Candidate, POLICIES, TIER_ENERGY_MULT, TRAIN_FLOP_MULTIPLIER,
    local_train, fedavg, count_forward_flops, model_bytes,
    topk_sparsify, sparse_payload_bytes,
)
from models import ResNet1D
from engine import Params, account_all
from data_mitbih import load_mitbih

N_CLASSES = 5
N_LEADS = 2


def balanced_accuracy(model, state, x_eval, y_eval, n_classes,
                       device=torch.device("cpu")):
    """
    Mean per-class recall, computed on WHATEVER (x_eval, y_eval) is passed
    in. This function does no subsampling of its own -- the caller fixes
    the evaluation set once and reuses it across every call within a run,
    so the only thing that can change balanced accuracy between calls is
    the model itself, not sampling noise in which rare classes happen to
    land in a given draw.

    Classes absent from x_eval/y_eval altogether (not just from one draw
    -- genuinely absent from the fixed evaluation set) are excluded from
    the mean, not scored as 0, since that reflects a gap in the evaluation
    pool itself, not a model failure on that class.
    """
    rep = evaluate_full(model, state, x_eval, y_eval, n_classes, device=device)
    return rep["macro_recall"], rep["raw_acc"], rep["n_classes_present"]


def evaluate_full(model, state, x_eval, y_eval, n_classes,
                  device=torch.device("cpu")):
    """
    Single forward pass over the fixed evaluation set, returning everything
    the per-round record needs: macro recall (balanced accuracy), raw
    accuracy, per-class recall (None for a class absent from the set), the
    confusion matrix (rows = true class, columns = predicted class), and the
    macro F1 over classes present in the set. balanced_accuracy() above is a
    thin wrapper, so the stopping rule uses exactly the same numbers.
    """
    model.to(device)
    model.load_state_dict(state)
    model.eval()
    batch = 64
    preds = []
    with torch.no_grad():
        for i in range(0, x_eval.shape[0], batch):
            xb = x_eval[i:i + batch].to(device, non_blocking=True)
            preds.append(model(xb).argmax(1).cpu())
    pred = torch.cat(preds)
    recalls, per_class, f1s = [], [], []
    confusion = [[0] * n_classes for _ in range(n_classes)]
    for tcls in range(n_classes):
        row = torch.bincount(pred[y_eval == tcls], minlength=n_classes).tolist()
        confusion[tcls] = row
    for c in range(n_classes):
        mask = (y_eval == c)
        if mask.sum() == 0:
            per_class.append(None)
            continue
        rec = (pred[mask] == c).float().mean().item()
        recalls.append(rec)
        per_class.append(rec)
        tp = confusion[c][c]
        fp = sum(confusion[o][c] for o in range(n_classes)) - tp
        fn = sum(confusion[c]) - tp
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    raw_acc = (pred == y_eval).float().mean().item()
    return {"macro_recall": macro_recall, "raw_acc": raw_acc,
            "n_classes_present": len(recalls), "per_class_recall": per_class,
            "confusion": confusion,
            "macro_f1": (sum(f1s) / len(f1s)) if f1s else 0.0}


def amplify_non_iid(clients, keep_top_n_classes=3, seed=0):
    """
    Restricts each client to its `keep_top_n_classes` most frequent local
    AAMI classes. Used only under --stress_test; the baseline trains on the
    unmodified per-recording split. `seed` only breaks ties among equally
    frequent classes, so the same partition is used for every training seed.
    """
    rng = np.random.RandomState(seed)
    out = []
    for c in clients:
        y = c["y"]
        counts = np.bincount(y, minlength=5).astype(np.float64)
        counts += rng.uniform(0, 1e-6, size=counts.shape)  # break ties deterministically
        keep = set(np.argsort(-counts)[:keep_top_n_classes].tolist())
        mask = np.isin(c["y"], list(keep))
        if mask.sum() == 0:
            out.append(c)
            continue
        out.append({**c, "X": c["X"][mask], "y": c["y"][mask]})
    return out


def draw_stress_availability(cstates, rng, avail_state, tier_rates, rho):
    """
    Two-state Markov chain per client, replacing the baseline's
    independent per-round Bernoulli(participation_p) draw under
    --stress_test. For a client whose tier gives it long-run availability
    rate p:
        P(available this round | available last round)   = p + rho*(1-p)
        P(available this round | unavailable last round) = p*(1-rho)
    This is the standard two-state parameterization whose stationary
    probability is exactly p and whose lag-1 autocorrelation is exactly
    rho, for any rho in [0, 1) -- so raising rho introduces bursty,
    correlated dropout without changing any client's long-run average
    availability. Low-tier devices also get a lower p than high-tier
    (see tier_rates), so slower devices are both less often selected
    (existing system_utility scoring) AND less often even available --
    the baseline protocol only models the first effect.
    `avail_state` is the caller-owned {cid: bool} map from the previous
    round; a client with no entry yet (first round) is drawn from the
    marginal rate p directly.
    """
    available = []
    new_state = {}
    for c in cstates:
        p = tier_rates[c.device_tier]
        was_avail = avail_state.get(c.cid)
        if was_avail is None:
            is_avail = rng.random() < p
        elif was_avail:
            is_avail = rng.random() < (p + rho * (1 - p))
        else:
            is_avail = rng.random() < (p * (1 - rho))
        new_state[c.cid] = is_avail
        if is_avail:
            available.append(c)
    return available, new_state


def split_heldout(clients, n_heldout, seed):
    """
    Recording-disjoint split. Recordings are shuffled with `seed` and taken
    until at least `n_heldout` are held out; 201 and 202 come from the same
    subject and are never separated. Returns (train_clients, heldout_clients).
    """
    groups = {}
    for i, c in enumerate(clients):
        key = "201+202" if c["patient_id"] in ("201", "202") else c["patient_id"]
        groups.setdefault(key, []).append(i)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    held_idx = []
    for k in keys:
        if len(held_idx) >= n_heldout:
            break
        held_idx.extend(groups[k])
    held = set(held_idx)
    train = [c for i, c in enumerate(clients) if i not in held]
    heldout = [c for i, c in enumerate(clients) if i in held]
    return train, heldout


DEFAULT_STRESS_TIER_RATES = {"low": 0.06, "medium": 0.10, "high": 0.14}  # mean 0.10,
                                                                          # matching the
                                                                          # baseline's flat
                                                                          # participation_p


def run_to_target(clients, model_fn, input_shape, n_classes,
                   target, max_rounds, K, mu, policy_name, seed,
                   epochs=2, lr=0.001, batch_size=32, sparsify_frac=0.3,
                   participation_p=0.1, device=None,
                   eval_every=1, progress_every=25, progress_tag="",
                   stability_window=5, policy_kwargs=None,
                   stress_test=False, dropout_rho=0.5, tier_rates=None,
                   paired_availability=False, eval_clients=None):
    """
    Client sampling, selection policy, local FedProx training,
    sparsification, and aggregation are the same FMCL protocol described
    in Section 4. The only thing specific to this script is the stopping
    rule: evaluation uses balanced_accuracy() every eval_every rounds
    (default every round, so the stopping point is not overshot), and the
    loop stops once balanced accuracy has held at or above target for
    stability_window consecutive evaluations.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if stress_test and tier_rates is None:
        tier_rates = DEFAULT_STRESS_TIER_RATES
    # Default: one rng drives both the availability draws and any randomness
    # inside the selection policy. With paired_availability=True availability
    # has its own stream, identical across policies for a given seed.
    rng = random.Random(seed if not paired_availability else seed + 1_000_003)
    avail_rng = random.Random(seed) if paired_availability else rng
    torch.manual_seed(seed)
    avail_state = {}  # {cid: bool}, carried round to round; only used under stress_test
    t_start = time.time()
    model = model_fn()
    global_state = {k: v.clone() for k, v in model.state_dict().items()}

    n_clients = len(clients)
    tiers = [["low", "medium", "high"][i % 3] for i in range(n_clients)]

    Xs, ys = [], []
    for c in clients:
        X = c["X"] if isinstance(c["X"], torch.Tensor) else torch.from_numpy(c["X"])
        y = c["y"] if isinstance(c["y"], torch.Tensor) else torch.from_numpy(c["y"])
        Xs.append(X); ys.append(y)
    if eval_clients is None:
        # Default: the evaluation sample is drawn from the pooled training data.
        x_all = torch.cat(Xs[:min(len(Xs), 200)])
        y_all = torch.cat(ys[:min(len(ys), 200)])
    else:
        # Recording-disjoint evaluation: the sample is drawn only from held-out
        # recordings that no client trains on.
        x_all = torch.cat([torch.from_numpy(c["X"]) if not isinstance(c["X"], torch.Tensor) else c["X"] for c in eval_clients])
        y_all = torch.cat([torch.from_numpy(c["y"]) if not isinstance(c["y"], torch.Tensor) else c["y"] for c in eval_clients])

    # Fixed evaluation set, drawn once per run and stratified by class so
    # that a rare AAMI class (particularly Q) is always represented in the
    # balanced-accuracy denominator. Up to per_class_cap examples of each
    # class are included; the remaining budget is filled with a random draw
    # from what is left.
    n_pool = x_all.shape[0]
    max_eval = 2000
    if n_pool > max_eval:
        strat_rng = random.Random(seed)
        per_class_cap = max(50, max_eval // (2 * n_classes))
        stratified_idx = []
        for c in range(n_classes):
            class_idx = (y_all == c).nonzero(as_tuple=True)[0].tolist()
            strat_rng.shuffle(class_idx)
            stratified_idx.extend(class_idx[:per_class_cap])
        stratified_set = set(stratified_idx)
        remaining_budget = max_eval - len(stratified_idx)
        if remaining_budget > 0:
            remaining_pool = [i for i in range(n_pool) if i not in stratified_set]
            strat_rng.shuffle(remaining_pool)
            stratified_idx.extend(remaining_pool[:remaining_budget])
        eval_idx = torch.tensor(stratified_idx, dtype=torch.long)
        x_eval, y_eval = x_all[eval_idx], y_all[eval_idx]
        classes_in_eval = sorted(set(y_eval.tolist()))
        if len(classes_in_eval) < n_classes:
            missing = sorted(set(range(n_classes)) - set(classes_in_eval))
            print(f"  WARNING [{progress_tag}]: classes {missing} have ZERO examples "
                  f"in the entire data pool, not just the eval draw -- balanced "
                  f"accuracy for this run is out of {len(classes_in_eval)} classes, "
                  f"not {n_classes}.", flush=True)
    else:
        x_eval, y_eval = x_all, y_all

    cstates = [ClientState(cid=i, x=Xs[i], y=ys[i], device_tier=tiers[i])
               for i in range(n_clients)]

    dense_bytes = model_bytes(global_state)
    history = []
    fresh_evals = []  # only genuinely fresh evaluations, never carried-forward
                       # repeats -- kept separate from history so the stability
                       # window below counts real confirmations even if
                       # eval_every is ever set above 1
    total_client_rounds = 0
    total_flops = 0
    total_bytes = 0
    reached_round = None
    reached_balanced_acc = None

    for t in range(max_rounds):
        if stress_test:
            available, avail_state = draw_stress_availability(
                cstates, avail_rng, avail_state, tier_rates, dropout_rho)
        else:
            available = [c for c in cstates if avail_rng.random() < participation_p]
        if len(available) < 2:
            available = cstates[:2]

        cands = []
        for c in available:
            with torch.no_grad():
                model.load_state_dict(global_state)
                model.to(device)
                if c.x.shape[0] == 0:
                    du = 0.0
                else:
                    xb = c.x[:min(8, c.x.shape[0])].to(device, non_blocking=True)
                    out = model(xb)
                    yb = c.y[:min(8, c.y.shape[0])].to(device, non_blocking=True)
                    du = float(F.cross_entropy(out, yb))
            su = 1.0 / TIER_ENERGY_MULT[c.device_tier]
            me = TIER_ENERGY_MULT[c.device_tier]
            lab = int(c.y[0].item()) if c.y.shape[0] > 0 else -1
            cands.append(Candidate(cid=c.cid, data_utility=du, system_utility=su,
                                   marginal_energy_J=me, grid_intensity=0.156,
                                   label=lab))

        k = min(K, len(cands))
        selected_ids = set(POLICIES[policy_name](cands, k, rng, **(policy_kwargs or {})))
        selected = [c for c in available if c.cid in selected_ids]

        states, weights = [], []
        for c in selected:
            if c.x.shape[0] == 0:
                continue
            new_state, _, fwd_fl, _ = local_train(
                model, c, global_state, mu=mu, epochs=epochs,
                lr=lr, batch_size=batch_size, device=device)
            _, nnz = topk_sparsify(new_state, sparsify_frac)
            states.append(new_state)
            weights.append(int(c.x.shape[0]))
            n_batches = max(1, c.x.shape[0] // batch_size) * epochs
            total_flops += fwd_fl * TRAIN_FLOP_MULTIPLIER * n_batches
            total_bytes += sparse_payload_bytes(nnz) * 2
            total_client_rounds += 1

        if states:
            global_state = fedavg(states, weights)

        if t % eval_every == 0 or t == max_rounds - 1:
            rep = evaluate_full(model, global_state, x_eval, y_eval, n_classes,
                                device=device)
            bal_acc, raw_acc = rep["macro_recall"], rep["raw_acc"]
            n_classes_seen = rep["n_classes_present"]
            fresh_evals.append({"round": t, "balanced_acc": bal_acc})
        else:
            rep = {k: history[-1][k] for k in ("per_class_recall", "confusion", "macro_f1")}
            bal_acc, raw_acc = history[-1]["balanced_acc"], history[-1]["raw_acc"]
            n_classes_seen = history[-1]["n_classes_seen"]

        energy_mult = (sum(TIER_ENERGY_MULT[c.device_tier] for c in selected) / len(selected)
                       if selected else 0.0)
        history.append({"round": t, "balanced_acc": bal_acc, "raw_acc": raw_acc,
                        "n_classes_seen": n_classes_seen, "n_selected": len(selected),
                        "n_candidates": len(cands),
                        "per_class_recall": rep["per_class_recall"],
                        "confusion": rep["confusion"], "macro_f1": rep["macro_f1"],
                        "selected_ids": sorted(c.cid for c in selected),
                        "selected_energy_mult_mean": energy_mult})

        if progress_every and (t % progress_every == 0 or t == max_rounds - 1):
            elapsed = time.time() - t_start
            rate = (t + 1) / elapsed if elapsed > 0 else 0.0
            eta = (max_rounds - t - 1) / rate if rate > 0 else float("nan")
            print(f"  [{progress_tag}] round {t:4d}/{max_rounds}  "
                  f"balanced_acc={bal_acc:.3f}  raw_acc={raw_acc:.3f}  "
                  f"elapsed={elapsed:6.0f}s  ~{rate:.2f} rounds/s  "
                  f"ETA={eta:6.0f}s", flush=True)

        if reached_round is None and bal_acc >= target:
            recent = [e["balanced_acc"] for e in fresh_evals[-stability_window:]]
            if len(recent) >= stability_window and all(v >= target for v in recent):
                reached_round = fresh_evals[-stability_window]["round"]  # the round the
                                                                           # qualifying
                                                                           # streak actually
                                                                           # started
                reached_balanced_acc = min(recent)  # the weakest round in the qualifying
                                                       # streak, not the peak, so this
                                                       # number is not flattered by
                                                       # whichever round happened to be
                                                       # highest within the streak
                break

    fpc = total_flops / max(total_client_rounds, 1)
    upload_mb = (total_bytes / max(total_client_rounds, 1) / 2) / 1e6
    rounds_used = (reached_round + 1) if reached_round is not None else max_rounds

    p_eng = Params(
        rounds=rounds_used,
        clients_per_round=K,
        flops_per_client_round=fpc,
        upload_MB=upload_mb,
        download_MB=upload_mb,
    )
    lifecycle_to_target = {
        arch: {k: float(v) for k, v in r.items()}
        for arch, r in account_all(p_eng).items()
    } if reached_round is not None else None

    n_rounds_run = len(history)
    binding = sum(1 for h in history if h["n_candidates"] > K)
    return {
        "eval_protocol": ("heldout_recordings" if eval_clients is not None
                          else "pooled_training_sample"),
        "heldout_ids": ([c["patient_id"] for c in eval_clients]
                        if eval_clients is not None else []),
        "selection_binding_fraction": binding / max(n_rounds_run, 1),
        "mean_selected_energy_mult": (sum(h["selected_energy_mult_mean"] for h in history)
                                       / max(n_rounds_run, 1)),
        "history": history,
        "reached_target": reached_round is not None,
        "rounds_to_target": (reached_round + 1) if reached_round is not None else None,
        "balanced_acc_at_target": reached_balanced_acc,
        "final_balanced_acc": history[-1]["balanced_acc"],
        "final_raw_acc": history[-1]["raw_acc"],
        "lifecycle_to_target_kg_CO2e": (
            {k: v["GWP100_kg_CO2e"] for k, v in lifecycle_to_target.items()}
            if lifecycle_to_target else None
        ),
        "counted": {"flops_per_client_round": fpc, "upload_MB": upload_mb,
                     "total_client_rounds": total_client_rounds,
                     "dense_model_MB": dense_bytes / 1e6},
        "config": {"target": target, "max_rounds": max_rounds, "K": K, "mu": mu,
                   "policy": policy_name, "seed": seed, "epochs": epochs, "lr": lr,
                   "policy_kwargs": policy_kwargs, "stress_test": stress_test,
                   "dropout_rho": dropout_rho if stress_test else None,
                   "tier_rates": tier_rates if stress_test else None,
                   "paired_availability": paired_availability,
                   "participation_p": participation_p},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True,
                        help="Path to the local MIT-BIH Arrhythmia Database "
                             "download (see data_mitbih.py).")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--n_seeds", type=int, default=10,
                        help="Seeds 0..n_seeds-1. Seed i starts from the same model "
                             "initialisation under every --policy; with "
                             "--paired_availability it also sees the same availability "
                             "draws. Table 5 uses 10.")
    parser.add_argument("--max_rounds", type=int, default=150,
                        help="Round ceiling per run. Table 5 uses 150.")
    parser.add_argument("--K", type=int, default=6,
                        help="Clients selected per round. Table 5 used 6.")
    parser.add_argument("--target", type=float, default=0.30,
                        help="Balanced-accuracy stopping target. Table 5 uses 0.30, "
                             "which lies above the 0.20 chance level of the "
                             "five-class problem.")
    parser.add_argument("--probe_only", action="store_true",
                        help="Ignore --target: run the full --max_rounds with no "
                             "early stopping and report the final balanced "
                             "accuracy of each seed.")
    parser.add_argument("--policy", default="energy_aware",
                        choices=["random", "oort_lite", "energy_aware",
                                 "carbon_aware", "class_balanced"],
                        help="Table 5 compares energy_aware against random.")
    parser.add_argument("--mu", type=float, default=0.01,
                        help="FedProx proximal term weight. Table 5 used 0.01.")
    parser.add_argument("--stability_window", type=int, default=5,
                        help="Balanced accuracy must hold at or above --target "
                             "for this many CONSECUTIVE evaluations before the "
                             "run is considered to have reached it. Table 5 "
                             "used 5.")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=2,
                        help="Local epochs per client per round. Table 5 used 2.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=None,
                        help="Weight-grid ablation: override the energy-aware "
                             "score's alpha (data-utility weight). Only used "
                             "with --policy energy_aware. Must be given "
                             "together with --beta and --gamma.")
    parser.add_argument("--beta", type=float, default=None,
                        help="Weight-grid ablation: override the energy-aware "
                             "score's beta (system-utility weight).")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Weight-grid ablation: override the energy-aware "
                             "score's gamma (marginal-energy weight).")
    parser.add_argument("--paired_availability", action="store_true",
                        help="Draw availability from its own RNG stream, identical across "
                             "policies for a given seed. Default off: availability and "
                             "policy randomness share one stream.")
    parser.add_argument("--stress_test", action="store_true",
                        help="Run the heterogeneity/availability stress "
                             "condition (class restriction, tier-dependent "
                             "availability, correlated dropout). See module "
                             "docstring.")
    parser.add_argument("--keep_top_n_classes", type=int, default=3,
                        help="--stress_test only: how many of each client's "
                             "most frequent local classes are kept. Default 3 "
                             "leaves all five AAMI classes present somewhere in "
                             "the fleet; 2 removes class Q fleet-wide.")
    parser.add_argument("--dropout_rho", type=float, default=0.5,
                        help="--stress_test only: lag-1 autocorrelation of "
                             "the per-client availability chain.")
    parser.add_argument("--participation_p", type=float, default=0.1,
                        help="Per-round availability probability of each client "
                             "(baseline protocol). Table 5 uses 0.1, about 4.4 "
                             "candidates per round for K = 6; larger values make "
                             "selection bind in more rounds.")
    parser.add_argument("--heldout_records", type=int, default=0,
                        help="Hold out this many recordings (recording-disjoint "
                             "evaluation). They are not trained on, and the "
                             "evaluation sample is drawn from them. Records 201 "
                             "and 202 (one subject) are always held out together. "
                             "Default 0 keeps the pooled-sample evaluation of "
                             "Table 5.")
    parser.add_argument("--split_seed", type=int, default=0,
                        help="Seed for choosing the held-out recordings; the same "
                             "recordings are held out for every training seed "
                             "and policy.")
    args = parser.parse_args()

    if args.target != 0.30 and not args.probe_only:
        print(f"  --target={args.target:.2f} differs from the 0.30 used for "
              f"Table 5; results are not directly comparable.")

    policy_kwargs = None
    given = [v is not None for v in (args.alpha, args.beta, args.gamma)]
    if any(given):
        if not all(given):
            raise SystemExit("--alpha/--beta/--gamma must be given together.")
        if args.policy != "energy_aware":
            raise SystemExit("--alpha/--beta/--gamma only apply to --policy energy_aware.")
        total = args.alpha + args.beta + args.gamma
        if abs(total - 1.0) > 1e-6:
            print(f"  alpha+beta+gamma={total:.4f}, not 1.0; the weights are "
                  f"used unnormalized.")
        policy_kwargs = {"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma}

    meta, clients = load_mitbih(args.data_dir)
    print(f"Clients: {len(clients)}  Beats: {meta['n_beats']}  "
          f"Window length: {meta['window_len']}  Classes: {N_CLASSES}")

    eval_clients = None
    if args.heldout_records > 0:
        clients, eval_clients = split_heldout(clients, args.heldout_records, args.split_seed)
        print(f"  Held-out evaluation: {len(eval_clients)} recordings "
              f"({', '.join(c['patient_id'] for c in eval_clients)}); "
              f"{len(clients)} training clients")

    if args.stress_test:
        clients = amplify_non_iid(clients, keep_top_n_classes=args.keep_top_n_classes)
        print(f"  Stress test: top {args.keep_top_n_classes} classes per client, "
              f"tier-scaled availability {DEFAULT_STRESS_TIER_RATES}, "
              f"dropout_rho={args.dropout_rho}")

    os.makedirs(args.out_dir, exist_ok=True)

    for model_name in ("resnet",):
        model_fn = lambda: ResNet1D(n_leads=N_LEADS, n_classes=N_CLASSES)
        n_params = sum(p.numel() for p in model_fn().parameters())
        print(f"\n=== Model: {model_name} ({n_params} parameters) ===")
        for seed in range(args.n_seeds):
            fname = f"mitbih_{model_name}_seed{seed:02d}.json"
            fpath = os.path.join(args.out_dir, fname)
            if os.path.exists(fpath):
                print(f"  seed {seed:2d}  [already complete, skipping]")
                continue
            t0 = time.time()
            result = run_to_target(
                clients, model_fn, (N_LEADS, meta["window_len"]), N_CLASSES,
                target=(1.01 if args.probe_only else args.target),  # >1.0 =>
                                                                      # never
                                                                      # triggers
                                                                      # early
                                                                      # stop in
                                                                      # probe
                                                                      # mode
                max_rounds=args.max_rounds, K=args.K, mu=args.mu,
                policy_name=args.policy, seed=seed, epochs=args.epochs,
                lr=args.lr, batch_size=args.batch_size,
                stability_window=args.stability_window,
                policy_kwargs=policy_kwargs, stress_test=args.stress_test,
                dropout_rho=args.dropout_rho,
                paired_availability=args.paired_availability,
                participation_p=args.participation_p, eval_clients=eval_clients,
                progress_tag=f"mitbih/{model_name}/seed{seed}")
            elapsed = time.time() - t0
            if args.probe_only:
                print(f"  seed {seed:2d}  final balanced_acc="
                      f"{result['final_balanced_acc']:.3f}  "
                      f"raw_acc={result['final_raw_acc']:.3f}  ({elapsed:.0f}s)")
            elif result["reached_target"]:
                fmcl_kg = result["lifecycle_to_target_kg_CO2e"]["FMCL"]
                print(f"  seed {seed:2d}  reached target at round "
                      f"{result['rounds_to_target']}  FMCL={fmcl_kg:.4f} kg  "
                      f"({elapsed:.0f}s)")
            else:
                print(f"  seed {seed:2d}  DID NOT reach target={args.target:.2f} "
                      f"within {args.max_rounds} rounds (final balanced_acc="
                      f"{result['final_balanced_acc']:.3f})  ({elapsed:.0f}s)")
            result["domain"] = "mitbih"
            result["model_name"] = model_name
            with open(fpath, "w") as f:
                json.dump(result, f)

    print(f"\nDone. Results saved to {args.out_dir}")
    print("Per-seed fields: rounds_to_target, balanced_acc_at_target, "
          "lifecycle_to_target_kg_CO2e['FMCL'].")
