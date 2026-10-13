"""
fl_harness.py -- the federated-learning mechanics used by the experiment of
Section 5.2.4: FLOP and byte counting, client selection policies, FedProx
local training, and FedAvg aggregation.

Design:
  - FLOP counting is a forward-pass hook-based analytic counter, extended
    to Conv1d/Conv2d/Linear layers, multiplied by TRAIN_FLOP_MULTIPLIER to
    approximate forward+backward cost (the standard fwd+bwd ~= 3x forward
    approximation).
  - Byte counting is exact: it serialises the actual state_dict tensors
    that would be transmitted, optionally after top-k sparsification, and
    counts real (value, index) pairs for the sparse case rather than
    assuming a payload size.
  - Client selection policies are pluggable and score candidates using a
    standard sustainability-score form: s_i = alpha*U_data + beta*U_sys -
    gamma*E_marginal (Section 4.3, Proposition 1).
  - FedProx is implemented as a genuine proximal term in the local loss,
    not approximated.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import torch
# Cap CPU thread usage: PyTorch/OpenMP/MKL default to spawning one thread
# per available core with no limit set, which on a multi-core server can
# make a single training process claim far more CPU than intended even
# though training runs on GPU. GPU compute is unaffected by this cap; it
# only bounds CPU-side orchestration and any operations that fall back to
# CPU.
torch.set_num_threads(4)
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ==========================================================================
# FLOP counting
# ==========================================================================
def count_forward_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    flops = {"n": 0}
    hooks = []

    def conv2d_hook(m, inp, out):
        kh, kw = m.kernel_size
        cin_per_group = m.in_channels // m.groups
        flops["n"] += 2 * out.numel() * cin_per_group * kh * kw

    def conv1d_hook(m, inp, out):
        # out shape: (N, C_out, L_out); kernel_size is an int or 1-tuple
        k = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
        cin_per_group = m.in_channels // m.groups
        flops["n"] += 2 * out.numel() * cin_per_group * k

    def lin_hook(m, inp, out):
        flops["n"] += 2 * out.numel() * m.in_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv2d_hook))
        elif isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv1d_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))

    model.eval()
    # Match the dummy input's device to wherever the model's own parameters
    # currently live, rather than always defaulting to CPU -- this call
    # runs from inside local_train() after the model has been moved to the
    # training device (CPU or GPU).
    dummy_device = next(model.parameters()).device
    with torch.no_grad():
        model(torch.zeros(1, *input_shape, device=dummy_device))
    for h in hooks:
        h.remove()
    return flops["n"]


TRAIN_FLOP_MULTIPLIER = 3.0   # fwd + bwd ~= 3x forward, standard approximation


def model_bytes(state_dict, dtype_bytes: int = 4) -> int:
    """Exact serialized size of a state dict at the given element width."""
    return sum(t.numel() for t in state_dict.values()) * dtype_bytes


def topk_sparsify(state_dict, keep_fraction: float):
    """
    Zero all but the top-k elements by magnitude, per tensor. Returns a new
    state dict and the number of NONZERO elements (what a real sparse
    encoding would transmit), not the dense tensor size.
    """
    if keep_fraction >= 1.0:
        total = sum(t.numel() for t in state_dict.values())
        return state_dict, total
    out = {}
    nnz = 0
    for k, t in state_dict.items():
        flat = t.flatten()
        n_keep = max(1, int(len(flat) * keep_fraction))
        if n_keep >= len(flat):
            out[k] = t
            nnz += t.numel()
            continue
        thresh = flat.abs().kthvalue(len(flat) - n_keep + 1).values
        mask = flat.abs() >= thresh
        sparse_flat = torch.where(mask, flat, torch.zeros_like(flat))
        out[k] = sparse_flat.view_as(t)
        nnz += int(mask.sum().item())
    return out, nnz


def sparse_payload_bytes(nnz: int, dtype_bytes: int = 4, index_bytes: int = 4) -> int:
    """A real sparse encoding needs a value AND an index per nonzero element."""
    return nnz * (dtype_bytes + index_bytes)


# ==========================================================================
# Client
# ==========================================================================
@dataclass
class ClientState:
    cid: int
    x: torch.Tensor
    y: torch.Tensor
    device_tier: str = "medium"          # low/medium/high compute-capability tier
    grid_intensity: float = 0.1556       # kg CO2/kWh, ties to engine default


TIER_ENERGY_MULT = {"low": 1.70, "medium": 1.10, "high": 1.00}   # relative marginal-energy multipliers per device tier (modeling assumption)


def local_train(model: nn.Module, client: ClientState, global_state,
                mu: float, epochs: int, lr: float,
                batch_size: int, device: torch.device = torch.device("cpu"),
                max_batches_per_client: int = None) -> Tuple[Dict, float, int, float]:
    """
    One client's local FedProx training round.
    Returns (updated_state_dict, final_loss, forward_flops_per_batch, 0.0).

    `device` defaults to CPU. Client data (client.x, client.y) stays on
    its original device (typically CPU, as loaded) and only the active
    mini-batch is moved to `device` each iteration.

    `max_batches_per_client`, if set, caps how many mini-batches are drawn
    from a client's local data per epoch. Defaults to None (no cap, full
    local epochs), which is what the Section 5.2.4 experiment uses.
    """
    model.to(device)
    model.load_state_dict(global_state)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    global_params = [p.clone().detach() for p in model.parameters()]

    n = client.x.shape[0]
    if n == 0:
        return {k: v.cpu() for k, v in model.state_dict().items()}, float("nan"), 0, 0.0

    last_loss = float("nan")
    for ep in range(epochs):
        perm = torch.randperm(n)
        if max_batches_per_client is not None:
            perm = perm[:max_batches_per_client * batch_size]
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb = client.x[idx].to(device, non_blocking=True)
            yb = client.y[idx].to(device, non_blocking=True)
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            if mu > 0:
                prox = sum(((p - gp) ** 2).sum()
                          for p, gp in zip(model.parameters(), global_params))
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            opt.step()
            last_loss = float(loss.item())

    fwd_flops = count_forward_flops(model, tuple(client.x.shape[1:]))
    # State dict is returned on CPU so fedavg() and downstream JSON-
    # serialisable metadata never have to reason about device placement.
    return {k: v.cpu() for k, v in model.state_dict().items()}, last_loss, fwd_flops, 0.0


# ==========================================================================
# Selection policies
# ==========================================================================
@dataclass
class Candidate:
    cid: int
    data_utility: float      # e.g. local loss magnitude, higher = more useful
    system_utility: float    # inverse expected completion time
    marginal_energy_J: float
    grid_intensity: float
    label: int = -1           # dominant/only class label of this client's
                               # local data; -1 if unknown. Used only by
                               # policy_class_balanced, which groups
                               # candidates by class.


def policy_random(cands: List[Candidate], k: int, rng) -> List[int]:
    ids = [c.cid for c in cands]
    rng.shuffle(ids)
    return ids[:k]


def policy_oort_lite(cands: List[Candidate], k: int, rng) -> List[int]:
    """Utility + speed, no energy or carbon term."""
    ranked = sorted(cands, key=lambda c: -(c.data_utility + c.system_utility))
    return [c.cid for c in ranked[:k]]


def policy_energy_aware(cands: List[Candidate], k: int, rng,
                        alpha=0.4, beta=0.3, gamma=0.3) -> List[int]:
    """s_i = alpha*U_data + beta*U_sys - gamma*E_marginal (Section 4.3, Proposition 1)."""
    if cands:
        emax = max(c.marginal_energy_J for c in cands) or 1.0
    else:
        emax = 1.0
    def score(c):
        return alpha * c.data_utility + beta * c.system_utility - gamma * (c.marginal_energy_J / emax)
    ranked = sorted(cands, key=lambda c: -score(c))
    return [c.cid for c in ranked[:k]]


def policy_carbon_aware(cands: List[Candidate], k: int, rng) -> List[int]:
    """Prefer low grid-intensity candidates, ignoring utility."""
    ranked = sorted(cands, key=lambda c: c.grid_intensity)
    return [c.cid for c in ranked[:k]]


def policy_class_balanced(cands: List[Candidate], k: int, rng) -> List[int]:
    """
    Selects clients so that all represented classes contribute as close to
    equally as possible, rather than sampling proportionally to their
    natural frequency.

    Grouping is by whatever `label` was populated on each Candidate -- for
    clients with more than one record this is their single dominant label,
    a reasonable simplification given the great majority of clients here
    hold exactly one record.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for c in cands:
        groups[c.label].append(c.cid)
    for g in groups.values():
        rng.shuffle(g)

    classes = list(groups.keys())
    rng.shuffle(classes)
    selected = []
    idx_per_class = {cl: 0 for cl in classes}
    # Round-robin across classes until k is reached or every class's pool
    # is exhausted, so scarce classes are not crowded out early and
    # abundant classes do not dominate the fill order either.
    while len(selected) < k and classes:
        progressed = False
        for cl in list(classes):
            i = idx_per_class[cl]
            if i < len(groups[cl]):
                selected.append(groups[cl][i])
                idx_per_class[cl] += 1
                progressed = True
                if len(selected) >= k:
                    break
        if not progressed:
            break
    return selected[:k]


POLICIES: Dict[str, Callable] = {
    "random": policy_random,
    "oort_lite": policy_oort_lite,
    "energy_aware": policy_energy_aware,
    "carbon_aware": policy_carbon_aware,
    "class_balanced": policy_class_balanced,
}


# ==========================================================================
# Aggregation
# ==========================================================================
def fedavg(states: List[Dict], weights: List[float]) -> Dict:
    total = sum(weights)
    out = {k: torch.zeros_like(v, dtype=torch.float32)
           for k, v in states[0].items()}
    for state, w in zip(states, weights):
        for k in out:
            out[k] += state[k].float() * (w / total)
    # Cast back to original dtype for integer buffers (e.g. BatchNorm num_batches_tracked)
    for k, v in states[0].items():
        if v.dtype in (torch.int32, torch.int64, torch.int16, torch.int8):
            out[k] = out[k].round().to(v.dtype)
    return out
