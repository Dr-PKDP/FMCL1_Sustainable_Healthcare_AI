"""
sensitivity_analysis.py -- one-way sweeps and a joint Monte Carlo analysis over
the eight parameters of Section 5.3, built on the Table 4 / Fig. 4 baseline of
table3_accounting.py.

Parameters
    device wear            epsilon     Proposition 6
    communication mode     Wi-Fi/LTE   Section 5.3.2, device fleet composition
    grid carbon intensity  I           Section 5.3.2
    dropout                q           Section 5.3.2, participation churn
    fleet size             M (edge)    Table 4
    convergence penalty    kappa       Proposition 5
    training frequency     f           Proposition 4
    edge idle attribution  rho         Section 5.3.2

One-way sweeps vary one parameter over its range with all others at the Table 4
baseline and report whether FMCL has the lower footprint than EDGE and than
CLOUD. The joint Monte Carlo draws all eight parameters simultaneously,
independently and uniformly over their ranges, and reports the fraction of
draws in which FMCL has the lower footprint against each alternative.

Ranges
    epsilon (wear fraction): 0 to 0.05
    comm_multiplier: 1x (Wi-Fi) to 10x (LTE), the five-to-tenfold penalty of
        Section 5.3.2
    grid_intensity_kg_per_kwh: 0.05 to 0.70, from low-carbon to coal-heavy grids
    dropout_q: 0 to 0.7
    edge_fleet_M: 1 to 100 servers, bracketing the 1-server (Fig. 5) and
        50-server (Fig. 4) scenarios
    kappa: 1 to 50, within the tolerance stated in Remark 6
    training_frequency_f: 10 to 20,000 rounds/year, bracketing the Proposition 4
        crossover f* of about 19,462 rounds/year
    rho: 0 to 1

All other inputs (campaign duration, round length, device and cloud payload and
compute figures, embodied-carbon point estimates) stay at the Table 4 baseline.
"""
import csv
import random
from dataclasses import dataclass, replace
from typing import Optional

from table3_accounting import (
    ROUNDS, CAMPAIGN_YEARS, ROUND_SECONDS, GRID_INTENSITY_KG_PER_KWH,
    EDGE_SERVERS, EDGE_SRV_EMBODIED_KG, EDGE_SRV_LIFETIME_YEARS,
    EDGE_IDLE_W, EDGE_ACTIVE_W, EDGE_RHO,
    FMCL_COMPUTE_WH_PER_ROUND, FMCL_COMM_WH_PER_ROUND,
    CLOUD_K, CLOUD_PAYLOAD_MB_PER_DEVICE_ROUND, CLOUD_WAN_KWH_PER_GB,
    CLOUD_COMPUTE_WH_PER_ROUND, CLOUD_PUE, CLOUD_EMBODIED_KG_PER_YEAR,
    PHONE_EMBODIED_KG_CO2E, FMCL_POPULATION, SECONDS_PER_YEAR,
)


@dataclass
class Scenario:
    """A full parameter set for one comparison. Defaults are the Table 4 /
    Fig. 4 baseline; a sweep or Monte Carlo draw overrides one or more
    fields and leaves the rest at baseline."""
    rounds: int = ROUNDS
    campaign_years: float = CAMPAIGN_YEARS
    round_seconds: float = ROUND_SECONDS
    grid_intensity: float = GRID_INTENSITY_KG_PER_KWH

    edge_servers: float = EDGE_SERVERS
    edge_srv_embodied_kg: float = EDGE_SRV_EMBODIED_KG
    edge_srv_lifetime_years: float = EDGE_SRV_LIFETIME_YEARS
    edge_idle_W: float = EDGE_IDLE_W
    edge_active_W: float = EDGE_ACTIVE_W
    rho: float = EDGE_RHO

    fmcl_compute_Wh_per_round: float = FMCL_COMPUTE_WH_PER_ROUND
    comm_multiplier: float = 1.0           # 1x = Wi-Fi baseline (0.55 Wh/round)
    dropout_q: float = 0.0                 # fraction of invited clients that
                                            # never complete
    kappa: float = 1.0                     # convergence penalty: FMCL's
                                            # round count is kappa times
                                            # edge's at equal model quality
    wear_epsilon: float = 0.0              # participation-attributable
                                            # wear fraction, charged once
                                            # per campaign (Proposition 6)

    cloud_K: float = CLOUD_K
    cloud_payload_MB: float = CLOUD_PAYLOAD_MB_PER_DEVICE_ROUND
    cloud_wan_kWh_per_GB: float = CLOUD_WAN_KWH_PER_GB
    cloud_compute_Wh_per_round: float = CLOUD_COMPUTE_WH_PER_ROUND
    cloud_PUE: float = CLOUD_PUE
    cloud_embodied_kg_per_year: float = CLOUD_EMBODIED_KG_PER_YEAR

    # Optional architecture-specific grid intensities (kg CO2e/kWh). When left
    # as None, the common `grid_intensity` applies to every architecture.
    grid_device: Optional[float] = None
    grid_edge: Optional[float] = None
    grid_cloud: Optional[float] = None

    # Pre-existing edge: the servers already exist and run whether or not the
    # campaign is scheduled, so the campaign is charged only the power it adds
    # above idle during active rounds, with no induced embodied carbon.
    edge_preexisting: bool = False


def _g(s: Scenario, which: str) -> float:
    v = getattr(s, "grid_" + which)
    return s.grid_intensity if v is None else v


def carbon_fmcl_kg(s: Scenario) -> float:
    """C^FMCL under Proposition 6's wear term and Proposition 5's
    convergence penalty: rounds actually run = kappa * s.rounds, dropout
    inflates the number of clients invited (and therefore communication
    and compute energy) by 1/(1 - q)."""
    rounds_run = s.kappa * s.rounds
    dropout_inflation = 1.0 / max(1e-6, (1.0 - s.dropout_q))
    comm_wh = FMCL_COMM_WH_PER_ROUND * s.comm_multiplier
    per_round_wh = (s.fmcl_compute_Wh_per_round + comm_wh) * dropout_inflation
    op_kwh = per_round_wh * rounds_run / 1000.0
    op_co2 = op_kwh * _g(s, 'device')
    wear_co2 = s.wear_epsilon * FMCL_POPULATION * PHONE_EMBODIED_KG_CO2E
    return op_co2 + wear_co2


def carbon_edge_kg(s: Scenario) -> float:
    """C^edge under Eq. 8's active/idle split and rho."""
    campaign_s = s.campaign_years * SECONDS_PER_YEAR
    active_s = s.rounds * s.round_seconds   # edge runs the SAME round count
                                             # as the campaign calls for;
                                             # kappa only inflates FMCL's
                                             # round count (the convergence
                                             # penalty is specifically
                                             # FMCL's, per Proposition 5)
    if s.edge_preexisting:
        incr_kwh = (s.edge_servers * max(0.0, s.edge_active_W - s.edge_idle_W)
                    * active_s) / 3.6e6
        return incr_kwh * _g(s, 'edge')
    idle_s = max(0.0, campaign_s - active_s)
    active_kwh = (s.edge_servers * s.edge_active_W * active_s) / 3.6e6
    idle_kwh = (s.edge_servers * s.edge_idle_W * idle_s * s.rho) / 3.6e6
    op_co2 = (active_kwh + idle_kwh) * _g(s, 'edge')
    emb_co2 = s.edge_servers * s.edge_srv_embodied_kg * min(
        1.0, s.campaign_years / s.edge_srv_lifetime_years)
    return op_co2 + emb_co2


def carbon_cloud_kg(s: Scenario) -> float:
    data_gb_per_round = (s.cloud_payload_MB * s.cloud_K) / 1000.0
    transfer_kwh = data_gb_per_round * s.rounds * s.cloud_wan_kWh_per_GB
    compute_kwh = (s.cloud_compute_Wh_per_round * s.cloud_PUE * s.rounds) / 1000.0
    op_co2 = (transfer_kwh + compute_kwh) * _g(s, 'cloud')
    emb_co2 = s.cloud_embodied_kg_per_year * s.campaign_years
    return op_co2 + emb_co2


def fmcl_wins(s: Scenario):
    c_fmcl, c_edge, c_cloud = carbon_fmcl_kg(s), carbon_edge_kg(s), carbon_cloud_kg(s)
    return c_fmcl < c_edge, c_fmcl < c_cloud, c_fmcl, c_edge, c_cloud


# --------------------------------------------------------------------------
# One-way sweeps
# --------------------------------------------------------------------------
SWEEPS = {
    "wear_epsilon":        [0.0, 0.005, 0.01, 0.02, 0.03, 0.05],
    "comm_multiplier":     [1.0, 2.5, 5.0, 7.5, 10.0],
    "grid_intensity":      [0.05, 0.15, 0.25, 0.40, 0.55, 0.70],
    "dropout_q":           [0.0, 0.2, 0.4, 0.5, 0.6, 0.7],
    "edge_servers":        [1, 5, 10, 25, 50, 100],
    "kappa":               [1, 2, 5, 10, 25, 50],
    "rho":                 [0.0, 0.25, 0.5, 0.75, 1.0],
}
# Training frequency is swept differently: it changes BOTH rounds and
# campaign_years is held fixed at 1 year, so f (rounds/year) = s.rounds.
FREQUENCY_SWEEP = [10, 100, 500, 1500, 5000, 19462, 20000]


def run_one_way_sweeps(out_csv: str = "sensitivity_one_way.csv") -> None:
    rows = []
    baseline = Scenario()
    for param, values in SWEEPS.items():
        for v in values:
            s = replace(baseline, **{param: v})
            fmcl_v_edge, fmcl_v_cloud, c_fmcl, c_edge, c_cloud = fmcl_wins(s)
            rows.append({
                "parameter": param, "value": v,
                "C_FMCL_kg": round(c_fmcl, 4), "C_EDGE_kg": round(c_edge, 2),
                "C_CLOUD_kg": round(c_cloud, 4),
                "FMCL_beats_EDGE": fmcl_v_edge, "FMCL_beats_CLOUD": fmcl_v_cloud,
            })
    for f in FREQUENCY_SWEEP:
        s = replace(baseline, rounds=f)
        fmcl_v_edge, fmcl_v_cloud, c_fmcl, c_edge, c_cloud = fmcl_wins(s)
        rows.append({
            "parameter": "training_frequency_f", "value": f,
            "C_FMCL_kg": round(c_fmcl, 4), "C_EDGE_kg": round(c_edge, 2),
            "C_CLOUD_kg": round(c_cloud, 4),
            "FMCL_beats_EDGE": fmcl_v_edge, "FMCL_beats_CLOUD": fmcl_v_cloud,
        })
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


# --------------------------------------------------------------------------
# Joint Monte Carlo (uniform over each parameter's stated range,
# independent draws -- a Latin Hypercube would reduce sampling variance
# at the same draw count but is not required for a stable estimate at
# n_draws=20000; swap in scipy.stats.qmc.LatinHypercube here if finer
# variance control is wanted)
# --------------------------------------------------------------------------
RANGES = {
    "wear_epsilon": (0.0, 0.05),
    "comm_multiplier": (1.0, 10.0),
    "grid_intensity": (0.05, 0.70),
    "dropout_q": (0.0, 0.7),
    "edge_servers": (1, 100),
    "kappa": (1, 50),
    "rho": (0.0, 1.0),
    "rounds": (10, 20000),   # training frequency, rounds/year at D=1 year
}


def monte_carlo(n_draws: int = 20000, seed: int = 0,
                asymmetric_grid: bool = False, edge_preexisting: bool = False):
    """asymmetric_grid=True draws the device, edge, and cloud grid intensities
    independently over the grid range instead of one common value.
    edge_preexisting=True charges the edge only its incremental active power."""
    rng = random.Random(seed)
    n_beats_edge = n_beats_cloud = 0
    for _ in range(n_draws):
        kw = {}
        for param, (lo, hi) in RANGES.items():
            v = rng.uniform(lo, hi)
            kw[param] = int(round(v)) if param in ("edge_servers", "rounds") else v
        if asymmetric_grid:
            lo, hi = RANGES["grid_intensity"]
            kw["grid_device"] = rng.uniform(lo, hi)
            kw["grid_edge"] = rng.uniform(lo, hi)
            kw["grid_cloud"] = rng.uniform(lo, hi)
        kw["edge_preexisting"] = edge_preexisting
        s = replace(Scenario(), **kw)
        fmcl_v_edge, fmcl_v_cloud, _, _, _ = fmcl_wins(s)
        n_beats_edge += int(fmcl_v_edge)
        n_beats_cloud += int(fmcl_v_cloud)
    return {
        "n_draws": n_draws,
        "P(FMCL beats EDGE)": n_beats_edge / n_draws,
        "P(FMCL beats CLOUD)": n_beats_cloud / n_draws,
    }


if __name__ == "__main__":
    print("=== One-way sweeps (each parameter varied alone, rest at Table 4 baseline) ===")
    rows = run_one_way_sweeps()
    cur_param = None
    for r in rows:
        if r["parameter"] != cur_param:
            cur_param = r["parameter"]
            print(f"\n-- {cur_param} --")
        print(f"  {r['value']!s:>10}  C_FMCL={r['C_FMCL_kg']:>10}  "
              f"C_EDGE={r['C_EDGE_kg']:>10}  C_CLOUD={r['C_CLOUD_kg']:>10}  "
              f"beats_EDGE={r['FMCL_beats_EDGE']!s:<5} beats_CLOUD={r['FMCL_beats_CLOUD']!s:<5}")
    print("\n(full table written to sensitivity_one_way.csv)")

    print("\n=== Joint Monte Carlo (all eight parameters varied simultaneously) ===")
    result = monte_carlo(n_draws=20000)
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n=== Joint Monte Carlo, device/edge/cloud grid intensities drawn independently ===")
    for k, v in monte_carlo(n_draws=20000, asymmetric_grid=True).items():
        print(f"  {k}: {v}")

    print("\n=== Pre-existing edge: campaign charged incremental active power only ===")
    base = replace(Scenario(), edge_preexisting=True)
    for M in (1, 2, 5, 10, 50, 100):
        s = replace(base, edge_servers=M)
        _, _, cf, ce, cc = fmcl_wins(s)
        print(f"  M={M:>3}  C_FMCL={cf:.3f}  C_EDGE(incremental)={ce:.3f}  FMCL<EDGE: {cf < ce}")
    for k, v in monte_carlo(n_draws=20000, edge_preexisting=True).items():
        print(f"  {k}: {v}")
    for k, v in monte_carlo(n_draws=20000, edge_preexisting=True, asymmetric_grid=True).items():
        print(f"  (asymmetric grids) {k}: {v}")
