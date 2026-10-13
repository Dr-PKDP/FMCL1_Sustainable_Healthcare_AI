"""
engine.py -- lifecycle carbon/impact accounting engine.

Computes multi-indicator lifecycle impact for three architectures over a
training campaign:

    FMCL   : training on already-deployed consumer devices (reuse constraint)
    EDGE   : training on M provisioned servers at a clinical site
    CLOUD  : centralised training, data transferred to a hyperscale facility

Design choices:

 1. Work is counted and power is calibrated. FLOPs and bytes are inputs,
    counted exactly by an experiment as fl_harness.py does, and are converted
    with coefficients drawn from published measurements.

 2. Device compute energy uses the two-term model E = alpha*W + beta*t, with
    alpha carried as a wide-prior uncertain input because it is not
    identifiable from published data alone.

 3. Communication energy is transfer PLUS one radio tail per burst. The tail
    dominates for small payloads; a per-bit constant is not used.

 4. Embodied carbon is attributed marginally on reused devices and in full on
    provisioned hardware. The edge arm carries bounds, not a point estimate.

 5. Indicators beyond GWP are carried because the reuse advantage differs by
    indicator.

All coefficients are carried as the Params dataclass's own field defaults
below, each with its provenance noted in a comment at the point of
definition. Nothing is hard-coded elsewhere that is not either a physical
constant or an explicit modelling choice captured here.
"""
from dataclasses import dataclass, field
from typing import Dict

# --------------------------------------------------------------------------
# Indicators carried through the engine
# --------------------------------------------------------------------------
INDICATORS = ("GWP100_kg_CO2e", "ADPe_kg_Sb_eq", "ADPf_MJ",
              "EP_kg_PO4_eq", "water_use_kg", "ewaste_kg")


@dataclass
class Params:
    """Every quantity the engine needs. Monte Carlo perturbs this object."""

    # ---- campaign definition (counted in the experiment) ----
    rounds: int = 200                    # T, training rounds in the campaign
    campaign_years: float = 1.0          # D, calendar duration
    clients_per_round: int = 100         # K
    flops_per_client_round: float = 3.0e11   # W, exact from the model+batch
    upload_MB: float = 1.0               # compressed update size
    download_MB: float = 1.0             # global model size
    round_seconds: float = 90.0          # t, local training wall clock

    # ---- device energy (two-term model) ----
    alpha_J_per_FLOP: float = 0.0105e-9  # WIDE PRIOR - not identifiable
    beta_W: float = 6.01                 # time-proportional draw, well identified
    device_idle_W: float = 0.0           # marginal framing: idle excluded (A1)

    # ---- communication ----
    link: str = "WiFi"                   # or "LTE"
    wifi_transfer_W: float = 0.760
    wifi_rate_Mbps: float = 20.0
    wifi_tail_W: float = 0.435
    wifi_tail_s: float = 0.2
    lte_transfer_W: float = 1.520
    lte_rate_Mbps: float = 16.0
    lte_tail_W: float = 1.216
    lte_tail_s: float = 10.0             # NO-DRX upper bound; sweep this

    # ---- edge arm (Dell R740-class) ----
    edge_servers: int = 1                # M
    edge_peak_W: float = 510.0
    edge_idle_W: float = 201.0           # 39.4% of peak power
    edge_active_W: float = 369.0         # 50%-load draw during training
    edge_idle_attribution: float = 0.5   # rho: share of idle charged to the task
    edge_server_lifetime_years: float = 4.0
    edge_PUE: float = 1.55               # on-premises / university cluster
    edge_throughput_FLOPs: float = 5.0e11    # 2x Xeon Gold 6152, realised FP32
    edge_embodied_kg_CO2e: float = 4288.0    # bound: 910 storage-light .. 4288
    # Scales the whole server_prod inventory (see _server_inventory). It is the
    # single lever for edge embodied impact: server_prod holds the reference
    # shape and this holds the magnitude.

    # ---- cloud arm ----
    cloud_compute_W: float = 262.0       # V100+CPU class
    cloud_throughput_FLOPs: float = 5.0e12   # V100, realised not peak
    cloud_PUE: float = 1.11
    cloud_transfer_kWh_per_GB: float = 0.06
    cloud_data_GB_per_round: float = 5.0
    cloud_embodied_kg_CO2e: float = 5.0
    # Amortised across tenants, so small per task - but NOT zero. Setting it to
    # exactly zero while FMCL carries a non-zero wear term would hand cloud the
    # comparison by construction.

    # ---- carbon intensity (kg CO2e / kWh) ----
    # UK grid intensity, 14-day winter series (2025-01-01 to 2025-01-15, UK
    # Grid ESO API, 673 half-hourly readings): mean 0.1556, min 0.035, max
    # 0.253. A winter mean is likely above the annual mean, and no full-year
    # series was used.
    intensity_device: float = 0.1556
    intensity_edge: float = 0.1556
    intensity_cloud: float = 0.054       # low-carbon hyperscale siting

    # ---- device wear (derived, not assumed) ----
    # Wear from cycling is derived from campaign energy and rated battery
    # cycle life. Only the thermal pathway is a free parameter, because no
    # study measures battery health under sustained federated training.
    battery_Wh_usable: float = 12.43     # 4200 mAh x 3.7 V x 0.8
    battery_rated_cycles: float = 850.0  # Fairphone 3 ageing, 3/3
    thermal_wear_multiplier: float = 1.0 # 1 = cycling only; >1 adds thermal ageing
    wear_fraction_override: float = -1.0 # <0 means "derive it"; >=0 forces a value
    device_population: int = 1000        # N devices in the fleet
    phone_embodied_kg_CO2e: float = 32.7

    # ---- per-device production-phase inventory (Fairphone 5 LCA) ----
    phone_prod: Dict[str, float] = field(default_factory=lambda: {
        "GWP100_kg_CO2e": 32.7, "ADPe_kg_Sb_eq": 1.25e-3, "ADPf_MJ": 385.0,
        "EP_kg_PO4_eq": 1.19e-2, "water_use_kg": 43800.0, "ewaste_kg": 0.212})
    # ---- per-server manufacturing inventory (Dell R740 LCA) ----
    server_prod: Dict[str, float] = field(default_factory=lambda: {
        # GWP, ADPf, EP: Dell R740 LCA (thinkstep 2019), manufacturing phase.
        # ADPe: Boavizta API, rack server archetype, embedded 0.14 kgSbeq
        #   (range 0.098-0.213). Boavizta's "adp" combines mineral and fossil
        #   components and is not a pure ADPe, unlike the Fairphone CML figure.
        #   It is scaled here to the Dell embodied magnitude, so the ADPe row
        #   is approximate and not definitionally identical.
        "GWP100_kg_CO2e": 4288.0, "ADPe_kg_Sb_eq": 0.14, "ADPf_MJ": 48070.0,
        "EP_kg_PO4_eq": 1.209, "water_use_kg": 0.0, "ewaste_kg": 22.36})
    # ---- operational intensity per kWh, non-carbon indicators ----
    op_per_kWh: Dict[str, float] = field(default_factory=lambda: {
        "ADPe_kg_Sb_eq": 0.0, "ADPf_MJ": 9.0, "EP_kg_PO4_eq": 1.5e-4,
        "water_use_kg": 1.8, "ewaste_kg": 0.0})


# --------------------------------------------------------------------------
# Energy sub-models
# --------------------------------------------------------------------------
def device_compute_energy_J(p: Params) -> float:
    """Two-term marginal compute energy for one client-round."""
    return p.alpha_J_per_FLOP * p.flops_per_client_round + p.beta_W * p.round_seconds


def device_comm_energy_J(p: Params) -> float:
    """Transfer plus ONE radio tail per burst. Upload and download separately."""
    if p.link == "WiFi":
        pw, rate, tw, ts = p.wifi_transfer_W, p.wifi_rate_Mbps, p.wifi_tail_W, p.wifi_tail_s
    elif p.link == "LTE":
        pw, rate, tw, ts = p.lte_transfer_W, p.lte_rate_Mbps, p.lte_tail_W, p.lte_tail_s
    else:
        raise ValueError(f"unknown link type {p.link!r}")
    bits = (p.upload_MB + p.download_MB) * 8e6
    transfer = pw * (bits / (rate * 1e6))
    return transfer + tw * ts        # one tail per round, not per direction


def J_to_kWh(j: float) -> float:
    return j / 3.6e6


# --------------------------------------------------------------------------
# Work conservation
# --------------------------------------------------------------------------
def total_flops_per_round(p: Params) -> float:
    """The work one round represents, independent of who performs it."""
    return p.clients_per_round * p.flops_per_client_round


def device_throughput_FLOPs(p: Params) -> float:
    return p.flops_per_client_round / p.round_seconds


def edge_round_seconds(p: Params) -> float:
    """Time for the edge fleet to perform the SAME work as one FMCL round."""
    return total_flops_per_round(p) / (p.edge_servers * p.edge_throughput_FLOPs)


def cloud_round_seconds(p: Params) -> float:
    return total_flops_per_round(p) / p.cloud_throughput_FLOPs


# --------------------------------------------------------------------------
# Architecture accounting
# --------------------------------------------------------------------------
def _apply_operational(result: Dict[str, float], kwh: float, intensity: float,
                       p: Params) -> None:
    result["GWP100_kg_CO2e"] += kwh * intensity
    for k, v in p.op_per_kWh.items():
        result[k] += kwh * v


def _check_feasible(p: Params, arm: str, round_s: float) -> None:
    campaign_s = p.campaign_years * 365 * 24 * 3600
    active_s = p.rounds * round_s
    if active_s > campaign_s:
        raise OversubscribedCampaign(
            f"{arm}: {p.rounds} rounds x {round_s:.2f}s = {active_s:,.0f}s exceeds "
            f"the {campaign_s:,.0f}s campaign. Max for this arm: "
            f"{campaign_s/round_s:,.0f} rounds.")


def rounds_per_client(p: Params) -> float:
    """Expected participations per device over the campaign."""
    if p.device_population <= 0:
        return 0.0
    return p.rounds * p.clients_per_round / p.device_population


def derived_wear_fraction(p: Params) -> float:
    """
    Fraction of a device's remaining service life consumed by participation.

    Derived, not assumed: training draws energy from a battery with
    a finite rated cycle life, so

        eq. cycles = campaign energy / usable battery energy
        eps        = eq. cycles / rated cycle life  x  thermal multiplier

    The thermal multiplier is the ONLY free term, and it is free because no
    study measures battery health under sustained federated training against
    a matched control. At 1.0 it represents the cycling pathway alone.
    """
    if p.wear_fraction_override >= 0.0:
        return p.wear_fraction_override
    per_round_J = device_compute_energy_J(p) + device_comm_energy_J(p)
    campaign_Wh = per_round_J * rounds_per_client(p) / 3600.0
    eq_cycles = campaign_Wh / p.battery_Wh_usable
    return (eq_cycles / p.battery_rated_cycles) * p.thermal_wear_multiplier


def account_fmcl(p: Params) -> Dict[str, float]:
    _check_feasible(p, "FMCL", p.round_seconds)
    r = {k: 0.0 for k in INDICATORS}
    per_client = device_compute_energy_J(p) + device_comm_energy_J(p)
    total_J = per_client * p.clients_per_round * p.rounds
    kwh = J_to_kWh(total_J)
    _apply_operational(r, kwh, p.intensity_device, p)
    # Embodied: marginal only, derived from the same energy above,
    # so wear cannot be varied independently of the work that causes it.
    eps = derived_wear_fraction(p)
    for k in INDICATORS:
        r[k] += eps * p.device_population * p.phone_prod.get(k, 0.0)
    r["_operational_kWh"] = kwh
    r["_embodied_kg_CO2e"] = eps * p.device_population * p.phone_prod["GWP100_kg_CO2e"]
    r["_wear_fraction"] = eps
    return r


def max_feasible_rounds(p: Params) -> float:
    """
    Rounds that physically fit in the campaign, back-to-back with no gap.
    Set by the SLOWEST architecture, since all three must complete the same
    campaign for the comparison to mean anything.
    """
    campaign_s = p.campaign_years * 365 * 24 * 3600
    slowest = max(p.round_seconds, edge_round_seconds(p), cloud_round_seconds(p))
    return campaign_s / slowest


class OversubscribedCampaign(ValueError):
    """Raised when more rounds are requested than fit in the campaign duration."""


def _server_inventory(p: Params) -> Dict[str, float]:
    """
    Scale the reference server inventory to the sampled embodied magnitude.
    Keeps the relative shape across indicators fixed (only one full
    multi-indicator server LCA exists) while letting magnitude be uncertain.
    """
    ref = p.server_prod["GWP100_kg_CO2e"]
    if ref <= 0:
        return dict(p.server_prod)
    scale = p.edge_embodied_kg_CO2e / ref
    return {k: v * scale for k, v in p.server_prod.items()}


def account_edge(p: Params) -> Dict[str, float]:
    r = {k: 0.0 for k in INDICATORS}
    _check_feasible(p, "EDGE", edge_round_seconds(p))
    campaign_s = p.campaign_years * 365 * 24 * 3600
    active_s = p.rounds * edge_round_seconds(p)
    idle_s = campaign_s - active_s
    active_J = p.edge_servers * p.edge_active_W * active_s
    idle_J = p.edge_servers * p.edge_idle_W * idle_s * p.edge_idle_attribution
    kwh = J_to_kWh(active_J + idle_J) * p.edge_PUE
    _apply_operational(r, kwh, p.intensity_edge, p)
    # Embodied: charged in FULL, amortised over calendar life
    amort = p.campaign_years / p.edge_server_lifetime_years
    inv = _server_inventory(p)
    for k in INDICATORS:
        r[k] += p.edge_servers * inv.get(k, 0.0) * amort
    r["_operational_kWh"] = kwh
    r["_embodied_kg_CO2e"] = p.edge_servers * inv["GWP100_kg_CO2e"] * amort
    return r


def account_cloud(p: Params) -> Dict[str, float]:
    _check_feasible(p, "CLOUD", cloud_round_seconds(p))
    r = {k: 0.0 for k in INDICATORS}
    compute_J = p.cloud_compute_W * cloud_round_seconds(p) * p.rounds
    compute_kwh = J_to_kWh(compute_J) * p.cloud_PUE
    transfer_kwh = p.cloud_data_GB_per_round * p.rounds * p.cloud_transfer_kWh_per_GB
    kwh = compute_kwh + transfer_kwh
    _apply_operational(r, kwh, p.intensity_cloud, p)
    for k in INDICATORS:
        r[k] += p.cloud_embodied_kg_CO2e if k == "GWP100_kg_CO2e" else 0.0
    r["_operational_kWh"] = kwh
    r["_embodied_kg_CO2e"] = p.cloud_embodied_kg_CO2e
    return r


ARCHITECTURES = {"FMCL": account_fmcl, "EDGE": account_edge, "CLOUD": account_cloud}


def account_all(p: Params) -> Dict[str, Dict[str, float]]:
    return {name: fn(p) for name, fn in ARCHITECTURES.items()}


# --------------------------------------------------------------------------
# Crossover horizon
# --------------------------------------------------------------------------
def crossover_rounds(p: Params, competitor: str = "EDGE",
                     indicator: str = "GWP100_kg_CO2e"):
    """
    Total impact is fixed + per-round. Returns the round count at which FMCL
    stops being the lower-impact choice, or None if it never does.

    Computed numerically from the engine rather than from a closed form, so
    that it stays correct if the accounting changes.
    """
    def at(T):
        q = _copy_with(p, rounds=max(T, 1))
        return (ARCHITECTURES["FMCL"](q)[indicator],
                ARCHITECTURES[competitor](q)[indicator])

    f1, c1 = at(1)
    f2, c2 = at(1001)
    fmcl_slope = (f2 - f1) / 1000.0
    comp_slope = (c2 - c1) / 1000.0
    fmcl_fixed = f1 - fmcl_slope
    comp_fixed = c1 - comp_slope
    if fmcl_slope <= comp_slope:
        return None                       # FMCL wins at every horizon
    T = (comp_fixed - fmcl_fixed) / (fmcl_slope - comp_slope)
    if T <= 0:
        return 0.0
    # A crossover beyond the physically feasible horizon is not a crossover.
    # Report it as unreachable rather than as a number, which would invite
    # quoting a threshold no deployment could ever reach.
    if T > max_feasible_rounds(p):
        return None
    return T


def _copy_with(p: Params, **kw) -> Params:
    import copy
    q = copy.deepcopy(p)
    for k, v in kw.items():
        setattr(q, k, v)
    return q
