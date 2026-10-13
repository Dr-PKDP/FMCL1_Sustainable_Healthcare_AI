"""
table3_accounting.py -- four-architecture lifecycle-carbon comparison (cloud,
dedicated edge, FMCL, cloud-coordinated FMCL) under three embodied-carbon
allocation conventions (marginal, calendar-time, shared-infrastructure),
computed for the reference scenario of Table 4 and Fig. 4 (Section 5.2).

The scenario states per-round energy figures directly (for example 20 Wh per
round for cloud compute and about 5 Wh per round for FMCL device compute), so
this module reproduces that arithmetic; engine.py, which derives energy from
counted FLOPs and bytes, is used only for the Table 5 experiment.

Parameters from Table 4 and Sections 5.2.1-5.2.2
    Campaign: 200 rounds/year, one year, 90 s per round
    Grid carbon intensity: 400 g CO2e/kWh
    EDGE: 50 servers, 350 kg CO2e per server, 5-year life, 80 W idle and
        200 W active
    FMCL: about 5 Wh/round compute (100 devices, 5 local epochs, about 2 W,
        90 s) and 0.1-1 Wh/round Wi-Fi communication (midpoint 0.55 used);
        marginal embodied carbon exactly zero, as in Fig. 4
    CLOUD: 50 MB per device per round, 0.06 kWh/GB WAN transmission, 20 Wh per
        round compute at PUE 1.15, 2 kg CO2e/year of amortized hardware

Additional assumptions needed for the non-marginal conventions
    PHONE_EMBODIED_KG_CO2E = 35.98: production-phase GWP100 of the Fairphone 2
        life-cycle assessment (Fraunhofer IZM, 2016; reference [73]).
    FMCL_POPULATION = 100: a fixed cohort of 100 devices participating in
        every round. A larger pool would lower the calendar-time and
        shared-infrastructure FMCL figures, so this reading is the less
        favorable to FMCL.
    Calendar-time FMCL: embodied carbon charged over a 2.5-year smartphone
        service life.
    Shared-infrastructure FMCL: FMCL is one of 50 equally weighted roles of a
        smartphone; the edge fleet is shared by 4 campaigns.
    CLOUD: the 2 kg CO2e/year figure is already a per-task amortized value and
        is used unchanged under the calendar-time and shared-infrastructure
        conventions.
    MARGINAL_PROVISIONED_FLOOR = 0.005: share of full embodied carbon charged
        to provisioned hardware (edge servers, the cloud coordinator, cloud
        hardware) under the marginal convention.
    Cloud-coordinated FMCL: one aggregation server with 350 kg CO2e embodied
        carbon (the edge-server figure), a 5-year life, an 80 W draw for 5 s
        per round, shared among 20 tenants under the shared-infrastructure
        convention.
"""
from dataclasses import dataclass

SECONDS_PER_YEAR = 365 * 24 * 3600
CONVENTIONS = ("marginal", "calendar_time", "shared_infrastructure")
ARCH_NAMES = ("CLOUD", "EDGE", "FMCL", "FMCL_CLOUD_COORD")

# Table 4 / Fig. 4 scenario
ROUNDS = 200
CAMPAIGN_YEARS = 1.0
ROUND_SECONDS = 90.0
GRID_INTENSITY_KG_PER_KWH = 0.400

EDGE_SERVERS = 50
EDGE_SRV_EMBODIED_KG = 350.0
EDGE_SRV_LIFETIME_YEARS = 5.0
EDGE_IDLE_W = 80.0
EDGE_ACTIVE_W = 200.0
EDGE_RHO = 1.0  # idle-energy attribution factor

FMCL_COMPUTE_WH_PER_ROUND = 5.0
FMCL_COMM_WH_PER_ROUND = 0.55  # midpoint of the stated 0.1-1 Wh/round, Wi-Fi

CLOUD_K = 100
CLOUD_PAYLOAD_MB_PER_DEVICE_ROUND = 50.0
CLOUD_WAN_KWH_PER_GB = 0.06
CLOUD_COMPUTE_WH_PER_ROUND = 20.0
CLOUD_PUE = 1.15
CLOUD_EMBODIED_KG_PER_YEAR = 2.0  # per-task amortized hardware figure

PHONE_EMBODIED_KG_CO2E = 35.98  # Fairphone 2 LCA (Fraunhofer IZM, 2016),
                                # production-phase GWP100 (reference [73])
FMCL_POPULATION = CLOUD_K       # fixed cohort of K = 100 devices

# Share of full embodied carbon charged to provisioned, shared hardware
# under the marginal convention. FMCL's own marginal embodied term is
# exactly zero, as in Fig. 4.
MARGINAL_PROVISIONED_FLOOR = 0.005

# Cloud-coordinated FMCL aggregator
COORD_EMBODIED_KG_CO2E = 350.0   # same per-server figure as the edge fleet
COORD_LIFETIME_YEARS = 5.0       # same as the edge servers
COORD_COMPUTE_W = 80.0           # draw of one lightly loaded server
COORD_SECONDS_PER_ROUND = 5.0    # aggregation is I/O-bound
COORD_N_TENANTS = 20.0           # concurrent deployments sharing the server


def _edge_operational_kwh() -> float:
    campaign_s = CAMPAIGN_YEARS * SECONDS_PER_YEAR
    active_s = ROUNDS * ROUND_SECONDS
    idle_s = campaign_s - active_s
    active_kwh = (EDGE_SERVERS * EDGE_ACTIVE_W * active_s) / 3.6e6
    idle_kwh = (EDGE_SERVERS * EDGE_IDLE_W * idle_s * EDGE_RHO) / 3.6e6
    return active_kwh + idle_kwh


def _fmcl_operational_kwh() -> float:
    return (FMCL_COMPUTE_WH_PER_ROUND + FMCL_COMM_WH_PER_ROUND) * ROUNDS / 1000.0


def _cloud_operational_kwh() -> float:
    data_gb_per_round = (CLOUD_PAYLOAD_MB_PER_DEVICE_ROUND * CLOUD_K) / 1000.0
    transfer_kwh = data_gb_per_round * ROUNDS * CLOUD_WAN_KWH_PER_GB
    compute_kwh = (CLOUD_COMPUTE_WH_PER_ROUND * CLOUD_PUE * ROUNDS) / 1000.0
    return transfer_kwh + compute_kwh


def _coord_operational_kwh() -> float:
    return (COORD_COMPUTE_W * COORD_SECONDS_PER_ROUND * ROUNDS) / 3.6e6


def embodied_edge_kg(convention: str) -> float:
    full = EDGE_SERVERS * EDGE_SRV_EMBODIED_KG
    if convention == "marginal":
        return MARGINAL_PROVISIONED_FLOOR * full
    if convention == "calendar_time":
        # Amortized over calendar time: M * C_srv_emb * D / L_srv
        return full * min(1.0, CAMPAIGN_YEARS / EDGE_SRV_LIFETIME_YEARS)
    if convention == "shared_infrastructure":
        n_tenants = 4.0  # this campaign plus 3 others sharing the fleet
        return full / n_tenants
    raise ValueError(convention)


def embodied_cloud_kg(convention: str) -> float:
    home = CLOUD_EMBODIED_KG_PER_YEAR * CAMPAIGN_YEARS  # already amortized over tenants
    if convention == "shared_infrastructure":
        return home
    if convention == "marginal":
        return home * MARGINAL_PROVISIONED_FLOOR
    if convention == "calendar_time":
        # The 2 kg CO2e/year figure is already a per-task amortized value,
        # so it is used unchanged; this column equals shared_infrastructure.
        return home
    raise ValueError(convention)


def embodied_fmcl_kg(convention: str) -> float:
    if convention == "marginal":
        return 0.0   # exactly as Fig. 4 states
    if convention == "calendar_time":
        # Charged over an average smartphone service life of 2.5 years.
        phone_life_years = 2.5
        share = min(1.0, CAMPAIGN_YEARS / phone_life_years)
        return share * FMCL_POPULATION * PHONE_EMBODIED_KG_CO2E
    if convention == "shared_infrastructure":
        n_other_uses = 49.0  # FMCL is 1 of 50 equally weighted roles
        share = 1.0 / (n_other_uses + 1.0)
        return share * FMCL_POPULATION * PHONE_EMBODIED_KG_CO2E
    raise ValueError(convention)


def embodied_coord_kg(convention: str) -> float:
    if convention == "marginal":
        return MARGINAL_PROVISIONED_FLOOR * COORD_EMBODIED_KG_CO2E
    if convention == "calendar_time":
        return COORD_EMBODIED_KG_CO2E * min(1.0, CAMPAIGN_YEARS / COORD_LIFETIME_YEARS)
    if convention == "shared_infrastructure":
        return COORD_EMBODIED_KG_CO2E / COORD_N_TENANTS
    raise ValueError(convention)


def total_kg_co2e(arch: str, convention: str) -> float:
    if arch == "EDGE":
        return _edge_operational_kwh() * GRID_INTENSITY_KG_PER_KWH + embodied_edge_kg(convention)
    if arch == "CLOUD":
        return _cloud_operational_kwh() * GRID_INTENSITY_KG_PER_KWH + embodied_cloud_kg(convention)
    if arch == "FMCL":
        return _fmcl_operational_kwh() * GRID_INTENSITY_KG_PER_KWH + embodied_fmcl_kg(convention)
    if arch == "FMCL_CLOUD_COORD":
        device_part = _fmcl_operational_kwh() * GRID_INTENSITY_KG_PER_KWH + embodied_fmcl_kg(convention)
        # The coordinator is charged at the scenario's single grid intensity.
        coord_part = _coord_operational_kwh() * GRID_INTENSITY_KG_PER_KWH + embodied_coord_kg(convention)
        return device_part + coord_part
    raise ValueError(arch)


def full_grid():
    return {arch: {conv: total_kg_co2e(arch, conv) for conv in CONVENTIONS}
            for arch in ARCH_NAMES}


if __name__ == "__main__":
    grid = full_grid()
    print(f"{'Architecture':<20}" + "".join(f"{c:>22}" for c in CONVENTIONS))
    for arch in ARCH_NAMES:
        print(f"{arch:<20}" + "".join(f"{grid[arch][c]:>22.2f}" for c in CONVENTIONS))
    print()
    print("Fig. 4 scenario "
          "(FMCL=marginal, EDGE=calendar_time, CLOUD=shared_infrastructure):")
    print(f"  FMCL (marginal)              = {grid['FMCL']['marginal']:.3f} kg CO2e/year")
    print(f"  EDGE (calendar_time)           = {grid['EDGE']['calendar_time']:.1f} kg CO2e/year")
    print(f"  CLOUD (shared_infrastructure)= {grid['CLOUD']['shared_infrastructure']:.2f} kg CO2e/year")
