"""Carbon accounting: grid carbon-intensity traces, temporal and geographic
workload shifting, and embodied carbon.

Operational carbon of an energy demand ``E`` (kWh) drawn at grid carbon
intensity ``I`` (g CO2e/kWh) is ``C = E I``.  Two mitigation levers are
modelled, and their *constrained* and *idealised* variants are reported
separately because the manuscript's headline "89 % reduction" is an idealised
upper bound, not an achieved system result (CONTRACT honesty rule).

* **Temporal shifting** - defer the delay-tolerant part of the workload inside
  the host region to the lowest-intensity hour reachable within
  ``temporal_shift_window_h``.  Only ``temporal_shift_max_fraction`` of the
  offered load is delay tolerant; URLLC and the headline real-time task are
  not.
* **Geographic shifting** - migrate part of the *cloud* workload to the
  lowest-intensity region available.  Only ``geo_shift_max_fraction`` may be
  migrated once data-sovereignty and latency constraints are respected.

The idealised bound migrates 100 % of the cloud workload to the daily minimum
of the cleanest region.  It is reported, and plotted, as an upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import Config, DEFAULT, JOULES_PER_KWH, SECONDS_PER_DAY

# ---------------------------------------------------------------------------
# 24-hour grid carbon-intensity traces, g CO2e/kWh, hourly resolution.
# Representative annual-average diurnal shapes for four bidding zones.  They
# are stylised class values, not a measured dataset: see CITATIONS NEEDED.
# ---------------------------------------------------------------------------
REGION_TRACES: Dict[str, np.ndarray] = {
    "US-CAISO": np.array([
        268, 262, 258, 256, 258, 272, 288, 296, 258, 205, 162, 138,
        128, 132, 152, 196, 252, 312, 348, 352, 336, 316, 296, 280.]),
    "DE-LU": np.array([
        352, 340, 330, 326, 330, 348, 386, 412, 396, 358, 318, 296,
        286, 292, 312, 348, 396, 442, 468, 472, 452, 424, 396, 372.]),
    "FR": np.array([
        58, 55, 53, 52, 53, 57, 64, 70, 68, 63, 58, 55,
        53, 54, 57, 63, 71, 80, 86, 88, 83, 76, 69, 63.]),
    "SE-3": np.array([
        31, 30, 29, 28, 29, 31, 35, 38, 37, 34, 31, 29,
        27, 28, 30, 33, 37, 41, 44, 45, 42, 39, 36, 33.]),
}

HOST_REGION = "US-CAISO"      # region hosting the cloud tier and the users
CLEAN_REGION = "SE-3"         # lowest-intensity region reachable by migration


def intensity_g_per_kwh(region: str, t_s) -> np.ndarray:
    """Carbon intensity at wall-clock time ``t_s`` (seconds), interpolated."""
    trace = REGION_TRACES[region]
    hour = (np.asarray(t_s, dtype=float) % SECONDS_PER_DAY) / 3600.0
    x = np.concatenate([np.arange(24.0), [24.0]])
    y = np.concatenate([trace, trace[:1]])
    return np.interp(hour, x, y)


def energy_to_kwh(energy_j) -> np.ndarray:
    return np.asarray(energy_j) / JOULES_PER_KWH


def carbon_g_per_1000(energy_j_per_inference, intensity) -> np.ndarray:
    """g CO2e per 1000 inferences."""
    return 1000.0 * energy_to_kwh(energy_j_per_inference) * np.asarray(intensity)


# ---------------------------------------------------------------------------
# Shifting strategies
# ---------------------------------------------------------------------------
def _min_within_window(region: str, t_s: np.ndarray, window_h: float
                       ) -> np.ndarray:
    """Lowest intensity reachable from ``t_s`` within ``window_h`` hours."""
    offsets = np.arange(0.0, window_h + 1e-9, 0.25) * 3600.0
    grid = intensity_g_per_kwh(region, t_s[:, None] + offsets[None, :])
    return grid.min(axis=1)


@dataclass
class CarbonResult:
    strategy: str
    g_per_1000: float
    reduction_pct: float
    idealised: bool
    caveat: str


def carbon_strategies(cfg: Config, t_s: np.ndarray, energy_by_tier_j: Dict[str, float]
                      ) -> List[CarbonResult]:
    """Carbon per 1000 inferences under each shifting strategy.

    ``energy_by_tier_j`` gives the per-inference energy attributable to each
    tier, i.e. ``alpha_i * E_i`` summed over the mix, so the three values add
    up to the system's per-inference energy.
    """
    e_cloud = energy_by_tier_j["cloud"]
    e_local = energy_by_tier_j["edge"] + energy_by_tier_j["device"]
    e_tot = e_cloud + e_local

    i_host = intensity_g_per_kwh(HOST_REGION, t_s)
    i_host_mean = float(i_host.mean())
    i_shift = _min_within_window(HOST_REGION, t_s, cfg.temporal_shift_window_h)
    i_shift_mean = float(i_shift.mean())
    i_clean = intensity_g_per_kwh(CLEAN_REGION, t_s)
    i_clean_mean = float(i_clean.mean())
    i_clean_min = float(REGION_TRACES[CLEAN_REGION].min())

    f_t = cfg.temporal_shift_max_fraction
    f_g = cfg.geo_shift_max_fraction

    base = carbon_g_per_1000(e_tot, i_host_mean)

    # temporal: a fraction f_t of *all* tiers is deferred inside the region
    temporal = carbon_g_per_1000(
        e_tot * (1.0 - f_t), i_host_mean) + carbon_g_per_1000(
        e_tot * f_t, i_shift_mean)

    # geographic: a fraction f_g of the *cloud* energy is served abroad
    geographic = (carbon_g_per_1000(e_cloud * (1.0 - f_g), i_host_mean)
                  + carbon_g_per_1000(e_cloud * f_g, i_clean_mean)
                  + carbon_g_per_1000(e_local, i_host_mean))

    # both: migrate first, then defer what remains deferrable in each region
    cloud_home = e_cloud * (1.0 - f_g)
    cloud_away = e_cloud * f_g
    both = (carbon_g_per_1000(cloud_home * (1 - f_t), i_host_mean)
            + carbon_g_per_1000(cloud_home * f_t, i_shift_mean)
            + carbon_g_per_1000(cloud_away * (1 - f_t), i_clean_mean)
            + carbon_g_per_1000(cloud_away * f_t,
                                float(_min_within_window(
                                    CLEAN_REGION, t_s,
                                    cfg.temporal_shift_window_h).mean()))
            + carbon_g_per_1000(e_local * (1 - f_t), i_host_mean)
            + carbon_g_per_1000(e_local * f_t, i_shift_mean))

    # idealised upper bound: everything runs in the cleanest region at its
    # daily minimum intensity, with no migration, egress or sovereignty limit
    ideal = carbon_g_per_1000(e_tot, i_clean_min)

    def red(x):
        return 100.0 * (base - x) / base

    return [
        CarbonResult("No shifting", float(base), 0.0, False,
                     f"served in {HOST_REGION} at its diurnal intensity"),
        CarbonResult("Temporal shifting", float(temporal), float(red(temporal)),
                     False,
                     f"only {f_t:.0%} of the load is delay tolerant; window "
                     f"{cfg.temporal_shift_window_h:.0f} h"),
        CarbonResult("Geographic shifting", float(geographic),
                     float(red(geographic)), False,
                     f"only {f_g:.0%} of the cloud load may be migrated "
                     f"({HOST_REGION} -> {CLEAN_REGION})"),
        CarbonResult("Temporal + geographic", float(both), float(red(both)),
                     False, "levers are multiplicative, not additive"),
        CarbonResult("Idealised full migration", float(ideal),
                     float(red(ideal)), True,
                     "UPPER BOUND: 100 % of the workload migrated to "
                     f"{CLEAN_REGION} at its daily minimum "
                     f"({i_clean_min:.0f} g/kWh); ignores migration latency, "
                     "egress energy and data-sovereignty limits"),
    ]


# ---------------------------------------------------------------------------
# Embodied carbon
# ---------------------------------------------------------------------------
def embodied_g_per_1000(cfg: Config, alpha: Dict[str, float],
                        t_inf_s: Dict[str, float]) -> Dict[str, float]:
    """Amortised manufacturing carbon, g CO2e per 1000 inferences.

    Hardware embodied carbon is amortised over the service life and charged in
    proportion to the time each request occupies the hardware.
    """
    life_s = cfg.hardware_life_years * 365.0 * 24.0 * 3600.0
    dev_life_s = cfg.device_life_years * 365.0 * 24.0 * 3600.0
    cloud_node_kg = cfg.embodied_kg_per_a100 * cfg.a100_per_cloud_node
    out = {
        "cloud": alpha["cloud"] * 1000.0 * 1e3
                 * cloud_node_kg / life_s * t_inf_s["cloud"]
                 / cfg.cloud_target_utilisation,
        "edge": alpha["edge"] * 1000.0 * 1e3
                * cfg.embodied_kg_per_edge_board / life_s * t_inf_s["edge"]
                / cfg.edge_target_utilisation,
        "device": alpha["device"] * 1000.0 * 1e3
                  * cfg.embodied_kg_per_device_npu / dev_life_s
                  * t_inf_s["device"],
    }
    out["total"] = sum(out.values())
    return out
