"""Operational carbon model: geographic and temporal workload shifting.

Operational carbon for an energy demand ``E`` (kWh) at grid carbon intensity
``I`` (gCO2eq/kWh) is simply ``C = E * I``.  Two orthogonal mitigation levers
are modelled:

* **Geographic shifting** - migrate deferrable batch work from CAISO
  (210 gCO2eq/kWh at peak dispatch) to a Norwegian hydro-dominated grid
  (22 gCO2eq/kWh) whenever the local intensity exceeds
  ``I_th = 150 gCO2eq/kWh``.
* **Temporal shifting** - defer deferrable work inside one region from the
  evening fossil-peaker window (~300 gCO2eq/kWh, 18:00-22:00) to the solar
  peak (~198 gCO2eq/kWh, 10:00-14:00).

Honesty rules (CONTRACT):

* The "89 %" geographic figure is an **idealised full-migration upper bound**.
  It assumes every deferrable workload is moved, ignores migration latency,
  egress energy and data-sovereignty constraints.
* The two levers are **not additive**.  The combined case applies the temporal
  intensity ratio *within* the low-carbon region, it does not add the two
  percentages.
* Real-time URLLC inference cannot be deferred at all, so both levers apply
  only to the deferrable fraction of the workload.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

# --- Grid carbon intensities, gCO2eq/kWh (Electricity Maps class values) ----
I_CAISO_PEAK_AVG = 210.0    # CAISO peak-hour dispatch average
I_CAISO_EVENING = 300.0     # CAISO 18:00-22:00 fossil-peaker window
I_CAISO_SOLAR = 198.0       # CAISO 10:00-14:00 solar peak
I_NORWAY = 22.0             # Norwegian hydro-dominated grid
I_THRESHOLD = 150.0         # migration trigger

# Fraction of the MAIMO workload that is deferrable / migratable.
# Real-time URLLC and semantic-CE inference must be served locally and now;
# FL aggregation, pre-training and offline evaluation can be moved.
DEFERRABLE_FRACTION = 0.35


@dataclass
class CarbonCase:
    name: str
    intensity_g_per_kwh: float
    residual_pct_of_baseline: float
    reduction_pct: float
    caveat: str

    def as_dict(self) -> dict:
        return asdict(self)


def reduction_pct(baseline_g: float, shifted_g: float) -> float:
    """Relative carbon reduction (%) of ``shifted_g`` against ``baseline_g``."""
    return 100.0 * (baseline_g - shifted_g) / baseline_g


def geographic_case(baseline: float = I_CAISO_PEAK_AVG,
                    target: float = I_NORWAY) -> Dict[str, float]:
    """Geographic shifting CAISO -> Norway, triggered above ``I_THRESHOLD``."""
    triggered = baseline > I_THRESHOLD
    return {
        "baseline_g_per_kwh": baseline,
        "target_g_per_kwh": target,
        "threshold_g_per_kwh": I_THRESHOLD,
        "migration_triggered": triggered,
        "reduction_pct": reduction_pct(baseline, target) if triggered else 0.0,
    }


def temporal_case(peak: float = I_CAISO_EVENING,
                  trough: float = I_CAISO_SOLAR) -> Dict[str, float]:
    """Temporal shifting inside one region: evening peak -> solar trough."""
    return {
        "peak_g_per_kwh": peak,
        "trough_g_per_kwh": trough,
        "reduction_pct": reduction_pct(peak, trough),
    }


def combined_case(peak: float = I_CAISO_EVENING,
                  trough: float = I_CAISO_SOLAR,
                  target: float = I_NORWAY) -> Dict[str, float]:
    """Combined geographic + temporal shifting (multiplicative, not additive).

    The workload is first migrated to the low-carbon region and then, inside
    that region, deferred to its own low-carbon window.  The temporal lever is
    modelled as an intensity *ratio* ``trough/peak`` that is assumed to carry
    over to the destination grid's own diurnal profile.
    """
    ratio = trough / peak
    effective = target * ratio
    return {
        "peak_g_per_kwh": peak,
        "temporal_ratio": ratio,
        "target_g_per_kwh": target,
        "effective_g_per_kwh": effective,
        "reduction_pct": reduction_pct(peak, effective),
        "additive_would_be_pct": (reduction_pct(peak, target)
                                  + reduction_pct(peak, trough)),
    }


def figure_cases(baseline: float = I_CAISO_EVENING) -> List[CarbonCase]:
    """The four cases plotted in Figure 6, all against one common baseline.

    The common baseline is the California **peak-hour** intensity
    (300 gCO2eq/kWh) so that a single consistent percentage axis can be used;
    the manuscript's 89 % geographic figure instead uses the CAISO peak-hour
    *dispatch average* of 210 gCO2eq/kWh as its reference, which is reported
    separately.
    """
    geo = I_NORWAY
    temp = I_CAISO_SOLAR
    comb = combined_case(peak=baseline)["effective_g_per_kwh"]
    cases = [
        CarbonCase("Baseline (CAISO peak hour)", baseline,
                   100.0, 0.0,
                   "reference; no shifting applied"),
        CarbonCase("Temporal scheduling only", temp,
                   100.0 * temp / baseline, reduction_pct(baseline, temp),
                   "applies to the deferrable workload fraction only; "
                   "URLLC inference cannot be deferred"),
        CarbonCase("Geographic shifting only", geo,
                   100.0 * geo / baseline, reduction_pct(baseline, geo),
                   "idealised full-migration upper bound; ignores migration "
                   "latency, egress energy and data-sovereignty limits"),
        CarbonCase("Combined (geographic + temporal)", comb,
                   100.0 * comb / baseline, reduction_pct(baseline, comb),
                   "multiplicative, not additive; upper bound on an upper bound"),
    ]
    return cases


def operational_carbon_kg(energy_kwh: float, intensity_g_per_kwh: float) -> float:
    """Operational carbon in kg CO2eq."""
    return energy_kwh * intensity_g_per_kwh / 1000.0


def summary(energy_kwh_per_1000_users_hour: float) -> dict:
    """Full carbon summary for the reporting pipeline."""
    geo = geographic_case()
    temp = temporal_case()
    comb = combined_case()
    cases = [c.as_dict() for c in figure_cases()]
    e = energy_kwh_per_1000_users_hour
    return {
        "energy_kwh_per_1000_users_hour": e,
        "deferrable_fraction": DEFERRABLE_FRACTION,
        "geographic": geo,
        "temporal": temp,
        "combined": comb,
        "figure_cases": cases,
        "carbon_kg_caiso_peak_avg": operational_carbon_kg(e, I_CAISO_PEAK_AVG),
        "carbon_kg_norway": operational_carbon_kg(e, I_NORWAY),
        "system_level_reduction_pct_deferrable_only": (
            DEFERRABLE_FRACTION * geo["reduction_pct"]),
        "caveats": [
            "The 89 % geographic reduction is an idealised full-migration "
            "upper bound, not an achieved system result.",
            "Geographic and temporal strategies are separate and "
            "non-additive; the combined figure is multiplicative.",
            "Only the deferrable workload fraction "
            f"({DEFERRABLE_FRACTION:.0%}) can be shifted; applying the "
            "geographic reduction to the whole workload would overstate the "
            "system-level benefit by roughly 1/"
            f"{DEFERRABLE_FRACTION:.2f}.",
            "Embodied carbon of the additional edge hardware is not included "
            "in the operational figures.",
        ],
    }


if __name__ == "__main__":  # pragma: no cover
    from energy import hybrid_energy_j, aggregate_kwh
    e = aggregate_kwh(hybrid_energy_j())
    s = summary(e)
    print(f"MAIMO energy: {e:.4g} kWh per 1000 users per hour")
    print(f"geographic  : {s['geographic']['reduction_pct']:.1f} % "
          f"({I_CAISO_PEAK_AVG:g} -> {I_NORWAY:g} gCO2eq/kWh)")
    print(f"temporal    : {s['temporal']['reduction_pct']:.1f} % "
          f"({I_CAISO_EVENING:g} -> {I_CAISO_SOLAR:g} gCO2eq/kWh)")
    print(f"combined    : {s['combined']['reduction_pct']:.1f} % "
          f"(effective {s['combined']['effective_g_per_kwh']:.2f} gCO2eq/kWh)")
    print(f"  (naive additive sum would be "
          f"{s['combined']['additive_would_be_pct']:.1f} % - wrong)")
    for c in s["figure_cases"]:
        print(f"  {c['name']:36s} {c['intensity_g_per_kwh']:7.2f} g/kWh  "
              f"{c['residual_pct_of_baseline']:6.2f} % residual")
