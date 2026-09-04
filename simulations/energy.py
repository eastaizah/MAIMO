"""Corrected hierarchical per-inference energy model for MAIMO.

The model implemented here is the one locked in ``work/CONTRACT.md``::

    E_i = PUE_i * ( P_i^act * T_i^inf + P_i^idle * (T^slot - n_i * T_i^inf) / n_i )

with ``T^slot = 1 s`` and ``n_i`` the number of inferences a node of tier ``i``
serves during that slot.  The second term amortises the node's idle power over
the inferences actually served in the slot; the original manuscript charged a
full idle slot to every inference, which over-counted idle energy by a factor
``n_i``.

All energies are returned in **joules**.  Aggregates are reported in kWh.
Never report "Wh per inference": for the platforms considered here a single
inference costs O(1-20 J), i.e. O(1-5) mWh, and the manuscript's original
"25.9 Wh/inference" was physically impossible.

IMPORTANT (honesty rule, editor requirement R3/R7): the numbers produced here
are *analytic estimates* obtained by combining vendor thermal-design-power
envelopes with simulated inference times.  They are not wall-plug
measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

JOULES_PER_KWH = 3.6e6
T_SLOT_ENERGY_S = 1.0  # accounting slot for the idle-power amortisation, seconds


@dataclass(frozen=True)
class TierEnergyParams:
    """Per-tier energy-accounting parameters (CONTRACT Table 9 / Table 4)."""

    tier: str
    platform: str
    p_act_w: float          # active compute power, W
    p_idle_w: float         # idle power, W
    t_inf_s: float          # reference inference time, s
    n_per_slot: float       # inferences served by the node during T^slot
    pue: float              # power-usage effectiveness
    extra_j: float = 0.0    # non-compute additive term (e.g. uplink radio), J
    extra_note: str = ""


# ---------------------------------------------------------------------------
# CONTRACT-locked tier parameters.
# ---------------------------------------------------------------------------
CLOUD = TierEnergyParams(
    tier="cloud",
    platform="70 B MoE sharded over 8x NVIDIA A100 80 GB SXM (400 W each, "
             "~70 % utilisation) + host/fabric",
    p_act_w=2550.0,
    p_idle_w=900.0,
    t_inf_s=5.0e-3,
    n_per_slot=195.0,
    pue=1.30,
)

EDGE = TierEnergyParams(
    tier="edge",
    platform="7 B LoRA-adapted model on a 250 W MEC accelerator board "
             "(incl. host and NIC)",
    p_act_w=250.0,
    p_idle_w=90.0,
    t_inf_s=8.0e-3,
    n_per_slot=71.0,
    pue=1.00,
)

DEVICE = TierEnergyParams(
    tier="device",
    platform="50 M INT4 micro-model on a 6 TOPS mobile NPU + uplink radio",
    p_act_w=2.5,
    p_idle_w=0.0,          # power-gated between inferences
    t_inf_s=3.0e-3,
    n_per_slot=1.0,        # irrelevant while P_idle = 0
    pue=1.00,
    extra_j=1.0e-4,        # 0.1 mJ semantic-feature uplink
    extra_note="0.1 mJ uplink radio energy for the semantic feature vector",
)

TIERS: Dict[str, TierEnergyParams] = {"cloud": CLOUD, "edge": EDGE, "device": DEVICE}

# Headline routing split used for the hybrid figure (CONTRACT).
HEADLINE_SPLIT = {"cloud": 0.25, "edge": 0.50, "device": 0.25}

# Values asserted by the CONTRACT.  ``cloud`` is reproduced only to within
# 0.75 %: see ``CONTRACT_CLOUD_NOTE`` and the DEVIATIONS section of
# ``work/results.md``.
CONTRACT_VALUES_J = {
    "cloud": 16.6,
    "edge": 2.55,
    "device": 7.6e-3,
    "hybrid": 5.43,
}
CONTRACT_CLOUD_NOTE = (
    "CONTRACT line 53 writes 1.30 * (12.75 + 0.115) = 16.6 J, but "
    "1.30 * 12.865 = 16.72 J; the stated 16.6 J equals 1.30 * 12.75, i.e. the "
    "PUE multiplier was applied to the active term only and the amortised idle "
    "term was dropped. This module implements the formula as written, giving "
    "16.72 J for the cloud tier and 5.46 J for the hybrid."
)


def tier_energy_j(p: TierEnergyParams, t_slot_s: float = T_SLOT_ENERGY_S) -> float:
    """Per-inference energy in joules for one tier.

    Implements ``E_i = PUE_i * (P_act * T_inf + P_idle * (T_slot - n*T_inf)/n)``
    plus any additive non-compute term.
    """
    if p.n_per_slot <= 0:
        raise ValueError("n_per_slot must be positive")
    active_j = p.p_act_w * p.t_inf_s
    idle_window_s = t_slot_s - p.n_per_slot * p.t_inf_s
    if idle_window_s < 0.0:
        # The node is over-subscribed: no idle time to amortise.
        idle_window_s = 0.0
    idle_j = p.p_idle_w * idle_window_s / p.n_per_slot
    return p.pue * (active_j + idle_j) + p.extra_j


def reference_tier_energies_j() -> Dict[str, float]:
    """The three CONTRACT reference per-inference energies, in joules."""
    return {name: tier_energy_j(p) for name, p in TIERS.items()}


def hybrid_energy_j(split: Dict[str, float] | None = None,
                    tier_energies: Dict[str, float] | None = None) -> float:
    """Weighted per-inference energy of the hybrid configuration.

    ``split`` must satisfy the simplex constraint sum(alpha_i) = 1.
    """
    split = HEADLINE_SPLIT if split is None else split
    tot = sum(split.values())
    if abs(tot - 1.0) > 1e-6:
        raise ValueError(f"routing split must sum to 1, got {tot}")
    e = reference_tier_energies_j() if tier_energies is None else tier_energies
    return sum(split[k] * e[k] for k in split)


def aggregate_kwh(energy_per_inference_j: float,
                  n_users: int = 1000,
                  inferences_per_user_per_s: float = 1.0,
                  hours: float = 1.0) -> float:
    """Aggregate energy in kWh for ``n_users`` users over ``hours`` hours."""
    n_inf = n_users * inferences_per_user_per_s * hours * 3600.0
    return n_inf * energy_per_inference_j / JOULES_PER_KWH


def request_energy_j(tier: str,
                     t_inf_s: float,
                     n_per_slot: float | None = None,
                     uplink_j: float = 0.0,
                     device_encode_s: float = 0.0) -> float:
    """Energy of a single simulated inference request.

    Unlike :func:`tier_energy_j` this uses the *simulated* inference time of the
    selected (model, compression) variant and the *simulated* per-node load,
    so a request served by a lightly loaded node carries a larger amortised
    idle share, exactly as the formula prescribes.

    ``device_encode_s`` charges the on-device semantic encoder (always run on
    the device NPU, whatever tier eventually serves the request).
    """
    base = TIERS[tier]
    n = base.n_per_slot if n_per_slot is None else max(n_per_slot, 1.0)
    p = TierEnergyParams(
        tier=base.tier,
        platform=base.platform,
        p_act_w=base.p_act_w,
        p_idle_w=base.p_idle_w,
        t_inf_s=t_inf_s,
        n_per_slot=n,
        pue=base.pue,
        extra_j=0.0,
    )
    e = tier_energy_j(p)
    if device_encode_s > 0.0:
        e += DEVICE.p_act_w * device_encode_s
    e += uplink_j
    return e


def parameter_table() -> list[dict]:
    """Machine-readable version of CONTRACT Table 9 with the derived E_i."""
    rows = []
    for name in ("cloud", "edge", "device"):
        p = TIERS[name]
        row = asdict(p)
        row["E_i_J"] = tier_energy_j(p)
        row["contract_E_i_J"] = CONTRACT_VALUES_J[name]
        rows.append(row)
    return rows


if __name__ == "__main__":  # pragma: no cover - human-readable summary
    e = reference_tier_energies_j()
    for k, v in e.items():
        print(f"{k:7s} E = {v:12.6g} J  ({v * 1e3:.4g} mJ)   contract "
              f"{CONTRACT_VALUES_J[k]:g} J")
    eh = hybrid_energy_j()
    print(f"hybrid  E = {eh:.4g} J/inference  (contract "
          f"{CONTRACT_VALUES_J['hybrid']:g} J)")
    print(f"aggregate 1000 users, 1 inf/s, 1 h = {aggregate_kwh(eh):.4g} kWh")
    print(f"cloud-only reference                = "
          f"{aggregate_kwh(e['cloud']):.4g} kWh")
    print(f"energy reduction vs cloud-only      = "
          f"{100 * (e['cloud'] - eh) / e['cloud']:.3g} %")
    print()
    print(CONTRACT_CLOUD_NOTE)
