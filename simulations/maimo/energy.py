"""The hierarchical per-inference energy model locked in ``work/CONTRACT.md``.

.. math::

    E_i = \\mathrm{PUE}_i \\left( P_i^{act} T_i^{inf}
          + P_i^{idle} \\frac{T^{slot} - n_i T_i^{inf}}{n_i} \\right)

with ``T^slot = 1 s`` and ``n_i`` the number of inferences a node of tier ``i``
serves during that slot.  The second term amortises the node's idle power over
the inferences actually served; the submitted manuscript charged a *full* idle
slot to every inference, which over-counted idle energy by a factor ``n_i`` and
produced the physically impossible figure of 25.9 Wh per inference
(= 93.2 kJ).  All energies here are in **joules**; aggregates are in kWh.

Writing ``n_i T_i^{inf} = rho_i T^{slot}`` with ``rho_i`` the node utilisation
gives the equivalent per-request form used inside the simulator,

.. math::

    E_i = \\mathrm{PUE}_i P_i^{act} T_i^{inf}
          \\left( 1 + \\frac{P_i^{idle}}{P_i^{act}}
                        \\frac{1-\\rho_i}{\\rho_i} \\right),

which is what allows a *mixture* of request sizes to be charged consistently:
the node's idle energy in the slot is shared out in proportion to each
request's occupancy of the node.

HONESTY RULE (editor requirements R3 and R7): every number produced by this
module is an **analytic estimate** obtained by combining vendor
thermal-design-power envelopes with simulated inference times.  These are not
wall-plug measurements and must never be described as such.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np

from .config import JOULES_PER_KWH

T_SLOT_ACCOUNTING_S = 1.0


@dataclass(frozen=True)
class TierEnergyParams:
    """Per-tier energy-accounting parameters (CONTRACT Table 9)."""

    tier: str
    platform: str
    p_act_w: float
    p_idle_w: float
    t_inf_s: float
    n_per_slot: float
    pue: float
    extra_j: float = 0.0
    extra_note: str = ""
    source: str = ""


CLOUD = TierEnergyParams(
    tier="cloud",
    platform="70 B MoE sharded over 8x NVIDIA A100 80 GB SXM (400 W each, "
             "~70 % utilisation) plus host and fabric",
    p_act_w=2550.0, p_idle_w=900.0, t_inf_s=5.0e-3, n_per_slot=195.0, pue=1.30,
    source="A100 SXM board power 400 W (vendor TDP); 8 x 400 W x 0.70 "
           "utilisation + ~510 W host/NIC/fabric = 2.55 kW; datacentre PUE "
           "1.30 is the industry average for a modern hyperscale facility.",
)

EDGE = TierEnergyParams(
    tier="edge",
    platform="7 B LoRA-adapted model on a 250 W MEC accelerator board "
             "(including host and NIC)",
    p_act_w=250.0, p_idle_w=90.0, t_inf_s=8.0e-3, n_per_slot=71.0, pue=1.00,
    source="250 W single-board MEC inference accelerator; PUE 1.00 because "
           "the board is deployed in a passively cooled street cabinet whose "
           "overhead is already inside the 250 W envelope.",
)

DEVICE = TierEnergyParams(
    tier="device",
    platform="50 M INT4 micro-model on a 6 TOPS mobile NPU plus uplink radio",
    p_act_w=2.5, p_idle_w=0.0, t_inf_s=3.0e-3, n_per_slot=1.0, pue=1.00,
    extra_j=1.0e-4,
    extra_note="0.1 mJ uplink radio energy for the semantic feature vector",
    source="6 TOPS mobile NPU at ~2.5 W sustained; power-gated between "
           "inferences, hence zero idle term.",
)

TIER_PARAMS: Dict[str, TierEnergyParams] = {
    "cloud": CLOUD, "edge": EDGE, "device": DEVICE}

HEADLINE_SPLIT = {"cloud": 0.25, "edge": 0.50, "device": 0.25}

# Values asserted by the CONTRACT table (revision of 2026-08-06).
CONTRACT_VALUES_J = {"cloud": 16.73, "edge": 2.55, "device": 7.6e-3,
                     "hybrid": 5.46}

CONTRACT_CLOUD_NOTE = (
    "The first version of the CONTRACT locked the cloud tier at 16.6 J, which "
    "does not follow from its own formula and parameters: the bracket is "
    "2550*0.005 = 12.75 J plus 900*(1-195*0.005)/195 = 0.1154 J, and "
    "1.30 * 12.8654 = 16.725 J. The quoted 16.6 J equalled 1.30 * 12.75, "
    "i.e. the amortised idle term had been dropped after the multiplication. "
    "This module implements the formula as written; the coordinator adopted "
    "the computed value in the 2026-08-06 revision of the CONTRACT, so the "
    "locked figures are now E_cloud = 16.73 J, E_hybrid = 5.46 J and a "
    "67.4 % reduction."
)


def tier_energy_j(p: TierEnergyParams,
                  t_slot_s: float = T_SLOT_ACCOUNTING_S) -> float:
    """Per-inference energy of one tier, in joules, from the locked formula."""
    if p.n_per_slot <= 0:
        raise ValueError("n_per_slot must be positive")
    active = p.p_act_w * p.t_inf_s
    idle_window = max(t_slot_s - p.n_per_slot * p.t_inf_s, 0.0)
    idle = p.p_idle_w * idle_window / p.n_per_slot
    return p.pue * (active + idle) + p.extra_j


def reference_tier_energies_j() -> Dict[str, float]:
    return {k: tier_energy_j(v) for k, v in TIER_PARAMS.items()}


def hybrid_energy_j(split: Dict[str, float] | None = None) -> float:
    split = HEADLINE_SPLIT if split is None else split
    if abs(sum(split.values()) - 1.0) > 1e-9:
        raise ValueError("routing split must sum to 1")
    e = reference_tier_energies_j()
    return sum(split[k] * e[k] for k in split)


def request_energy_j(tier: str, t_inf_s, rho, device_encode_s: float = 0.0,
                     uplink_j: float = 0.0):
    """Per-request energy for a *simulated* inference.

    ``t_inf_s`` is the realised inference time of the selected variant and
    ``rho`` the realised utilisation of the serving node in that slot, so a
    request served by a lightly loaded node carries a larger amortised idle
    share, exactly as the locked formula prescribes.  Both may be arrays.
    """
    p = TIER_PARAMS[tier]
    rho = np.clip(rho, 1e-6, 1.0)
    e = p.pue * p.p_act_w * np.asarray(t_inf_s) * (
        1.0 + (p.p_idle_w / p.p_act_w) * (1.0 - rho) / rho)
    return e + p.extra_j + DEVICE.p_act_w * device_encode_s + uplink_j


def aggregate_kwh(energy_per_inference_j: float, n_users: int = 1000,
                  inferences_per_user_per_s: float = 1.0,
                  hours: float = 1.0) -> float:
    n = n_users * inferences_per_user_per_s * hours * 3600.0
    return n * energy_per_inference_j / JOULES_PER_KWH


def parameter_table() -> list:
    rows = []
    for name in ("cloud", "edge", "device"):
        p = TIER_PARAMS[name]
        row = asdict(p)
        row["E_i_J"] = tier_energy_j(p)
        row["contract_E_i_J"] = CONTRACT_VALUES_J[name]
        row["utilisation"] = p.n_per_slot * p.t_inf_s / T_SLOT_ACCOUNTING_S
        rows.append(row)
    return rows


if __name__ == "__main__":   # pragma: no cover
    e = reference_tier_energies_j()
    for k, v in e.items():
        print(f"{k:7s} {v:12.6g} J   (contract {CONTRACT_VALUES_J[k]:g} J)")
    eh = hybrid_energy_j()
    print(f"hybrid  {eh:12.6g} J   (contract {CONTRACT_VALUES_J['hybrid']:g} J)")
    print(f"reduction vs cloud-only "
          f"{100 * (e['cloud'] - eh) / e['cloud']:.3f} %")
    print(f"1000 users, 1 inf/s, 1 h: {aggregate_kwh(eh):.4g} kWh "
          f"(cloud-only {aggregate_kwh(e['cloud']):.4g} kWh)")
    print()
    print(CONTRACT_CLOUD_NOTE)
