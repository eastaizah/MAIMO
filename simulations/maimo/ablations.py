"""The locked ablation catalogue A0-A5 (CONTRACT, editor requirement R6).

Each ablation switches off exactly one component of the full system; every
other parameter is identical to A0, and every ablation is evaluated on the
same traffic traces and the same channel realisations as A0, so the
difference is attributable to the removed component alone.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

from .baselines import BASELINE_BY_ID
from .sim import PolicySpec

_FULL = BASELINE_BY_ID["B10"]

ABLATIONS: List[PolicySpec] = [
    replace(_FULL, ident="A0", name="Full MAIMO",
            description="reference configuration (identical to B10)"),
    replace(_FULL, ident="A1", name="w/o BiLSTM predictor",
            predictor="persistence",
            description="traffic prediction replaced by the last observed "
                        "load (persistence); proactive loading still runs, "
                        "but on a much worse forecast"),
    replace(_FULL, ident="A2", name="w/o PPO controller",
            controller="threshold",
            description="PPO replaced by the B6 threshold rule; the BiLSTM "
                        "predictor and proactive loading are retained"),
    replace(_FULL, ident="A3", name="w/o proactive model loading",
            proactive_loading=False,
            description="models are loaded on demand at the first request "
                        "that needs them (cold start)"),
    replace(_FULL, ident="A4", name="w/o adaptive compression",
            adaptive_compression=False,
            description="all tiers run FP16 dense weights: no LoRA, no INT8 "
                        "or INT4 quantisation, no pruning"),
    replace(_FULL, ident="A5", name="w/o early-exit inference",
            early_exit=False,
            description="every request runs the full depth of its model"),
]

ABLATION_BY_ID: Dict[str, PolicySpec] = {a.ident: a for a in ABLATIONS}
