"""Ablation configurations (editor requirement R6).

Seven configurations are compared:

==== ==================================================================
 a   full MAIMO
 b   no BiLSTM forecast (reactive state only)
 c   no DRL controller (rule-based selection, forecast retained)
 d   no proactive model loading (reactive on-demand loading)
 e   no compression (FP16 only, no LoRA/INT8/INT4)
 f   no semantic compression (raw payload uplink)
 g   unidirectional LSTM instead of BiLSTM
==== ==================================================================

How the learned policy is handled
---------------------------------
Configurations (b), (d), (e), (f) and (g) remove a *mechanism* from the
deployed system and keep the orchestration policy that full MAIMO learned; the
policy's action mask is restricted to the actions the ablated system can still
execute (this matters for (e), where the compressed variants no longer exist).
Configuration (c) replaces the learned policy by the rule-based controller and
trains nothing.  This is a deployment-time ablation and is what the numbers in
Table 8 measure; it is stated explicitly there and in the README, because a
retrained-per-configuration ablation would answer a different question ("how
well can the system do without the mechanism") and costs seven times the
training budget.  Configuration (b) is the one where the distinction matters
most, so it is *also* run with a controller retrained without the forecast, and
both numbers are reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from env import RunConfig


@dataclass
class Ablation:
    key: str
    tag: str                     # (a)..(g), as in Table 8
    label: str
    cfg: RunConfig
    controller: str              # "maimo" | "rule" | "retrain_no_forecast"
    predictor: str               # "bilstm" | "lstm" | "none"
    note: str = ""


def ablation_registry() -> Dict[str, Ablation]:
    return {
        "full": Ablation(
            "full", "(a)", "Full MAIMO",
            RunConfig(name="abl_full"),
            controller="maimo", predictor="bilstm",
            note="reference configuration; identical to the MAIMO row of "
                 "Table 7"),
        "no_forecast": Ablation(
            "no_forecast", "(b)", "No BiLSTM forecast (reactive state only)",
            RunConfig(name="abl_no_forecast", use_forecast=False,
                      proactive_loading=False, predictor_kind="none"),
            controller="maimo", predictor="none",
            note="the forecast block of the observation is zero and pre-loading "
                 "has no signal to act on, so removing the forecast also "
                 "disables proactive loading; a variant with the controller "
                 "retrained without the forecast is reported alongside"),
        "no_drl": Ablation(
            "no_drl", "(c)", "No DRL controller (rule-based selection)",
            RunConfig(name="abl_no_drl"),
            controller="rule", predictor="bilstm",
            note="deterministic threshold rules over the same observation; the "
                 "forecast is retained and still drives pre-loading, so the "
                 "ablation isolates the learned controller"),
        "no_preload": Ablation(
            "no_preload", "(d)", "No proactive model loading",
            RunConfig(name="abl_no_preload", proactive_loading=False),
            controller="maimo", predictor="bilstm",
            note="variants are fetched on demand when a request misses the MEC "
                 "cache; the forecast is still available to the controller"),
        "no_compression": Ablation(
            "no_compression", "(e)", "No compression (FP16 only)",
            RunConfig(name="abl_no_compression",
                      allowed_compressions=("none",)),
            controller="maimo", predictor="bilstm",
            note="only the dense FP16 variants exist.  The FP16 7 B model needs "
                 "14 GB of weights and the MEC board can serve at most 12 GB, "
                 "so the edge tier becomes unavailable and the system degrades "
                 "to cloud plus device serving: that is the quantitative "
                 "argument for the compression pipeline of Sec. 3.3"),
        "no_semantic": Ablation(
            "no_semantic", "(f)", "No semantic compression (raw uplink)",
            RunConfig(name="abl_no_semantic", semantic_compression=False),
            controller="maimo", predictor="bilstm",
            note="the raw observation is uploaded instead of the semantic "
                 "feature vector; the semantic encoder is not run, so its "
                 "0.81 ms and its device energy are saved, but the uplink "
                 "payload grows by the semantic compression ratio"),
        "lstm": Ablation(
            "lstm", "(g)", "Unidirectional LSTM instead of BiLSTM",
            RunConfig(name="abl_lstm", predictor_kind="lstm"),
            controller="maimo", predictor="lstm",
            note="same controller, same pre-loading, forecast produced by the "
                 "unidirectional predictor"),
    }


ABLATION_ORDER = ("full", "no_forecast", "no_drl", "no_preload",
                  "no_compression", "no_semantic", "lstm")


def ablation_predictor_kind(a: Ablation) -> Optional[str]:
    return None if a.predictor == "none" else a.predictor
