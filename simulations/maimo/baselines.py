"""The locked baseline catalogue B1-B10 (CONTRACT, editor requirement R5).

B1-B3 are conventional single-tier deployments: they do **not** include the
MAIMO semantic-communication front-end, because that front-end is part of the
three-tier architecture under evaluation and a monolithic cloud or device
deployment does not have it.  B4-B9 are orchestration policies that run on the
same MAIMO substrate (semantic encoder, model zoo, compression variants) and
differ only in how they decide where to run each request; B8 and B9 also share
MAIMO's predictor, proactive loading and early exit, so that the comparison
against B10 isolates the *controller*.  This is stated in the Materials and
Methods section and is the fairest reading of the editor's request for
"stronger, more representative baselines".
"""

from __future__ import annotations

from typing import Dict, List

from .sim import PolicySpec

BASELINES: List[PolicySpec] = [
    PolicySpec(
        ident="B1", name="Cloud-only monolithic", controller="fixed",
        fixed_alpha=(1.0, 0.0, 0.0), use_affinity=False, semantic_comm=False,
        predictor="none", proactive_loading=False, adaptive_compression=False,
        early_exit=False, cloud_always_warm=True,
        description="every request served by the 70 B MoE model in the cloud, "
                    "raw payload uploaded over the air"),
    PolicySpec(
        ident="B2", name="Edge-only static", controller="fixed",
        fixed_alpha=(0.0, 1.0, 0.0), use_affinity=False, semantic_comm=False,
        predictor="none", proactive_loading=False, adaptive_compression=True,
        early_exit=False, cloud_always_warm=False,
        description="every request served by a fixed 7 B LoRA model at the "
                    "MEC host, on-demand model loading"),
    PolicySpec(
        ident="B3", name="Device-only", controller="fixed",
        fixed_alpha=(0.0, 0.0, 1.0), use_affinity=False, semantic_comm=False,
        predictor="none", proactive_loading=False, adaptive_compression=True,
        early_exit=False, cloud_always_warm=False,
        description="every request served by the 50 M INT4 micro-model on the "
                    "device NPU"),
    PolicySpec(
        ident="B4", name="Static proportional split", controller="fixed",
        fixed_alpha=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), use_affinity=True,
        semantic_comm=True, predictor="none", proactive_loading=False,
        adaptive_compression=True, early_exit=False,
        description="fixed routing alpha = (1/3, 1/3, 1/3), no adaptation"),
    PolicySpec(
        ident="B5", name="Greedy least-latency", controller="greedy",
        semantic_comm=True, predictor="none", proactive_loading=False,
        adaptive_compression=True, early_exit=False,
        description="per-interval myopic choice of the tier mix with the "
                    "lowest instantaneous predicted latency"),
    PolicySpec(
        ident="B6", name="SINR/load threshold heuristic", controller="threshold",
        semantic_comm=True, predictor="none", proactive_loading=False,
        adaptive_compression=True, early_exit=False,
        description="rule-based MEC offloading with hysteresis on the edge "
                    "utilisation and the radio condition"),
    PolicySpec(
        ident="B7", name="Lyapunov drift-plus-penalty", controller="lyapunov",
        semantic_comm=True, predictor="none", proactive_loading=False,
        adaptive_compression=True, early_exit=False,
        description="online queue-stability optimiser with an energy and "
                    "latency penalty, control parameter V swept"),
    PolicySpec(
        ident="B8", name="DQN orchestrator", controller="dqn",
        semantic_comm=True, predictor="bilstm", proactive_loading=True,
        adaptive_compression=True, early_exit=True,
        description="value-based DRL on the same state and action space as "
                    "MAIMO, same substrate"),
    PolicySpec(
        ident="B9", name="LinUCB contextual bandit", controller="linucb",
        semantic_comm=True, predictor="bilstm", proactive_loading=True,
        adaptive_compression=True, early_exit=True,
        description="disjoint LinUCB bandit orchestrator on the same context "
                    "vector, same substrate"),
    PolicySpec(
        ident="B10", name="MAIMO (proposed)", controller="ppo",
        semantic_comm=True, predictor="bilstm", proactive_loading=True,
        adaptive_compression=True, early_exit=True,
        description="BiLSTM predictor + PPO controller + proactive loading + "
                    "adaptive compression + early exit"),
]

BASELINE_BY_ID: Dict[str, PolicySpec] = {b.ident: b for b in BASELINES}
