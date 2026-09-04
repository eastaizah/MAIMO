"""MAIMO reference simulator.

A self-contained, seed-reproducible slotted simulator of the three-tier
Massive AI Model Orchestration (MAIMO) system: cloud (70 B MoE), edge (7 B
LoRA) and device (50 M INT4) tiers, a 3GPP TR 38.901 UMa radio layer,
non-stationary traffic, a BiLSTM traffic predictor, a PPO orchestrator, nine
baseline orchestration schemes and five ablations.

Everything reported by this package is a **simulation** result.  Per-inference
energies are analytic estimates from vendor power envelopes combined with
simulated inference times; they are not wall-plug measurements.  Accuracy
figures are a task-success-rate proxy derived from accuracy-versus-compression
curves, not measurements on a real dataset.
"""

__version__ = "1.0.0"
