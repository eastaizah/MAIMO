"""Model zoo: FLOPs, inference time, memory footprint, load time, the
accuracy-versus-compression curve and the early-exit profile.

Inference time uses the corrected forward-pass expression

``tau_inf = 2 P N_tok / (F_eff s_comp)``

where ``P`` is the parameter count, ``N_tok`` the sequence length, ``F_eff``
the tier's effective throughput in FLOP/s and ``s_comp`` the compression
speed-up.  The factor 2 is the multiply-accumulate count of one forward pass
per parameter per token.  The manuscript's original expression divided a
parameter count by a throughput in FLOP/s, which is dimensionally wrong (a
parameter count is dimensionless, so the quotient is not a time).

**Accuracy is a task-success-rate proxy.**  The numbers in
:data:`SUCCESS_RATE_BY_COMPLEXITY` and :data:`CAPABILITY_LOSS_PP` are read off
an accuracy-versus-model-scale curve and an accuracy-versus-compression curve;
they are *not* measurements on a real dataset and must never be presented as
such.  They exist so that the orchestration trade-off (route a hard request to
a small model and you lose accuracy) is represented quantitatively.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .config import (Config, DEFAULT, COMPRESSIONS, COMP_IDX, MODEL_BY_TIER,
                     SERVICE_CLASSES, TIERS, N_TIER, N_CLASS, N_COMPLEXITY,
                     COMPLEXITY_MIX)

# ---------------------------------------------------------------------------
# Accuracy-vs-compression / accuracy-vs-scale proxy curves
# ---------------------------------------------------------------------------
# Task success rate (%) of the 70 B cloud reference model on each complexity
# stratum of the joint semantic-communication + channel-estimation task.
SUCCESS_RATE_BY_COMPLEXITY: Tuple[float, float, float] = (91.0, 94.5, 97.5)

# Capability loss (percentage points) of the tier's model relative to the
# 70 B reference *on the same complexity stratum*, before any compression
# penalty.  Ordered (hard, medium, easy).
CAPABILITY_LOSS_PP: Dict[str, Tuple[float, float, float]] = {
    "cloud":  (0.00, 0.00, 0.00),
    "edge":   (6.60, 0.35, 0.00),
    "device": (7.00, 3.00, 0.50),
}

# Which compression variant each tier runs.
COMPRESSION_BY_MODE: Dict[str, Tuple[str, str, str]] = {
    # mode                cloud   edge    device
    "adaptive_default": ("fp16", "lora", "int4"),
    "adaptive_fast":    ("fp16", "int8", "int4"),
    "dense_fp16":       ("fp16", "fp16", "fp16"),
}

# Probability that the difficulty estimator produced by the semantic encoder
# is uninformative, in which case the request is routed by proportion only.
ROUTER_CONFUSION = 0.12   # [CAL]

# Task success rate of the uncompressed 70 B cloud model over the complexity
# mixture: the best achievable value in this model, used as the accuracy
# reference in the orchestrator's reward.
REFERENCE_ACCURACY_PCT = float(
    sum(w * a for w, a in zip(COMPLEXITY_MIX, SUCCESS_RATE_BY_COMPLEXITY)))


def inference_time_s(params: float, tokens: int, flops: float,
                     speedup: float, exit_factor: float = 1.0) -> float:
    return 2.0 * params * tokens * exit_factor / (flops * speedup)


def tier_flops(cfg: Config) -> Tuple[float, float, float]:
    return (cfg.flops_cloud, cfg.flops_edge, cfg.flops_device)


def memory_gb(params: float, comp_key: str) -> float:
    return params * COMPRESSIONS[COMP_IDX[comp_key]].bytes_per_param / 1e9


def load_time_ms(cfg: Config, tier: str, comp_key: str) -> float:
    gb = memory_gb(MODEL_BY_TIER[tier].params, comp_key)
    return gb * cfg.cold_start_ms_per_gb


def semantic_encode_ms(cfg: Config) -> float:
    """On-device semantic encoding time (always executed on the UE NPU)."""
    return 1e3 * inference_time_s(
        cfg.semantic_encoder_params, cfg.semantic_encoder_tokens,
        cfg.flops_device, COMPRESSIONS[COMP_IDX["int4"]].speedup)


def inference_time_table_ms(cfg: Config, mode: str,
                            early_exit: bool) -> np.ndarray:
    """Per-class, per-tier inference time in milliseconds, shape ``(C, T)``."""
    comps = COMPRESSION_BY_MODE[mode]
    flops = tier_flops(cfg)
    out = np.zeros((N_CLASS, N_TIER))
    for ci, sc in enumerate(SERVICE_CLASSES):
        for ti, tier in enumerate(TIERS):
            use_exit = early_exit and (
                sc.early_exit or (tier == "device" and cfg.device_early_exit))
            ef = cfg.early_exit_factor if use_exit else 1.0
            out[ci, ti] = 1e3 * inference_time_s(
                MODEL_BY_TIER[tier].params, sc.tokens, flops[ti],
                COMPRESSIONS[COMP_IDX[comps[ti]]].speedup, ef)
    return out


def early_exit_mask(cfg: Config, early_exit: bool) -> np.ndarray:
    """``(C, T)`` indicator of where the early-exit heads are actually used."""
    out = np.zeros((N_CLASS, N_TIER))
    if not early_exit:
        return out
    for ci, sc in enumerate(SERVICE_CLASSES):
        for ti, tier in enumerate(TIERS):
            if sc.early_exit or (tier == "device" and cfg.device_early_exit):
                out[ci, ti] = 1.0
    return out


def compression_penalty_pp(mode: str) -> np.ndarray:
    """Accuracy penalty (pp) of the compression variant used at each tier."""
    comps = COMPRESSION_BY_MODE[mode]
    return np.array([COMPRESSIONS[COMP_IDX[c]].acc_penalty_pp for c in comps])


# ---------------------------------------------------------------------------
# Complexity-aware assignment and the resulting accuracy
# ---------------------------------------------------------------------------
_CUM = np.concatenate([[0.0], np.cumsum(COMPLEXITY_MIX)])


def assignment_matrix(alpha: np.ndarray,
                      confusion: float = ROUTER_CONFUSION) -> np.ndarray:
    """Fraction of each complexity stratum served by each tier.

    ``alpha`` has shape ``(G, T)`` and sums to one along the tier axis.  The
    complexity-aware router sorts requests by estimated difficulty and fills
    the cloud with the hardest ``alpha_cloud`` fraction of the load, the edge
    with the next ``alpha_edge`` and the device with the remainder.  With
    probability ``confusion`` the difficulty estimate carries no information
    and the request is assigned by proportion only.

    Returns an array of shape ``(G, K, T)`` whose ``[g, :, :]`` block sums to
    one and whose row sums are the complexity mixture weights.
    """
    alpha = np.asarray(alpha, dtype=float)
    g = alpha.shape[0]
    upper = np.cumsum(alpha, axis=1)                       # (G, T)
    lower = upper - alpha
    lo = _CUM[:-1][None, :, None]                          # (1, K, 1)
    hi = _CUM[1:][None, :, None]
    sorted_m = np.maximum(0.0, np.minimum(hi, upper[:, None, :])
                          - np.maximum(lo, lower[:, None, :]))
    w = np.asarray(COMPLEXITY_MIX)[None, :, None]
    random_m = w * alpha[:, None, :]
    return (1.0 - confusion) * sorted_m + confusion * random_m


def tier_quality_pp(mode: str, early_exit: bool, cfg: Config) -> np.ndarray:
    """Task success rate (%) for each (complexity, tier), shape ``(K, T)``."""
    base = np.asarray(SUCCESS_RATE_BY_COMPLEXITY)[:, None]      # (K, 1)
    loss = np.stack([np.asarray(CAPABILITY_LOSS_PP[t]) for t in TIERS],
                    axis=1)                                     # (K, T)
    pen = compression_penalty_pp(mode)[None, :]                 # (1, T)
    exit_pen = np.zeros((1, N_TIER))
    if early_exit and cfg.device_early_exit:
        exit_pen[0, TIERS.index("device")] = cfg.early_exit_acc_pp
    return base - loss - pen - exit_pen


def expected_accuracy(alpha: np.ndarray, mode: str, early_exit: bool,
                      cfg: Config, confusion: float = ROUTER_CONFUSION
                      ) -> np.ndarray:
    """Expected task success rate (%) of the headline class, shape ``(G,)``."""
    m = assignment_matrix(alpha, confusion)                     # (G, K, T)
    q = tier_quality_pp(mode, early_exit, cfg)[None, :, :]      # (1, K, T)
    return np.sum(m * q, axis=(1, 2))
