"""The slotted simulation engine.

Design notes (these are what make a 3.6e5-slot horizon x 20 seeds x 16
configurations affordable on a CPU laptop):

* **Vectorised over replications.**  The 20 seeds are simulated as a batch
  dimension.  Every seed has its own traffic trace, its own channel pool, its
  own trained predictor and its own controller parameters, so the replications
  remain statistically independent; they merely share the Python loop.
* **Two timescales.**  The physical slot is ``t_slot_s`` = 1 s, which is the
  window over which the CONTRACT energy model amortises idle power.  The
  orchestrator acts once per ``control_interval_slots`` = 10 slots, which is
  the cadence at which a real MEC orchestrator can move models.  The ten slots
  inside a control interval are advanced with closed-form vectorised
  arithmetic, not a Python loop.
* **Lindley recursion in closed form.**  Tier backlog obeys
  ``Q_s = max(0, Q_{s-1} + W_s - C_s)``.  Its solution
  ``Q_s = S_s + max(Q_0, -min_{u<=s} S_u)`` with ``S`` the cumulative sum of
  ``W - C`` is evaluated with ``cumsum`` and ``minimum.accumulate``.
* **Fluid arrivals, stochastic queues.**  Thousands of requests arrive per
  slot, so requests are carried as (Poisson) counts and the *per-request*
  latency distribution is reconstructed by drawing a small number of weighted
  latency samples per (seed, class, tier) group each control interval.  Means
  are computed analytically; p95/p99 and SLA violation come from a weighted
  histogram of those samples.

Everything in this module is a simulation.  Nothing here is measured on real
hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import energy as en
from .cache import build_cache_model
from .carbon import HOST_REGION, intensity_g_per_kwh
from .channel import ChannelPool
from .config import (Config, DEFAULT, SERVICE_CLASSES, CLASSES, TIERS,
                     N_CLASS, N_TIER, HEADLINE_CLASS, JOULES_PER_KWH)
from .models import (COMPRESSION_BY_MODE, REFERENCE_ACCURACY_PCT,
                     expected_accuracy, inference_time_table_ms,
                     semantic_encode_ms)
from .traffic import TrafficTrace, make_trace, sample_arrivals

# ---------------------------------------------------------------------------
# Routing codebook and per-class affinities
# ---------------------------------------------------------------------------
# Candidate global routing splits (cloud, edge, device) on a 1/8 simplex grid.
ALPHA_CODEBOOK = np.array([
    [0.00, 0.75, 0.25],
    [0.10, 0.70, 0.20],
    [0.15, 0.60, 0.25],
    [0.20, 0.60, 0.20],
    [0.25, 0.50, 0.25],     # the CONTRACT headline split
    [0.25, 0.65, 0.10],
    [0.30, 0.50, 0.20],
    [0.35, 0.45, 0.20],
    [0.20, 0.45, 0.35],
    [0.15, 0.50, 0.35],
    [0.10, 0.40, 0.50],
    [0.00, 0.50, 0.50],
])
N_ALPHA = ALPHA_CODEBOOK.shape[0]
COMPRESSION_MODES = ("adaptive_default", "adaptive_fast")
N_ACTION = N_ALPHA * len(COMPRESSION_MODES)

# Per-class tier affinity.  The headline class is deliberately neutral, so the
# routing split reported for it is exactly the codebook entry the controller
# selected.  The other classes tilt the split towards the tiers that can
# actually serve them: the cloud cannot meet a 5 ms V2X budget, and the 50 M
# device model cannot carry a 256-token video-semantics workload.
CLASS_AFFINITY = np.array([
    [1.00, 1.00, 1.00],     # semantic_ce  (neutral)
    [0.08, 1.55, 1.15],     # urllc_v2x
    [1.30, 1.20, 0.52],     # embb
    [0.50, 1.30, 1.00],     # mmtc
])


def per_class_alpha(alpha_global: np.ndarray) -> np.ndarray:
    """Tilt a global split ``(G, T)`` into a per-class split ``(G, C, T)``."""
    a = alpha_global[:, None, :] * CLASS_AFFINITY[None, :, :]
    return a / np.maximum(a.sum(axis=2, keepdims=True), 1e-12)


# ---------------------------------------------------------------------------
# Policy specification
# ---------------------------------------------------------------------------
@dataclass
class PolicySpec:
    """One evaluated configuration: a baseline, an ablation or full MAIMO."""

    ident: str                       # B1..B10, A0..A5
    name: str
    controller: str                  # fixed|greedy|threshold|lyapunov|ppo|dqn|linucb
    fixed_alpha: Optional[Tuple[float, float, float]] = None
    use_affinity: bool = True
    semantic_comm: bool = True
    predictor: str = "bilstm"        # bilstm|persistence|none
    proactive_loading: bool = True
    adaptive_compression: bool = True
    early_exit: bool = True
    cloud_always_warm: bool = True
    description: str = ""

    def compression_modes(self) -> Tuple[str, ...]:
        if not self.adaptive_compression:
            return ("dense_fp16",)
        return COMPRESSION_MODES


# ---------------------------------------------------------------------------
# Per-seed static precomputation
# ---------------------------------------------------------------------------
@dataclass
class SeedContext:
    cfg: Config
    seeds: List[int]
    pools: List[ChannelPool]
    ul_per_bit_ms: np.ndarray        # (G, C, N) ms per bit
    ul_mean_per_bit_ms: np.ndarray   # (G, C)
    traces: List[TrafficTrace]
    lam: np.ndarray                  # (G, K, C)
    t_s: np.ndarray                  # (G, K)


def build_context(cfg: Config, seeds: Sequence[int], n_intervals: int,
                  time_offset_s: float | None = None) -> SeedContext:
    pools = [ChannelPool(cfg, s) for s in seeds]
    ul = np.stack([np.stack([1e3 / p.rate_bps[c.key] for c in SERVICE_CLASSES])
                   for p in pools])                            # (G, C, N)
    traces = [make_trace(cfg, s, n_intervals, time_offset_s) for s in seeds]
    lam = np.stack([t.lam for t in traces])
    t_s = np.stack([t.t_s for t in traces])
    return SeedContext(cfg=cfg, seeds=list(seeds), pools=pools,
                       ul_per_bit_ms=ul, ul_mean_per_bit_ms=ul.mean(axis=2),
                       traces=traces, lam=lam, t_s=t_s)


# ---------------------------------------------------------------------------
# Static per-policy tables
# ---------------------------------------------------------------------------
@dataclass
class PolicyTables:
    t_inf_ms: np.ndarray             # (M, C, T) one table per compression mode
    payload_bits: np.ndarray         # (C, T) uplink payload actually sent
    result_bits: np.ndarray          # (C,)
    encode_ms: float
    encode_used: np.ndarray          # (C, T) 1 where the semantic encoder runs
    fixed_ms: np.ndarray             # (T,)
    transport_ms: np.ndarray         # (T,)
    access_ms: np.ndarray            # (C,) tti/2 + ran processing
    harq_p: np.ndarray               # (C,)
    aggregation_ms: np.ndarray       # (C,)
    deadline_ms: np.ndarray          # (C,)
    cache_eligible: np.ndarray       # (C,) 0 where the variant is pinned
    accuracy_modes: Tuple[str, ...]


def build_tables(cfg: Config, spec: PolicySpec) -> PolicyTables:
    modes = spec.compression_modes()
    t_inf = np.stack([inference_time_table_ms(cfg, m, spec.early_exit)
                      for m in modes])
    payload = np.zeros((N_CLASS, N_TIER))
    encode_used = np.zeros((N_CLASS, N_TIER))
    result = np.array([sc.result_bits for sc in SERVICE_CLASSES])
    dev = TIERS.index("device")
    for ci, sc in enumerate(SERVICE_CLASSES):
        up = sc.raw_bits / sc.semantic_ratio if spec.semantic_comm else sc.raw_bits
        for ti in range(N_TIER):
            if ti == dev:
                # served locally; only the result crosses the air interface
                payload[ci, ti] = sc.result_bits
            else:
                payload[ci, ti] = up
                encode_used[ci, ti] = 1.0 if spec.semantic_comm else 0.0
    fixed = np.array([cfg.cloud_fixed_ms, cfg.edge_fixed_ms, cfg.device_fixed_ms])
    transport = np.array([cfg.wan_rtt_ms, 0.0, 0.0])
    access = np.array([sc.tti_ms / 2.0 + sc.ran_proc_ms for sc in SERVICE_CLASSES])
    return PolicyTables(
        t_inf_ms=t_inf, payload_bits=payload, result_bits=result,
        encode_ms=semantic_encode_ms(cfg), encode_used=encode_used,
        fixed_ms=fixed, transport_ms=transport, access_ms=access,
        harq_p=np.array([sc.harq_retx_prob for sc in SERVICE_CLASSES]),
        aggregation_ms=np.array([sc.aggregation_ms for sc in SERVICE_CLASSES]),
        deadline_ms=np.array([sc.deadline_ms for sc in SERVICE_CLASSES]),
        cache_eligible=np.array([0.0 if sc.pinned_cache else 1.0
                                 for sc in SERVICE_CLASSES]),
        accuracy_modes=modes)


# ---------------------------------------------------------------------------
# Queueing
# ---------------------------------------------------------------------------
def sakasegawa_wait_ms(c_active, rho, es_ms, cs2):
    """Sakasegawa's M/G/c waiting-time approximation, in milliseconds.

    ``W = ((1 + C_s^2)/2) * rho^(sqrt(2(c+1)) - 1) / (c (1 - rho)) * E[S]``

    Exact for M/M/1 and reduces to Pollaczek-Khinchine for M/D/1.
    """
    c = np.maximum(c_active, 1.0)
    r = np.clip(rho, 1e-9, 0.999)
    expo = np.sqrt(2.0 * (c + 1.0)) - 1.0
    return 0.5 * (1.0 + cs2) * np.power(r, expo) / (c * (1.0 - r)) * es_ms


def lindley(q0: np.ndarray, work: np.ndarray, cap: np.ndarray) -> np.ndarray:
    """Backlog trajectory of ``Q_s = max(0, Q_{s-1} + W_s - C)``.

    ``q0`` is ``(G, T)``, ``work`` is ``(G, S, T)``, ``cap`` is ``(T,)``.
    Returns ``(G, S, T)`` backlog *after* each slot.
    """
    x = work - cap[None, None, :]
    s = np.cumsum(x, axis=1)
    zero = np.zeros((s.shape[0], 1, s.shape[2]))
    s_full = np.concatenate([zero, s], axis=1)          # S_0 = 0
    run_min = np.minimum.accumulate(s_full, axis=1)[:, 1:, :]
    return s + np.maximum(q0[:, None, :], -run_min)


# ---------------------------------------------------------------------------
# Accumulators
# ---------------------------------------------------------------------------
class Accumulator:
    """Weighted metric accumulation over the measured (post warm-up) window."""

    def __init__(self, cfg: Config, n_seeds: int):
        self.cfg = cfg
        self.g = n_seeds
        b = cfg.latency_hist_bins
        self.edges = np.logspace(math.log10(cfg.latency_hist_min_ms),
                                 math.log10(cfg.latency_hist_max_ms), b + 1)
        self.centres = np.sqrt(self.edges[:-1] * self.edges[1:])
        self.hist = np.zeros((n_seeds, N_CLASS, b))
        self.count = np.zeros((n_seeds, N_CLASS))
        self.lat_sum = np.zeros((n_seeds, N_CLASS))
        self.energy_sum = np.zeros((n_seeds, N_CLASS))
        self.energy_tier_sum = np.zeros((n_seeds, N_TIER))
        self.count_tier = np.zeros((n_seeds, N_TIER))
        # Headline-class energy split into the four terms that make it up, so
        # that the simulated figure can be reconciled against the
        # inference-only reference of the locked energy model (editor R3).
        self.e_parts_sum = np.zeros((n_seeds, 4))   # compute, encode, radio, load
        self.carbon_sum = np.zeros(n_seeds)          # grams CO2e
        self.acc_sum = np.zeros(n_seeds)
        self.acc_w = np.zeros(n_seeds)
        self.alpha_sum = np.zeros((n_seeds, N_TIER))
        self.alpha_w = np.zeros(n_seeds)
        self.cache_hit_sum = np.zeros(n_seeds)
        self.cache_w = np.zeros(n_seeds)
        self.pred_err_sum = np.zeros(n_seeds)
        self.pred_w = np.zeros(n_seeds)
        self.sim_time_s = 0.0
        self.drop = np.zeros(n_seeds)

        self._base = None

    def add_latency_samples(self, samples: np.ndarray, weights: np.ndarray):
        """``samples`` and ``weights`` have shape ``(G, C, T, S)``."""
        b = self.cfg.latency_hist_bins
        if self._base is None or self._base.shape != samples.shape:
            gi = np.arange(self.g)[:, None, None, None]
            ci = np.arange(N_CLASS)[None, :, None, None]
            self._base = np.broadcast_to((gi * N_CLASS + ci) * b,
                                         samples.shape).copy()
        idx = np.clip(np.searchsorted(self.edges, samples, side="right") - 1,
                      0, b - 1)
        flat = (self._base + idx).ravel()
        self.hist += np.bincount(
            flat, weights=weights.ravel(),
            minlength=self.g * N_CLASS * b).reshape(self.g, N_CLASS, b)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
def run_batch(cfg: Config, ctx: SeedContext, spec: PolicySpec,
              controller, predictions: np.ndarray,
              pred_err: np.ndarray, measure_from: int,
              n_intervals: int, collect_reward: bool = False,
              rng_seed: int = 0) -> dict:
    """Run every seed of ``ctx`` under ``spec`` for ``n_intervals`` intervals.

    ``predictions`` and ``pred_err`` are ``(G, K)`` arrays produced ahead of
    time by the traffic predictor (a single batched forward pass per seed).
    ``controller`` must expose ``act(state) -> action`` and, when training,
    ``observe(reward)``.
    """
    g = len(ctx.seeds)
    rng = np.random.default_rng(7_000_000 + rng_seed)
    tab = build_tables(cfg, spec)
    acc = Accumulator(cfg, g)
    n_slots = cfg.control_interval_slots
    t_slot = cfg.t_slot_s
    n_modes = tab.t_inf_ms.shape[0]

    # The device tier has no shared server: every UE owns its NPU, so its
    # capacity is the session population itself.  A large finite number keeps
    # the Lindley recursion well defined (inf - inf would be a NaN).
    dev_cap = float(cfg.n_sessions())
    cap = np.array([cfg.cloud_nodes_max * t_slot,
                    cfg.edge_boards_max() * t_slot, dev_cap * t_slot])
    c_max = np.array([float(cfg.cloud_nodes_max), float(cfg.edge_boards_max()),
                      dev_cap])
    rho_target = np.array([cfg.cloud_target_utilisation,
                           cfg.edge_target_utilisation, 1.0])
    p_act = np.array([en.CLOUD.p_act_w, en.EDGE.p_act_w, en.DEVICE.p_act_w])
    p_idle = np.array([en.CLOUD.p_idle_w, en.EDGE.p_idle_w, en.DEVICE.p_idle_w])
    pue = np.array([en.CLOUD.pue, en.EDGE.pue, en.DEVICE.pue])

    lam_ref = float(np.mean(ctx.lam.sum(axis=2)))
    q = np.zeros((g, N_TIER))
    prev_alpha = np.tile(ALPHA_CODEBOOK[2], (g, 1))
    prev_action = np.full(g, 2 * len(COMPRESSION_MODES), dtype=np.int64)
    ns = cfg.latency_samples_per_group
    rewards: List[np.ndarray] = []

    dev = TIERS.index("device")
    cache_model = build_cache_model(cfg)
    encode_j = en.DEVICE.p_act_w * tab.encode_ms * 1e-3
    ue_radio_w = cfg.ue_radio_power_w
    e_load_per_miss = pue * p_idle * cfg.cold_start_penalty_ms * 1e-3
    ul_flat = np.ascontiguousarray(
        ctx.ul_per_bit_ms.reshape(g * N_CLASS, -1))
    not_dev = (np.arange(N_TIER) != dev).astype(float)
    # Task success rate of every (compression mode, codebook entry) pair,
    # precomputed once.  The headline class has neutral tier affinity, so its
    # routing split is exactly the codebook entry the controller selected.
    if spec.fixed_alpha is not None:
        fixed = np.asarray(spec.fixed_alpha)[None, :]
        acc_table = np.stack([expected_accuracy(fixed, m, spec.early_exit, cfg)
                              for m in tab.accuracy_modes])       # (M, 1)
    else:
        acc_table = np.stack([expected_accuracy(ALPHA_CODEBOOK, m,
                                                spec.early_exit, cfg)
                              for m in tab.accuracy_modes])       # (M, A)

    for k in range(n_intervals):
        lam = ctx.lam[:, k, :]                                    # (G, C)
        tot = lam.sum(axis=1)
        pred = predictions[:, k]
        perr = pred_err[:, k]

        # -- state -----------------------------------------------------
        tod = 2.0 * math.pi * (ctx.t_s[:, k] % 86400.0) / 86400.0
        state = np.concatenate([
            (pred / lam_ref)[:, None],
            (tot / lam_ref)[:, None],
            lam / np.maximum(tot[:, None], 1e-9),
            (q[:, :2] / c_max[None, :2]),
            np.clip(np.stack([np.sin(tod), np.cos(tod)], axis=1), -1, 1),
            prev_alpha,
            perr[:, None],
        ], axis=1)

        action = controller.act(state, k)
        alpha_idx = action // n_modes if n_modes > 1 else action
        mode_idx = action % n_modes if n_modes > 1 else np.zeros_like(action)
        alpha_g = ALPHA_CODEBOOK[alpha_idx]                       # (G, T)
        if spec.fixed_alpha is not None:
            alpha_g = np.tile(np.asarray(spec.fixed_alpha), (g, 1))
        alpha = (per_class_alpha(alpha_g) if spec.use_affinity
                 else np.broadcast_to(alpha_g[:, None, :],
                                      (g, N_CLASS, N_TIER)).copy())
        t_inf = tab.t_inf_ms[mode_idx]                            # (G, C, T)

        # -- arrivals and routing --------------------------------------
        arr = sample_arrivals(rng, lam, n_slots, cfg)             # (G, S, C)
        n_req = arr[:, :, :, None] * alpha[:, None, :, :]         # (G, S, C, T)
        work = np.einsum('gsct,gct->gst', n_req, t_inf) * 1e-3    # seconds

        # -- backlog, utilisation, waiting time ------------------------
        qs = lindley(q, work, cap)
        q_prev = np.concatenate([q[:, None, :], qs[:, :-1, :]], axis=1)
        served = q_prev + work - qs                               # (G, S, T)
        served = np.maximum(served, 0.0)
        c_active = np.minimum(
            np.ceil(served / np.maximum(rho_target * t_slot, 1e-12)),
            c_max[None, None, :])
        c_active = np.maximum(c_active, 1.0)
        rho = np.clip(served / np.maximum(c_active * t_slot, 1e-12), 1e-6, 1.0)
        rho[:, :, dev] = 1.0                                      # power-gated

        n_tier = n_req.sum(axis=2)                                # (G, S, T)
        es = np.einsum('gsct,gct->gst', n_req, t_inf) / np.maximum(n_tier, 1e-9)
        es2 = np.einsum('gsct,gct->gst', n_req, t_inf ** 2) / np.maximum(n_tier, 1e-9)
        cs2 = np.clip(es2 / np.maximum(es ** 2, 1e-12) - 1.0, 0.0, 20.0)
        w_q = sakasegawa_wait_ms(c_active, rho, es, cs2)
        w_q += 1e3 * q_prev / np.maximum(c_max[None, None, :], 1e-9)
        w_q[:, :, dev] = 0.0
        w_mean = np.average(w_q, axis=1, weights=None)            # (G, T)
        rho_mean = rho.mean(axis=1)

        # -- model loading ---------------------------------------------
        # Two different rates.  ``miss`` is how often a request *stalls*
        # waiting for a variant, which is what costs latency; proactive
        # staging hides most of those.  ``miss_load`` is how often a load
        # actually has to be performed, which is what costs energy, and
        # staging a model early does not make fetching it free.
        if spec.proactive_loading and spec.predictor != "none":
            miss = cache_model.proactive_miss(perr)
            miss_load = np.full(g, cache_model.resident_miss)
        else:
            miss = np.full(g, cache_model.reactive_miss)
            miss_load = miss
        # Only the edge hosts a zoo that outgrows its cache.  The device runs
        # one permanently resident 50 M INT4 model, so it never cold-starts,
        # and the cloud is provisioned to keep every variant warm unless the
        # policy under test says otherwise.
        edge_t = TIERS.index("edge")
        miss_tier = np.zeros((g, N_CLASS, N_TIER))
        load_tier = np.zeros((g, N_CLASS, N_TIER))
        miss_tier[:, :, edge_t] = miss[:, None]
        load_tier[:, :, edge_t] = miss_load[:, None]
        if not spec.cloud_always_warm:
            miss_tier[:, :, 0] = miss[:, None]
            load_tier[:, :, 0] = miss_load[:, None]
        # classes whose variant is pinned resident are never cold-started
        miss_tier *= tab.cache_eligible[None, :, None]
        load_tier *= tab.cache_eligible[None, :, None]

        # -- per-request latency ---------------------------------------
        n_ct = n_req.sum(axis=1)                                  # (G, C, T)
        deterministic = (tab.aggregation_ms[None, :, None]
                         + tab.access_ms[None, :, None]
                         + tab.encode_ms * tab.encode_used[None, :, :]
                         + tab.fixed_ms[None, None, :]
                         + tab.transport_ms[None, None, :]
                         + t_inf
                         + w_mean[:, None, :])                    # (G, C, T)

        idx = rng.integers(0, ctx.ul_per_bit_ms.shape[2],
                           size=(g * N_CLASS, N_TIER * ns))
        ul_pb = np.take_along_axis(ul_flat, idx, axis=1).reshape(
            g, N_CLASS, N_TIER, ns)                               # (G,C,T,ns)
        harq = 1.0 + (rng.random((g, N_CLASS, N_TIER, ns))
                      < tab.harq_p[None, :, None, None])
        ul_ms = ul_pb * tab.payload_bits[None, :, :, None] * harq
        dl_ms = (ul_pb * tab.result_bits[None, :, None, None]
                 * not_dev[None, None, :, None])
        qjit = (rng.exponential(size=(g, N_CLASS, N_TIER, ns))
                * w_mean[:, None, :, None])
        cold = ((rng.random((g, N_CLASS, N_TIER, ns))
                 < miss_tier[:, :, :, None])
                * cfg.cold_start_penalty_ms)
        samples = (deterministic[:, :, :, None] - w_mean[:, None, :, None]
                   + qjit + ul_ms + dl_ms + cold)

        mean_ul = (ctx.ul_mean_per_bit_ms[:, :, None]
                   * tab.payload_bits[None, :, :]
                   * (1.0 + tab.harq_p[None, :, None]))
        mean_dl = (ctx.ul_mean_per_bit_ms[:, :, None]
                   * tab.result_bits[None, :, None] * not_dev[None, None, :])
        lat_mean = (deterministic + mean_ul + mean_dl
                    + miss_tier * cfg.cold_start_penalty_ms)

        # -- energy ----------------------------------------------------
        e_compute = (pue[None, None, :] * p_act[None, None, :]
                     * t_inf * 1e-3
                     * (1.0 + (p_idle / p_act)[None, None, :]
                        * (1.0 - rho_mean[:, None, :])
                        / rho_mean[:, None, :]))
        e_radio = ue_radio_w * (mean_ul + mean_dl) * 1e-3
        # A cold start also costs energy: the serving node holds the backhaul
        # transfer and the deserialisation for cold_start_penalty_ms while the
        # accelerator itself is stalled, so the host, NIC and memory subsystem
        # draw approximately the idle envelope for that whole window.
        e_load = load_tier * e_load_per_miss[None, None, :]
        e_encode = encode_j * tab.encode_used[None, :, :]
        e_req = e_compute + e_encode + e_radio + e_load

        # -- accuracy --------------------------------------------------
        acc_pct = acc_table[mode_idx,
                            np.zeros(g, dtype=np.int64)
                            if spec.fixed_alpha is not None else alpha_idx]

        # -- reward ----------------------------------------------------
        w_all = n_ct.sum(axis=(1, 2))
        lat_norm = np.sum(n_ct * lat_mean
                          / tab.deadline_ms[None, :, None], axis=(1, 2)) \
            / np.maximum(w_all, 1e-9)
        e_mean = np.sum(n_ct * e_req, axis=(1, 2)) / np.maximum(w_all, 1e-9)
        viol = np.mean(samples > tab.deadline_ms[None, :, None, None],
                       axis=(1, 2, 3))
        acc_short = np.maximum(0.0, cfg.accuracy_floor_pct - acc_pct)
        reward = -(cfg.w_latency * lat_norm
                   + cfg.w_energy * e_mean / cfg.energy_norm_j
                   + cfg.w_accuracy * (REFERENCE_ACCURACY_PCT - acc_pct)
                   / cfg.accuracy_norm_pp
                   + cfg.reward_accuracy_penalty * acc_short
                   / cfg.accuracy_norm_pp
                   + cfg.reward_sla_penalty * viol)
        controller.observe(state, action, reward, k)
        if collect_reward:
            rewards.append(reward.copy())

        # -- accumulate ------------------------------------------------
        if k >= measure_from:
            acc.count += n_ct.sum(axis=2)
            acc.lat_sum += np.sum(n_ct * lat_mean, axis=2)
            acc.energy_sum += np.sum(n_ct * e_req, axis=2)
            acc.energy_tier_sum += np.sum(n_ct * e_req, axis=1)
            acc.count_tier += n_ct.sum(axis=1)
            weights = np.repeat((n_ct / ns)[:, :, :, None], ns, axis=3)
            acc.add_latency_samples(samples, weights)
            ci_g = intensity_g_per_kwh(HOST_REGION, ctx.t_s[:, k])
            acc.carbon_sum += (np.sum(n_ct * e_req, axis=(1, 2))
                               / JOULES_PER_KWH * ci_g)
            hw = n_ct[:, HEADLINE_CLASS, :].sum(axis=1)
            acc.acc_sum += acc_pct * hw
            acc.acc_w += hw
            acc.alpha_sum += alpha[:, HEADLINE_CLASS, :] * hw[:, None]
            acc.alpha_w += hw
            # Weight the stall rate by where the requests of the headline
            # class actually go: a policy that never uses a cache-eligible
            # tier cannot suffer a cold start, and reporting the bare
            # per-tier rate for it would be misleading.
            stall = np.einsum("gt,gt->g", n_ct[:, HEADLINE_CLASS, :],
                              miss_tier[:, HEADLINE_CLASS, :])
            acc.cache_hit_sum += hw - stall
            nh = n_ct[:, HEADLINE_CLASS, :]
            for j, part in enumerate((e_compute, e_encode, e_radio, e_load)):
                acc.e_parts_sum[:, j] += np.einsum(
                    "gt,gt->g", nh, np.broadcast_to(
                        part[:, HEADLINE_CLASS, :] if part.ndim == 3
                        else part, nh.shape))
            acc.cache_w += hw
            acc.pred_err_sum += perr * hw
            acc.pred_w += hw
            acc.sim_time_s += n_slots * t_slot

        q = qs[:, -1, :]
        q[:, dev] = 0.0
        prev_alpha = alpha_g
        prev_action = action

    out = finalise(cfg, acc, spec)
    if collect_reward:
        out["reward_trace"] = np.stack(rewards, axis=1)
    return out


def finalise(cfg: Config, acc: Accumulator, spec: PolicySpec) -> dict:
    g = acc.g
    cnt = np.maximum(acc.count, 1e-9)
    lat_mean = acc.lat_sum / cnt
    energy = acc.energy_sum / cnt
    tier_energy = acc.energy_tier_sum / np.maximum(acc.count_tier, 1e-9)

    p95 = np.zeros((g, N_CLASS))
    p99 = np.zeros((g, N_CLASS))
    sla = np.zeros((g, N_CLASS))
    from .stats import weighted_percentile, weighted_tail_fraction
    deadline = np.array([sc.deadline_ms for sc in SERVICE_CLASSES])
    for i in range(g):
        for c in range(N_CLASS):
            h = acc.hist[i, c]
            p95[i, c] = weighted_percentile(acc.edges, h, 95.0)
            p99[i, c] = weighted_percentile(acc.edges, h, 99.0)
            sla[i, c] = 100.0 * weighted_tail_fraction(acc.edges, h,
                                                       deadline[c])

    total_req = acc.count.sum(axis=1)
    headline = HEADLINE_CLASS
    return {
        "ident": spec.ident,
        "name": spec.name,
        "accuracy_pct": acc.acc_sum / np.maximum(acc.acc_w, 1e-9),
        "latency_mean_ms": lat_mean[:, headline],
        "latency_p95_ms": p95[:, headline],
        "latency_p99_ms": p99[:, headline],
        "energy_j": energy[:, headline],
        "sla_violation_pct": sla[:, headline],
        "cache_hit_pct": 100.0 * acc.cache_hit_sum / np.maximum(acc.cache_w, 1e-9),
        "throughput_per_s": total_req / max(acc.sim_time_s, 1e-9),
        "carbon_g_per_1000": 1000.0 * acc.carbon_sum / np.maximum(total_req, 1e-9),
        "pred_error": acc.pred_err_sum / np.maximum(acc.pred_w, 1e-9),
        "alpha": acc.alpha_sum / np.maximum(acc.alpha_w[:, None], 1e-9),
        "tier_energy_j": tier_energy,
        "energy_parts_j": acc.e_parts_sum / np.maximum(
            acc.count[:, HEADLINE_CLASS][:, None], 1e-9),
        "per_class": {
            "latency_mean_ms": lat_mean,
            "latency_p95_ms": p95,
            "latency_p99_ms": p99,
            "sla_violation_pct": sla,
            "energy_j": energy,
            "count": acc.count,
        },
    }
