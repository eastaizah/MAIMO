"""Parametric discrete-time system-level environment for MAIMO.

Seven-cell hexagonal deployment, 30 UEs per cell, mixed URLLC/eMBB/mMTC
traffic, 3GPP UMa propagation with CDL-like fading, random-waypoint mobility,
per-node priority queues with finite buffers, a three-tier model registry with
compression variants, and per-edge-node model caches with cold-start penalties
and forecast-driven proactive pre-loading.

The slot loop is deliberately split into three phases so that an *offline
oracle* can enumerate candidate actions against exactly the same random
realisation that the online schemes see:

    begin_slot()                 # draw arrivals, channels, complexity classes
    evaluate(group, action)      # pure function: cost of a candidate action
    commit(group, action)        # apply the decision, update queues and caches

Two dimensional corrections relative to the submitted manuscript are applied
and are documented in the README:

* SNR is formed in the linear domain (``channel.linear_snr``), not by dividing
  by a dB-valued path loss.
* Inference time follows ``tau = 2 P N_tok / (Throughput * s)``, not
  ``P / FLOPS``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import energy as energy_mod
from channel import RadioEnvironment, shannon_rate_bps
from config import (BASE_ACC_LOSS_PCT, COMPLEXITY_MIX, COMPRESSIONS, MODELS,
                    N_ACTIONS, N_USE_CASES, Params, TIERS, USE_CASES,
                    UC_INDEX, decode_action, feature_layout,
                    inference_time_s, model_memory_gb, tier_flops)

# Complexity-dependent multiplier on the compression accuracy penalty.
# Quantisation and pruning hurt hard inputs far more than easy ones.
COMPLEXITY_COMP_MULT = (0.35, 1.00, 2.50)

UE_TX_W = 10.0 ** ((23.0 - 30.0) / 10.0)      # 0.2 W, for uplink energy


@dataclass
class RunConfig:
    """Switches that define a scheme or an ablation configuration."""

    name: str = "maimo"
    semantic_compression: bool = True
    proactive_loading: bool = True
    use_forecast: bool = True
    allowed_compressions: Tuple[str, ...] = ("none", "lora", "int8", "int4")
    predictor_kind: str = "bilstm"            # bilstm | lstm | none
    forecast_interval: int = 5
    mobility_interval: int = 5


# ---------------------------------------------------------------------------
# Feasibility: model <-> tier binding
# ---------------------------------------------------------------------------
def _base_feasible(p: Params) -> np.ndarray:
    """Architecturally and physically realisable (model, tier, compression).

    The device micro-model is served only at the device tier and the 7 B
    task-adapted model only at the edge (or, as a split-inference fallback, at
    the cloud): that binding is the three-layer architecture of Section 3, not
    a free choice.  Structured pruning followed by rank-16 LoRA re-adaptation to
    local channel statistics is defined in Section 3.3 as the *edge* compression
    pipeline: the cloud serves the reference model (optionally quantised for
    throughput) and Section 3.4 obtains the device variants by post-training
    integer quantisation, so ``lora`` exists only at the edge tier.  Everything
    else is ruled out by serving memory.

    Fourteen of the 36 joint actions survive.
    """
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for a in range(N_ACTIONS):
        mi, ti, ci = decode_action(a)
        m, t, c = MODELS[mi], TIERS[ti], COMPRESSIONS[ci]
        gb = model_memory_gb(m.params, c)
        if t == "device":
            ok = (m.name == "device_50M") and gb <= p.device_mem_gb
        elif t == "edge":
            ok = (m.name == "edge_7B") and gb <= p.edge_serving_mem_gb
        else:
            ok = m.name in ("edge_7B", "cloud_70B")
        mask[a] = ok and (c.name != "lora" or t == "edge")
    return mask


# ---------------------------------------------------------------------------
# Traffic seasonality
# ---------------------------------------------------------------------------
class TrafficProfile:
    """Diurnal + weekly seasonality with a bursty special-event component."""

    def __init__(self, p: Params, rng: np.random.Generator):
        self.p = p
        self.rng = rng
        self.event_left = 0
        self.event_total = 1
        # Random phase so that replications sample different parts of the day.
        self.t0_s = rng.random() * 7 * 86400.0
        # Mean-reverting local-demand process (correlated user activity).
        self.rho = math.exp(-1.0 / p.demand_ar_tau_slots)
        self.u = rng.normal()

    def clock_s(self, slot: int) -> float:
        return self.t0_s + slot * (self.p.t_slot_ms / 1000.0
                                   * self.p.sim_seconds_per_slot_scale)

    def intensity(self, slot: int) -> float:
        p = self.p
        t = self.clock_s(slot)
        hour = (t / 3600.0) % 24.0
        day = int((t / 86400.0) % 7)
        ph = 2 * math.pi * (hour - p.diurnal_peak_hour) / 24.0
        diurnal = (1.0 + p.diurnal_amplitude * math.cos(ph)
                   + p.diurnal_second_harmonic * math.cos(2 * ph))
        weekly = p.weekend_factor if day >= 5 else 1.0

        # Special-event component with a raised-cosine envelope: crowd-sourced
        # AR/video traffic ramps up and decays, it does not switch on as a step.
        if self.event_left > 0:
            frac = 1.0 - self.event_left / self.event_total
            env_shape = 0.5 * (1.0 - math.cos(2 * math.pi * frac))
            burst = 1.0 + (p.event_amplitude - 1.0) * env_shape
            self.event_left -= 1
        else:
            burst = 1.0
            if self.rng.random() < p.event_prob_per_slot:
                self.event_total = p.event_duration_slots
                self.event_left = p.event_duration_slots

        # Mean-reverting multiplicative demand factor.
        self.u = (self.rho * self.u
                  + math.sqrt(max(1.0 - self.rho ** 2, 0.0))
                  * self.rng.normal())
        local = math.exp(p.demand_ar_sigma * self.u
                         - 0.5 * p.demand_ar_sigma ** 2)
        return max(0.05, diurnal * weekly * burst * local)

    def time_features(self, slot: int) -> Tuple[float, float, float]:
        t = self.clock_s(slot)
        hour = (t / 3600.0) % 24.0
        day = int((t / 86400.0) % 7)
        return (math.sin(2 * math.pi * hour / 24.0),
                math.cos(2 * math.pi * hour / 24.0),
                1.0 if day >= 5 else 0.0)


# ---------------------------------------------------------------------------
# Priority queue with a finite buffer
# ---------------------------------------------------------------------------
class PriorityNode:
    """Work-conserving non-preemptive priority server.

    ``backlog[p]`` is the unfinished work (in seconds of service) of priority
    class ``p``.  A request of priority ``p`` waits for all work of priority
    ``<= p`` that is already queued, shared over ``streams`` parallel servers.
    """

    __slots__ = ("backlog", "streams", "capacity_s", "buffer_s", "drops")

    def __init__(self, streams: float, t_slot_s: float, buffer_slots: float,
                 n_priorities: int):
        self.backlog = np.zeros(n_priorities)
        self.streams = float(streams)
        self.capacity_s = streams * t_slot_s
        self.buffer_s = buffer_slots * self.capacity_s
        self.drops = 0

    def wait_s(self, prio: int) -> float:
        return float(self.backlog[:prio + 1].sum()) / self.streams

    def would_overflow(self, work_s: float) -> bool:
        return self.backlog.sum() + work_s > self.buffer_s

    def enqueue(self, prio: int, work_s: float) -> bool:
        if self.would_overflow(work_s):
            self.drops += 1
            return False
        self.backlog[prio] += work_s
        return True

    def drain(self) -> None:
        """Serve one slot of work, strict priority order."""
        b = self.backlog
        if b[0] == 0.0 and b.sum() == 0.0:
            return
        cum = np.cumsum(b)
        served = np.clip(self.capacity_s - (cum - b), 0.0, b)
        b -= served

    def utilisation(self) -> float:
        return float(self.backlog.sum() / self.buffer_s) if self.buffer_s else 0.0


# ---------------------------------------------------------------------------
# Edge model cache
# ---------------------------------------------------------------------------
class ModelCache:
    """Finite LRU model cache with asynchronous delta loading.

    A miss does not stall the request: following Algorithm 1 (lines 8-10) the
    orchestrator starts an asynchronous delta transfer and serves the current
    batch from the best cached variant of the same model, paying a bounded
    control-plane detour.  Only if *no* variant of the model is cached does the
    request stall for the remaining transfer time.
    """

    __slots__ = ("capacity_gb", "resident", "last_used", "loading",
                 "load_ms_per_gb", "delta_comp", "n_models", "n_comp",
                 "t_slot_ms", "misses", "requests", "loads", "n_pending")

    def __init__(self, p: Params, warm: Tuple[Tuple[int, int], ...] = ()):
        self.n_models = len(MODELS)
        self.n_comp = len(COMPRESSIONS)
        self.capacity_gb = p.edge_cache_gb
        self.resident = np.zeros((self.n_models, self.n_comp), dtype=bool)
        self.last_used = np.zeros((self.n_models, self.n_comp))
        self.loading = np.full((self.n_models, self.n_comp), -1.0)  # eta slot
        self.load_ms_per_gb = p.cold_start_ms_per_gb
        self.delta_comp = p.cache_delta_compression
        self.t_slot_ms = p.t_slot_ms
        self.misses = 0
        self.requests = 0
        self.loads = 0
        self.n_pending = 0
        # Warm baseline: the statically deployed variants the node ships with.
        for mi, ci in warm:
            self._install(mi, ci, 0.0)

    def _footprint(self, mi: int, ci: int) -> float:
        return model_memory_gb(MODELS[mi].params, COMPRESSIONS[ci])

    def load_time_ms(self, mi: int, ci: int) -> float:
        """Delta-transfer time for one variant.

        Only the sparse quantised weight delta relative to the resident base
        model is shipped (Sec. 3.3), so the transferred volume is the variant
        footprint divided by ``cache_delta_compression``.
        """
        return (self._footprint(mi, ci) / self.delta_comp
                * self.load_ms_per_gb)

    def _used_gb(self) -> float:
        idx = np.nonzero(self.resident)
        return float(sum(self._footprint(m, c) for m, c in zip(*idx)))

    def _install(self, mi: int, ci: int, now: float) -> None:
        """Make a variant resident, evicting least-recently-used variants.

        The capacity constraint is hard: variants are evicted until the new one
        fits.  A variant larger than the whole cache cannot be installed (it is
        also excluded from the action set by ``edge_serving_mem_gb``).
        """
        need = self._footprint(mi, ci)
        if need > self.capacity_gb:
            return
        while self._used_gb() + need > self.capacity_gb:
            idx = [(m, c) for m, c in zip(*np.nonzero(self.resident))
                   if (m, c) != (mi, ci)]
            if not idx:
                break
            victim = min(idx, key=lambda mc: self.last_used[mc])
            self.resident[victim] = False
        self.resident[mi, ci] = True
        self.last_used[mi, ci] = now

    def tick(self, slot: int) -> None:
        """Complete any asynchronous transfers that have finished."""
        if self.n_pending == 0:
            return
        done = (self.loading >= 0) & (self.loading <= slot)
        if done.any():
            self.n_pending -= int(done.sum())
            for mi, ci in zip(*np.nonzero(done)):
                self._install(int(mi), int(ci), float(slot))
                self.loading[mi, ci] = -1.0
                self.loads += 1

    def start_load(self, mi: int, ci: int, slot: int) -> None:
        if self.resident[mi, ci] or self.loading[mi, ci] >= 0:
            return
        n_slots = max(1.0, self.load_time_ms(mi, ci) / self.t_slot_ms)
        self.loading[mi, ci] = slot + n_slots
        self.n_pending += 1

    def resolve(self, mi: int, ci: int, slot: int,
                allowed: np.ndarray) -> Tuple[int, bool, float]:
        """Return (served compression index, was_miss, stall_ms)."""
        self.requests += 1
        if self.resident[mi, ci]:
            self.last_used[mi, ci] = slot
            return ci, False, 0.0
        self.misses += 1
        self.start_load(mi, ci, slot)
        cand = np.nonzero(self.resident[mi] & allowed)[0]
        if cand.size:
            # Fall back to the cached variant closest in compression level.
            pick = int(cand[np.argmin(np.abs(cand - ci))])
            self.last_used[mi, pick] = slot
            return pick, True, 0.0
        stall = self.load_time_ms(mi, ci)
        return ci, True, stall

    def hit_rate(self) -> float:
        return 1.0 - self.misses / max(self.requests, 1)


# ---------------------------------------------------------------------------
# Per-request outcome of a candidate action
# ---------------------------------------------------------------------------
@dataclass
class ActionOutcome:
    latency_ms: float
    energy_j: float
    energy_j_loadaware: float
    acc_loss_pct: float
    sla_met: float
    tier: str
    served_comp: int
    cold_start: float
    t_inf_s: float
    work_s: float
    node_index: int
    n_req: int
    scalar_cost: float


class MAIMOEnv:
    """The MAIMO orchestration environment."""

    def __init__(self, p: Params, seed: int, cfg: Optional[RunConfig] = None):
        self.p = p
        self.cfg = cfg or RunConfig()
        self.rng = np.random.default_rng(seed)
        self.radio = RadioEnvironment(p, np.random.default_rng(seed + 977))
        self.traffic = TrafficProfile(p, np.random.default_rng(seed + 3313))
        self.t_slot_s = p.t_slot_ms / 1000.0

        self.feasible = _base_feasible(p)
        allowed = np.zeros(len(COMPRESSIONS), dtype=bool)
        for cn in self.cfg.allowed_compressions:
            allowed[[c.name for c in COMPRESSIONS].index(cn)] = True
        self.comp_allowed = allowed
        # Which (model, compression) pairs the MEC board can actually serve.
        # Used both for cache-miss fallback and for proactive pre-loading, so
        # that neither can install a variant the edge cannot run.
        self.edge_servable = np.array(
            [[allowed[ci]
              and model_memory_gb(m.params, COMPRESSIONS[ci])
              <= p.edge_serving_mem_gb
              for ci in range(len(COMPRESSIONS))] for m in MODELS])
        for a in range(N_ACTIONS):
            _, _, ci = decode_action(a)
            if not allowed[ci]:
                self.feasible[a] = False
        self.feasible_idx = np.nonzero(self.feasible)[0]

        self.cloud = PriorityNode(p.cloud_replicas, self.t_slot_s,
                                  p.buffer_slots_cloud, p.n_priorities)
        self.edges = [PriorityNode(p.edge_streams_per_node, self.t_slot_s,
                                   p.buffer_slots_edge, p.n_priorities)
                      for _ in range(p.edge_nodes)]
        # Statically deployed warm set of each MEC node: the most accurate
        # servable variant of the edge model (the pruned + LoRA-re-adapted 7 B
        # model of Sec. 3.3) and the fastest one, shipped as the low-latency
        # fallback.  With the default registry that is LoRA (7.7 GB) + INT4
        # (3.5 GB) = 11.2 GB of the 16 GB cache, so any third variant forces an
        # eviction: this is the cache pressure proactive loading manages.
        srv = np.nonzero(self.edge_servable[1])[0]
        warm = ()
        if srv.size:
            best_acc = int(srv[np.argmin([COMPRESSIONS[c].acc_penalty_pct
                                          for c in srv])])
            fastest = int(srv[np.argmax([COMPRESSIONS[c].speedup
                                        for c in srv])])
            warm = tuple(dict.fromkeys(((1, best_acc), (1, fastest))))
        self.warm_set = warm
        self.caches = [ModelCache(p, warm) for _ in range(p.edge_nodes)]

        self._layout = feature_layout(p)
        self._cx_cdf = np.cumsum(COMPLEXITY_MIX)
        self.slot = 0
        self._epoch_acc = 0.0
        self.mean_epoch_load = (mean_slot_arrivals(p, self.radio.ue_class)
                                * p.predictor_epoch_slots)
        self.history: List[float] = [self.mean_epoch_load] * p.history_window
        self.forecast = np.zeros(p.horizon)
        self.predictor = None                     # injected by the driver
        self._demand_ema = np.zeros((len(MODELS), len(COMPRESSIONS)))

        # per-inference reference energies at the CONTRACT reference load
        self._n_ref = {"cloud": energy_mod.CLOUD.n_per_slot,
                       "edge": energy_mod.EDGE.n_per_slot,
                       "device": energy_mod.DEVICE.n_per_slot}
        # (PUE, P_act, P_idle, n_ref, extra_j) per tier, inlined for speed;
        # identical arithmetic to energy.tier_energy_j (asserted in the tests).
        self._e_par = {
            t: (q.pue, q.p_act_w, q.p_idle_w, q.n_per_slot, q.extra_j)
            for t, q in (("cloud", energy_mod.CLOUD), ("edge", energy_mod.EDGE),
                         ("device", energy_mod.DEVICE))}
        self._served_this_slot = {"cloud": 0, "edge": 0, "device": 0}

        self.t_encode_s = inference_time_s(
            p.semantic_encoder_params, p.semantic_encoder_tokens,
            p.flops_device, COMPRESSIONS[3].speedup)
        self._build_tables()

        self.stats = _Accumulator()
        self._slot_state: Dict[int, dict] = {}
        self.collecting = False

    # ------------------------------------------------------------------
    def _build_tables(self) -> None:
        """Precompute everything that depends only on (use case, action)."""
        p = self.p
        nA, nG = N_ACTIONS, N_USE_CASES
        self._decoded = [decode_action(a) for a in range(nA)]
        self._tier_of = [TIERS[self._decoded[a][1]] for a in range(nA)]
        self._t_inf = np.zeros((nG, nA))
        self._e_base = np.zeros((nG, nA))       # energy at reference load
        self._acc = np.zeros((nG, nA, 3))
        self._fixed_ms = np.zeros(nA)
        for a in range(nA):
            mi, ti, ci = self._decoded[a]
            tier = TIERS[ti]
            comp = COMPRESSIONS[ci]
            self._fixed_ms[a] = {"cloud": p.cloud_fixed_ms,
                                 "edge": p.edge_fixed_ms,
                                 "device": p.device_fixed_ms}[tier]
            pue, pa, pi, n_ref, extra = self._e_par[tier]
            for g, uc in enumerate(USE_CASES):
                t = inference_time_s(MODELS[mi].params, uc.tokens,
                                     tier_flops(p, tier), comp.speedup)
                self._t_inf[g, a] = t
                idle = max(1.0 - n_ref * t, 0.0) / n_ref
                self._e_base[g, a] = pue * (pa * t + pi * idle) + extra
                for cx in range(3):
                    self._acc[g, a, cx] = (
                        BASE_ACC_LOSS_PCT[MODELS[mi].name][cx]
                        + comp.acc_penalty_pct * COMPLEXITY_COMP_MULT[cx])
        self._enc_ms = self.t_encode_s * 1e3
        self._enc_j = energy_mod.DEVICE.p_act_w * self.t_encode_s

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------
    def base_feature_dim(self) -> int:
        return self._layout["_dim"][0]

    def feature_dim(self) -> int:
        return self.base_feature_dim() * (1 + N_USE_CASES)

    def expand(self, f: np.ndarray) -> np.ndarray:
        """Tensor the observation with the use-case one-hot.

        The controllers are linear, so a single shared weight per (feature,
        action) pair forces one trade-off on all four use cases at once.  Their
        deadlines differ by two orders of magnitude, so the shared weights are
        dominated by the highest-rate use case and the others can only be
        corrected through their own one-hot.  Concatenating the observation with
        (use-case one-hot) x (observation) gives each use case its own block of
        parameters while keeping the shared block for statistical strength; the
        policy remains linear, in a product basis.
        """
        i, w = self._layout["use_case"]
        return np.concatenate((f, np.outer(f[i:i + w], f).ravel()))

    def features(self, g: int) -> np.ndarray:
        st = self._slot_state[g]
        p = self.p
        L = self._layout
        f = np.zeros(self.base_feature_dim())
        prio = USE_CASES[g].priority - 1
        node = st["node"]
        f[L["use_case"][0] + g] = 1.0
        f[L["complexity"][0] + st["complexity"]] = 1.0
        f[L["uc_x_cx"][0] + g * 3 + st["complexity"]] = 1.0
        f[L["arrivals"][0]] = min(st["n"] / 8.0, 2.0)
        f[L["cloud_util"][0]] = self.cloud.utilisation()
        f[L["edge_util"][0]] = self._mean_edge_util()
        f[L["device_busy"][0]] = st["device_busy"]
        f[L["cloud_wait"][0]] = self.cloud.wait_s(prio) / self.t_slot_s
        f[L["edge_wait"][0]] = self.edges[node].wait_s(prio) / self.t_slot_s
        f[L["snr_mean"][0]] = min(st["snr_db_mean"] / 30.0, 2.0)
        f[L["snr_low"][0]] = min(st["snr_db_p10"] / 30.0, 2.0)
        i, w = L["cache"]
        f[i:i + w] = self.caches[node].resident[1]
        i, w = L["forecast"]
        if self.cfg.use_forecast:
            f[i:i + w] = np.clip(self.forecast, 0.0, 4.0)
        i, _ = L["time"]
        f[i], f[i + 1], f[i + 2] = self.traffic.time_features(self.slot)
        f[L["bias"][0]] = 1.0
        return self.expand(f)

    def _mean_edge_util(self) -> float:
        s = 0.0
        for e in self.edges:
            s += e.utilisation()
        return s / len(self.edges)

    # ------------------------------------------------------------------
    # slot lifecycle
    # ------------------------------------------------------------------
    def begin_slot(self) -> List[int]:
        """Draw this slot's randomness and return the active decision groups."""
        p = self.p
        if self.slot % self.cfg.mobility_interval == 0:
            self.radio.step_mobility(self.cfg.mobility_interval)
        snr = self.radio.snr_linear()
        rate = shannon_rate_bps(snr, p.bandwidth_hz)
        lam = self.traffic.intensity(self.slot)
        for c in self.caches:
            c.tick(self.slot)
        self._served_this_slot = {"cloud": 0, "edge": 0, "device": 0}

        self._slot_state = {}
        active: List[int] = []
        total_arrivals = 0.0
        for g, uc in enumerate(USE_CASES):
            if uc.population == "all":
                pop = np.arange(self.radio.n)
            else:
                cls = {"urllc": 0, "embb": 1, "mmtc": 2}[uc.population]
                pop = np.nonzero(self.radio.ue_class == cls)[0]
            # Independent per-UE Poisson counts at rate mu have a Poisson sum,
            # and conditioned on that sum the requests are spread uniformly and
            # independently over the population.  Drawing the total once and
            # then the originating UEs uniformly is therefore an exact sample of
            # the same process at a fraction of the cost of one Poisson draw per
            # UE per slot.
            mu = uc.rate_per_ue_s * lam * self.t_slot_s * pop.size
            n = int(self.rng.poisson(mu))
            total_arrivals += n
            if n == 0:
                continue
            ue = pop[self.rng.integers(0, pop.size, n)]
            cell = self.radio.cell_of_ue[ue]
            if uc.fixed_complexity is not None:
                comp = uc.fixed_complexity
            else:
                comp = int(np.searchsorted(self._cx_cdf, self.rng.random()))
            snr_db = 10.0 * np.log10(np.maximum(snr[ue], 1e-12))
            m = float(snr_db.mean())
            counts = np.bincount(cell, minlength=p.n_cells)
            if uc.bandwidth_frac >= 1.0:
                r_ue = rate[ue]
            else:
                # Narrowband allocation: the UE concentrates its transmit power
                # in a fraction ``f`` of the carrier, so the per-Hz SNR rises by
                # 1/f while the usable bandwidth falls by f.
                f = uc.bandwidth_frac
                r_ue = shannon_rate_bps(snr[ue] / f, p.bandwidth_hz * f)
            st = {
                "n": n,
                "ue": ue,
                "cell": cell,
                "cell_counts": counts,
                "rate": r_ue,
                "node": int(counts.argmax()),
                "complexity": comp,
                "snr_db_mean": m,
                "snr_db_p10": m - float(snr_db.std()),
                "tti_offset": self.rng.random(n) * uc.tti_ms,
                "wan_jitter": self.rng.exponential(p.wan_jitter_ms, n),
                "device_busy": min(n / max(pop.size, 1) * 10.0, 1.0),
            }
            r = np.maximum(r_ue, 1e3)
            t_up_sem = uc.payload_semantic_bits / r * 1e3
            t_up_raw = uc.payload_raw_bits / r * 1e3
            st["t_up"] = (t_up_sem, t_up_raw)
            st["up_j"] = (float(UE_TX_W * t_up_sem.mean() / 1e3),
                          float(UE_TX_W * t_up_raw.mean() / 1e3))
            self._slot_state[g] = st
            active.append(g)

        # The predictor observes the offered load aggregated over an
        # *orchestration epoch* of ``predictor_epoch_slots`` radio slots, not the
        # raw per-slot Poisson count: at 10 ms granularity the count is almost
        # pure shot noise (Var = mean) and a 5-slot (50 ms) forecast horizon is
        # shorter than a single model cold start.  See README, "Predictor
        # timescale".
        self._epoch_acc += total_arrivals
        if (self.slot + 1) % p.predictor_epoch_slots == 0:
            self.history.append(self._epoch_acc)
            self._epoch_acc = 0.0
            if self.predictor is not None and self.cfg.use_forecast:
                self.forecast = self.predictor.predict(
                    np.asarray(self.history[-p.history_window:], dtype=float))
            if self.cfg.proactive_loading:
                self._proactive_preload()
        return active

    # ------------------------------------------------------------------
    def evaluate(self, g: int, action: int, apply_cache: bool = False
                 ) -> ActionOutcome:
        """Exact latency/energy/accuracy of ``action`` for group ``g``.

        Pure with respect to queues and statistics unless ``apply_cache`` is
        set, in which case the cache LRU state and miss counters are updated
        (used by :meth:`commit`).
        """
        p = self.p
        uc = USE_CASES[g]
        st = self._slot_state[g]
        mi, ti, ci = self._decoded[action]
        tier = self._tier_of[action]
        node = st["node"]
        prio = uc.priority - 1

        cold = 0.0
        served_ci = ci
        miss = False
        if tier == "edge":
            if apply_cache:
                served_ci, miss, cold = self.caches[node].resolve(
                    mi, ci, self.slot, self.edge_servable[mi])
            else:
                if self.caches[node].resident[mi, ci]:
                    served_ci = ci
                else:
                    miss = True
                    cand = np.nonzero(self.caches[node].resident[mi]
                                      & self.edge_servable[mi])[0]
                    if cand.size:
                        served_ci = int(cand[np.argmin(np.abs(cand - ci))])
                    else:
                        cold = self.caches[node].load_time_ms(mi, ci)
            if miss:
                cold += p.cold_start_detour_ms

        served_action = action + (served_ci - ci)
        t_inf = self._t_inf[g, served_action]

        # ---- communication ------------------------------------------------
        if tier == "device":
            t_air = 0.0
            up_j = p.semantic_uplink_energy_j
            enc_j = 0.0
            t_enc = 0.0
        else:
            t_enc = self._enc_ms if self.cfg.semantic_compression else 0.0
            t_air = uc.ran_proc_ms + st["tti_offset"] + st["t_up"][
                0 if self.cfg.semantic_compression else 1]
            up_j = st["up_j"][0 if self.cfg.semantic_compression else 1]
            enc_j = self._enc_j if self.cfg.semantic_compression else 0.0

        t_transport = 0.0
        if tier == "cloud":
            t_transport = p.wan_rtt_ms + st["wan_jitter"]
            if not self.cfg.semantic_compression:
                t_transport += (uc.payload_raw_bits / (p.backhaul_gbps * 1e9)
                                * 1e3)
        t_fixed = self._fixed_ms[served_action]

        # ---- queueing -----------------------------------------------------
        if tier == "cloud":
            t_queue = self.cloud.wait_s(prio) * 1e3
        elif tier == "edge":
            # Requests are served by the MEC server of their own cell; the
            # group's mean waiting time is the request-count-weighted mean.
            cc = st["cell_counts"]
            tot = 0.0
            for i in np.nonzero(cc)[0]:
                tot += cc[i] * self.edges[i].wait_s(prio)
            t_queue = tot / st["n"] * 1e3
        else:
            t_queue = max(0.0, (st["n"] / max(len(st["ue"]), 1) - 1.0)) * \
                t_inf * 1e3

        base = (uc.agg_window_ms + t_enc + t_transport + t_queue
                + t_inf * 1e3 + t_fixed + cold)
        latency = base + t_air              # scalar or (n,) array

        # ---- accuracy -----------------------------------------------------
        cx = st["complexity"]
        acc = self._acc[g, served_action, cx]

        # ---- energy -------------------------------------------------------
        pue, pa, pi, n_ref, extra = self._e_par[tier]
        e_ref = self._e_base[g, served_action] + up_j + enc_j
        served_now = max(max(self._served_this_slot[tier], 1) / self.t_slot_s,
                         1.0)
        idle = max(1.0 - served_now * t_inf, 0.0) / served_now
        e_load = pue * (pa * t_inf + pi * idle) + extra + up_j + enc_j

        if np.isscalar(latency):
            sla = 1.0 if latency <= uc.deadline_ms else 0.0
            lat_mean = float(latency)
        else:
            sla = float(np.count_nonzero(latency <= uc.deadline_ms)
                        / latency.size)
            lat_mean = float(latency.mean())
        cost = (p.alpha[0] * min(lat_mean / uc.deadline_ms, 3.0)
                + p.alpha[1] * min(e_ref / p.energy_norm_j, 6.0)
                + p.alpha[2] * min(acc / p.acc_norm_pct, 6.0)
                - p.sla_bonus * sla)

        return ActionOutcome(
            latency_ms=lat_mean, energy_j=e_ref, energy_j_loadaware=e_load,
            acc_loss_pct=acc, sla_met=sla, tier=tier, served_comp=served_ci,
            cold_start=1.0 if miss else 0.0, t_inf_s=t_inf,
            work_s=t_inf * st["n"], node_index=node, n_req=st["n"],
            scalar_cost=cost)

    # ------------------------------------------------------------------
    def commit(self, g: int, action: int) -> ActionOutcome:
        out = self.evaluate(g, action, apply_cache=True)
        st = self._slot_state[g]
        uc = USE_CASES[g]
        prio = uc.priority - 1
        admitted = True
        if out.tier == "cloud":
            admitted = self.cloud.enqueue(prio, out.work_s)
        elif out.tier == "edge":
            ok = 0
            cc = st["cell_counts"]
            for i in np.nonzero(cc)[0]:
                ok += cc[i] * self.edges[i].enqueue(prio, cc[i] * out.t_inf_s)
            admitted = ok >= 0.5 * st["n"]
        self._served_this_slot[out.tier] += st["n"]
        mi, ti, ci = self._decoded[action]
        self._demand_ema *= 0.995
        if TIERS[ti] == "edge":
            self._demand_ema[mi, ci] += st["n"]
        if self.collecting:
            self.stats.add(g, out, admitted, uc)
        return out

    def end_slot(self) -> None:
        self.cloud.drain()
        for e in self.edges:
            e.drain()
        self.slot += 1

    # ------------------------------------------------------------------
    def _proactive_preload(self) -> None:
        """Forecast-driven pre-loading of the variants about to be demanded.

        Called once per orchestration epoch.  Pre-loading is triggered when the
        forecast demand over the next ``H`` epochs exceeds the recently observed
        demand, which is exactly the condition under which a reactive scheme
        would incur cold starts.
        """
        growth = 1.0
        if self.cfg.use_forecast and self.history:
            recent = float(np.mean(self.history[-self.p.history_window:]))
            if recent > 1e-9:
                growth = float(np.mean(self.forecast)
                               * self.mean_epoch_load / recent)
        if growth < self.p.preload_trigger:
            return
        score = self._demand_ema[1].copy()          # edge_7B variants
        score[~self.edge_servable[1]] = -1.0
        order = np.argsort(-score)
        for c in self.caches:
            for ci in order[:2]:
                if score[ci] > 0:
                    c.start_load(1, int(ci), self.slot)

    # ------------------------------------------------------------------
    def start_collecting(self) -> None:
        self.collecting = True
        self.stats = _Accumulator()

    def results(self) -> dict:
        r = self.stats.summary()
        hit = np.mean([c.hit_rate() for c in self.caches])
        r["cache_hit_rate"] = float(hit)
        r["cold_start_rate_pct"] = float(100.0 * (1.0 - hit))
        r["cloud_drops"] = self.cloud.drops
        r["edge_drops"] = int(sum(e.drops for e in self.edges))
        return r


# ---------------------------------------------------------------------------
class _Accumulator:
    """Streaming accumulation of per-use-case and aggregate metrics."""

    MAIN_MIX = tuple(uc.key for uc in USE_CASES if uc.in_main_mix)

    def __init__(self):
        self.n = np.zeros(N_USE_CASES)
        self.lat = np.zeros(N_USE_CASES)
        self.lat2 = np.zeros(N_USE_CASES)
        self.energy = np.zeros(N_USE_CASES)
        self.energy_load = np.zeros(N_USE_CASES)
        self.acc = np.zeros(N_USE_CASES)
        self.met = np.zeros(N_USE_CASES)
        self.cold = np.zeros(N_USE_CASES)
        self.dropped = np.zeros(N_USE_CASES)
        self.tier_count = np.zeros((N_USE_CASES, 3))
        self.comp_count = np.zeros((N_USE_CASES, len(COMPRESSIONS)))

    def add(self, g: int, out: ActionOutcome, admitted: bool, uc) -> None:
        w = out.n_req
        self.n[g] += w
        self.lat[g] += out.latency_ms * w
        self.lat2[g] += out.latency_ms ** 2 * w
        self.energy[g] += out.energy_j * w
        self.energy_load[g] += out.energy_j_loadaware * w
        self.acc[g] += out.acc_loss_pct * w
        self.met[g] += (out.sla_met if admitted else 0.0) * w
        self.cold[g] += out.cold_start * w
        if not admitted:
            self.dropped[g] += w
        self.tier_count[g, TIERS.index(out.tier)] += w
        self.comp_count[g, out.served_comp] += w

    def summary(self) -> dict:
        n = np.maximum(self.n, 1e-9)
        per_uc = {}
        for g, uc in enumerate(USE_CASES):
            per_uc[uc.key] = {
                "n_requests": float(self.n[g]),
                "latency_ms": float(self.lat[g] / n[g]),
                "energy_j": float(self.energy[g] / n[g]),
                "energy_j_loadaware": float(self.energy_load[g] / n[g]),
                "acc_loss_pct": float(self.acc[g] / n[g]),
                "sla_violation_pct": float(100.0 * (1.0 - self.met[g] / n[g])),
                "cold_start_pct": float(100.0 * self.cold[g] / n[g]),
                "drop_pct": float(100.0 * self.dropped[g] / n[g]),
                "tier_split": (self.tier_count[g] / n[g]).tolist(),
            }
        mask = np.array([uc.key in self.MAIN_MIX for uc in USE_CASES])
        nm = max(self.n[mask].sum(), 1e-9)
        na = max(self.n.sum(), 1e-9)
        return {
            "per_use_case": per_uc,
            "latency_ms": float(self.lat[mask].sum() / nm),
            "energy_j": float(self.energy.sum() / na),
            "energy_j_loadaware": float(self.energy_load.sum() / na),
            "acc_loss_pct": float(self.acc.sum() / na),
            "sla_violation_pct": float(100.0 * (1.0 - self.met[mask].sum() / nm)),
            "tier_split": (self.tier_count.sum(0) / na).tolist(),
            "comp_split": (self.comp_count.sum(0) / na).tolist(),
            "n_requests": float(na),
        }


# ---------------------------------------------------------------------------
def mean_slot_arrivals(p: Params, ue_class: np.ndarray) -> float:
    """Expected total request arrivals per radio slot at unit seasonality."""
    t = p.t_slot_ms / 1000.0
    tot = 0.0
    for uc in USE_CASES:
        if uc.population == "all":
            n = ue_class.size
        else:
            n = int((ue_class == {"urllc": 0, "embb": 1,
                                  "mmtc": 2}[uc.population]).sum())
        tot += uc.rate_per_ue_s * n * t
    return tot


def generate_traffic_trace(p: Params, seed: int, n_epochs: int) -> np.ndarray:
    """Per-epoch arrival-count trace used to train the traffic predictor.

    Only the traffic process is exercised (seasonality x Poisson arrivals), not
    the full radio/queue/cache machinery, because the predictor observes nothing
    else.  This keeps trace generation to a fraction of a second per seed.
    """
    rng = np.random.default_rng(seed + 8821)
    prof = TrafficProfile(p, np.random.default_rng(seed + 3313))
    radio_classes = np.repeat(
        np.concatenate([np.full(int(round(f * p.ue_per_cell)), i)
                        for i, f in enumerate(p.traffic_mix)]), p.n_cells)
    base = mean_slot_arrivals(p, radio_classes)
    k = p.predictor_epoch_slots
    out = np.empty(n_epochs)
    slot = 0
    for e in range(n_epochs):
        tot = 0.0
        for _ in range(k):
            tot += rng.poisson(base * prof.intensity(slot))
            slot += 1
        out[e] = tot
    return out
