"""The baseline family, plus the shared training and evaluation loops.

Nine schemes are compared (editor requirement R5):

1. ``cloud_only``      centralised inference, no edge layer
2. ``edge_only``       static edge deployment, no dynamic model selection
3. ``random_greedy``   cloud by default, random non-cloud tier when the
                       estimated latency would miss the deadline
4. ``threshold``       serve at the edge while the edge queue is below a tuned
                       threshold, otherwise offload to the cloud
5. ``edf``             deadline-aware earliest-deadline-first scheduling with
                       static model selection
6. ``lyapunov``        drift-plus-penalty online control with a tuned ``V``
7. ``dqn``             reactive value-based DRL, no traffic forecast
8. ``maimo``           proposed: BiLSTM forecast + PPO + proactive loading +
                       compression
9. ``oracle``          per-slot optimal assignment computed with full knowledge
                       of the current slot's arrivals and channel realisations.
                       **This is a bound, not an achievable scheme.**

The oracle enumerates all 14 physically realisable actions against exactly the
same random realisation the online schemes see, and picks the exact minimiser of
the scalarised objective.  Because the per-slot decision is a choice of one
action per decision group with no coupling *within* the slot, exhaustive
enumeration is the exact solution of the per-slot assignment problem; a
``scipy.optimize.linear_sum_assignment`` formulation would return the same
values at higher cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from config import (COMPRESSIONS, MODELS, N_ACTIONS, Params, TIERS, USE_CASES,
                    encode_action)
from controller import DQNController, PPOController, RuleBasedController
from env import MAIMOEnv, RunConfig, generate_traffic_trace
import predictor as predictor_mod


def _act_index(model: str, tier: str, comp: str) -> int:
    return encode_action([m.name for m in MODELS].index(model),
                         list(TIERS).index(tier),
                         [c.name for c in COMPRESSIONS].index(comp))


A_CLOUD_FULL = _act_index("cloud_70B", "cloud", "none")
A_CLOUD_INT8 = _act_index("cloud_70B", "cloud", "int8")
A_EDGE_LORA = _act_index("edge_7B", "edge", "lora")
A_EDGE_INT4 = _act_index("edge_7B", "edge", "int4")
A_EDGE_INT8 = _act_index("edge_7B", "edge", "int8")
A_DEVICE_INT4 = _act_index("device_50M", "device", "int4")


# ---------------------------------------------------------------------------
# Scheme definitions
# ---------------------------------------------------------------------------
@dataclass
class Scheme:
    key: str
    label: str
    cfg: RunConfig
    kind: str                      # fixed | policy | heuristic | oracle
    is_bound: bool = False
    needs_training: Optional[str] = None      # None | "ppo" | "dqn"
    note: str = ""


def scheme_registry(p: Params) -> Dict[str, Scheme]:
    return {
        "cloud_only": Scheme(
            "cloud_only", "Cloud-only centralised inference",
            RunConfig(name="cloud_only", semantic_compression=False,
                      proactive_loading=False, use_forecast=False),
            "fixed",
            note="all requests served by the 70 B FP16 cloud model; raw "
                 "payload upload, since the semantic encoder/decoder pair is "
                 "co-designed with the edge tier"),
        "edge_only": Scheme(
            "edge_only", "Edge-only static deployment",
            RunConfig(name="edge_only", proactive_loading=False,
                      use_forecast=False,
                      allowed_compressions=("lora",)),
            "fixed",
            note="single statically deployed 7 B LoRA variant, no dynamic "
                 "model selection, no cloud fallback"),
        "random_greedy": Scheme(
            "random_greedy", "Random-tier greedy offloading",
            RunConfig(name="random_greedy", proactive_loading=False,
                      use_forecast=False),
            "heuristic",
            note="cloud by default; on a predicted deadline miss the tier is "
                 "drawn uniformly from the non-cloud tiers"),
        "threshold": Scheme(
            "threshold", "Threshold-based offloading heuristic",
            RunConfig(name="threshold", proactive_loading=False,
                      use_forecast=False),
            "heuristic",
            note="serve at the edge while the edge queue occupancy is below a "
                 "grid-search-tuned threshold, otherwise offload to the cloud"),
        "edf": Scheme(
            "edf", "Deadline-aware EDF with static model selection",
            RunConfig(name="edf", proactive_loading=False, use_forecast=False),
            "heuristic",
            note="decision groups served in earliest-deadline order; the "
                 "cheapest tier that still meets the deadline is selected, "
                 "with one fixed model variant per tier"),
        "lyapunov": Scheme(
            "lyapunov", "Lyapunov drift-plus-penalty controller",
            RunConfig(name="lyapunov", proactive_loading=False,
                      use_forecast=False),
            "heuristic",
            note="minimises Q * (queue drift) + V * (penalty) per slot with a "
                 "grid-search-tuned V"),
        "dqn": Scheme(
            "dqn", "DQN reactive controller (no forecast)",
            RunConfig(name="dqn", proactive_loading=False, use_forecast=False,
                      predictor_kind="none"),
            "policy", needs_training="dqn",
            note="linear Q-learning with epsilon-greedy exploration, replay "
                 "and a target network; reacts to the observed state only"),
        "maimo": Scheme(
            "maimo", "MAIMO (proposed)",
            RunConfig(name="maimo"),
            "policy", needs_training="ppo",
            note="BiLSTM forecast + PPO orchestration + proactive model "
                 "pre-loading + compression-aware model selection"),
        "oracle": Scheme(
            "oracle", "Offline oracle (bound, not achievable)",
            RunConfig(name="oracle"),
            "oracle", is_bound=True,
            note="per-slot exact minimiser of the scalarised objective with "
                 "full knowledge of the current slot's arrivals and channel "
                 "realisations; an upper bound on achievable performance"),
    }


# ---------------------------------------------------------------------------
# Decision functions
# ---------------------------------------------------------------------------
class HeuristicPolicy:
    """Stateless decision rules that read the environment directly."""

    def __init__(self, key: str, p: Params, feasible: np.ndarray, seed: int,
                 threshold: float = 0.5, lyapunov_v: float = 1.0):
        self.key = key
        self.p = p
        self.feasible = np.nonzero(feasible)[0]
        self.rng = np.random.default_rng(seed + 2027)
        self.threshold = threshold
        self.v = lyapunov_v
        self.by_tier = [[int(a) for a in self.feasible
                         if TIERS[(a // 4) % 3] == t] for t in TIERS]

    def decide(self, env: MAIMOEnv, g: int) -> int:
        if self.key == "cloud_only":
            return A_CLOUD_FULL
        if self.key == "edge_only":
            return A_EDGE_LORA
        if self.key == "random_greedy":
            # The serving tier is drawn uniformly (no orchestration
            # intelligence); within the drawn tier the cheapest feasible
            # variant is taken greedily.
            cand = self.by_tier[int(self.rng.integers(len(TIERS)))]
            return min(cand, key=lambda a: env.evaluate(g, a).scalar_cost)
        if self.key == "threshold":
            # Offload to the cloud once the edge queueing delay would consume
            # more than a tuned fraction of the request's deadline.
            st = env._slot_state[g]
            prio = USE_CASES[g].priority - 1
            wait_ms = env.edges[st["node"]].wait_s(prio) * 1e3
            if wait_ms < self.threshold * USE_CASES[g].deadline_ms:
                return A_EDGE_LORA
            return A_CLOUD_FULL
        if self.key == "edf":
            # Static model selection: one fixed variant per tier.  Deadline
            # awareness means spending the available slack on accuracy, so the
            # most capable variant that still fits the deadline is chosen and
            # the scheme degrades to smaller models only under time pressure.
            for a in (A_CLOUD_FULL, A_EDGE_LORA, A_DEVICE_INT4):
                if env.evaluate(g, a).latency_ms <= USE_CASES[g].deadline_ms:
                    return a
            return A_DEVICE_INT4
        if self.key == "lyapunov":
            # Drift-plus-penalty.  Minimise
            #     Q_n * (work added to node n)  +  V * penalty,
            # with queues and work expressed in slots of service so the drift is
            # dimensionless.  Delay is controlled by the drift term, as in the
            # classical formulation, so the penalty carries only the remaining
            # objectives (energy and accuracy); the controller is therefore not
            # a re-parameterisation of the scalarised cost.
            p = self.p
            best, best_j = None, np.inf
            for a in self.feasible:
                out = env.evaluate(g, int(a))
                if out.tier == "cloud":
                    node = env.cloud
                elif out.tier == "edge":
                    node = env.edges[out.node_index]
                else:
                    node = None
                if node is None:
                    drift = 0.0
                else:
                    q = node.backlog.sum() / node.capacity_s
                    drift = q * out.work_s / node.capacity_s
                pen = (p.alpha[1] * out.energy_j / p.energy_norm_j
                       + p.alpha[2] * out.acc_loss_pct / p.acc_norm_pct)
                j = drift + self.v * pen
                if j < best_j:
                    best_j, best = j, int(a)
            return best
        raise KeyError(self.key)


class OraclePolicy:
    """Exact per-slot minimiser of the scalarised objective (a bound)."""

    def __init__(self, p: Params, feasible: np.ndarray):
        self.feasible = np.nonzero(feasible)[0]

    def decide(self, env: MAIMOEnv, g: int) -> int:
        best, best_c = int(self.feasible[0]), np.inf
        for a in self.feasible:
            c = env.evaluate(g, int(a)).scalar_cost
            if c < best_c:
                best_c, best = c, int(a)
        return best


class PolicyWrapper:
    """Adapts a learned controller to the ``decide(env, g)`` interface."""

    def __init__(self, ctrl):
        self.ctrl = ctrl

    def decide(self, env: MAIMOEnv, g: int) -> int:
        a, _ = self.ctrl.act(env.features(g), greedy=True)
        return a


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _reward(out) -> float:
    return -out.scalar_cost


def train_ppo(p: Params, seed: int, cfg: RunConfig, predictor,
              total_episodes: Optional[int] = None,
              curve_bins: int = 200) -> Tuple[PPOController, np.ndarray]:
    """Train the PPO controller; return it and the episodic-return curve."""
    total_episodes = total_episodes or p.ppo_total_episodes
    env = MAIMOEnv(p, seed + 50000, cfg)
    env.predictor = predictor
    ctrl = PPOController(p, env.feature_dim(), env.feasible, seed)

    obs: List[np.ndarray] = []
    acts: List[int] = []
    logps: List[float] = []
    rews: List[float] = []
    dones: List[float] = []
    groups: List[int] = []
    curve = np.zeros(curve_bins)
    counts = np.zeros(curve_bins)
    ep_in_rollout = 0

    for ep in range(total_episodes):
        ctrl.set_progress(ep / max(total_episodes - 1, 1))
        ep_ret = 0.0
        for s in range(p.ppo_episode_slots):
            active = env.begin_slot()
            last = (s == p.ppo_episode_slots - 1)
            for g in active:
                x = env.features(g)
                a, lp = ctrl.act(x)
                out = env.commit(g, a)
                r = _reward(out)
                obs.append(x)
                acts.append(a)
                logps.append(lp)
                rews.append(r)
                dones.append(1.0 if last else 0.0)
                groups.append(g)
                ep_ret += r
            env.end_slot()
        b = min(int(ep / total_episodes * curve_bins), curve_bins - 1)
        curve[b] += ep_ret
        counts[b] += 1
        ctrl.returns.append(ep_ret)
        ep_in_rollout += 1
        if ep_in_rollout >= p.ppo_rollout_episodes and obs:
            ctrl.update(np.array(obs), np.array(acts), np.array(logps),
                        np.array(rews), np.array(dones), np.array(groups))
            obs, acts, logps, rews, dones, groups = [], [], [], [], [], []
            ep_in_rollout = 0
    return ctrl, curve / np.maximum(counts, 1)


def train_dqn(p: Params, seed: int, cfg: RunConfig, predictor,
              total_episodes: Optional[int] = None
              ) -> Tuple[DQNController, np.ndarray]:
    total_episodes = total_episodes or p.dqn_total_episodes
    env = MAIMOEnv(p, seed + 50000, cfg)
    env.predictor = predictor
    ctrl = DQNController(p, env.feature_dim(), env.feasible, seed)
    curve = []
    for ep in range(total_episodes):
        ctrl.decay_epsilon(ep)
        ep_ret = 0.0
        # One transition chain per decision group, for the same reason PPO
        # accumulates GAE per group: successive decisions of *different* use
        # cases are not successive states of one Markov chain.
        prev: Dict[int, Tuple[np.ndarray, int, float]] = {}
        for s in range(p.ppo_episode_slots):
            active = env.begin_slot()
            for g in active:
                x = env.features(g)
                a, _ = ctrl.act(x)
                out = env.commit(g, a)
                r = _reward(out)
                if g in prev:
                    ctrl.store(prev[g][0], prev[g][1], prev[g][2], x, 0.0)
                prev[g] = (x, a, r)
                ctrl.train_step()
                ep_ret += r
            env.end_slot()
        for pv in prev.values():
            ctrl.store(pv[0], pv[1], pv[2], pv[0], 1.0)
        curve.append(ep_ret)
    ctrl.returns = curve
    return ctrl, np.array(curve)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(p: Params, seed: int, cfg: RunConfig, policy, predictor=None,
             warmup: Optional[int] = None, slots: Optional[int] = None) -> dict:
    """Run one replication and return its summary metrics."""
    warmup = p.warmup_slots if warmup is None else warmup
    slots = p.eval_slots if slots is None else slots
    env = MAIMOEnv(p, seed, cfg)
    env.predictor = predictor
    for _ in range(warmup):
        for g in env.begin_slot():
            env.commit(g, policy.decide(env, g))
        env.end_slot()
    env.start_collecting()
    for _ in range(slots):
        for g in env.begin_slot():
            env.commit(g, policy.decide(env, g))
        env.end_slot()
    r = env.results()
    r["seed"] = seed
    r["warmup_slots"] = warmup
    r["eval_slots"] = slots
    return r


# ---------------------------------------------------------------------------
# Baseline hyper-parameter tuning (reported in the paper)
# ---------------------------------------------------------------------------
def tune_threshold(p: Params, seeds=(1, 2, 3), slots: int = 700) -> Tuple[float, list]:
    reg = scheme_registry(p)["threshold"]
    trace = []
    for th in p.threshold_grid:
        cost = 0.0
        for s in seeds:
            env_pol = HeuristicPolicy("threshold", p,
                                      MAIMOEnv(p, s, reg.cfg).feasible, s,
                                      threshold=th)
            r = evaluate(p, s, reg.cfg, env_pol, warmup=150, slots=slots)
            cost += _objective(p, r)
        trace.append((th, cost / len(seeds)))
    best = min(trace, key=lambda t: t[1])[0]
    return best, trace


def tune_lyapunov(p: Params, seeds=(1, 2, 3), slots: int = 700) -> Tuple[float, list]:
    reg = scheme_registry(p)["lyapunov"]
    trace = []
    for v in p.lyapunov_v_grid:
        cost = 0.0
        for s in seeds:
            pol = HeuristicPolicy("lyapunov", p,
                                  MAIMOEnv(p, s, reg.cfg).feasible, s,
                                  lyapunov_v=v)
            r = evaluate(p, s, reg.cfg, pol, warmup=150, slots=slots)
            cost += _objective(p, r)
        trace.append((v, cost / len(seeds)))
    best = min(trace, key=lambda t: t[1])[0]
    return best, trace


def _objective(p: Params, r: dict) -> float:
    """The scalarised objective used for baseline tuning (same weights)."""
    return (p.alpha[0] * r["latency_ms"] / 30.0
            + p.alpha[1] * r["energy_j"] / p.energy_norm_j
            + p.alpha[2] * r["acc_loss_pct"] / p.acc_norm_pct
            + 0.01 * r["sla_violation_pct"])


# ---------------------------------------------------------------------------
def build_policy(scheme: Scheme, p: Params, seed: int, feasible: np.ndarray,
                 controller=None, threshold: float = 0.5,
                 lyapunov_v: float = 1.0):
    if scheme.kind == "oracle":
        return OraclePolicy(p, feasible)
    if scheme.kind == "policy":
        return PolicyWrapper(controller)
    return HeuristicPolicy(scheme.key, p, feasible, seed, threshold, lyapunov_v)


def make_predictor(kind: str, p: Params, seed: int):
    if kind == "none":
        return predictor_mod.build_predictor("none", p, seed)
    trace = generate_traffic_trace(p, seed, p.predictor_train_epochs_data)
    return predictor_mod.build_predictor(kind, p, seed).fit(trace)
