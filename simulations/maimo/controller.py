"""Orchestration controllers.

All controllers expose the same two-method interface used by
:func:`maimo.sim.run_batch`::

    act(state: (G, d) -> action: (G,) int64)
    observe(state, action, reward, k) -> None

and all of them are *grouped*: the ``G`` replications carry independent
parameters and independent learning state, so batching them into one tensor
does not couple the seeds.  The learned controllers (PPO, DQN, LinUCB) are
trained on a training trace and then frozen for the evaluation run, so the
reported numbers are out-of-sample.

State layout (14 features), produced by :func:`maimo.sim.run_batch`:

===== =========================================================
index feature
===== =========================================================
0     predicted aggregate load over the next horizon / reference
1     current aggregate load / reference
2-5   share of the offered load in each service class
6-7   cloud and edge backlog, normalised by the node count
8-9   sine and cosine of the time of day
10-12 previous routing split (cloud, edge, device)
13    normalised prediction error of the last interval
===== =========================================================
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .config import (Config, SERVICE_CLASSES, HEADLINE_CLASS, N_CLASS,
                     N_TIER)
from .energy import reference_tier_energies_j
from .models import expected_accuracy
from .sim import (ALPHA_CODEBOOK, COMPRESSION_MODES, N_ACTION, N_ALPHA,
                  per_class_alpha, sakasegawa_wait_ms)

STATE_DIM = 14


# ---------------------------------------------------------------------------
# Analytic model shared by the non-learning heuristics
# ---------------------------------------------------------------------------
class HeuristicModel:
    """The internal model a rule-based orchestrator would carry.

    It maps a candidate routing split and the observed load and backlog to an
    estimated per-tier waiting time, compute load and energy.  It uses the
    same M/G/c approximation as the simulator, i.e. the heuristics are given
    an honest, not a crippled, view of the system.
    """

    def __init__(self, cfg: Config, t_inf_ms: np.ndarray, lam_ref: float):
        self.cfg = cfg
        self.t_inf = t_inf_ms                                  # (C, T)
        self.lam_ref = lam_ref
        self.c_max = np.array([float(cfg.cloud_nodes_max),
                               float(cfg.edge_boards_max()), 1e12])
        self.rho_target = np.array([cfg.cloud_target_utilisation,
                                    cfg.edge_target_utilisation, 1.0])
        self.base_ms = np.array([cfg.wan_rtt_ms + cfg.cloud_fixed_ms,
                                 cfg.edge_fixed_ms, cfg.device_fixed_ms])
        e_ref = reference_tier_energies_j()
        self.tier_energy_j = np.array([e_ref["cloud"], e_ref["edge"],
                                       e_ref["device"]])
        self.alpha_c = np.stack([per_class_alpha(ALPHA_CODEBOOK[i:i + 1])[0]
                                 for i in range(N_ALPHA)])     # (A, C, T)
        # Task-success-rate proxy of every candidate split, so that the
        # rule-based baselines are held to the same quality-of-result floor as
        # the learned controllers instead of being allowed to collapse onto
        # the cheapest tier.  Without this B5 and B7 are not comparable with
        # B8-B10, and the comparison the editor asked for is not a fair one.
        self.acc = expected_accuracy(
            self.alpha_c[:, HEADLINE_CLASS, :], "adaptive_default", False, cfg)
        self.acc_short = np.maximum(0.0, cfg.accuracy_floor_pct - self.acc)
        self.feasible = self.acc >= cfg.accuracy_floor_pct

    def evaluate(self, state: np.ndarray):
        """Return ``(latency (G,A), work (G,A,T), energy (G,A))``."""
        g = state.shape[0]
        load = state[:, 1] * self.lam_ref                      # (G,)
        share = state[:, 2:6]                                  # (G, C)
        q = np.concatenate([state[:, 6:8] * self.c_max[None, :2],
                            np.zeros((g, 1))], axis=1)         # (G, T)
        n = load[:, None, None, None] * share[:, None, :, None] \
            * self.alpha_c[None, :, :, :]                      # (G,A,C,T)
        work = np.einsum('gact,ct->gat', n, self.t_inf) * 1e-3
        demand = work + q[:, None, :]
        c_act = np.maximum(np.minimum(
            np.ceil(demand / self.rho_target[None, None, :]),
            self.c_max[None, None, :]), 1.0)
        rho = np.clip(demand / c_act, 1e-6, 0.999)
        nt = np.maximum(n.sum(axis=2), 1e-9)
        es = np.einsum('gact,ct->gat', n, self.t_inf) / nt
        es2 = np.einsum('gact,ct->gat', n, self.t_inf ** 2) / nt
        cs2 = np.clip(es2 / np.maximum(es ** 2, 1e-12) - 1.0, 0.0, 20.0)
        w = sakasegawa_wait_ms(c_act, rho, es, cs2)
        w[:, :, 2] = 0.0
        lat_t = self.base_ms[None, None, :] + w + es
        lat = np.sum(n.sum(axis=2) * lat_t, axis=2) / nt.sum(axis=2)
        e = np.sum(n.sum(axis=2) * self.tier_energy_j[None, None, :], axis=2) \
            / nt.sum(axis=2)
        return lat, work, e


# ---------------------------------------------------------------------------
# Non-learning controllers
# ---------------------------------------------------------------------------
class FixedController:
    """B1-B4: a constant action."""

    def __init__(self, g: int, action: int):
        self.a = np.full(g, action, dtype=np.int64)

    def act(self, state, k):
        return self.a

    def observe(self, state, action, reward, k):
        pass


class GreedyController:
    """B5: myopic choice of the lowest instantaneous predicted latency among
    the splits that still meet the quality-of-result floor."""

    def __init__(self, model: HeuristicModel, n_modes: int):
        self.m = model
        self.n_modes = n_modes
        self._fallback = int(np.argmax(model.acc))

    def act(self, state, k):
        lat, _, _ = self.m.evaluate(state)
        masked = np.where(self.m.feasible[None, :], lat, np.inf)
        a = np.argmin(masked, axis=1)
        a = np.where(np.isfinite(masked.min(axis=1)), a, self._fallback)
        return (a * self.n_modes).astype(np.int64)

    def observe(self, state, action, reward, k):
        pass


class ThresholdController:
    """B6: SINR/load threshold heuristic with hysteresis on the edge queue.

    The rule walks along the routing codebook: when the edge utilisation
    exceeds the upper threshold the split moves one step towards the cloud;
    when it falls below the lower threshold it moves one step back towards the
    edge.  Hysteresis prevents oscillation.  Device offload is enabled only
    while the radio conditions are good enough, which the controller reads
    from the fraction of load in the delay-tolerant classes.
    """

    ORDER = np.array([0, 1, 3, 4, 6, 7])   # increasing cloud share

    def __init__(self, cfg: Config, model: HeuristicModel, n_modes: int):
        self.cfg = cfg
        self.m = model
        self.n_modes = n_modes
        self.pos: Optional[np.ndarray] = None

    def act(self, state, k):
        g = state.shape[0]
        if self.pos is None:
            self.pos = np.full(g, 2, dtype=np.int64)
        _, work, _ = self.m.evaluate(state)
        cur = self.ORDER[self.pos]
        util_edge = work[np.arange(g), cur, 1] / self.m.c_max[1]
        hi = self.cfg.edge_target_utilisation + self.cfg.threshold_queue_hysteresis
        lo = self.cfg.edge_target_utilisation - self.cfg.threshold_queue_hysteresis
        self.pos = np.clip(self.pos + (util_edge > hi).astype(np.int64)
                           - (util_edge < lo).astype(np.int64),
                           0, self.ORDER.size - 1)
        return (self.ORDER[self.pos] * self.n_modes).astype(np.int64)

    def observe(self, state, action, reward, k):
        pass


class LyapunovController:
    """B7: drift-plus-penalty with control parameter ``V``.

    Minimises ``sum_t Q_t (W_t(a) - C_t) + V * (E(a)/E_ref + L(a)/L_ref)``.
    """

    def __init__(self, cfg: Config, model: HeuristicModel, n_modes: int,
                 v: float | None = None):
        self.cfg = cfg
        self.m = model
        self.n_modes = n_modes
        self.v = cfg.lyapunov_v if v is None else v

    def act(self, state, k):
        lat, work, e = self.m.evaluate(state)
        g = state.shape[0]
        q = np.concatenate([state[:, 6:8] * self.m.c_max[None, :2],
                            np.zeros((g, 1))], axis=1)
        drift = np.sum(q[:, None, :] * (work - self.m.c_max[None, None, :]),
                       axis=2)
        pen = (e / self.cfg.energy_norm_j
               + lat / self.cfg.latency_norm_ms
               + self.cfg.reward_accuracy_penalty
               * self.m.acc_short[None, :] / self.cfg.accuracy_norm_pp)
        return (np.argmin(drift + self.v * pen, axis=1)
                * self.n_modes).astype(np.int64)

    def observe(self, state, action, reward, k):
        pass


# ---------------------------------------------------------------------------
# Grouped neural networks: one independent parameter set per replication
# ---------------------------------------------------------------------------
class GroupedLinear(nn.Module):
    def __init__(self, g: int, din: int, dout: int, gen: torch.Generator):
        super().__init__()
        bound = 1.0 / math.sqrt(din)
        w = (torch.rand(g, din, dout, generator=gen) * 2 - 1) * bound
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(g, 1, dout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.baddbmm(self.bias, x, self.weight)


class GroupedMLP(nn.Module):
    def __init__(self, g, din, hidden, dout, gen, n_hidden=2):
        super().__init__()
        layers: List[nn.Module] = []
        d = din
        for _ in range(n_hidden):
            layers += [GroupedLinear(g, d, hidden, gen), nn.Tanh()]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.head = GroupedLinear(g, d, dout, gen)

    def forward(self, x):
        return self.head(self.trunk(x))


class GroupedActorCritic(nn.Module):
    def __init__(self, g, din, hidden, n_action, gen, n_hidden=2):
        super().__init__()
        layers: List[nn.Module] = []
        d = din
        for _ in range(n_hidden):
            layers += [GroupedLinear(g, d, hidden, gen), nn.Tanh()]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.pi = GroupedLinear(g, d, n_action, gen)
        self.v = GroupedLinear(g, d, 1, gen)

    def forward(self, x):
        h = self.trunk(x)
        return self.pi(h), self.v(h).squeeze(-1)


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------
class PPOController:
    """B10 / A0: clipped-surrogate PPO with GAE, one policy per replication."""

    def __init__(self, cfg: Config, g: int, n_action: int, seed: int = 0,
                 training: bool = True):
        self.cfg = cfg
        self.g = g
        self.n_action = n_action
        self.training = training
        gen = torch.Generator().manual_seed(5_000_000 + seed)
        torch.manual_seed(5_000_000 + seed)
        self.net = GroupedActorCritic(g, STATE_DIM, cfg.ppo_hidden, n_action,
                                      gen, cfg.ppo_layers)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.ppo_lr,
                                    eps=cfg.ppo_adam_eps)
        self.buf: dict = {k: [] for k in ("s", "a", "logp", "v", "r")}
        self.returns: List[float] = []
        self._ep_return = np.zeros(g)
        self._ep_len = 0
        self.torch_gen = torch.Generator().manual_seed(6_000_000 + seed)

    def act(self, state, k):
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(1)   # (G,1,d)
        with torch.no_grad():
            logits, v = self.net(x)
            logits = logits.squeeze(1)
            if not self.training:
                a = torch.argmax(logits, dim=-1)
                self._last = None
                return a.numpy().astype(np.int64)
            p = torch.softmax(logits, dim=-1)
            a = torch.multinomial(p, 1, generator=self.torch_gen).squeeze(-1)
            logp = torch.log_softmax(logits, dim=-1).gather(
                1, a[:, None]).squeeze(1)
        self._last = (x, a, logp, v.squeeze(1))
        return a.numpy().astype(np.int64)

    def observe(self, state, action, reward, k):
        if not self.training or self._last is None:
            return
        x, a, logp, v = self._last
        self.buf["s"].append(x)
        self.buf["a"].append(a)
        self.buf["logp"].append(logp)
        self.buf["v"].append(v)
        self.buf["r"].append(torch.tensor(reward, dtype=torch.float32))
        self._ep_return += reward
        self._ep_len += 1
        if len(self.buf["r"]) >= self.cfg.ppo_rollout:
            self.returns.append(float(np.mean(self._ep_return)))
            self._ep_return[:] = 0.0
            self._update()

    def _update(self):
        cfg = self.cfg
        s = torch.cat(self.buf["s"], dim=1)                  # (G, R, d)
        a = torch.stack(self.buf["a"], dim=1)                # (G, R)
        logp_old = torch.stack(self.buf["logp"], dim=1)
        v_old = torch.stack(self.buf["v"], dim=1)
        r = torch.stack(self.buf["r"], dim=1)
        R = r.shape[1]

        adv = torch.zeros_like(r)
        last = torch.zeros(self.g)
        next_v = v_old[:, -1]
        for t in range(R - 1, -1, -1):
            nv = v_old[:, t + 1] if t + 1 < R else next_v
            delta = r[:, t] + cfg.ppo_gamma * nv - v_old[:, t]
            last = delta + cfg.ppo_gamma * cfg.ppo_gae_lambda * last
            adv[:, t] = last
        ret = adv + v_old
        adv = (adv - adv.mean(dim=1, keepdim=True)) / (
            adv.std(dim=1, keepdim=True) + 1e-8)

        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(R, generator=self.torch_gen)
            for i in range(0, R, cfg.ppo_minibatch):
                idx = perm[i:i + cfg.ppo_minibatch]
                logits, v = self.net(s[:, idx, :])
                logp_all = torch.log_softmax(logits, dim=-1)
                logp = logp_all.gather(2, a[:, idx, None]).squeeze(-1)
                ratio = torch.exp(logp - logp_old[:, idx])
                a1 = ratio * adv[:, idx]
                a2 = torch.clamp(ratio, 1 - cfg.ppo_clip,
                                 1 + cfg.ppo_clip) * adv[:, idx]
                pol = -torch.min(a1, a2).mean()
                vloss = ((v - ret[:, idx]) ** 2).mean()
                ent = -(logp_all.exp() * logp_all).sum(-1).mean()
                loss = pol + cfg.ppo_value_coef * vloss - cfg.ppo_entropy_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.ppo_grad_clip)
                self.opt.step()
        for v in self.buf.values():
            v.clear()

    def freeze(self):
        self.training = False
        return self


# ---------------------------------------------------------------------------
# DQN (B8)
# ---------------------------------------------------------------------------
class DQNController:
    def __init__(self, cfg: Config, g: int, n_action: int, seed: int = 0,
                 training: bool = True):
        self.cfg = cfg
        self.g = g
        self.n_action = n_action
        self.training = training
        gen = torch.Generator().manual_seed(7_100_000 + seed)
        torch.manual_seed(7_100_000 + seed)
        self.q = GroupedMLP(g, STATE_DIM, cfg.dqn_hidden, n_action, gen)
        self.tgt = GroupedMLP(g, STATE_DIM, cfg.dqn_hidden, n_action, gen)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.dqn_lr)
        self.rng = np.random.default_rng(7_200_000 + seed)
        cap = cfg.dqn_replay
        self.s = np.zeros((cap, g, STATE_DIM), dtype=np.float32)
        self.a = np.zeros((cap, g), dtype=np.int64)
        self.r = np.zeros((cap, g), dtype=np.float32)
        self.s2 = np.zeros((cap, g, STATE_DIM), dtype=np.float32)
        self.n = 0
        self.ptr = 0
        self.steps = 0
        self._prev = None

    def _eps(self):
        f = min(1.0, self.steps / max(self.cfg.dqn_eps_decay_steps, 1))
        return self.cfg.dqn_eps_start + f * (self.cfg.dqn_eps_end
                                             - self.cfg.dqn_eps_start)

    def act(self, state, k):
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            qv = self.q(x).squeeze(1).numpy()
        a = np.argmax(qv, axis=1).astype(np.int64)
        if self.training:
            eps = self._eps()
            rand = self.rng.random(self.g) < eps
            a = np.where(rand, self.rng.integers(0, self.n_action, self.g), a)
        return a

    def observe(self, state, action, reward, k):
        if not self.training:
            return
        if self._prev is not None:
            ps, pa, pr = self._prev
            i = self.ptr
            self.s[i] = ps
            self.a[i] = pa
            self.r[i] = pr
            self.s2[i] = state
            self.ptr = (self.ptr + 1) % self.cfg.dqn_replay
            self.n = min(self.n + 1, self.cfg.dqn_replay)
        self._prev = (state.astype(np.float32), action, reward)
        self.steps += 1
        if self.n >= self.cfg.dqn_batch:
            self._update()
        if self.steps % self.cfg.dqn_target_sync == 0:
            self.tgt.load_state_dict(self.q.state_dict())

    def _update(self):
        idx = self.rng.integers(0, self.n, self.cfg.dqn_batch)
        s = torch.tensor(self.s[idx].transpose(1, 0, 2))
        a = torch.tensor(self.a[idx].T)
        r = torch.tensor(self.r[idx].T)
        s2 = torch.tensor(self.s2[idx].transpose(1, 0, 2))
        with torch.no_grad():
            tq = self.tgt(s2).max(dim=2).values
            target = r + self.cfg.dqn_gamma * tq
        qv = self.q(s).gather(2, a[:, :, None]).squeeze(-1)
        loss = ((qv - target) ** 2).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def freeze(self):
        self.training = False
        return self


# ---------------------------------------------------------------------------
# LinUCB (B9)
# ---------------------------------------------------------------------------
class LinUCBController:
    """Disjoint LinUCB with rank-one (Sherman-Morrison) inverse updates."""

    def __init__(self, cfg: Config, g: int, n_action: int, seed: int = 0,
                 training: bool = True):
        self.cfg = cfg
        self.g = g
        self.n_action = n_action
        self.training = training
        d = STATE_DIM
        self.Ainv = np.tile(np.eye(d) / cfg.linucb_lambda,
                            (g, n_action, 1, 1))
        self.b = np.zeros((g, n_action, d))
        self.rng = np.random.default_rng(7_300_000 + seed)

    def act(self, state, k):
        x = state
        theta = np.einsum('gaij,gaj->gai', self.Ainv, self.b)
        mean = np.einsum('gai,gi->ga', theta, x)
        var = np.einsum('gi,gaij,gj->ga', x, self.Ainv, x)
        score = mean + self.cfg.linucb_alpha * np.sqrt(np.maximum(var, 0.0))
        return np.argmax(score, axis=1).astype(np.int64)

    def observe(self, state, action, reward, k):
        if not self.training:
            return
        gi = np.arange(self.g)
        Ai = self.Ainv[gi, action]                     # (G, d, d)
        x = state
        Ax = np.einsum('gij,gj->gi', Ai, x)
        denom = 1.0 + np.einsum('gi,gi->g', x, Ax)
        self.Ainv[gi, action] = Ai - (Ax[:, :, None] * Ax[:, None, :]
                                      / denom[:, None, None])
        self.b[gi, action] += reward[:, None] * x

    def freeze(self):
        self.training = False
        return self
