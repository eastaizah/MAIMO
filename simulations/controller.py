"""Orchestration controllers.

* :class:`PPOController` - Proximal Policy Optimization with a **linear-softmax**
  policy over the joint action space (model variant x serving tier x compression
  level), a clipped surrogate objective, generalised advantage estimation, an
  entropy bonus and a linear value baseline.  Gradients are derived analytically;
  no deep-learning framework is used.
* :class:`DQNController` - value-based reactive baseline: linear Q-function with
  epsilon-greedy exploration, experience replay and a target network.  Used for
  the "reactive DRL, no forecast" baseline.
* :class:`RuleBasedController` - deterministic threshold rules over the same
  observation, used for the "no DRL controller" ablation (the forecast is
  retained and drives pre-loading and the load thresholds).

Scope statement (editor requirement R7): the policy is linear in the observation
features, not a deep network.  The action space has 14 physically realisable
elements and the observation is 28-dimensional, so a linear-softmax policy is
sufficient to represent the routing rules that the environment rewards, and it
trains to a stable policy within the runtime budget of the released artefact.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from config import (COMPRESSIONS, MODELS, N_ACTIONS, Params, TIERS, USE_CASES,
                    feature_layout)


# ---------------------------------------------------------------------------
def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z = np.where(mask, logits, -np.inf)
    z = z - z.max()
    e = np.exp(z)
    e[~mask] = 0.0
    s = e.sum()
    return e / s if s > 0 else mask / mask.sum()


class _Adam:
    def __init__(self, shape, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0

    def step(self, w: np.ndarray, g: np.ndarray) -> None:
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        w -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ---------------------------------------------------------------------------
class PPOController:
    """PPO with a linear-softmax policy and a linear value baseline."""

    name = "ppo"
    stochastic = True

    def __init__(self, p: Params, n_features: int, feasible: np.ndarray,
                 seed: int = 0):
        self.p = p
        self.mask = feasible.astype(bool)
        rng = np.random.default_rng(seed + 4409)
        self.theta = rng.normal(0.0, 0.01, (n_features, N_ACTIONS))
        self.w_v = np.zeros(n_features)
        self.opt_pi = _Adam(self.theta.shape, p.ppo_lr, p.ppo_adam_beta1,
                            p.ppo_adam_beta2, p.ppo_adam_eps)
        self.opt_v = _Adam(self.w_v.shape, p.ppo_value_lr, p.ppo_adam_beta1,
                           p.ppo_adam_beta2, p.ppo_adam_eps)
        self.rng = rng
        self.returns: List[float] = []
        self.updates = 0
        self.explore_eps = p.ppo_explore_start
        self._uniform = self.mask / self.mask.sum()

    # ------------------------------------------------------------------
    def with_mask(self, feasible: np.ndarray) -> "PPOController":
        """A view of this policy restricted to a smaller action set.

        Used by the ablations that remove actions from the deployed system
        (Sec. "How the learned policy is handled" in ``ablation.py``): the
        learned weights are kept and the softmax is renormalised over the
        actions the ablated system can still execute.
        """
        import copy
        c = copy.copy(self)
        c.mask = np.asarray(feasible).astype(bool)
        c._uniform = c.mask / c.mask.sum()
        return c

    def set_progress(self, frac: float) -> None:
        """Anneal the exploration floor; ``frac`` runs 0 -> 1 over training."""
        p = self.p
        f = min(max(frac, 0.0), 1.0)
        self.explore_eps = (p.ppo_explore_start
                            + f * (p.ppo_explore_end - p.ppo_explore_start))

    def probs(self, x: np.ndarray) -> np.ndarray:
        return _masked_softmax(x @ self.theta, self.mask)

    def act(self, x: np.ndarray, greedy: bool = False) -> Tuple[int, float]:
        """Sample an action and return it with its *behaviour* log-probability.

        During training the softmax is mixed with the uniform distribution over
        feasible actions.  Without the floor the probability of an action that
        is briefly disfavoured decays geometrically, it stops being sampled, and
        its weights stop receiving gradient - the policy locks onto whatever it
        preferred early.  The mixture is the behaviour policy, so its
        log-probability is what the PPO importance ratio must be formed against.
        """
        pr = self.probs(x)
        if greedy:
            return int(np.argmax(pr)), float(math.log(max(pr.max(), 1e-12)))
        e = self.explore_eps
        beh = (1.0 - e) * pr + e * self._uniform
        a = int(np.searchsorted(np.cumsum(beh), self.rng.random()))
        a = min(a, N_ACTIONS - 1)
        return a, float(math.log(max(beh[a], 1e-12)))

    def value(self, x: np.ndarray) -> float:
        return float(x @ self.w_v)

    # ------------------------------------------------------------------
    def update(self, obs: np.ndarray, acts: np.ndarray, logp_old: np.ndarray,
               rews: np.ndarray, dones: np.ndarray,
               groups: Optional[np.ndarray] = None) -> dict:
        """One PPO update from a rollout.

        ``obs`` (T, F), ``acts`` (T,), ``rews`` (T,), ``dones`` (T,) with
        ``dones[t] = 1`` on the last transition of an episode.

        ``groups`` (T,) identifies the decision group (use case) each transition
        belongs to.  Several groups are decided in the same slot and their
        rewards are independent given the queue state, so GAE is accumulated
        along one chain *per group* rather than along the interleaved rollout.
        Interleaving would add every other group's reward to each return, and
        those rewards differ by more than an order of magnitude between use
        cases, which swamps the differences the policy has to resolve.
        """
        p = self.p
        T = obs.shape[0]
        vals = obs @ self.w_v
        adv = np.zeros(T)
        if groups is None:
            groups = np.zeros(T, dtype=int)
        for gid in np.unique(groups):
            sel = np.nonzero(groups == gid)[0]
            last = 0.0
            for k in range(sel.size - 1, -1, -1):
                t = sel[k]
                nonterm = 0.0 if dones[t] else 1.0
                v_next = (vals[sel[k + 1]]
                          if (k + 1 < sel.size and nonterm) else 0.0)
                delta = rews[t] + p.ppo_gamma * v_next - vals[t]
                last = delta + p.ppo_gamma * p.ppo_gae_lambda * nonterm * last
                adv[t] = last
        ret = adv + vals
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idx = np.arange(T)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "clip_frac": 0.0}
        n_batches = 0
        for _ in range(p.ppo_epochs):
            self.rng.shuffle(idx)
            for s in range(0, T, p.ppo_minibatch):
                mb = idx[s:s + p.ppo_minibatch]
                if mb.size < 2:
                    continue
                x = obs[mb]
                a = acts[mb]
                logits = x @ self.theta
                pr = np.where(self.mask, logits, -np.inf)
                pr = pr - pr.max(axis=1, keepdims=True)
                e = np.exp(pr)
                e[:, ~self.mask] = 0.0
                pr = e / e.sum(axis=1, keepdims=True)
                lp = np.log(np.maximum(pr[np.arange(mb.size), a], 1e-12))
                ratio = np.exp(lp - logp_old[mb])
                A = adv[mb]
                clipped = np.clip(ratio, 1 - p.ppo_clip, 1 + p.ppo_clip)
                use_unclipped = (ratio * A) <= (clipped * A)
                # d/dlogits of the surrogate, only where the ratio is active.
                coef = np.where(use_unclipped, ratio * A, 0.0)
                g_logits = -pr * coef[:, None]
                g_logits[np.arange(mb.size), a] += coef
                # entropy bonus
                with np.errstate(divide="ignore", invalid="ignore"):
                    logp_all = np.where(pr > 0, np.log(np.maximum(pr, 1e-12)),
                                        0.0)
                ent = -(pr * logp_all).sum(axis=1)
                g_ent = pr * (-(logp_all) - ent[:, None])
                g_logits = g_logits + p.ppo_entropy_coef * g_ent
                g_theta = -(x.T @ g_logits) / mb.size
                gn = np.sqrt(float((g_theta ** 2).sum()))
                if gn > p.ppo_grad_clip:
                    g_theta *= p.ppo_grad_clip / gn
                self.opt_pi.step(self.theta, g_theta)

                v = x @ self.w_v
                err = v - ret[mb]
                g_v = 2.0 * (x.T @ err) / mb.size * p.ppo_value_coef
                gn = np.sqrt(float((g_v ** 2).sum()))
                if gn > p.ppo_grad_clip:
                    g_v *= p.ppo_grad_clip / gn
                self.opt_v.step(self.w_v, g_v)

                stats["policy_loss"] += -float((ratio * A).mean())
                stats["value_loss"] += float((err ** 2).mean())
                stats["entropy"] += float(ent.mean())
                stats["clip_frac"] += float(np.mean(ratio != clipped))
                n_batches += 1
        self.updates += 1
        for k in stats:
            stats[k] /= max(n_batches, 1)
        return stats


# ---------------------------------------------------------------------------
class DQNController:
    """Linear Q-learning with epsilon-greedy exploration, replay and a target
    network: the reactive value-based DRL baseline (no traffic forecast)."""

    name = "dqn"
    stochastic = False

    def __init__(self, p: Params, n_features: int, feasible: np.ndarray,
                 seed: int = 0):
        self.p = p
        self.mask = feasible.astype(bool)
        rng = np.random.default_rng(seed + 6607)
        self.W = rng.normal(0.0, 0.01, (n_features, N_ACTIONS))
        self.W_t = self.W.copy()
        self.opt = _Adam(self.W.shape, p.dqn_lr, p.ppo_adam_beta1,
                         p.ppo_adam_beta2, p.ppo_adam_eps)
        self.rng = rng
        self.eps = p.dqn_eps_start
        self.buf_x = np.zeros((p.dqn_replay_size, n_features))
        self.buf_a = np.zeros(p.dqn_replay_size, dtype=int)
        self.buf_r = np.zeros(p.dqn_replay_size)
        self.buf_x2 = np.zeros((p.dqn_replay_size, n_features))
        self.buf_d = np.zeros(p.dqn_replay_size)
        self.n = 0
        self.ptr = 0
        self.steps = 0
        self.returns: List[float] = []

    def q(self, x: np.ndarray) -> np.ndarray:
        return np.where(self.mask, x @ self.W, -np.inf)

    def act(self, x: np.ndarray, greedy: bool = False) -> Tuple[int, float]:
        if not greedy and self.rng.random() < self.eps:
            cand = np.nonzero(self.mask)[0]
            return int(self.rng.choice(cand)), 0.0
        return int(np.argmax(self.q(x))), 0.0

    def store(self, x, a, r, x2, d) -> None:
        i = self.ptr
        self.buf_x[i] = x
        self.buf_a[i] = a
        self.buf_r[i] = r
        self.buf_x2[i] = x2
        self.buf_d[i] = d
        self.ptr = (i + 1) % self.p.dqn_replay_size
        self.n = min(self.n + 1, self.p.dqn_replay_size)

    def train_step(self) -> None:
        p = self.p
        self.steps += 1
        if self.n < p.dqn_batch * 4:
            return
        mb = self.rng.integers(0, self.n, p.dqn_batch)
        x, a, r = self.buf_x[mb], self.buf_a[mb], self.buf_r[mb]
        x2, d = self.buf_x2[mb], self.buf_d[mb]
        q_next = np.where(self.mask, x2 @ self.W_t, -np.inf).max(axis=1)
        target = r + p.dqn_gamma * (1.0 - d) * q_next
        q = (x @ self.W)[np.arange(p.dqn_batch), a]
        err = q - target
        g = np.zeros_like(self.W)
        np.add.at(g.T, a, 2.0 * err[:, None] * x)
        g /= p.dqn_batch
        gn = np.sqrt(float((g ** 2).sum()))
        if gn > 1.0:
            g /= gn
        self.opt.step(self.W, g)
        if self.steps % p.dqn_target_sync == 0:
            self.W_t = self.W.copy()

    def decay_epsilon(self, episode: int) -> None:
        p = self.p
        frac = min(1.0, episode / max(p.dqn_eps_decay_episodes, 1))
        self.eps = p.dqn_eps_start + frac * (p.dqn_eps_end - p.dqn_eps_start)


# ---------------------------------------------------------------------------
class RuleBasedController:
    """Deterministic rules, used for the "no DRL controller" ablation.

    The rules encode the obvious engineering heuristic and still consume the
    traffic forecast (it gates the aggressiveness of edge offloading), so the
    ablation isolates the *learned* controller rather than the forecast.
    """

    name = "rule"
    stochastic = False

    def __init__(self, p: Params, n_features: int, feasible: np.ndarray,
                 seed: int = 0):
        self.p = p
        self.mask = feasible.astype(bool)
        self.returns: List[float] = []
        names = [(MODELS[a // 12].name, TIERS[(a // 4) % 3],
                  COMPRESSIONS[a % 4].name) for a in range(N_ACTIONS)]
        self._by_name = {n: i for i, n in enumerate(names)}
        self._L = feature_layout(p)

    def _pick(self, model: str, tier: str, comp: str) -> int:
        a = self._by_name[(model, tier, comp)]
        return a if self.mask[a] else int(np.nonzero(self.mask)[0][0])

    def act_from_state(self, cx: int, edge_util: float, cloud_util: float,
                       forecast: float) -> int:
        """cx: complexity class (0 easy, 1 medium, 2 hard)."""
        if cx == 2:
            if cloud_util < 0.85:
                return self._pick("cloud_70B", "cloud", "none")
            return self._pick("edge_7B", "edge", "int8")
        if cx == 1:
            if edge_util < 0.7 and forecast < 1.25:
                return self._pick("edge_7B", "edge", "lora")
            return self._pick("edge_7B", "edge", "int4")
        return self._pick("device_50M", "device", "int4")

    def act(self, x: np.ndarray, greedy: bool = True) -> Tuple[int, float]:
        L = self._L
        i, w = L["complexity"]
        cx = int(np.argmax(x[i:i + w]))
        i, w = L["forecast"]
        fc = float(x[i:i + w].mean())
        return self.act_from_state(cx, float(x[L["edge_util"][0]]),
                                   float(x[L["cloud_util"][0]]),
                                   fc if fc > 0 else 1.0), 0.0


def build_controller(kind: str, p: Params, n_features: int,
                     feasible: np.ndarray, seed: int = 0):
    return {"ppo": PPOController, "dqn": DQNController,
            "rule": RuleBasedController}[kind](p, n_features, feasible, seed)
