"""Experiment driver: build a controller for a policy, train it, evaluate it.

Variance reduction: every policy is evaluated on the **same** traffic traces
and the same channel realisations (common random numbers).  This makes the
comparison paired, which removes the between-seed traffic variability from the
differences between policies.  Welch's two-sided t-test is still used, as the
CONTRACT prescribes; on positively correlated samples Welch is conservative,
so any significance we report is not an artefact of the pairing.

Learned controllers (PPO, DQN, LinUCB) are trained on a *separate* traffic
window and separate channel realisations and are then frozen, so every number
reported for them is out of sample.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import predictor as pred_mod
from .config import Config, SERVICE_CLASSES
from .controller import (DQNController, FixedController, GreedyController,
                         HeuristicModel, LinUCBController, LyapunovController,
                         PPOController, STATE_DIM, ThresholdController)
from .models import inference_time_table_ms
from .sim import (COMPRESSION_MODES, N_ACTION, N_ALPHA, PolicySpec,
                  SeedContext, build_context, run_batch)


class PredictorCache:
    """One trained BiLSTM per seed, reused by every policy."""

    def __init__(self, cfg: Config, seeds: Sequence[int]):
        self.cfg = cfg
        self.seeds = list(seeds)
        t0 = time.perf_counter()
        self.models = [pred_mod.train_predictor(cfg, s) for s in seeds]
        self.train_seconds = time.perf_counter() - t0
        self.history = [m.history for m in self.models]
        self._cache: Dict[int, tuple] = {}

    def predictions(self, ctx: SeedContext, kind: str):
        key = (id(ctx), kind)
        if key in self._cache:
            return self._cache[key]
        outs, errs = [], []
        for i, tr in enumerate(ctx.traces):
            if kind == "bilstm":
                o, e = pred_mod.predict_trace(self.models[i], self.cfg, tr)
            elif kind == "persistence":
                o, e = pred_mod.persistence_predict(self.cfg, tr)
            else:
                # ``predictor="none"``: the controller sees the true load but
                # has no forecast to pre-stage models with, so the cache falls
                # back to reactive loading.  ``maimo.sim`` selects the reactive
                # branch on the spec itself, and the zero error here just
                # records that the observation is exact.
                o, e = pred_mod.oracle_predict(self.cfg, tr)
                e = np.zeros_like(e)
            outs.append(o)
            errs.append(e)
        res = (np.stack(outs), np.stack(errs))
        self._cache[key] = res
        return res


def n_actions_for(spec: PolicySpec) -> int:
    return N_ALPHA * len(spec.compression_modes())


def make_controller(cfg: Config, spec: PolicySpec, g: int, seed: int,
                    training: bool):
    n_modes = len(spec.compression_modes())
    n_action = N_ALPHA * n_modes
    t_inf = inference_time_table_ms(cfg, spec.compression_modes()[0],
                                    spec.early_exit)
    lam_ref = float(cfg.n_sessions() * sum(
        s.share_of_sessions * s.rate_per_session for s in SERVICE_CLASSES))
    model = HeuristicModel(cfg, t_inf, lam_ref)
    c = spec.controller
    if c == "fixed":
        return FixedController(g, 2 * n_modes)
    if c == "greedy":
        return GreedyController(model, n_modes)
    if c == "threshold":
        return ThresholdController(cfg, model, n_modes)
    if c == "lyapunov":
        return LyapunovController(cfg, model, n_modes)
    if c == "ppo":
        return PPOController(cfg, g, n_action, seed, training=training)
    if c == "dqn":
        return DQNController(cfg, g, n_action, seed, training=training)
    if c == "linucb":
        return LinUCBController(cfg, g, n_action, seed, training=training)
    raise ValueError(f"unknown controller {c}")


LEARNED = {"ppo", "dqn", "linucb"}


def run_policy(cfg: Config, spec: PolicySpec, eval_ctx: SeedContext,
               train_ctx: Optional[SeedContext], cache: PredictorCache,
               seed: int = 0, verbose: bool = True) -> dict:
    g = len(eval_ctx.seeds)
    t0 = time.perf_counter()
    ctrl = make_controller(cfg, spec, g, seed, training=True)
    convergence = None

    if spec.controller in LEARNED and train_ctx is not None:
        p_tr, e_tr = cache.predictions(train_ctx, spec.predictor)
        n_train = train_ctx.lam.shape[1]
        out = run_batch(cfg, train_ctx, spec, ctrl, p_tr, e_tr,
                        measure_from=n_train, n_intervals=n_train,
                        collect_reward=True, rng_seed=17)
        convergence = {
            "reward_trace": out["reward_trace"],
            "ppo_returns": getattr(ctrl, "returns", None),
        }
        ctrl.freeze()
    train_seconds = time.perf_counter() - t0

    p_ev, e_ev = cache.predictions(eval_ctx, spec.predictor)
    t1 = time.perf_counter()
    res = run_batch(cfg, eval_ctx, spec, ctrl, p_ev, e_ev,
                    measure_from=cfg.warmup_intervals(),
                    n_intervals=eval_ctx.lam.shape[1], rng_seed=23)
    res["train_seconds"] = train_seconds
    res["eval_seconds"] = time.perf_counter() - t1
    res["convergence"] = convergence
    res["spec"] = spec
    if verbose:
        print(f"  {spec.ident:3s} {spec.name:32s} "
              f"lat {np.mean(res['latency_mean_ms']):7.2f} ms  "
              f"E {np.mean(res['energy_j']):7.3f} J  "
              f"acc {np.mean(res['accuracy_pct']):5.2f} %  "
              f"SLA {np.mean(res['sla_violation_pct']):5.2f} %  "
              f"({train_seconds:.1f}+{res['eval_seconds']:.1f} s)")
    return res
