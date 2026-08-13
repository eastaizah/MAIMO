"""Non-stationary request arrivals.

The offered load of each service class is the product of four factors:

``lambda_c(t) = N_c r_c * D(t) * S(t) * B_c(t)``

* ``N_c r_c`` - population of the class times its per-session request rate;
* ``D(t)``    - deterministic diurnal profile, two harmonics;
* ``S(t)``    - weekly seasonality (weekday/weekend);
* ``B_c(t)``  - a two-state Markov-modulated Poisson process (MMPP-2) that
  produces heavy, correlated bursts on top of the seasonal mean.

Arrivals in a slot are then Poisson with that instantaneous rate, so the
process is a doubly stochastic (Cox) point process: seasonal at long
timescales, bursty at medium timescales, Poisson at the slot timescale.

The traffic process is **exogenous**: it does not depend on the orchestration
policy.  It is therefore generated once per seed, before the control loop
starts, which is what allows the BiLSTM predictor to be evaluated with a
single batched forward pass over the whole trace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import (Config, DEFAULT, SERVICE_CLASSES, N_CLASS,
                     SECONDS_PER_DAY, SECONDS_PER_WEEK)


def diurnal_factor(t_s: np.ndarray, cfg: Config) -> np.ndarray:
    """Two-harmonic daily profile, mean 1."""
    ph1 = 2.0 * math.pi * (t_s / SECONDS_PER_DAY
                           - cfg.diurnal_peak_hour / 24.0)
    ph2 = 4.0 * math.pi * (t_s / SECONDS_PER_DAY
                           - cfg.diurnal_second_peak_hour / 24.0)
    return 1.0 + cfg.diurnal_amp1 * np.cos(ph1) + cfg.diurnal_amp2 * np.cos(ph2)


def weekly_factor(t_s: np.ndarray, cfg: Config) -> np.ndarray:
    """Weekday/weekend seasonality with a smooth (12 h) transition."""
    day = (t_s % SECONDS_PER_WEEK) / SECONDS_PER_DAY      # 0 = Monday 00:00
    # weekend = days 5 and 6; smooth with a raised cosine over 0.5 day.
    def _ramp(x):
        return 0.5 * (1.0 + np.tanh(x / 0.25))
    into = _ramp(day - 5.0)
    out = _ramp(day - 7.0)
    w = into - out
    return 1.0 - (1.0 - cfg.weekend_factor) * w


def mmpp2_multiplier(rng: np.random.Generator, n: int, dt_s: float,
                     cfg: Config) -> np.ndarray:
    """Two-state Markov-modulated burst multiplier sampled on a regular grid.

    Sojourn times are exponential with means ``mmpp_mean_quiet_s`` and
    ``mmpp_mean_burst_s``; the multiplier is 1 in the quiet state and
    ``mmpp_burst_multiplier`` in the burst state.  The realised long-run mean
    is divided out so that the MMPP factor does not shift the mean offered
    load, only its correlation structure and tail.
    """
    total = n * dt_s
    means = (cfg.mmpp_mean_quiet_s, cfg.mmpp_mean_burst_s)
    vals = (1.0, cfg.mmpp_burst_multiplier)
    edges = [0.0]
    states = []
    s = 0
    t = 0.0
    while t < total:
        t += rng.exponential(means[s])
        edges.append(t)
        states.append(s)
        s = 1 - s
    grid = (np.arange(n) + 0.5) * dt_s
    idx = np.searchsorted(np.asarray(edges), grid, side="right") - 1
    idx = np.clip(idx, 0, len(states) - 1)
    m = np.asarray([vals[st] for st in states], dtype=float)[idx]
    p_burst = means[1] / (means[0] + means[1])
    return m / (1.0 * (1.0 - p_burst) + vals[1] * p_burst)


@dataclass
class TrafficTrace:
    """Offered load per control interval, per class, for one replication."""

    t_s: np.ndarray                # (K,) wall-clock time at interval start
    lam: np.ndarray                # (K, C) requests per second, class c
    observed: np.ndarray           # (K,) noisy aggregate load measurement
    cfg: Config

    def total(self) -> np.ndarray:
        return self.lam.sum(axis=1)


def make_trace(cfg: Config, seed: int, n_intervals: int,
               time_offset_s: float | None = None) -> TrafficTrace:
    """Generate one exogenous traffic trace."""
    rng = np.random.default_rng(2_000_000 + seed)
    dt = cfg.t_control_s()
    if time_offset_s is None:
        # Each replication starts at an independent phase of the week, so the
        # 20 replications together cover the weekly seasonality.
        time_offset_s = float(rng.random() * SECONDS_PER_WEEK)
    t = time_offset_s + np.arange(n_intervals, dtype=float) * dt

    base = np.array([cfg.n_sessions() * sc.share_of_sessions
                     * sc.rate_per_session for sc in SERVICE_CLASSES])
    season = diurnal_factor(t, cfg) * weekly_factor(t, cfg)
    lam = np.empty((n_intervals, N_CLASS))
    for c in range(N_CLASS):
        burst = mmpp2_multiplier(rng, n_intervals, dt, cfg)
        lam[:, c] = base[c] * season * burst
    lam = np.maximum(lam, 1.0)

    tot = lam.sum(axis=1)
    observed = tot * (1.0 + cfg.pred_obs_noise * rng.standard_normal(n_intervals))
    return TrafficTrace(t_s=t, lam=lam, observed=np.maximum(observed, 1.0),
                        cfg=cfg)


def sample_arrivals(rng: np.random.Generator, lam_slice: np.ndarray,
                    n_slots: int, cfg: Config) -> np.ndarray:
    """Poisson arrivals for ``n_slots`` slots given per-second rates.

    ``lam_slice`` has shape ``(G, C)``; the result has shape ``(G, S, C)``.
    """
    shape = (lam_slice.shape[0], n_slots, lam_slice.shape[1])
    mean = np.broadcast_to(lam_slice[:, None, :] * cfg.t_slot_s, shape)
    if not cfg.poisson_arrivals:
        return np.array(mean, dtype=float)
    return rng.poisson(mean).astype(float)
