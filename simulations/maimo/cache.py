"""Edge model-cache occupancy and cold-start miss rates.

The MEC site cannot hold the whole model zoo.  Earlier revisions of this
simulator carried the reactive and proactive miss rates as free parameters,
which is indefensible: a miss rate is a *consequence* of how many variants
compete for how much cache and of how skewed their popularity is, and a
reviewer checking the arithmetic would notice that the asserted rates did not
follow from the stated cache size.  This module derives them instead.

Resident-set occupancy
----------------------
``K = floor(cache_gb / variant_gb)`` variants fit.  Variant popularity follows
a Zipf law, so the fraction of requests whose variant is *not* resident is
obtained from Che's approximation, the standard analytic model for LRU under
the independent-reference model: the cache behaves as if each variant were
held for a common characteristic time ``T`` after every reference, variant
``i`` with request probability ``p_i`` is therefore resident with probability
``1 - exp(-p_i T)``, and ``T`` is fixed by requiring the expected occupancy to
equal ``K``.

What a miss costs, and why proactive loading helps
--------------------------------------------------
The two loading policies differ in *when* the load happens, not in how many
variants they can hold, and that is where the benefit comes from.

*Reactive* loading is demand-driven: the absence of a variant is discovered by
the request that needs it, which then waits for the fetch.  Every resident-set
miss is therefore a latency event, and the latency-relevant miss rate is the
full LRU miss rate.

*Proactive* loading uses the traffic forecast to stage variants at the start
of a control interval, before the demand arrives, so the fetch overlaps with
useful work instead of sitting on the critical path.  Two things follow.  A
variant the controller knowingly does not host is not a stall either: the
orchestrator can see that it is absent and routes those requests to a tier
that does hold them rather than blocking for a fetch.  What proactive loading
*cannot* absorb is demand it failed to anticipate -- a variant that becomes
hot faster than the forecast expected is discovered the same way a reactive
cache would discover it, by a request that stalls.  The unanticipated share of
demand is the predictor's relative error, and the probability that such demand
needs a non-resident variant is the resident-set miss rate, so

    proactive latency-miss = (relative forecast error) x (resident-set miss)

This makes the value of the predictor explicit and bounded: a perfect forecast
removes the cold-start stalls entirely, a poor one degrades gracefully towards
the reactive rate, and neither policy can escape the fact that only ``K``
variants fit.  Note that the *energy* of a load is charged in either case --
staging a model early moves the cost off the critical path, it does not make
it free -- and ``maimo.sim`` accounts for that separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def zipf_pmf(n: int, s: float) -> np.ndarray:
    """Popularity of ``n`` variants ranked by a Zipf law of exponent ``s``."""
    r = np.arange(1, n + 1, dtype=float)
    w = r ** (-s)
    return w / w.sum()


def che_characteristic_time(p: np.ndarray, k: float) -> float:
    """Solve ``sum_i (1 - exp(-p_i T)) = k`` for the LRU characteristic time.

    The left-hand side increases strictly in ``T`` from 0 to ``n``, so a
    bisection is safe and converges quickly.
    """
    n = p.size
    if k >= n:
        return math.inf
    hi = 1.0
    while np.sum(1.0 - np.exp(-p * hi)) < k:
        hi *= 2.0
        if hi > 1e18:
            return hi
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.sum(1.0 - np.exp(-p * mid)) < k:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def lru_miss_rate(p: np.ndarray, k: int) -> float:
    """Steady-state LRU miss rate under Che's approximation."""
    if k >= p.size:
        return 0.0
    t = che_characteristic_time(p, float(k))
    if not math.isfinite(t):
        return 0.0
    hit = float(np.sum(p * (1.0 - np.exp(-p * t))))
    return float(min(max(1.0 - hit, 0.0), 1.0))


def static_miss_rate(p: np.ndarray, k: int) -> float:
    """Miss rate of an oracle that pins the ``k`` most popular variants.  This
    is the floor no online placement can beat under a stationary popularity
    law, and it is the resident-set miss rate the proactive policy attains."""
    if k >= p.size:
        return 0.0
    order = np.sort(p)[::-1]
    return float(max(1.0 - order[:k].sum(), 0.0))


@dataclass(frozen=True)
class CacheModel:
    """Cache behaviour derived from one configuration."""

    n_variants: int
    variant_gb: float
    capacity_gb: float
    slots: int                 # variants that physically fit
    zipf_s: float
    resident_miss: float       # proactive placement, stationary popularity
    reactive_miss: float       # LRU on demand: every miss is a stall

    def proactive_miss(self, rel_error) -> np.ndarray:
        """Latency-relevant miss rate of proactive loading.

        Only the demand the forecast failed to anticipate can stall, and it
        stalls only if it needs a variant outside the resident set.  Capped by
        the reactive rate: anticipating badly cannot be worse than not
        anticipating at all.
        """
        e = np.asarray(rel_error, dtype=float)
        return np.minimum(np.abs(e) * self.resident_miss, self.reactive_miss)

    def summary(self) -> str:
        return (f"{self.n_variants} variants x {self.variant_gb:.2f} GB "
                f"({self.n_variants * self.variant_gb:.0f} GB) vs. "
                f"{self.capacity_gb:.0f} GB cache -> {self.slots} resident "
                f"({100.0 * self.slots / self.n_variants:.0f} % of the zoo); "
                f"Zipf s = {self.zipf_s}; resident-set miss "
                f"{100 * self.resident_miss:.2f} %, reactive (LRU) miss "
                f"{100 * self.reactive_miss:.2f} %")


def build_cache_model(cfg) -> CacheModel:
    p = zipf_pmf(cfg.edge_variants, cfg.variant_zipf_s)
    slots = int(cfg.edge_cache_gb // cfg.variant_gb)
    slots = max(1, min(slots, cfg.edge_variants))
    return CacheModel(
        n_variants=cfg.edge_variants,
        variant_gb=cfg.variant_gb,
        capacity_gb=cfg.edge_cache_gb,
        slots=slots,
        zipf_s=cfg.variant_zipf_s,
        resident_miss=static_miss_rate(p, slots),
        reactive_miss=lru_miss_rate(p, slots),
    )
