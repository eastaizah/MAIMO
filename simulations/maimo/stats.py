"""Statistics protocol locked in ``work/CONTRACT.md``.

* 20 independent replications (seeds 1..20).
* Report ``mean +/- half-width of the 95 % confidence interval`` using
  Student's t with 19 degrees of freedom, ``t_{0.975,19} = 2.093``.
* Report the standard deviation in the reproducibility table.
* Welch's two-sided t-test for MAIMO against each baseline.
* Holm-Bonferroni correction across the baseline family.
* No claim of significance without a p-value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats as sps


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    sd: float
    ci95: float          # half-width of the 95 % confidence interval
    minimum: float
    maximum: float

    def pm(self, digits: int = 2) -> str:
        return f"{self.mean:.{digits}f} ± {self.ci95:.{digits}f}"


def t_critical(n: int) -> float:
    """Two-sided 97.5 % Student-t critical value with ``n-1`` d.o.f."""
    return float(sps.t.ppf(0.975, n - 1))


def summarise(x: Sequence[float], t_crit: float | None = None) -> Summary:
    a = np.asarray(list(x), dtype=float)
    n = a.size
    if n < 2:
        raise ValueError("need at least two replications")
    m = float(a.mean())
    sd = float(a.std(ddof=1))
    tc = t_critical(n) if t_crit is None else float(t_crit)
    return Summary(n=n, mean=m, sd=sd, ci95=tc * sd / np.sqrt(n),
                   minimum=float(a.min()), maximum=float(a.max()))


def welch_t_test(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Welch's two-sided t-test; returns ``(t statistic, p value)``.

    Several policies here are deterministic given the seed, so a metric can be
    bit-identical across replications.  Handing such a sample to SciPy makes it
    compute a t statistic out of catastrophic cancellation and warn that the
    result is unreliable, so the degenerate cases are resolved directly: two
    constant samples with the same value are not different, and two constant
    samples with different values differ with certainty.
    """
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    scale = max(abs(float(x.mean())), abs(float(y.mean())), 1e-300)
    sx, sy = float(x.std(ddof=1)), float(y.std(ddof=1))
    if sx <= 1e-12 * scale and sy <= 1e-12 * scale:
        same = abs(float(x.mean()) - float(y.mean())) <= 1e-12 * scale
        return (0.0, 1.0) if same else (float("inf"), 0.0)
    res = sps.ttest_ind(x, y, equal_var=False)
    t, p = float(res.statistic), float(res.pvalue)
    if not np.isfinite(p):
        return (0.0, 1.0)
    return t, p


def holm_bonferroni(pvalues: Dict[str, float], alpha: float = 0.05
                    ) -> Dict[str, dict]:
    """Holm-Bonferroni step-down correction over a family of tests."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, dict] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[key] = {"p_raw": p, "p_holm": adj, "significant": adj < alpha,
                    "rank": i + 1, "family_size": m}
    return out


def ci_half_width(sd: float, n: int, t_crit: float) -> float:
    return t_crit * sd / np.sqrt(n)


def weighted_percentile(bin_edges: np.ndarray, weights: np.ndarray,
                        q: float) -> float:
    """Percentile of a weighted histogram (``q`` in [0, 100]).

    ``bin_edges`` has one more entry than ``weights``.  The mass of a bin is
    assumed uniform in log-latency across that bin, which is the right
    assumption for the logarithmically spaced grid used by the simulator and
    which removes the bin-width quantisation that would otherwise make the
    tail percentiles of different seeds collapse onto the same value.
    """
    w = np.asarray(weights, dtype=float)
    e = np.asarray(bin_edges, dtype=float)
    total = w.sum()
    if total <= 0:
        return float("nan")
    c = np.cumsum(w) / total
    target = q / 100.0
    i = int(np.searchsorted(c, target, side="left"))
    i = min(i, w.size - 1)
    lo = c[i - 1] if i > 0 else 0.0
    hi = c[i]
    frac = (target - lo) / (hi - lo) if hi > lo else 0.0
    frac = min(max(frac, 0.0), 1.0)
    return float(math.exp(math.log(e[i])
                          + frac * (math.log(e[i + 1]) - math.log(e[i]))))


def weighted_tail_fraction(bin_edges: np.ndarray, weights: np.ndarray,
                           threshold: float) -> float:
    """Fraction of a weighted histogram's mass strictly above ``threshold``.

    The bin that straddles the threshold is split in proportion to its
    log-width, matching the interpolation used by :func:`weighted_percentile`.
    """
    w = np.asarray(weights, dtype=float)
    e = np.asarray(bin_edges, dtype=float)
    total = w.sum()
    if total <= 0:
        return float("nan")
    above = float(w[e[:-1] >= threshold].sum())
    i = int(np.searchsorted(e, threshold, side="right")) - 1
    if 0 <= i < w.size and e[i] < threshold < e[i + 1]:
        frac = ((math.log(e[i + 1]) - math.log(threshold))
                / (math.log(e[i + 1]) - math.log(e[i])))
        above += w[i] * frac
    return float(above / total)
