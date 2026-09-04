"""Statistics protocol locked in ``work/CONTRACT.md``.

* 20 independent replications (seeds 1..20).
* Report ``mean +/- half-width of the 95 % confidence interval`` using
  Student's t with 19 degrees of freedom, ``t_{0.975,19} = 2.093``.
* Report the standard deviation in the reproducibility table.
* Common random numbers: replication ``s`` presents an identical traffic and
  channel realisation to every policy, so observations are paired by seed.
  The comparison protocol is therefore the **paired** (one-sample-on-
  differences) two-sided Student t-test on the 20 per-seed differences, with
  ``n - 1 = 19`` degrees of freedom, reported together with the mean paired
  difference and the half-width of its 95 % confidence interval.
* Wilcoxon signed-rank test on the same differences as a distribution-free
  robustness check.
* Holm-Bonferroni correction across the baseline family.
* No claim of significance without a p-value.

``welch_t_test`` is retained for reference and for unpaired situations, but it
is *not* the protocol used for the policy comparisons: discarding the pairing
would ignore the positive between-policy correlation that the common random
numbers deliberately induce.
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


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Two-sided paired Student t-test; returns ``(t statistic, p value)``.

    This is the test the experiment design licenses.  Every policy is driven
    by the same traffic traces and the same channel realisations, so entry
    ``i`` of ``a`` and entry ``i`` of ``b`` are two measurements of the *same*
    environment realisation.  The test is the one-sample t-test on the
    differences ``d = a - b``, ``t = mean(d) / (sd(d) / sqrt(n))`` with
    ``n - 1`` degrees of freedom.

    The degenerate cases are resolved directly, as in :func:`welch_t_test`:
    several policies are deterministic given the seed, so ``d`` can be
    bit-identical across replications and handing it to SciPy would produce a
    statistic built out of catastrophic cancellation.  An all-zero difference
    vector is no difference at all, ``(0.0, 1.0)``; a constant non-zero
    difference vector differs with certainty, ``(+/-inf, 0.0)``, signed with
    the direction of the difference.
    """
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if x.size != y.size:
        raise ValueError("paired samples must have the same length")
    if x.size < 2:
        raise ValueError("need at least two replications")
    d = x - y
    scale = max(abs(float(x.mean())), abs(float(y.mean())), 1e-300)
    sd = float(d.std(ddof=1))
    md = float(d.mean())
    if sd <= 1e-12 * scale:
        if abs(md) <= 1e-12 * scale:
            return (0.0, 1.0)
        return (math.copysign(float("inf"), md), 0.0)
    res = sps.ttest_rel(x, y)
    t, p = float(res.statistic), float(res.pvalue)
    if not np.isfinite(p):
        return (0.0, 1.0)
    return t, p


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]
                         ) -> Tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test on the paired differences.

    The distribution-free companion to :func:`paired_t_test`, used as a
    robustness check against the normality assumption on the differences.
    Returns ``(statistic, p value)``.  An all-zero difference vector carries
    no signed ranks at all and is reported as ``(0.0, 1.0)`` rather than being
    handed to SciPy, which rejects it.
    """
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if x.size != y.size:
        raise ValueError("paired samples must have the same length")
    if x.size < 2:
        raise ValueError("need at least two replications")
    d = x - y
    scale = max(abs(float(x.mean())), abs(float(y.mean())), 1e-300)
    if np.all(np.abs(d) <= 1e-12 * scale):
        return (0.0, 1.0)
    res = sps.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    stat, p = float(res.statistic), float(res.pvalue)
    if not np.isfinite(p):
        return (stat, 1.0)
    return stat, p


def paired_difference(a: Sequence[float], b: Sequence[float],
                      t_crit: float | None = None) -> Summary:
    """Summary of the paired differences ``d = a - b``.

    The paired design estimates ``mean(d)``, not the difference of two
    independent means, and the interval ``mean(d) +/- t_{0.975,n-1} sd(d) /
    sqrt(n)`` is the quantity a reader should be given alongside the p-value.
    The returned :class:`Summary` carries ``mean``, ``sd`` and ``ci95`` of the
    difference vector.
    """
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if x.size != y.size:
        raise ValueError("paired samples must have the same length")
    return summarise(x - y, t_crit)


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
