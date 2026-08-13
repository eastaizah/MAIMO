"""Statistics protocol of the artefact, as locked in ``work/CONTRACT.md``.

Every reported quantity is the mean over ``n = 20`` independent replications
(seeds 1..20) after a discarded warm-up, reported as

    mean +- half-width of the 95 % confidence interval,

with the half-width formed from the Student-t critical value
``t_{0.975,19} = 2.093`` and the sample standard deviation over replications.
MAIMO is compared against every other scheme with a **Welch** two-sided t-test
(unequal variances, Welch-Satterthwaite degrees of freedom) and the resulting
p-values are corrected across the baseline family with the **Holm-Bonferroni**
step-down procedure.

Only :func:`scipy.stats.t` is used, for the Student-t distribution function; the
estimators themselves are written out explicitly so that the arithmetic in the
manuscript can be checked line by line.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats as sps

T_CRIT_19_DF = 2.093        # t_{0.975,19}, the value quoted in the manuscript


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Summary:
    """Mean, spread and 95 % confidence interval of one metric."""

    n: int
    mean: float
    sd: float
    ci95: float             # half-width
    lo: float
    hi: float

    def as_dict(self) -> dict:
        return asdict(self)

    def fmt(self, digits: int = 2, unit: str = "") -> str:
        u = (" " + unit) if unit else ""
        return f"{self.mean:.{digits}f} ± {self.ci95:.{digits}f}{u}"


def summarise(x: Sequence[float], t_crit: float | None = None) -> Summary:
    """Mean, sample sd and 95 % CI half-width of a sample of replications.

    ``t_crit`` defaults to the exact Student-t critical value for ``n - 1``
    degrees of freedom; with ``n = 20`` that is 2.093, the value the manuscript
    quotes.
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if n == 0:
        return Summary(0, float("nan"), float("nan"), float("nan"),
                       float("nan"), float("nan"))
    m = float(a.mean())
    if n == 1:
        return Summary(1, m, 0.0, 0.0, m, m)
    sd = float(a.std(ddof=1))
    tc = float(sps.t.ppf(0.975, n - 1)) if t_crit is None else float(t_crit)
    h = tc * sd / np.sqrt(n)
    return Summary(n, m, sd, float(h), m - float(h), m + float(h))


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WelchTest:
    """Welch two-sided t-test of ``a`` (MAIMO) against ``b`` (a baseline)."""

    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    diff: float               # mean_b - mean_a
    t: float
    df: float
    p: float
    p_holm: float = float("nan")
    reject_holm: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def welch_t(a: Sequence[float], b: Sequence[float]) -> WelchTest:
    """Welch's unequal-variance two-sided t-test, written out explicitly."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    nx, ny = x.size, y.size
    mx, my = float(x.mean()), float(y.mean())
    vx = float(x.var(ddof=1)) if nx > 1 else 0.0
    vy = float(y.var(ddof=1)) if ny > 1 else 0.0
    sx, sy = vx / max(nx, 1), vy / max(ny, 1)
    denom = np.sqrt(sx + sy)
    if denom <= 0.0:
        # Both samples are constant: the difference is exact, not statistical.
        p = 0.0 if mx != my else 1.0
        return WelchTest(nx, ny, mx, my, my - mx,
                         float("inf") if mx != my else 0.0,
                         float(nx + ny - 2), p)
    t = (mx - my) / denom
    df_num = (sx + sy) ** 2
    df_den = (sx ** 2 / max(nx - 1, 1)) + (sy ** 2 / max(ny - 1, 1))
    df = df_num / df_den if df_den > 0 else float(nx + ny - 2)
    p = float(2.0 * sps.t.sf(abs(t), df))
    return WelchTest(nx, ny, mx, my, my - mx, float(t), float(df), p)


# ---------------------------------------------------------------------------
def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05
                    ) -> Tuple[List[float], List[bool]]:
    """Holm-Bonferroni step-down correction.

    Returns the adjusted p-values (monotone non-decreasing in the sorted order,
    clipped at 1) and the reject/accept decisions at level ``alpha``.
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p, kind="stable")
    adj_sorted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj_sorted[rank] = min(running, 1.0)
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj.tolist(), (adj <= alpha).tolist()


def compare_family(reference: Sequence[float],
                   others: Dict[str, Sequence[float]],
                   alpha: float = 0.05) -> Dict[str, WelchTest]:
    """Welch tests of ``reference`` against a family of samples, Holm-adjusted.

    The correction is applied across the whole family, which is the multiple
    comparison the editor's statistical-variability requirement asks for.
    """
    keys = list(others)
    tests = [welch_t(reference, others[k]) for k in keys]
    adj, rej = holm_bonferroni([t.p for t in tests], alpha)
    out: Dict[str, WelchTest] = {}
    for k, t, a, r in zip(keys, tests, adj, rej):
        out[k] = WelchTest(t.n_a, t.n_b, t.mean_a, t.mean_b, t.diff, t.t, t.df,
                           t.p, float(a), bool(r))
    return out


# ---------------------------------------------------------------------------
def fmt_p(p: float) -> str:
    """Compact p-value rendering for the manuscript tables."""
    if not np.isfinite(p):
        return "n/a"
    if p == 0.0:
        return "<1e-300"
    if p < 1e-4:
        return f"{p:.1e}"
    if p < 0.001:
        return f"{p:.5f}"
    return f"{p:.4f}"


def reduction_pct(reference: float, value: float) -> float:
    """Percentage reduction of ``value`` relative to ``reference``."""
    return 100.0 * (reference - value) / reference if reference else float("nan")


def convergence_episode(curve: np.ndarray, window: int = 10,
                        tol: float = 0.05) -> int:
    """First index from which a learning curve stays within ``tol`` of its end.

    The plateau level is the mean of the last ``window`` points.  The returned
    index is the earliest point after which every subsequent windowed mean lies
    within ``tol`` (relative) of that level, i.e. the empirical onset of the
    plateau phase.
    """
    c = np.asarray(curve, dtype=float)
    if c.size <= window:
        return int(c.size)
    smooth = np.convolve(c, np.ones(window) / window, mode="valid")
    level = float(smooth[-1])
    scale = max(abs(level), 1e-9)
    ok = np.abs(smooth - level) <= tol * scale
    # earliest index such that all later windowed means are within tolerance
    idx = int(np.argmax(np.cumprod(ok[::-1])[::-1] > 0)) if ok.any() else c.size
    for i in range(ok.size - 1, -1, -1):
        if not ok[i]:
            idx = i + 1
            break
    return int(min(idx + window, c.size))
