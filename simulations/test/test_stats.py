"""The statistics helpers must match scipy and the locked protocol."""

import numpy as np
import pytest
from scipy import stats as sps

from maimo import stats as st


def test_t_critical_19_df_matches_contract():
    assert st.t_critical(20) == pytest.approx(2.093, abs=5e-4)


def test_ci_half_width_matches_scipy():
    rng = np.random.default_rng(3)
    x = rng.normal(10.0, 2.0, 20)
    s = st.summarise(x)
    lo, hi = sps.t.interval(0.95, len(x) - 1, loc=x.mean(),
                            scale=sps.sem(x))
    assert s.ci95 == pytest.approx((hi - lo) / 2.0, rel=1e-12)
    assert s.sd == pytest.approx(x.std(ddof=1), rel=1e-12)
    assert s.minimum == x.min() and s.maximum == x.max()


def test_welch_matches_scipy():
    rng = np.random.default_rng(4)
    a = rng.normal(0.0, 1.0, 20)
    b = rng.normal(0.6, 2.5, 20)
    t, p = st.welch_t_test(a, b)
    ref = sps.ttest_ind(a, b, equal_var=False)
    assert t == pytest.approx(float(ref.statistic), rel=1e-12)
    assert p == pytest.approx(float(ref.pvalue), rel=1e-12)


def test_holm_bonferroni_step_down():
    p = {"a": 0.001, "b": 0.013, "c": 0.020, "d": 0.600}
    out = st.holm_bonferroni(p, alpha=0.05)
    assert out["a"]["p_holm"] == pytest.approx(0.004)
    assert out["b"]["p_holm"] == pytest.approx(0.039)
    assert out["c"]["p_holm"] == pytest.approx(0.040)
    assert out["d"]["p_holm"] == pytest.approx(0.600)
    assert out["a"]["significant"] and out["b"]["significant"]
    assert out["c"]["significant"] and not out["d"]["significant"]


def test_holm_is_monotone():
    p = {"a": 0.04, "b": 0.0001, "c": 0.049}
    out = st.holm_bonferroni(p)
    ordered = sorted(out.items(), key=lambda kv: kv[1]["rank"])
    vals = [v["p_holm"] for _, v in ordered]
    assert all(vals[i] <= vals[i + 1] + 1e-15 for i in range(len(vals) - 1))


def test_weighted_percentile_matches_numpy_on_a_log_grid():
    edges = np.logspace(np.log10(0.05), np.log10(5000.0), 1201)
    rng = np.random.default_rng(7)
    x = rng.lognormal(mean=2.5, sigma=0.8, size=400_000)
    w, _ = np.histogram(x, bins=edges)
    for q in (50.0, 90.0, 95.0, 99.0):
        got = st.weighted_percentile(edges, w.astype(float), q)
        assert got == pytest.approx(float(np.percentile(x, q)), rel=0.01)


def test_weighted_percentile_interpolates_inside_the_bin():
    """Two histograms differing only in sub-bin mass must not report the same
    percentile; this is the quantisation the interpolation removes."""
    edges = np.array([1.0, 2.0, 4.0, 8.0])
    a = st.weighted_percentile(edges, np.array([90.0, 9.0, 1.0]), 95.0)
    b = st.weighted_percentile(edges, np.array([90.0, 6.0, 4.0]), 95.0)
    assert 2.0 < a < 4.0 and 2.0 < b < 4.0
    assert a != b
    # the bin-centre rule would have returned sqrt(2*4) for both
    assert abs(a - np.sqrt(8.0)) > 1e-6


def test_weighted_tail_fraction():
    edges = np.array([1.0, 2.0, 4.0, 8.0])
    w = np.array([50.0, 30.0, 20.0])
    assert st.weighted_tail_fraction(edges, w, 4.0) == pytest.approx(0.20)
    assert st.weighted_tail_fraction(edges, w, 0.5) == pytest.approx(1.00)
    assert st.weighted_tail_fraction(edges, w, 16.0) == pytest.approx(0.00)
    # threshold at the geometric midpoint of the middle bin splits it in half
    mid = st.weighted_tail_fraction(edges, w, np.sqrt(2.0 * 4.0))
    assert mid == pytest.approx((20.0 + 15.0) / 100.0)
