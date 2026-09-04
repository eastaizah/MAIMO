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


# --- paired protocol (common random numbers) -------------------------------


def test_paired_t_matches_scipy_ttest_rel():
    rng = np.random.default_rng(11)
    shared = rng.normal(0.0, 3.0, 20)          # the seed effect
    a = shared + rng.normal(0.0, 0.4, 20)
    b = shared + rng.normal(0.5, 0.4, 20)
    t, p = st.paired_t_test(a, b)
    ref = sps.ttest_rel(a, b)
    assert t == pytest.approx(float(ref.statistic), rel=1e-12)
    assert p == pytest.approx(float(ref.pvalue), rel=1e-12)


def test_paired_t_is_the_one_sample_t_on_the_differences():
    rng = np.random.default_rng(12)
    a = rng.normal(4.0, 1.5, 20)
    b = rng.normal(3.4, 2.7, 20)
    t, p = st.paired_t_test(a, b)
    ref = sps.ttest_1samp(np.asarray(a) - np.asarray(b), 0.0)
    assert t == pytest.approx(float(ref.statistic), rel=1e-12)
    assert p == pytest.approx(float(ref.pvalue), rel=1e-12)


def test_paired_t_equals_the_hand_formula():
    """t = mean(d) / (sd(d) / sqrt(n)) on a hand-computed example.

    d = a - b = (1, 2, 3, 4, 5): mean 3, sum of squared deviations 10, so
    sd = sqrt(10/4) = sqrt(2.5) and t = 3 sqrt(5) / sqrt(2.5) = 3 sqrt(2) =
    4.2426406..., on n - 1 = 4 degrees of freedom.
    """
    a = np.array([11.0, 12.0, 13.0, 14.0, 15.0])
    b = np.full(5, 10.0)
    d = a - b
    assert d.mean() == pytest.approx(3.0)
    assert d.std(ddof=1) == pytest.approx(np.sqrt(2.5))
    t, p = st.paired_t_test(a, b)
    assert t == pytest.approx(d.mean() / (d.std(ddof=1) / np.sqrt(5)), rel=1e-12)
    assert t == pytest.approx(3.0 * np.sqrt(2.0), rel=1e-12)
    assert t == pytest.approx(4.242640687119285, rel=1e-12)
    assert p == pytest.approx(2.0 * sps.t.sf(abs(t), 4), rel=1e-12)
    assert p == pytest.approx(0.0132356, abs=1e-7)


def test_paired_t_all_zero_differences_is_no_difference():
    x = np.array([1.0, 5.0, 2.0, 9.0, 3.0])
    t, p = st.paired_t_test(x, x.copy())
    assert (t, p) == (0.0, 1.0)
    # also when both samples are constant and equal
    c = np.full(20, 7.25)
    assert st.paired_t_test(c, c.copy()) == (0.0, 1.0)


def test_paired_t_constant_non_zero_difference_is_certainty():
    x = np.array([1.0, 5.0, 2.0, 9.0, 3.0, 4.0])
    t, p = st.paired_t_test(x + 0.5, x)
    assert p == 0.0 and t == float("inf")
    t, p = st.paired_t_test(x, x + 0.5)
    assert p == 0.0 and t == float("-inf")


def test_paired_t_is_more_powerful_under_common_random_numbers():
    """Positively correlated pairs: the paired test must see what Welch cannot."""
    rng = np.random.default_rng(13)
    shared = rng.normal(0.0, 10.0, 20)     # dominant between-seed variance
    a = shared
    b = shared + 0.5                       # a small, perfectly consistent shift
    _, p_paired = st.paired_t_test(a, b)
    _, p_welch = st.welch_t_test(a, b)
    assert p_paired < 1e-6 < p_welch
    assert p_welch > 0.5


def test_wilcoxon_matches_scipy():
    rng = np.random.default_rng(14)
    a = rng.normal(0.0, 1.0, 20)
    b = rng.normal(0.4, 1.0, 20)
    w, p = st.wilcoxon_signed_rank(a, b)
    ref = sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    assert w == pytest.approx(float(ref.statistic), rel=1e-12)
    assert p == pytest.approx(float(ref.pvalue), rel=1e-12)


def test_wilcoxon_degenerate_cases():
    x = np.array([1.0, 5.0, 2.0, 9.0, 3.0, 4.0, 8.0])
    assert st.wilcoxon_signed_rank(x, x.copy()) == (0.0, 1.0)
    # a constant non-zero shift puts every signed rank on the same side
    w, p = st.wilcoxon_signed_rank(x + 0.5, x)
    assert w == 0.0 and p == pytest.approx(2.0 / 2 ** x.size, rel=1e-9)


def test_paired_difference_reports_mean_sd_and_half_width():
    rng = np.random.default_rng(15)
    a = rng.normal(6.0, 1.0, 20)
    b = rng.normal(5.0, 1.0, 20)
    d = np.asarray(a) - np.asarray(b)
    s = st.paired_difference(a, b)
    assert s.n == 20
    assert s.mean == pytest.approx(float(d.mean()), rel=1e-12)
    assert s.sd == pytest.approx(float(d.std(ddof=1)), rel=1e-12)
    lo, hi = sps.t.interval(0.95, 19, loc=d.mean(), scale=sps.sem(d))
    assert s.ci95 == pytest.approx((hi - lo) / 2.0, rel=1e-12)
    # the interval covers zero exactly when the paired t-test does not reject
    _, p = st.paired_t_test(a, b)
    assert (abs(s.mean) > s.ci95) == (p < 0.05)


def test_paired_helpers_reject_mismatched_lengths():
    a, b = np.arange(5.0), np.arange(6.0)
    for fn in (st.paired_t_test, st.wilcoxon_signed_rank, st.paired_difference):
        with pytest.raises(ValueError):
            fn(a, b)
