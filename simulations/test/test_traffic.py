"""The traffic generator must be non-stationary, bursty and reproducible."""

import numpy as np
import pytest

from maimo import traffic as tr
from maimo.config import DEFAULT, SECONDS_PER_DAY, SERVICE_CLASSES


def test_diurnal_profile_has_unit_mean_over_a_day():
    t = np.linspace(0.0, SECONDS_PER_DAY, 20000, endpoint=False)
    assert tr.diurnal_factor(t, DEFAULT).mean() == pytest.approx(1.0, abs=1e-3)


def test_diurnal_peak_is_at_the_configured_hour():
    t = np.arange(0.0, SECONDS_PER_DAY, 60.0)
    d = tr.diurnal_factor(t, DEFAULT)
    peak_hour = t[int(np.argmax(d))] / 3600.0
    assert abs(peak_hour - DEFAULT.diurnal_peak_hour) < 1.5


def test_weekly_factor_drops_at_the_weekend():
    weekday = float(tr.weekly_factor(np.array([2.5 * SECONDS_PER_DAY]), DEFAULT)[0])
    weekend = float(tr.weekly_factor(np.array([5.5 * SECONDS_PER_DAY]), DEFAULT)[0])
    assert weekday == pytest.approx(1.0, abs=1e-3)
    assert weekend == pytest.approx(DEFAULT.weekend_factor, abs=1e-2)


def test_mmpp_preserves_the_mean_and_adds_correlation():
    rng = np.random.default_rng(11)
    m = tr.mmpp2_multiplier(rng, 40000, 10.0, DEFAULT)
    assert m.mean() == pytest.approx(1.0, rel=0.15)
    assert m.max() > 1.5                       # bursts really happen
    lag1 = np.corrcoef(m[:-1], m[1:])[0, 1]
    assert lag1 > 0.9                          # bursts are correlated in time


def test_trace_is_reproducible_and_has_the_right_mean_load():
    a = tr.make_trace(DEFAULT, 3, 2000)
    b = tr.make_trace(DEFAULT, 3, 2000)
    assert np.array_equal(a.lam, b.lam)
    c = tr.make_trace(DEFAULT, 4, 2000)
    assert not np.array_equal(a.lam, c.lam)
    nominal = sum(DEFAULT.n_sessions() * s.share_of_sessions
                  * s.rate_per_session for s in SERVICE_CLASSES)
    # a 2000-interval window is only 5.6 h, so the diurnal factor need not
    # average to one; require the same order of magnitude
    assert 0.4 * nominal < a.total().mean() < 1.8 * nominal


def test_poisson_arrivals_have_the_right_mean_and_variance():
    rng = np.random.default_rng(5)
    lam = np.full((4, 2), 100.0)
    a = tr.sample_arrivals(rng, lam, 2000, DEFAULT)
    assert a.shape == (4, 2000, 2)
    assert a.mean() == pytest.approx(100.0, rel=0.02)
    assert a.var() == pytest.approx(100.0, rel=0.15)
