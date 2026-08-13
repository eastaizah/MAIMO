"""The radio layer must match hand-computed 3GPP TR 38.901 UMa values."""

import math

import numpy as np
import pytest

from maimo import channel as ch
from maimo.config import DEFAULT


def test_breakpoint_distance():
    # d'_BP = 4 h'_BS h'_UT f_c / c = 4 * 24 * 0.5 * 3.5e9 / 2.99792458e8
    assert ch.breakpoint_distance_m(DEFAULT) == pytest.approx(560.36, abs=0.1)


def test_uma_los_hand_computed():
    """f_c = 3.5 GHz, h_BS = 25 m, h_UT = 1.5 m, d_2D = 100 m.

    d_3D = sqrt(100^2 + 23.5^2) = 102.7241 m, below the 560 m breakpoint, so
    PL = 28 + 22 log10(102.7241) + 20 log10(3.5)
       = 28 + 44.2568 + 10.8814 = 83.1381 dB.
    """
    pl = float(ch.uma_los_pathloss_db(100.0, DEFAULT))
    assert pl == pytest.approx(83.1381, abs=1e-3)


def test_uma_nlos_hand_computed():
    """PL'_NLOS = 13.54 + 39.08 log10(102.7241) + 20 log10(3.5) - 0.6*(1.5-1.5)
              = 13.54 + 78.6162 + 10.8814 = 103.0375 dB, and NLOS is the
    maximum of that and the LOS value."""
    pl = float(ch.uma_nlos_pathloss_db(100.0, DEFAULT))
    assert pl == pytest.approx(103.0375, abs=1e-3)
    assert pl > float(ch.uma_los_pathloss_db(100.0, DEFAULT))


def test_los_probability_limits():
    assert float(ch.uma_los_probability(1.0, DEFAULT)) == pytest.approx(1.0, abs=1e-6)
    assert float(ch.uma_los_probability(18.0, DEFAULT)) == pytest.approx(1.0, abs=1e-6)
    far = float(ch.uma_los_probability(1000.0, DEFAULT))
    assert 0.0 < far < 0.05


def test_beyond_breakpoint_slope_increases():
    d = np.array([100.0, 560.0, 2000.0])
    pl = ch.uma_los_pathloss_db(d, DEFAULT)
    assert pl[2] - pl[1] > pl[1] - pl[0]


def test_sinr_is_formed_in_the_linear_domain():
    """Doubling the transmit power must raise the SINR by exactly 3 dB."""
    a = ch.linear_sinr(23.0, 100.0, 0.0, 1.0, -95.0)
    b = ch.linear_sinr(26.0, 100.0, 0.0, 1.0, -95.0)
    assert 10.0 * math.log10(b / a) == pytest.approx(3.0, abs=1e-9)


def test_band_averaged_gain_has_unit_mean_and_ordered_variance():
    rng = np.random.default_rng(0)
    c = ch.cdl_band_averaged_gain(rng, 20000, 300.0, 12, 20e6)
    a = ch.cdl_band_averaged_gain(rng, 20000, 30.0, 6, 20e6)
    assert c.mean() == pytest.approx(1.0, abs=0.03)
    assert a.mean() == pytest.approx(1.0, abs=0.03)
    # a longer delay spread over the same band gives more frequency diversity
    assert c.var() < a.var()


def test_rate_is_capped_and_monotone():
    r_low = ch.shannon_rate_bps(1.0, 20e6, 1, DEFAULT)
    r_high = ch.shannon_rate_bps(1e6, 20e6, 1, DEFAULT)
    assert r_high > r_low
    assert r_high <= DEFAULT.max_spectral_efficiency * 20e6 + 1.0


def test_channel_pool_is_deterministic():
    a = ch.ChannelPool(DEFAULT, 7)
    b = ch.ChannelPool(DEFAULT, 7)
    for k in a.rate_bps:
        assert np.array_equal(a.rate_bps[k], b.rate_bps[k])
