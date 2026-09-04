"""The energy arithmetic must reproduce the CONTRACT table to 3 significant
figures, and the one entry that cannot be reproduced must be documented."""

import math

import numpy as np
import pytest

from maimo import energy as en
from maimo.config import JOULES_PER_KWH


def sig3(x):
    if x == 0:
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + 2)


def test_edge_matches_contract_to_3sf():
    assert sig3(en.tier_energy_j(en.EDGE)) == 2.55


def test_device_matches_contract_to_3sf():
    assert sig3(en.tier_energy_j(en.DEVICE)) == 0.0076


def test_cloud_reproduces_the_formula():
    """The CONTRACT's original 16.6 J dropped the amortised idle term; the
    2026-08-06 revision adopted the value the formula actually gives."""
    e = en.tier_energy_j(en.CLOUD)
    bracket = 2550.0 * 0.005 + 900.0 * (1.0 - 195.0 * 0.005) / 195.0
    assert bracket == pytest.approx(12.8654, abs=1e-3)
    assert e == pytest.approx(1.30 * bracket, rel=1e-12)
    assert sig3(e) == 16.7
    assert e == pytest.approx(en.CONTRACT_VALUES_J["cloud"], abs=0.01)
    assert sig3(1.30 * 2550.0 * 0.005) == 16.6   # what the old value computed


def test_hybrid_and_reduction():
    eh = en.hybrid_energy_j()
    assert eh == pytest.approx(en.CONTRACT_VALUES_J["hybrid"], abs=5e-3)
    e = en.reference_tier_energies_j()
    red = 100.0 * (e["cloud"] - eh) / e["cloud"]
    assert red == pytest.approx(67.4, abs=0.1)


def test_aggregate_cross_check():
    """1000 users at 1 inference/s for 1 h."""
    eh = en.hybrid_energy_j()
    kwh = en.aggregate_kwh(eh)
    assert kwh == pytest.approx(3.6e6 * eh / JOULES_PER_KWH, rel=1e-12)
    assert kwh == pytest.approx(5.46, abs=0.02)
    assert en.aggregate_kwh(en.reference_tier_energies_j()["cloud"]) == \
        pytest.approx(16.7, abs=0.05)


def test_utilisation_form_is_equivalent_to_the_contract_form():
    for name, p in en.TIER_PARAMS.items():
        if p.p_idle_w == 0.0 and p.n_per_slot == 1.0:
            continue
        rho = p.n_per_slot * p.t_inf_s
        a = en.tier_energy_j(p)
        b = en.request_energy_j(name, p.t_inf_s, rho)
        assert float(b) == pytest.approx(a, rel=1e-12)


def test_idle_share_grows_as_utilisation_falls():
    hi = en.request_energy_j("cloud", 5e-3, 0.975)
    lo = en.request_energy_j("cloud", 5e-3, 0.400)
    assert lo > hi


def test_no_wh_per_inference_nonsense():
    """25.9 Wh per inference is 93.2 kJ; the corrected model is ~4 orders
    of magnitude smaller."""
    eh = en.hybrid_energy_j()
    assert eh < 10.0
    assert 25.9 * 3600.0 / eh > 1e4
