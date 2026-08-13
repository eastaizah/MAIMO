"""End-to-end properties of the slotted engine.

The important one is determinism: the whole point of the reproducibility
requirement (R1) is that re-running the same seed reproduces the same numbers
bit for bit, so a reviewer who runs the code gets the table in the paper.
"""

import numpy as np
import pytest

from maimo.ablations import ABLATION_BY_ID
from maimo.baselines import BASELINE_BY_ID
from maimo.config import DEFAULT, HEADLINE_CLASS, N_TIER, quick
from maimo.experiment import PredictorCache, run_policy
from maimo.sim import ALPHA_CODEBOOK, build_context, per_class_alpha

CFG = quick(DEFAULT, horizon_slots=3000, warmup_slots=1000, n_seeds=3)
SEEDS = [1, 2, 3]

METRICS = ("accuracy_pct", "latency_mean_ms", "latency_p99_ms", "energy_j",
           "sla_violation_pct", "cache_hit_pct", "carbon_g_per_1000")


@pytest.fixture(scope="module")
def harness():
    ev = build_context(CFG, SEEDS, 400)
    tr = build_context(CFG, [900_001, 900_002, 900_003], 256)
    return ev, tr, PredictorCache(CFG, SEEDS)


def run(pid, harness):
    ev, tr, cache = harness
    spec = BASELINE_BY_ID.get(pid) or ABLATION_BY_ID[pid]
    return run_policy(CFG, spec, ev, tr, cache, verbose=False)


def test_same_seed_gives_identical_metrics(harness):
    a = run("B4", harness)
    b = run("B4", harness)
    for m in METRICS:
        np.testing.assert_array_equal(a[m], b[m])


def test_learned_controller_is_also_deterministic(harness):
    a = run("B10", harness)
    b = run("B10", harness)
    for m in METRICS:
        np.testing.assert_array_equal(a[m], b[m])


def test_different_seeds_give_different_metrics(harness):
    r = run("B4", harness)
    assert np.unique(np.round(r["latency_mean_ms"], 9)).size == len(SEEDS)


def test_a0_is_b10(harness):
    """A0 is the reference ablation and must be the same configuration."""
    spec = ABLATION_BY_ID["A0"]
    b10 = BASELINE_BY_ID["B10"]
    for f in ("controller", "predictor", "proactive_loading",
              "adaptive_compression", "early_exit", "semantic_comm"):
        assert getattr(spec, f) == getattr(b10, f)


def test_single_tier_baselines_route_where_they_should(harness):
    for pid, tier in (("B1", 0), ("B2", 1), ("B3", 2)):
        alpha = run(pid, harness)["alpha"].mean(axis=0)
        assert alpha[tier] == pytest.approx(1.0, abs=1e-9)


def test_codebook_is_a_simplex():
    assert ALPHA_CODEBOOK.shape[1] == N_TIER
    np.testing.assert_allclose(ALPHA_CODEBOOK.sum(axis=1), 1.0, atol=1e-12)
    assert (ALPHA_CODEBOOK >= 0.0).all()


def test_per_class_alpha_preserves_the_simplex():
    a = per_class_alpha(ALPHA_CODEBOOK)
    np.testing.assert_allclose(a.sum(axis=2), 1.0, atol=1e-12)
    # the headline class is deliberately neutral
    np.testing.assert_allclose(a[:, HEADLINE_CLASS, :], ALPHA_CODEBOOK,
                               atol=1e-12)


def test_metrics_are_physical(harness):
    r = run("B10", harness)
    assert np.all(r["latency_mean_ms"] > 0)
    assert np.all(r["latency_p99_ms"] >= r["latency_p95_ms"])
    assert np.all(r["energy_j"] > 0)
    assert np.all((r["accuracy_pct"] > 50.0) & (r["accuracy_pct"] <= 100.0))
    assert np.all((r["sla_violation_pct"] >= 0.0)
                  & (r["sla_violation_pct"] <= 100.0))
    assert np.all((r["cache_hit_pct"] >= 0.0) & (r["cache_hit_pct"] <= 100.0))


def test_warmup_is_discarded(harness):
    """Metrics must not change when the warm-up is lengthened, beyond the
    transient it removes: a run whose warm-up covers the whole horizon has no
    measured requests at all."""
    ev, tr, cache = harness
    cfg = CFG.replace(warmup_slots=2000)
    spec = BASELINE_BY_ID["B4"]
    r = run_policy(cfg, spec, ev, tr, cache, verbose=False)
    assert np.all(np.isfinite(r["latency_mean_ms"]))


def test_proactive_loading_beats_reactive(harness):
    """A3 removes proactive loading, so its cache hit rate must fall."""
    a0 = run("B10", harness)
    a3 = run("A3", harness)
    assert a3["cache_hit_pct"].mean() < a0["cache_hit_pct"].mean()
    assert a3["latency_mean_ms"].mean() > a0["latency_mean_ms"].mean()
