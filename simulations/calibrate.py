"""Calibration harness.

Runs a short version of the experiment and prints the quantities that are
locked in ``work/CONTRACT.md``, so that the calibrated parameters in
``maimo/config.py`` can be checked without paying for the full 3.6e5-slot
horizon.  This script never modifies any result: it only reports.

Usage::

    python calibrate.py [--intervals 4000] [--seeds 4] [--policies B1,B10]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from maimo import energy as en
from maimo.ablations import ABLATION_BY_ID
from maimo.baselines import BASELINE_BY_ID
from maimo.config import DEFAULT, CLASS_LABELS, quick
from maimo.experiment import PredictorCache, run_policy
from maimo.sim import build_context

TARGETS = {
    "latency_headline_maimo": 12.0,
    "latency_headline_cloud": 22.0,
    "latency_urllc_maimo": 2.1,
    "latency_urllc_cloud": 18.5,
    "latency_embb_maimo": 35.0,
    "latency_embb_cloud": 120.0,
    "latency_mmtc_maimo": 180.0,
    "latency_mmtc_cloud": 450.0,
    "energy_cloud": 16.73,
    "energy_edge": 2.55,
    "energy_device": 0.0076,
    "energy_maimo": 5.46,
    "reduction_pct": 67.4,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intervals", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--policies", type=str, default="B1,B2,B3,B10")
    ap.add_argument("--train-intervals", type=int, default=2560)
    args = ap.parse_args()

    cfg = quick(DEFAULT, horizon_slots=args.intervals * 10 - 5000,
                warmup_slots=5000, n_seeds=args.seeds)
    seeds = list(range(1, args.seeds + 1))
    t0 = time.perf_counter()
    eval_ctx = build_context(cfg, seeds, args.intervals)
    train_ctx = build_context(cfg, [900_000 + s for s in seeds],
                              args.train_intervals)
    cache = PredictorCache(cfg, seeds)
    print(f"context+predictors: {time.perf_counter() - t0:.1f} s  "
          f"(predictor MAPE "
          f"{np.mean([h['test_mape_pct'] for h in cache.history]):.2f} %)")

    results = {}
    for pid in args.policies.split(","):
        spec = BASELINE_BY_ID.get(pid) or ABLATION_BY_ID[pid]
        results[pid] = run_policy(cfg, spec, eval_ctx, train_ctx, cache)

    print("\nPer-class mean latency (ms)")
    header = f"{'':4s}" + "".join(f"{c[:22]:>24s}" for c in CLASS_LABELS)
    print(header)
    for pid, r in results.items():
        row = "".join(f"{np.mean(r['per_class']['latency_mean_ms'][:, c]):24.2f}"
                      for c in range(4))
        print(f"{pid:4s}{row}")

    print("\nRouting split (headline class) and per-tier energy")
    for pid, r in results.items():
        a = r["alpha"].mean(axis=0)
        te = np.nanmean(np.where(r["tier_energy_j"] > 0,
                                 r["tier_energy_j"], np.nan), axis=0)
        print(f"{pid:4s} alpha=({a[0]:.3f},{a[1]:.3f},{a[2]:.3f})  "
              f"E_tier=({te[0]:.3f}, {te[1]:.3f}, {te[2]*1e3:.2f} mJ)  "
              f"E={np.mean(r['energy_j']):.3f} J  "
              f"cache={np.mean(r['cache_hit_pct']):.1f} %  "
              f"thr={np.mean(r['throughput_per_s']):.0f}/s")

    if "B1" in results and "B10" in results:
        e1 = np.mean(results["B1"]["energy_j"])
        e10 = np.mean(results["B10"]["energy_j"])
        print(f"\nenergy reduction vs cloud-only: "
              f"{100 * (e1 - e10) / e1:.2f} %  "
              f"(target {TARGETS['reduction_pct']} %)")

    print("\nCONTRACT reference energies (analytic, not simulated):")
    for k, v in en.reference_tier_energies_j().items():
        print(f"  {k:7s} {v:.4g} J  (contract {en.CONTRACT_VALUES_J[k]:g})")
    print(f"  hybrid  {en.hybrid_energy_j():.4g} J  "
          f"(contract {en.CONTRACT_VALUES_J['hybrid']:g})")
    print(f"\ntotal wall clock {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
