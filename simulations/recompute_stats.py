"""Recompute the comparison statistics from the stored per-seed results.

The per-seed metric vectors written by ``run_all.py`` (``results/<ID>.json``)
carry the 20 replications of every configuration in a fixed seed order, so the
pairing induced by the common random numbers is available directly on disk and
the statistical protocol can be evaluated without re-running the campaign.

This script rebuilds ``results/comparisons.json`` and ``results/per_class.json``
under the paired protocol (paired two-sided Student t-test on the per-seed
differences, mean paired difference with its 95 % confidence half-width, and
the Wilcoxon signed-rank test as a distribution-free check), applying
Holm-Bonferroni exactly as ``run_all.py`` does.  The marginal summaries --
every mean, standard deviation and confidence interval in ``summary.csv`` --
are untouched: pairing changes the comparisons, not the per-policy summaries.

Usage::

    python recompute_stats.py [--results results] [--no-backup]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict

import numpy as np

from maimo.ablations import ABLATIONS
from maimo.baselines import BASELINES
from maimo.config import CLASS_LABELS
from maimo.stats import (holm_bonferroni, paired_difference, paired_t_test,
                         t_critical, wilcoxon_signed_rank)

SCALAR_METRICS = (
    "accuracy_pct", "latency_mean_ms", "latency_p95_ms", "latency_p99_ms",
    "energy_j", "sla_violation_pct", "cache_hit_pct", "throughput_per_s",
    "carbon_g_per_1000", "pred_error",
)


def load_results(out: str) -> Dict[str, dict]:
    res = {}
    for pid in [s.ident for s in BASELINES] + [a.ident for a in ABLATIONS]:
        with open(f"{out}/{pid}.json", encoding="utf-8") as f:
            res[pid] = json.load(f)
    return res


def comparisons_for(results: Dict[str, dict], tc: float) -> Dict[str, dict]:
    comparisons: Dict[str, dict] = {}
    for family, members in (
            ("baselines", [s.ident for s in BASELINES if s.ident != "B10"]),
            ("ablations", [a.ident for a in ABLATIONS if a.ident != "A0"])):
        ref = results["B10" if family == "baselines" else "A0"]
        fam: Dict[str, dict] = {}
        for m in SCALAR_METRICS:
            praw, extra = {}, {}
            for pid in members:
                x = np.asarray(ref[m], dtype=float)
                y = np.asarray(results[pid][m], dtype=float)
                t, p = paired_t_test(x, y)
                d = paired_difference(x, y, tc)
                w, pw = wilcoxon_signed_rank(x, y)
                praw[pid] = p
                extra[pid] = {"t": t, "test": "paired_t",
                              "mean_diff": d.mean, "diff_ci95": d.ci95,
                              "diff_sd": d.sd,
                              "wilcoxon_stat": w, "p_wilcoxon": pw}
            corrected = holm_bonferroni(praw)
            for pid in members:
                corrected[pid].update(extra[pid])
            fam[m] = corrected
        comparisons[family] = fam
    return comparisons


def per_class_for(results: Dict[str, dict], tc: float) -> dict:
    per_class: dict = {}
    for pid in ("B1", "B10"):
        r = results[pid]["per_class"]
        per_class[pid] = {k: np.asarray(v).tolist() for k, v in r.items()}
    per_class["class_labels"] = list(CLASS_LABELS)
    per_class["tests"] = {}
    for c, label in enumerate(CLASS_LABELS):
        entry = {}
        for m in ("latency_mean_ms", "latency_p95_ms", "latency_p99_ms"):
            a = np.asarray(results["B10"]["per_class"][m])[:, c]
            b = np.asarray(results["B1"]["per_class"][m])[:, c]
            t, p = paired_t_test(a, b)
            d = paired_difference(a, b, tc)
            w, pw = wilcoxon_signed_rank(a, b)
            entry[m] = {"t": t, "p_raw": p, "test": "paired_t",
                        "mean_diff": d.mean, "diff_ci95": d.ci95,
                        "diff_sd": d.sd,
                        "wilcoxon_stat": w, "p_wilcoxon": pw}
        per_class["tests"][label] = entry
    flat = {f"{lab}|{m}": per_class["tests"][lab][m]["p_raw"]
            for lab in CLASS_LABELS
            for m in ("latency_mean_ms", "latency_p95_ms", "latency_p99_ms")}
    for key, v in holm_bonferroni(flat).items():
        lab, m = key.split("|")
        per_class["tests"][lab][m].update(v)
    return per_class


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--no-backup", action="store_true",
                    help="do not keep the superseded files")
    args = ap.parse_args()

    results = load_results(args.results)
    n = len(results["B10"]["latency_mean_ms"])
    tc = t_critical(n)

    if not args.no_backup:
        for name, keep in (("comparisons", "comparisons_welch"),
                           ("per_class", "per_class_welch")):
            src = f"{args.results}/{name}.json"
            dst = f"{args.results}/{keep}.json"
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copyfile(src, dst)

    with open(f"{args.results}/comparisons.json", "w", encoding="utf-8") as f:
        json.dump(comparisons_for(results, tc), f, indent=1)
    with open(f"{args.results}/per_class.json", "w", encoding="utf-8") as f:
        json.dump(per_class_for(results, tc), f, indent=1)
    print(f"recomputed paired statistics from {n} replications "
          f"into {args.results}/comparisons.json and per_class.json")


if __name__ == "__main__":
    main()
