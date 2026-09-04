"""Run the full locked experiment: every policy over every seed.

Protocol (``work/CONTRACT.md``): 20 independent replications, seeds 1..20,
each 3.6e5 simulated slots after a 1e4-slot warm-up that is discarded.  Every
policy sees the *same* traffic traces and the same channel realisations
(common random numbers), so the comparison is paired; the learned controllers
are trained on a disjoint traffic window and then frozen, so every reported
number is out of sample.

Outputs, all under ``results/``:

``<ID>.json``        per-seed metric vectors for one policy
``summary.csv``      one row per policy, mean and 95 % CI half-width
``comparisons.json`` paired t-tests against MAIMO with Holm-Bonferroni
``carbon.json``      carbon strategies with per-seed values
``convergence.json`` PPO/DQN learning curves
``meta.json``        configuration, environment and timing

Usage::

    python run_all.py                 # the locked protocol, ~N minutes
    python run_all.py --quick         # a short smoke run
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List

# The tensors here are tiny (a 64-unit MLP and a 32-unit BiLSTM over short
# sequences), so BLAS and torch intra-op parallelism cost far more in thread
# synchronisation than they save in arithmetic, and oversubscribing the cores
# also makes the wall-clock time depend on what else the machine is doing,
# which would undermine reproducibility.  This must run before numpy or torch
# is imported.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:      # already initialised by an earlier import
    pass

from maimo import __version__
from maimo import carbon as cb
from maimo import energy as en
from maimo.ablations import ABLATIONS
from maimo.baselines import BASELINES
from maimo.config import CLASS_LABELS, DEFAULT, TIERS, quick
from maimo.experiment import PredictorCache, run_policy
from maimo.sim import build_context
from maimo.stats import (holm_bonferroni, paired_difference, paired_t_test,
                         summarise, t_critical, wilcoxon_signed_rank)

RESULTS = "results"

# Metrics that are single numbers per seed (the locked metric set).
SCALAR_METRICS = (
    "accuracy_pct", "latency_mean_ms", "latency_p95_ms", "latency_p99_ms",
    "energy_j", "sla_violation_pct", "cache_hit_pct", "throughput_per_s",
    "carbon_g_per_1000", "pred_error",
)
# Lower is better for all of these except accuracy, cache hit and throughput.
HIGHER_IS_BETTER = {"accuracy_pct", "cache_hit_pct", "throughput_per_s"}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:                                   # pragma: no cover
        return "unknown"


def to_list(x):
    a = np.asarray(x)
    return [None if not np.isfinite(v) else float(v) for v in a.ravel()] \
        if a.ndim == 1 else a.tolist()


def policy_record(res: dict) -> dict:
    rec = {"ident": res["ident"], "name": res["name"],
           "description": res["spec"].description,
           "train_seconds": res["train_seconds"],
           "eval_seconds": res["eval_seconds"]}
    for m in SCALAR_METRICS:
        rec[m] = to_list(res[m])
    rec["alpha"] = np.asarray(res["alpha"]).tolist()
    rec["tier_energy_j"] = np.nan_to_num(
        np.asarray(res["tier_energy_j"]), nan=0.0).tolist()
    rec["energy_parts_j"] = np.nan_to_num(
        np.asarray(res["energy_parts_j"]), nan=0.0).tolist()
    rec["per_class"] = {
        k: np.asarray(v).tolist() for k, v in res["per_class"].items()}
    return rec


def carbon_for_policy(cfg, res, t_s: np.ndarray) -> Dict[str, list]:
    """Per-seed carbon under each shifting strategy."""
    alpha = np.asarray(res["alpha"])                       # (G, 3)
    e_tier = np.nan_to_num(np.asarray(res["tier_energy_j"]), nan=0.0)
    out: Dict[str, list] = {}
    caveats: Dict[str, str] = {}
    ideal: Dict[str, bool] = {}
    for i in range(alpha.shape[0]):
        split = {t: float(alpha[i, j] * e_tier[i, j])
                 for j, t in enumerate(TIERS)}
        for r in cb.carbon_strategies(cfg, t_s, split):
            out.setdefault(r.strategy, []).append(r.g_per_1000)
            out.setdefault(r.strategy + "|reduction_pct", []).append(
                r.reduction_pct)
            caveats[r.strategy] = r.caveat
            ideal[r.strategy] = r.idealised
    return {"values": out, "caveats": caveats, "idealised": ideal}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="short smoke run instead of the locked protocol")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--out", type=str, default=RESULTS)
    args = ap.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)

    cfg = quick(DEFAULT) if args.quick else DEFAULT
    if args.seeds:
        cfg = cfg.replace(n_seeds=args.seeds)
    seeds = list(range(1, cfg.n_seeds + 1))
    n_iv = cfg.control_intervals()
    n_train_iv = cfg.train_intervals()

    t_start = time.perf_counter()
    print(f"MAIMO simulator {__version__} | {cfg.n_seeds} seeds x {n_iv} "
          f"control intervals ({cfg.horizon_slots + cfg.warmup_slots} slots "
          f"of {cfg.t_slot_s} s)")

    eval_ctx = build_context(cfg, seeds, n_iv)
    train_ctx = build_context(cfg, [900_000 + s for s in seeds], n_train_iv)
    t_ctx = time.perf_counter() - t_start

    cache = PredictorCache(cfg, seeds)
    mape = [float(h["test_mape_pct"]) for h in cache.history]
    print(f"context {t_ctx:.1f} s | {cfg.n_seeds} BiLSTM predictors "
          f"{cache.train_seconds:.1f} s, held-out MAPE "
          f"{np.mean(mape):.2f} +/- {summarise(mape).ci95:.2f} %")

    specs = list(BASELINES) + [a for a in ABLATIONS if a.ident != "A0"]
    results: Dict[str, dict] = {}
    convergence: Dict[str, dict] = {}
    for spec in specs:
        res = run_policy(cfg, spec, eval_ctx, train_ctx, cache)
        results[spec.ident] = res
        if res["convergence"] is not None:
            conv = res["convergence"]
            convergence[spec.ident] = {
                "reward_trace": np.asarray(conv["reward_trace"]).tolist(),
                "returns": conv["ppo_returns"],
            }
    # A0 is the reference ablation and is by definition identical to B10.
    results["A0"] = dict(results["B10"], ident="A0", name="Full MAIMO")

    # ---- per-policy files --------------------------------------------
    t_s = eval_ctx.t_s[0, cfg.warmup_intervals():]
    carbon: Dict[str, dict] = {}
    for pid, res in results.items():
        rec = policy_record(res)
        with open(f"{args.out}/{pid}.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1)
        carbon[pid] = carbon_for_policy(cfg, res, t_s)

    # ---- summary table ------------------------------------------------
    tc = t_critical(cfg.n_seeds)
    lines = ["policy,name," + ",".join(
        f"{m}_mean,{m}_ci95,{m}_sd,{m}_min,{m}_max" for m in SCALAR_METRICS)]
    for pid in [s.ident for s in BASELINES] + [a.ident for a in ABLATIONS]:
        res = results[pid]
        cells = []
        for m in SCALAR_METRICS:
            s = summarise(np.asarray(res[m], dtype=float), tc)
            cells += [f"{s.mean:.6g}", f"{s.ci95:.6g}", f"{s.sd:.6g}",
                      f"{s.minimum:.6g}", f"{s.maximum:.6g}"]
        lines.append(f"{pid},\"{res['name']}\"," + ",".join(cells))
    with open(f"{args.out}/summary.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # ---- paired tests against MAIMO, Holm-Bonferroni per family -------
    # Common random numbers make replication s the same environment for every
    # policy, so the comparison is paired and the test is run on the per-seed
    # differences.  The mean paired difference and its 95 % CI half-width are
    # recorded alongside the p-value because they, not the p-value, are the
    # quantity the design estimates; Wilcoxon signed-rank is a
    # distribution-free robustness check on the same differences.
    comparisons: Dict[str, dict] = {}
    for family, members in (
            ("baselines", [s.ident for s in BASELINES if s.ident != "B10"]),
            ("ablations", [a.ident for a in ABLATIONS if a.ident != "A0"])):
        ref = results["B10" if family == "baselines" else "A0"]
        fam: Dict[str, dict] = {}
        for m in SCALAR_METRICS:
            praw = {}
            extra = {}
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
    with open(f"{args.out}/comparisons.json", "w", encoding="utf-8") as f:
        json.dump(comparisons, f, indent=1)

    # ---- per-class latency (Table 6 needs MAIMO vs cloud-only) --------
    per_class = {}
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
    # Holm across the 12 latency comparisons of Table 6
    flat = {f"{lab}|{m}": per_class["tests"][lab][m]["p_raw"]
            for lab in CLASS_LABELS
            for m in ("latency_mean_ms", "latency_p95_ms", "latency_p99_ms")}
    for key, v in holm_bonferroni(flat).items():
        lab, m = key.split("|")
        per_class["tests"][lab][m].update(v)
    with open(f"{args.out}/per_class.json", "w", encoding="utf-8") as f:
        json.dump(per_class, f, indent=1)

    with open(f"{args.out}/carbon.json", "w", encoding="utf-8") as f:
        json.dump({"policies": carbon,
                   "region_traces": {k: v.tolist()
                                     for k, v in cb.REGION_TRACES.items()},
                   "host_region": cb.HOST_REGION,
                   "clean_region": cb.CLEAN_REGION,
                   "embodied_g_per_1000": cb.embodied_g_per_1000(
                       cfg, dict(zip(TIERS, np.asarray(
                           results["B10"]["alpha"]).mean(axis=0))),
                       {"cloud": en.CLOUD.t_inf_s, "edge": en.EDGE.t_inf_s,
                        "device": en.DEVICE.t_inf_s})}, f, indent=1)

    with open(f"{args.out}/convergence.json", "w", encoding="utf-8") as f:
        json.dump(convergence, f, indent=1)

    wall = time.perf_counter() - t_start
    meta = {
        "version": __version__,
        "commit": git_commit(),
        "wall_clock_s": wall,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "torch": __import__("torch").__version__,
        "scipy": __import__("scipy").__version__,
        "seeds": seeds,
        "control_intervals": n_iv,
        "train_intervals": n_train_iv,
        "warmup_intervals": cfg.warmup_intervals(),
        "slots_per_seed": cfg.horizon_slots,
        "predictor_mape_pct": mape,
        "predictor_train_seconds": cache.train_seconds,
        "t_critical": tc,
        "config": cfg.as_dict(),
        "contract_reference_energy_j": en.reference_tier_energies_j(),
        "contract_hybrid_energy_j": en.hybrid_energy_j(),
        "contract_values_j": en.CONTRACT_VALUES_J,
        "contract_cloud_note": en.CONTRACT_CLOUD_NOTE,
    }
    with open(f"{args.out}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)

    print(f"\ntotal wall clock {wall / 60.0:.1f} min -> {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
