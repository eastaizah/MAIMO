"""Driver: runs every experiment reported in the manuscript and writes JSON.

Protocol (locked by ``work/CONTRACT.md``)
-----------------------------------------
* ``n = 20`` independent replications, seeds 1..20.  A seed fixes the traffic,
  channel, mobility, predictor initialisation, controller initialisation and
  exploration stream of that replication; every scheme in a replication sees the
  *same* environment seed, so the comparison is paired by construction even
  though the tests are unpaired (Welch), which is the conservative choice.
* ``warmup_slots`` slots are simulated and discarded before statistics are
  collected, so that queues, caches and the mobility process are in their
  stationary regime.
* All results are reported as mean +- 95 % CI half-width over the replications,
  with Welch two-sided t-tests of MAIMO against every other scheme and a
  Holm-Bonferroni correction across the family.

Outputs (``simulations/results/``)
----------------------------------
``raw_seeds.json``       every per-seed metric of every scheme and ablation
``summary.json``         means, CIs, statistical tests, tuned parameters
``learning_curves.npz``  PPO and DQN episodic-return curves, per seed
``nmse.json``            channel-estimation NMSE replications
``energy_carbon.json``   the contract energy table and the carbon cases

Usage
-----
    python run_experiments.py                  # full protocol, 20 seeds
    python run_experiments.py --seeds 3        # quick smoke run
    python run_experiments.py --quick          # smoke run, short episodes
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import replace
from typing import Dict, List

import numpy as np

import ablation as abl
import baselines as B
import carbon as carbon_mod
import channel as channel_mod
import energy as energy_mod
import predictor as predictor_mod
import stats as st
from config import (COMPRESSIONS, DEFAULT, MODELS, N_USE_CASES, Params, TIERS,
                    USE_CASES)
from env import MAIMOEnv, generate_traffic_trace

RESULTS_DIR = "results"
SCHEME_ORDER = ("cloud_only", "edge_only", "random_greedy", "threshold", "edf",
                "lyapunov", "dqn", "maimo", "oracle")
# Metrics carried through to the manuscript tables.
METRICS = ("latency_ms", "energy_j", "acc_loss_pct", "sla_violation_pct",
           "cold_start_rate_pct")
HEADLINE_UC = "semantic_ce"


# ---------------------------------------------------------------------------
def _ensure_dir(path: str) -> None:
    import os
    os.makedirs(path, exist_ok=True)


def _flatten(r: dict) -> dict:
    """Per-seed record: aggregate metrics plus the per-use-case block."""
    out = {k: r[k] for k in ("latency_ms", "energy_j", "energy_j_loadaware",
                            "acc_loss_pct", "sla_violation_pct",
                            "cold_start_rate_pct", "cache_hit_rate",
                            "n_requests", "cloud_drops", "edge_drops")}
    out["tier_split"] = r["tier_split"]
    out["comp_split"] = r["comp_split"]
    for key, d in r["per_use_case"].items():
        for m, v in d.items():
            out[f"{key}.{m}"] = v
    return out


def _collect(records: List[dict], field: str) -> List[float]:
    return [float(r[field]) for r in records]


# ---------------------------------------------------------------------------
def run(p: Params, seeds: List[int], verbose: bool = True) -> dict:
    t_start = time.perf_counter()
    reg = B.scheme_registry(p)
    abl_reg = abl.ablation_registry()
    timings: Dict[str, float] = {}

    def log(msg: str) -> None:
        if verbose:
            print(f"[{time.perf_counter() - t_start:7.1f}s] {msg}", flush=True)

    # ---- baseline hyper-parameter tuning (reported in the paper) ----------
    log("tuning the threshold heuristic and the Lyapunov weight")
    t0 = time.perf_counter()
    tune_seeds = tuple(seeds[:3])
    threshold, threshold_trace = B.tune_threshold(p, tune_seeds)
    lyap_v, lyap_trace = B.tune_lyapunov(p, tune_seeds)
    timings["tuning_s"] = time.perf_counter() - t0
    log(f"  tuned threshold = {threshold:g} (fraction of the deadline), "
        f"Lyapunov V = {lyap_v:g}")

    raw: Dict[str, List[dict]] = {k: [] for k in SCHEME_ORDER}
    raw_abl: Dict[str, List[dict]] = {k: [] for k in abl.ABLATION_ORDER}
    ppo_curves: List[np.ndarray] = []
    dqn_curves: List[np.ndarray] = []
    pred_nrmse: Dict[str, List[float]] = {"bilstm": [], "lstm": [],
                                          "persistence": []}
    t_pred = t_ppo = t_dqn = t_eval = t_abl = 0.0

    for si, seed in enumerate(seeds):
        log(f"seed {seed} ({si + 1}/{len(seeds)})")

        # ---- predictors --------------------------------------------------
        t0 = time.perf_counter()
        trace = generate_traffic_trace(p, seed, p.predictor_train_epochs_data)
        held = generate_traffic_trace(p, seed + 500, 1200)
        pred = {}
        for kind in ("bilstm", "lstm"):
            pr = predictor_mod.build_predictor(kind, p, seed).fit(trace)
            pred[kind] = pr
            pred_nrmse[kind].append(pr.nrmse(held))
        pred["none"] = predictor_mod.build_predictor("none", p, seed)
        pred_nrmse["persistence"].append(pred["none"].nrmse(held))
        t_pred += time.perf_counter() - t0

        # ---- controllers -------------------------------------------------
        t0 = time.perf_counter()
        ppo, curve = B.train_ppo(p, seed, reg["maimo"].cfg, pred["bilstm"])
        ppo_curves.append(curve)
        t_ppo += time.perf_counter() - t0

        t0 = time.perf_counter()
        dqn, dcurve = B.train_dqn(p, seed, reg["dqn"].cfg, pred["none"])
        dqn_curves.append(dcurve)
        t_dqn += time.perf_counter() - t0

        # ---- the nine schemes --------------------------------------------
        t0 = time.perf_counter()
        for key in SCHEME_ORDER:
            sch = reg[key]
            feasible = MAIMOEnv(p, seed, sch.cfg).feasible
            ctrl = {"maimo": ppo, "dqn": dqn}.get(key)
            pol = B.build_policy(sch, p, seed, feasible, ctrl, threshold,
                                 lyap_v)
            pr = pred["bilstm"] if sch.cfg.use_forecast else pred["none"]
            rec = _flatten(B.evaluate(p, seed, sch.cfg, pol, pr))
            rec["seed"] = seed
            raw[key].append(rec)
        t_eval += time.perf_counter() - t0

        # ---- ablations ---------------------------------------------------
        t0 = time.perf_counter()
        for key in abl.ABLATION_ORDER:
            a = abl_reg[key]
            env0 = MAIMOEnv(p, seed, a.cfg)
            if a.controller == "rule":
                ctrl = B.controller_mod.RuleBasedController(
                    p, env0.feature_dim(), env0.feasible, seed)
            elif a.controller == "retrain_no_forecast":
                ctrl, _ = B.train_ppo(p, seed, a.cfg, pred[a.predictor])
            else:
                ctrl = ppo.with_mask(env0.feasible)
            rec = _flatten(B.evaluate(p, seed, a.cfg, B.PolicyWrapper(ctrl),
                                      pred[a.predictor]))
            rec["seed"] = seed
            raw_abl[key].append(rec)
        # Ablation (b) is also run with a controller retrained without the
        # forecast, so that the deployment-time ablation can be checked against
        # the retrained one.
        a = abl_reg["no_forecast"]
        ctrl_nf, _ = B.train_ppo(p, seed, a.cfg, pred["none"])
        rec = _flatten(B.evaluate(p, seed, a.cfg, B.PolicyWrapper(ctrl_nf),
                                  pred["none"]))
        rec["seed"] = seed
        raw_abl.setdefault("no_forecast_retrained", []).append(rec)
        t_abl += time.perf_counter() - t0

    timings.update({"predictors_s": t_pred, "ppo_training_s": t_ppo,
                    "dqn_training_s": t_dqn, "scheme_eval_s": t_eval,
                    "ablation_s": t_abl})

    # ---- channel-estimation NMSE ----------------------------------------
    log("channel-estimation NMSE")
    t0 = time.perf_counter()
    nmse = [channel_mod.channel_estimation_nmse(s, p) for s in seeds]
    timings["nmse_s"] = time.perf_counter() - t0

    # ---- aggregate -------------------------------------------------------
    log("aggregating, testing and writing results")
    summary = _summarise(p, seeds, raw, raw_abl, ppo_curves, dqn_curves,
                         pred_nrmse, nmse,
                         {"threshold": threshold,
                          "threshold_trace": threshold_trace,
                          "lyapunov_v": lyap_v,
                          "lyapunov_trace": lyap_trace,
                          "tuning_seeds": list(tune_seeds)})
    timings["total_s"] = time.perf_counter() - t_start
    summary["timings_s"] = timings
    summary["environment"] = _environment(p, seeds)

    _ensure_dir(RESULTS_DIR)
    with open(f"{RESULTS_DIR}/raw_seeds.json", "w", encoding="utf-8") as fh:
        json.dump({"schemes": raw, "ablations": raw_abl}, fh, indent=1)
    with open(f"{RESULTS_DIR}/summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    with open(f"{RESULTS_DIR}/nmse.json", "w", encoding="utf-8") as fh:
        json.dump(nmse, fh, indent=1)
    np.savez_compressed(f"{RESULTS_DIR}/learning_curves.npz",
                        ppo=np.array(ppo_curves),
                        dqn=np.array([c[:min(len(x) for x in dqn_curves)]
                                      for c in dqn_curves]))
    with open(f"{RESULTS_DIR}/energy_carbon.json", "w", encoding="utf-8") as fh:
        json.dump({"energy": energy_mod.contract_table(),
                   "carbon": carbon_mod.figure_cases()}, fh, indent=1)
    log(f"done in {timings['total_s']:.1f} s")
    return summary


# ---------------------------------------------------------------------------
def _summarise(p, seeds, raw, raw_abl, ppo_curves, dqn_curves, pred_nrmse,
               nmse, tuned) -> dict:
    tc = p.t_crit_19_df if len(seeds) == 20 else None

    def s(x):
        return st.summarise(x, tc).as_dict()

    # --- Table 7: schemes, on the headline use case and on the whole mix ---
    schemes: Dict[str, dict] = {}
    for key in raw:
        rec = raw[key]
        block = {"n_seeds": len(rec)}
        for m in METRICS:
            block[m] = s(_collect(rec, m))
        for m in METRICS[:-1]:
            block[f"{HEADLINE_UC}.{m}"] = s(_collect(rec, f"{HEADLINE_UC}.{m}"))
        block["tier_split"] = np.mean([r["tier_split"] for r in rec],
                                      axis=0).tolist()
        block["comp_split"] = np.mean([r["comp_split"] for r in rec],
                                      axis=0).tolist()
        block["per_use_case"] = {
            uc.key: {m: s(_collect(rec, f"{uc.key}.{m}"))
                     for m in ("latency_ms", "energy_j", "acc_loss_pct",
                               "sla_violation_pct", "cold_start_pct",
                               "drop_pct")}
            for uc in USE_CASES}
        schemes[key] = block

    # --- statistical tests: MAIMO against every other scheme --------------
    tests: Dict[str, Dict[str, dict]] = {}
    for m in ("latency_ms", "energy_j", "acc_loss_pct", "sla_violation_pct",
              f"{HEADLINE_UC}.latency_ms", f"{HEADLINE_UC}.energy_j"):
        ref = _collect(raw["maimo"], m)
        fam = {k: _collect(raw[k], m) for k in raw if k != "maimo"}
        tests[m] = {k: v.as_dict()
                    for k, v in st.compare_family(ref, fam).items()}

    # --- Table 6: per-use-case latency, MAIMO vs cloud-only ---------------
    table6 = {}
    for uc in USE_CASES:
        f = f"{uc.key}.latency_ms"
        a, b = _collect(raw["maimo"], f), _collect(raw["cloud_only"], f)
        sa, sb = st.summarise(a, tc), st.summarise(b, tc)
        w = st.welch_t(a, b)
        table6[uc.key] = {"label": uc.label, "deadline_ms": uc.deadline_ms,
                          "maimo": sa.as_dict(), "cloud_only": sb.as_dict(),
                          "reduction_pct": st.reduction_pct(sb.mean, sa.mean),
                          "welch": w.as_dict()}
    # Holm correction across the four use cases of Table 6.
    p_raw = [table6[uc.key]["welch"]["p"] for uc in USE_CASES]
    p_adj, rej = st.holm_bonferroni(p_raw)
    for uc, pa, rj in zip(USE_CASES, p_adj, rej):
        table6[uc.key]["welch"]["p_holm"] = pa
        table6[uc.key]["welch"]["reject_holm"] = bool(rj)

    # --- Table 8: ablations ----------------------------------------------
    abl_reg = abl.ablation_registry()
    full_lat = _collect(raw_abl["full"], "latency_ms")
    ablations: Dict[str, dict] = {}
    for key, rec in raw_abl.items():
        a = abl_reg.get(key.replace("_retrained", ""))
        block = {"tag": a.tag if a else "(b')",
                 "label": (a.label if a else
                           "No BiLSTM forecast, controller retrained"),
                 "note": a.note if a else
                         "control experiment for the deployment-time ablation",
                 "n_seeds": len(rec)}
        for m in METRICS:
            block[m] = s(_collect(rec, m))
        for m in ("latency_ms", "acc_loss_pct", "sla_violation_pct"):
            block[f"{HEADLINE_UC}.{m}"] = s(_collect(rec, f"{HEADLINE_UC}.{m}"))
        lat = _collect(rec, "latency_ms")
        block["delta_latency_pct"] = s(
            [100.0 * (x - y) / y for x, y in zip(lat, full_lat)])
        block["welch_vs_full"] = st.welch_t(full_lat, lat).as_dict()
        block["tier_split"] = np.mean([r["tier_split"] for r in rec],
                                      axis=0).tolist()
        ablations[key] = block
    fam = {k: _collect(v, "latency_ms") for k, v in raw_abl.items()
           if k != "full"}
    adj = st.compare_family(full_lat, fam)
    for k, v in adj.items():
        ablations[k]["welch_vs_full"] = v.as_dict()

    # --- convergence ------------------------------------------------------
    curves = np.array(ppo_curves)
    conv = [st.convergence_episode(c) * (p.ppo_total_episodes / curves.shape[1])
            for c in curves]
    phases = _phases(curves, p)

    return {
        "protocol": {
            "seeds": list(seeds), "n_seeds": len(seeds),
            "warmup_slots": p.warmup_slots, "eval_slots": p.eval_slots,
            "t_crit": p.t_crit_19_df if len(seeds) == 20 else "exact per n",
            "t_slot_ms": p.t_slot_ms,
            "simulated_seconds_per_replication":
                (p.warmup_slots + p.eval_slots) * p.t_slot_ms / 1e3,
        },
        "tuned": tuned,
        "schemes": schemes,
        "table6_latency": table6,
        "ablations": ablations,
        "tests": tests,
        "predictor_nrmse": {k: s(v) for k, v in pred_nrmse.items()},
        "predictor_gain_pct": {
            "bilstm_vs_lstm": s([100.0 * (l - b) / l for b, l in
                                 zip(pred_nrmse["bilstm"],
                                     pred_nrmse["lstm"])]),
            "bilstm_vs_persistence": s([100.0 * (q - b) / q for b, q in
                                        zip(pred_nrmse["bilstm"],
                                            pred_nrmse["persistence"])]),
        },
        "nmse": {k: s([r[k] for r in nmse])
                 for k in ("ls_nmse_db", "edge_7b_lora_nmse_db",
                           "device_50m_int4_nmse_db", "ideal_lmmse_nmse_db")},
        "convergence": {
            "episodes_to_plateau": s(conv),
            "total_episodes": p.ppo_total_episodes,
            "phases": phases,
            "final_return": s([float(c[-5:].mean()) for c in curves]),
            "initial_return": s([float(c[:5].mean()) for c in curves]),
        },
        "energy": energy_mod.contract_table(),
        "carbon": carbon_mod.figure_cases(),
        "params": p.as_dict(),
    }


def _phases(curves: np.ndarray, p: Params) -> dict:
    """Empirical exploration / improvement / plateau boundaries.

    The mean curve is scaled to [0, 1] between its first and its plateau value;
    the exploration phase ends when 10 % of the total improvement has been
    realised and the improvement phase ends when 90 % has been.
    """
    m = curves.mean(axis=0)
    scale = p.ppo_total_episodes / m.size
    lo, hi = float(m[:3].mean()), float(m[-5:].mean())
    if hi <= lo:
        return {"exploration_end_episode": 0, "plateau_start_episode": 0}
    frac = (m - lo) / (hi - lo)
    def first(th):
        idx = np.nonzero(frac >= th)[0]
        return int((idx[0] + 1) * scale) if idx.size else int(p.ppo_total_episodes)
    return {"exploration_end_episode": first(0.10),
            "plateau_start_episode": first(0.90),
            "initial_return": lo, "plateau_return": hi}


def _environment(p: Params, seeds: List[int]) -> dict:
    import matplotlib
    import scipy
    return {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} "
                    f"({platform.machine()})",
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "n_seeds": len(seeds),
    }


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=20,
                    help="number of replications (seeds 1..N)")
    ap.add_argument("--quick", action="store_true",
                    help="short episodes and short replications (smoke test)")
    args = ap.parse_args(argv)

    p = DEFAULT
    if args.quick:
        p = replace(p, ppo_total_episodes=400, dqn_total_episodes=400,
                    eval_slots=600, warmup_slots=150,
                    predictor_epochs=40, predictor_train_epochs_data=800)
    run(p, list(range(1, args.seeds + 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
