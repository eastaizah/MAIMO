"""Calibration driver (development tool, not part of the released pipeline)."""
import sys
import time

import numpy as np

from config import DEFAULT, COMPRESSIONS, MODELS, TIERS, USE_CASES, decode_action
import baselines as B
from env import MAIMOEnv, RunConfig

TARGETS = {
    "semantic_ce": (12.0, 22.0),
    "urllc_v2x": (2.1, 18.5),
    "embb": (35.0, 120.0),
    "mmtc": (180.0, 450.0),
}

p = DEFAULT
episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
seeds = [int(s) for s in (sys.argv[2].split(",") if len(sys.argv) > 2
                          else ["1", "2", "3"])]

reg = B.scheme_registry(p)
t0 = time.perf_counter()
pred = {s: B.make_predictor("bilstm", p, s) for s in seeds}
print("predictors %.1f s" % (time.perf_counter() - t0))

ctrl = {}
t0 = time.perf_counter()
for s in seeds:
    c, curve = B.train_ppo(p, s, reg["maimo"].cfg, pred[s], episodes)
    ctrl[s] = c
    print("  seed %d: return %.3f -> %.3f" % (s, curve[:5].mean(),
                                              curve[-5:].mean()))
print("ppo train %.1f s (%d episodes/seed)" % (time.perf_counter() - t0, episodes))

# What did the policy learn?
env = MAIMOEnv(p, seeds[0], reg["maimo"].cfg)
env.predictor = pred[seeds[0]]
env.begin_slot()
L = env._layout
for g, uc in enumerate(USE_CASES):
    for cx, cxn in enumerate(("easy", "medium", "hard")):
        x = np.zeros(env.base_feature_dim())
        x[L["use_case"][0] + g] = 1.0
        x[L["complexity"][0] + cx] = 1.0
        x[L["uc_x_cx"][0] + g * 3 + cx] = 1.0
        x[L["arrivals"][0]] = 0.25
        x[L["snr_mean"][0]] = 0.8
        x[L["cache"][0]:L["cache"][0] + 4] = 1.0
        x[L["forecast"][0]:L["forecast"][0] + 5] = 1.0
        x[L["bias"][0]] = 1.0
        x = env.expand(x)
        a, _ = ctrl[seeds[0]].act(x, greedy=True)
        mi, ti, ci = decode_action(a)
        pr = ctrl[seeds[0]].probs(x)
        print("  %-12s %-6s -> %-11s %-6s %-5s  (p=%.2f)"
              % (uc.key, cxn, MODELS[mi].name, TIERS[ti], COMPRESSIONS[ci].name,
                 pr[a]))

print()
print("%-16s %9s %9s %9s %9s %9s" % ("scheme", "lat_ms", "energy_J",
                                     "acc_%", "sla_%", "cold_%"))
th, _ = B.tune_threshold(p, tuple(seeds))
lv, _ = B.tune_lyapunov(p, tuple(seeds))
print("tuned threshold %.2f   lyapunov V %.3f" % (th, lv))

t0 = time.perf_counter()
rows = {}
for key, sch in reg.items():
    accum = []
    for s in seeds:
        c = ctrl[s] if sch.needs_training == "ppo" else None
        if sch.needs_training == "dqn":
            c, _ = B.train_dqn(p, s, sch.cfg,
                               B.make_predictor("none", p, s),
                               max(episodes // 2, 400))
        pol = B.build_policy(sch, p, s, MAIMOEnv(p, s, sch.cfg).feasible, c,
                             th, lv)
        pr = pred[s] if sch.cfg.use_forecast else None
        accum.append(B.evaluate(p, s, sch.cfg, pol, pr))
    rows[key] = accum
    m = lambda k: float(np.mean([a[k] for a in accum]))
    print("%-16s %9.2f %9.3f %9.3f %9.2f %9.2f"
          % (key, m("latency_ms"), m("energy_j"), m("acc_loss_pct"),
             m("sla_violation_pct"), m("cold_start_rate_pct")))
print("eval %.1f s" % (time.perf_counter() - t0))

print()
print("%-12s %-10s %10s %10s   %10s %10s" % ("use case", "", "MAIMO", "target",
                                             "cloud-only", "target"))
for uc in USE_CASES:
    mm = float(np.mean([a["per_use_case"][uc.key]["latency_ms"]
                        for a in rows["maimo"]]))
    cc = float(np.mean([a["per_use_case"][uc.key]["latency_ms"]
                        for a in rows["cloud_only"]]))
    t = TARGETS[uc.key]
    print("%-23s %10.2f %10.1f   %10.2f %10.1f"
          % (uc.key, mm, t[0], cc, t[1]))

sp = np.mean([a["per_use_case"]["semantic_ce"]["tier_split"]
              for a in rows["maimo"]], axis=0)
print("MAIMO semantic_ce tier split (cloud/edge/device): %.3f %.3f %.3f"
      % tuple(sp))
so = np.mean([a["per_use_case"]["semantic_ce"]["tier_split"]
              for a in rows["oracle"]], axis=0)
print("oracle                                          : %.3f %.3f %.3f"
      % tuple(so))
print("target                                          : 0.250 0.500 0.250")
print("oracle comp split (none/lora/int8/int4): %s"
      % np.round(np.mean([a["comp_split"] for a in rows["oracle"]], axis=0), 3))

print()
print("headline use case (semantic_ce) per scheme")
print("%-16s %9s %9s %9s %9s" % ("scheme", "lat_ms", "energy_J", "acc_%",
                                 "sla_%"))
for key in reg:
    d = [a["per_use_case"]["semantic_ce"] for a in rows[key]]
    m = lambda k: float(np.mean([x[k] for x in d]))
    print("%-16s %9.2f %9.3f %9.3f %9.2f"
          % (key, m("latency_ms"), m("energy_j"), m("acc_loss_pct"),
             m("sla_violation_pct")))

print()
print("SLA violation %% by use case")
print("%-16s %10s %10s %10s %10s" % ("scheme", *[u.key for u in USE_CASES]))
for key in reg:
    v = [float(np.mean([a["per_use_case"][u.key]["sla_violation_pct"]
                        for a in rows[key]])) for u in USE_CASES]
    print("%-16s %10.2f %10.2f %10.2f %10.2f" % (key, *v))
