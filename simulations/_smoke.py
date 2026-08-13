"""Smoke test: environment steps, predictor trains, action space is sane."""
import time
import numpy as np

from config import DEFAULT, USE_CASES, decode_action, MODELS, TIERS, COMPRESSIONS
from env import MAIMOEnv, RunConfig, generate_traffic_trace
import predictor as pred

p = DEFAULT
env = MAIMOEnv(p, 1, RunConfig())
print("feature dim", env.feature_dim())
print("feasible actions", env.feasible_idx.size)
for a in env.feasible_idx:
    mi, ti, ci = decode_action(int(a))
    print("  %2d  %-11s %-7s %-5s" % (a, MODELS[mi].name, TIERS[ti],
                                      COMPRESSIONS[ci].name))

t0 = time.perf_counter()
env.start_collecting()
n = 2000
for _ in range(n):
    act = env.begin_slot()
    for g in act:
        env.commit(g, int(env.feasible_idx[0]))
    env.end_slot()
dt = time.perf_counter() - t0
print("slot cost %.1f us  (%d slots in %.2f s)" % (dt / n * 1e6, n, dt))
r = env.results()
print("aggregate latency %.2f ms  energy %.3f J  sla %.2f %%"
      % (r["latency_ms"], r["energy_j"], r["sla_violation_pct"]))
for k, v in r["per_use_case"].items():
    print("  %-12s n=%6.0f lat=%8.2f ms  e=%7.3f J  acc=%.2f %% sla=%.2f %%"
          % (k, v["n_requests"], v["latency_ms"], v["energy_j"],
             v["acc_loss_pct"], v["sla_violation_pct"]))

t0 = time.perf_counter()
trace = generate_traffic_trace(p, 1, 3000)
print("trace gen %.2f s  mean %.3f std %.3f"
      % (time.perf_counter() - t0, trace.mean(), trace.std()))
t0 = time.perf_counter()
bp = pred.build_predictor("bilstm", p, 1).fit(trace)
print("bilstm fit %.2f s  nrmse %.4f  loss %.4f -> %.4f"
      % (time.perf_counter() - t0, bp.nrmse(trace), bp.history_loss[0],
         bp.history_loss[-1]))
lp = pred.build_predictor("lstm", p, 1).fit(trace)
print("lstm   nrmse %.4f" % lp.nrmse(trace))
np_ = pred.build_predictor("none", p, 1).fit(trace)
print("persist nrmse %.4f" % np_.nrmse(trace))
