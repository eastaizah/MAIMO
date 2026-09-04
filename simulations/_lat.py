"""Latency CCDF of the cloud-only baseline, to place the SLA deadline.
Development tool, not part of the released pipeline."""
import numpy as np

from config import DEFAULT, USE_CASES
import baselines as B
from env import MAIMOEnv

p = DEFAULT
reg = B.scheme_registry(p)
for key in ("cloud_only",):
    sch = reg[key]
    lat = {u.key: [] for u in USE_CASES}
    for s in (1, 2, 3):
        env = MAIMOEnv(p, s, sch.cfg)
        pol = B.HeuristicPolicy(key, p, env.feasible, s)
        for i in range(2400):
            for g in env.begin_slot():
                out = env.commit(g, pol.decide(env, g))
                if i >= 400:
                    lat[USE_CASES[g].key].append((out.latency_ms, out.n_req))
            env.end_slot()
    print("=== %s" % key)
    for u in USE_CASES:
        a = np.array(lat[u.key])
        if not a.size:
            continue
        v, w = a[:, 0], a[:, 1]
        print("  %-12s mean %7.2f  p50 %7.2f  p90 %7.2f  p99 %7.2f" %
              (u.key, np.average(v, weights=w),
               *[np.percentile(v, q) for q in (50, 90, 99)]))
        for d in (20, 22, 24, 26, 28, 30, 35, 40, 120, 140, 380, 450, 520):
            print("      deadline %5.0f ms -> violations %5.2f %%"
                  % (d, 100.0 * w[v > d].sum() / w.sum()))
