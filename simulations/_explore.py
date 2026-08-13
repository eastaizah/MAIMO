"""Mean realised cost per (use case, complexity, action) under uniform
exploration -- i.e. the reward signal PPO actually sees early in training.
Development tool, not part of the released pipeline."""
import sys

import numpy as np

from config import (COMPRESSIONS, DEFAULT, MODELS, TIERS, USE_CASES,
                    decode_action)
from env import MAIMOEnv, RunConfig
import baselines as B

p = DEFAULT
cfg = B.scheme_registry(p)["maimo"].cfg
env = MAIMOEnv(p, 1, cfg)
env.predictor = B.make_predictor("bilstm", p, 1)
rng = np.random.default_rng(7)
feas = env.feasible_idx

n_slots = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
tot = np.zeros((len(USE_CASES), 3, 36))
cnt = np.zeros((len(USE_CASES), 3, 36))
for _ in range(n_slots):
    for g in env.begin_slot():
        cx = env._slot_state[g]["complexity"]
        a = int(rng.choice(feas))
        out = env.commit(g, a)
        tot[g, cx, a] += out.scalar_cost
        cnt[g, cx, a] += 1
    env.end_slot()

print("cloud util %.3f  edge util %.3f  hit %.3f"
      % (env.cloud.utilisation(), env._mean_edge_util(),
         np.mean([c.hit_rate() for c in env.caches])))
names = ("easy", "medium", "hard")
for g in (0,):
    for cx in range(3):
        m = np.where(cnt[g, cx] > 0, tot[g, cx] / np.maximum(cnt[g, cx], 1),
                     np.nan)
        order = np.argsort(np.nan_to_num(m, nan=1e9))
        print("\n=== %s / %s" % (USE_CASES[g].key, names[cx]))
        for a in order[:14]:
            if not np.isfinite(m[a]):
                continue
            mi, ti, ci = decode_action(int(a))
            print("  %-11s %-6s %-5s  cost %7.3f  (n=%d)"
                  % (MODELS[mi].name, TIERS[ti], COMPRESSIONS[ci].name,
                     m[a], cnt[g, cx, a]))
