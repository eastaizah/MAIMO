"""PPO hyper-parameter sweep (development tool, not part of the pipeline)."""
import dataclasses
import itertools
import time

import numpy as np

from config import DEFAULT, COMPRESSIONS, MODELS, TIERS, USE_CASES, decode_action
import baselines as B
from env import MAIMOEnv

reg = B.scheme_registry(DEFAULT)
cfg = reg["maimo"].cfg
seeds = (1, 2)
pred = {s: B.make_predictor("bilstm", DEFAULT, s) for s in seeds}

grid = list(itertools.product((0.95, 0.0), (0.0, 0.5, 0.95), (3000,)))
print("%8s %7s %6s | %7s %7s %7s | %s" % ("gamma", "lam", "eps", "cloud",
                                          "edge", "device",
                                          "greedy(semantic easy/med/hard)"))
for lr, ent, eps in grid:
    p = dataclasses.replace(DEFAULT, ppo_gamma=lr, ppo_gae_lambda=ent)
    t0 = time.perf_counter()
    res, tbl = [], []
    for s in seeds:
        c, _ = B.train_ppo(p, s, cfg, pred[s], eps)
        r = B.evaluate(p, s, cfg, B.PolicyWrapper(c), pred[s],
                       warmup=200, slots=1200)
        res.append(r["per_use_case"]["semantic_ce"])
        if s == seeds[0]:
            env = MAIMOEnv(p, s, cfg)
            env.predictor = pred[s]
            env.begin_slot()
            L = env._layout
            for cx in range(3):
                x = np.zeros(env.base_feature_dim())
                x[L["use_case"][0]] = 1.0
                x[L["complexity"][0] + cx] = 1.0
                x[L["uc_x_cx"][0] + cx] = 1.0
                x[L["arrivals"][0]] = 0.25
                x[L["snr_mean"][0]] = 0.8
                x[L["cache"][0] + 1] = 1.0
                x[L["cache"][0] + 3] = 1.0
                x[L["forecast"][0]:L["forecast"][0] + 5] = 1.0
                x[L["bias"][0]] = 1.0
                a, _ = c.act(env.expand(x), greedy=True)
                mi, ti, ci = decode_action(a)
                tbl.append("%s/%s/%s" % (MODELS[mi].name.split("_")[0],
                                         TIERS[ti][:3], COMPRESSIONS[ci].name))
    sp = np.mean([r["tier_split"] for r in res], axis=0)
    print("%8.1e %7.3f %6d | %7.3f %7.3f %7.3f | %s  lat %.2f acc %.2f  (%.0f s)"
          % (lr, ent, eps, sp[0], sp[1], sp[2], "  ".join(tbl),
             np.mean([r["latency_ms"] for r in res]),
             np.mean([r["acc_loss_pct"] for r in res]),
             time.perf_counter() - t0))
