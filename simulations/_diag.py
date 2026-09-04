"""Trained-policy probabilities vs realised costs on real states.
Development tool, not part of the released pipeline."""
import numpy as np

from config import (COMPRESSIONS, DEFAULT, MODELS, TIERS, USE_CASES,
                    decode_action)
from env import MAIMOEnv
import baselines as B

p = DEFAULT
cfg = B.scheme_registry(p)["maimo"].cfg
pred = B.make_predictor("bilstm", p, 1)
ctrl, curve = B.train_ppo(p, 1, cfg, pred, 3000)
print("return %.3f -> %.3f" % (curve[:5].mean(), curve[-5:].mean()))

env = MAIMOEnv(p, 1, cfg)
env.predictor = pred
for _ in range(300):
    for g in env.begin_slot():
        env.commit(g, ctrl.act(env.features(g), greedy=True)[0])
    env.end_slot()

seen = {}
for _ in range(4000):
    act = env.begin_slot()
    for g in act:
        cx = env._slot_state[g]["complexity"]
        if g == 0 and cx not in seen:
            x = env.features(g)
            seen[cx] = (x, ctrl.probs(x),
                        np.array([env.evaluate(g, int(a)).scalar_cost
                                  for a in range(36)]))
        env.commit(g, ctrl.act(env.features(g), greedy=True)[0])
    env.end_slot()
    if len(seen) == 3:
        break

names = ("easy", "medium", "hard")
for cx in sorted(seen):
    x, pr, cost = seen[cx]
    print("\n=== semantic_ce / %s     V(s) = %.3f" % (names[cx], ctrl.value(x)))
    for a in np.argsort(-pr)[:14]:
        if not env.feasible[a]:
            continue
        mi, ti, ci = decode_action(int(a))
        print("  %-11s %-6s %-5s  p=%.4f  logit=%8.3f  cost=%7.3f"
              % (MODELS[mi].name, TIERS[ti], COMPRESSIONS[ci].name, pr[a],
                 float(x @ ctrl.theta[:, a]), cost[a]))
