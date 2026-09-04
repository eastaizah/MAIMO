"""Scratch: inspect the routing split, headline metrics and energy split."""
import json
import sys

import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "results_smoke"
print(f"{'id':4s} {'alpha c/e/d':17s} {'lat':>6s} {'p99':>6s} {'E':>8s} "
      f"{'acc':>5s} {'SLA':>5s} {'nostall':>7s} | "
      f"{'compute':>8s} {'encode':>7s} {'radio':>7s} {'load':>7s}")
for p in ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10",
          "A0", "A1", "A2", "A3", "A4", "A5"):
    try:
        r = json.load(open(f"{d}/{p}.json"))
    except FileNotFoundError:
        continue
    a = np.array(r["alpha"]).mean(axis=0)
    q = np.array(r.get("energy_parts_j", np.zeros((1, 4)))).mean(axis=0)
    print(f"{p:4s} {a[0]:.3f}/{a[1]:.3f}/{a[2]:.3f}  "
          f"{np.mean(r['latency_mean_ms']):6.2f} "
          f"{np.mean(r['latency_p99_ms']):6.2f} "
          f"{np.mean(r['energy_j']):8.4f} "
          f"{np.mean(r['accuracy_pct']):5.2f} "
          f"{np.mean(r['sla_violation_pct']):5.2f} "
          f"{np.mean(r['cache_hit_pct']):7.2f} | "
          f"{q[0]:8.4f} {q[1]:7.4f} {q[2]:7.4f} {q[3]:7.4f}")
