# MAIMO simulator

A self-contained, seed-reproducible slotted simulator of the three-tier MAIMO
system (cloud / edge / device) described in *Massive AI Model Orchestration for
6G Networks: Architecture, Optimization, and Energy-Efficient Deployment*
(MDPI *Future Internet*, submission 4495624).

Everything the manuscript reports in Section 6 — Tables 6, 7 and 8, Figure 5
and Figure 6 — is produced by the two commands in
[Reproducing the paper](#reproducing-the-paper). Nothing is hand-entered.

## What this is, and what it is not

Read this before quoting any number.

| Component | Status |
|---|---|
| Radio link, traffic arrivals, queueing, routing, model loading | **simulated** in this repository |
| BiLSTM traffic predictor, PPO / DQN / LinUCB orchestrators | **implemented** here, in PyTorch on CPU, and trained per seed |
| Per-inference and aggregate energy | **analytic estimates**: vendor power envelopes combined with simulated inference times. *Not* wall-plug measurements |
| Task accuracy | a **task-success-rate proxy** read off published accuracy-versus-compression curves. It is *not* an evaluation on a real dataset and must never be described as one |
| Grid carbon intensity | **stylised** representative diurnal shapes for four bidding zones, not a measured dataset |
| Physical testbed, real 6G hardware, real user traffic | **not present**. No result here comes from a deployment |

The energy model is the one locked in `../work/CONTRACT.md`, implemented
literally in `maimo/energy.py`, and checked against the contract's own
arithmetic by `tests/test_energy.py`.

## Requirements

Python 3.13 with the packages pinned in `requirements.txt` (NumPy, SciPy,
Matplotlib, PyTorch CPU, pytest). No GPU, no pandas, no openpyxl.

```
python -m pip install -r requirements.txt
```

## Reproducing the paper

```
cd simulations
python run_all.py            # the locked 20-seed protocol -> results/
python plot_figures.py       # -> ../figures/Figura5.png, ../figures/Figura6.png
```

`run_all.py` is the only thing that consumes real time. It runs 16
configurations (B1-B10 and A0-A5) over 20 seeds, each seed covering
3.6 x 10^5 simulated slots of 1 s after a 10^4-slot warm-up that is discarded,
and it trains 20 BiLSTM predictors plus the PPO, DQN and LinUCB controllers.
See `results/meta.json` for the wall-clock time, the machine and the exact
package versions of the run that produced the committed numbers.

Useful flags:

```
python run_all.py --quick               # ~1 min smoke run, same physics
python run_all.py --seeds 5             # fewer replications
python run_all.py --out results_alt     # write somewhere else
python calibrate.py                     # print the CONTRACT operating points only
python -m pytest tests -q               # unit tests
```

## Determinism

Every source of randomness is drawn from an explicitly seeded generator:
traffic (`maimo/traffic.py`), channel realisations (`maimo/channel.py`),
per-request latency sampling (`maimo/sim.py`), and network initialisation and
action sampling (`maimo/controller.py`, `maimo/predictor.py`). Re-running the
same seed reproduces the same metrics exactly; `tests/test_sim.py` asserts
this for both a fixed policy and a learned one. Policies are compared under
common random numbers, so the comparison is paired and the reported
differences are not an artefact of traffic variability between runs.

Learned controllers are trained on a **disjoint** traffic window (seeds
900001-900020) and then frozen, so every evaluation number is out of sample.

## Layout

```
maimo/config.py       every parameter, with a justification or a source
maimo/channel.py      3GPP TR 38.901 UMa path loss, CDL-C fading, SINR, rate
maimo/traffic.py      diurnal + weekly + MMPP-2 bursty arrivals per class
maimo/models.py       model zoo: FLOPs, memory, load time, accuracy curves
maimo/energy.py       the CONTRACT energy model, verbatim
maimo/carbon.py       grid intensity, temporal/geographic shifting, embodied
maimo/predictor.py    BiLSTM traffic predictor
maimo/controller.py   PPO orchestrator plus DQN, LinUCB and the heuristics
maimo/sim.py          the slotted engine
maimo/baselines.py    B1-B10
maimo/ablations.py    A0-A5
maimo/stats.py        CI half-widths, Welch's t-test, Holm-Bonferroni
maimo/experiment.py   train-then-freeze driver, common random numbers
run_all.py            every policy x 20 seeds -> results/
plot_figures.py       -> ../figures/Figura5.png, ../figures/Figura6.png
calibrate.py          prints the CONTRACT operating points from a short run
tests/                unit tests
```

## Outputs

| File | Contents |
|---|---|
| `results/<ID>.json` | per-seed metric vectors for one policy |
| `results/summary.csv` | one row per policy: mean, 95 % CI half-width, sd, min, max |
| `results/comparisons.json` | Welch t-tests against MAIMO, Holm-Bonferroni corrected |
| `results/per_class.json` | per-service-class latency for Table 6 |
| `results/carbon.json` | carbon per shifting strategy, region traces, embodied carbon |
| `results/convergence.json` | PPO and DQN learning curves |
| `results/meta.json` | configuration, environment, timings, contract cross-check |

## Design notes

**Two timescales.** Physical slots are 1 s, matching the `T_slot = 1 s`
accounting window of the contract's energy model; the orchestrator decides
every 10 slots, because model (re)loading takes seconds and a per-slot
controller would be unimplementable. Per-slot arithmetic is vectorised over
seeds, service classes and tiers with NumPy; queue backlogs use a closed-form
Lindley recursion and waiting times use Sakasegawa's M/G/c approximation.

**One batched forward pass.** Traffic is exogenous, so each seed's BiLSTM runs
over the whole trace once and the predictions are fed into the cheap per-slot
control loop, rather than being recomputed per slot.

**Calibration.** Parameters marked `[CAL]` in `maimo/config.py` were chosen to
reproduce the operating points locked in the contract. Each one carries a
physical justification in its docstring. Calibration selects defensible
parameter values; it never hard-codes an output. Where a locked value could
not be reproduced, the discrepancy is reported in `../work/sim_results.md`
under `## DEVIATIONS` rather than being fudged.
