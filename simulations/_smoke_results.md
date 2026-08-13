## Reproduction

Every number and both figures in this file are produced by `simulations/run_all.py` followed by `simulations/plot_figures.py` and `simulations/make_report.py`, at commit da530f5, from the code in `simulations/`. The run covers 4 independent replications, seeds 1-4, each simulating 20000 slots of 1 s after a 10000-slot warm-up that is discarded, for all sixteen configurations B1-B10 and A0-A5; the learned controllers are additionally trained for 3200 control intervals on a disjoint traffic window and then frozen, so every reported number is out of sample. Wall-clock time was 0.5 minutes on Windows-11-10.0.26200-SP0 with Python 3.13.0, NumPy 2.4.4, SciPy 1.17.1 and PyTorch 2.11.0+cpu on CPU. All randomness is seeded explicitly and re-running a seed reproduces its metrics exactly; `simulations/tests/` asserts this along with the energy arithmetic, the 3GPP path loss and the confidence-interval computation.

## Table 6. End-to-end latency, MAIMO versus cloud-only baseline.

Mean of 4 seeds ± 95 % confidence half-width (Student's t, 3 d.o.f.). p-values from Welch's two-sided t-test, corrected with Holm-Bonferroni across the twelve comparisons in this table.

| Scenario | Metric | MAIMO (ms) | Cloud-only B1 (ms) | Reduction (%) | p (Holm-Bonferroni) |
|---|---|---|---|---|---|
| Joint semantic comm. + channel estimation | mean | 11.17 ± 1.25 | 22.03 ± 0.15 | 49.3 | 0.0003 |
| Joint semantic comm. + channel estimation | p95 | 22.26 ± 0.60 | 28.12 ± 0.25 | 20.8 | &lt; 0.0001 |
| Joint semantic comm. + channel estimation | p99 | 25.97 ± 1.42 | 34.29 ± 3.25 | 24.3 | 0.0030 |
| URLLC V2X | mean | 2.08 ± 0.26 | 18.52 ± 0.17 | 88.8 | &lt; 0.0001 |
| URLLC V2X | p95 | 2.34 ± 0.03 | 24.02 ± 0.63 | 90.3 | &lt; 0.0001 |
| URLLC V2X | p99 | 15.48 ± 0.45 | 31.46 ± 0.13 | 50.8 | &lt; 0.0001 |
| eMBB video streaming | mean | 34.3 ± 2.1 | 121.6 ± 0.9 | 71.8 | &lt; 0.0001 |
| eMBB video streaming | p95 | 58.9 ± 0.4 | 272.8 ± 0.1 | 78.4 | &lt; 0.0001 |
| eMBB video streaming | p99 | 76.0 ± 3.8 | 501.3 ± 0.2 | 84.8 | &lt; 0.0001 |
| mMTC batch sensor upload | mean | 176.0 ± 5.2 | 451.2 ± 4.7 | 61.0 | &lt; 0.0001 |
| mMTC batch sensor upload | p95 | 291.4 ± 19.7 | 1676.6 ± 1.6 | 82.6 | &lt; 0.0001 |
| mMTC batch sensor upload | p99 | 327.6 ± 2.9 | 1932.9 ± 456.1 | 83.1 | 0.0030 |

## Table 7. Comparison against baseline orchestration schemes.

Mean of 4 seeds ± 95 % confidence half-width. p-values compare each scheme's mean latency against MAIMO using Welch's two-sided t-test, corrected with Holm-Bonferroni across the family of nine comparisons. Energies are analytic estimates from vendor power envelopes combined with simulated inference times, not wall-plug measurements; accuracy is a task-success-rate proxy from the accuracy-versus-compression curves, not an evaluation on a dataset.

| ID | Scheme | Accuracy (%) | Latency mean (ms) | Latency p99 (ms) | Energy (J/inf.) | SLA violation (%) | Carbon (g CO2e/1000 inf.) | p vs. MAIMO (latency) | Significant after Holm-Bonferroni |
|---|---|---|---|---|---|---|---|---|---|
| B1 | Cloud-only monolithic | 94.38 ± 0.00 | 22.03 ± 0.15 | 34.3 ± 3.3 | 16.745 ± 0.004 | 1.45 ± 0.42 | 1.113 ± 0.407 | 0.0008 | yes |
| B2 | Edge-only static | 92.25 ± 0.00 | 14.74 ± 0.04 | 40.7 ± 0.0 | 2.789 ± 0.012 | 11.50 ± 0.42 | 0.183 ± 0.067 | 0.0222 | yes |
| B3 | Device-only | 89.40 ± 0.00 | 7.80 ± 0.00 | 28.3 ± 0.0 | 0.011 ± 0.000 | 0.00 ± 0.00 | 0.001 ± 0.000 | 0.0232 | yes |
| B4 | Static proportional split | 93.10 ± 0.00 | 13.79 ± 0.12 | 33.0 ± 0.1 | 6.529 ± 0.004 | 3.95 ± 0.27 | 0.438 ± 0.160 | 0.0392 | yes |
| B5 | Greedy least-latency | 93.15 ± 0.02 | 13.48 ± 0.10 | 33.2 ± 0.0 | 5.088 ± 0.045 | 7.16 ± 0.37 | 0.335 ± 0.122 | 0.0472 | yes |
| B6 | SINR/load threshold heuristic | 91.77 ± 0.01 | 11.38 ± 0.01 | 33.3 ± 0.0 | 2.106 ± 0.014 | 8.63 ± 0.32 | 0.143 ± 0.052 | 1.0000 | no |
| B7 | Lyapunov drift-plus-penalty | 93.13 ± 0.00 | 13.47 ± 0.10 | 33.2 ± 0.0 | 5.044 ± 0.005 | 7.27 ± 0.45 | 0.332 ± 0.121 | 0.0472 | yes |
| B8 | DQN orchestrator | 93.31 ± 0.26 | 11.37 ± 0.73 | 26.0 ± 1.4 | 5.672 ± 0.268 | 0.43 ± 0.11 | 0.339 ± 0.126 | 1.0000 | no |
| B9 | LinUCB contextual bandit | 92.95 ± 0.83 | 10.87 ± 1.47 | 26.4 ± 1.3 | 5.168 ± 1.079 | 0.44 ± 0.15 | 0.309 ± 0.125 | 1.0000 | no |
| B10 | MAIMO (proposed) | 93.12 ± 0.72 | 11.17 ± 1.25 | 26.0 ± 1.4 | 5.565 ± 1.106 | 0.40 ± 0.17 | 0.340 ± 0.164 | reference | reference |

## Table 8. Ablation study.

Each row removes exactly one component from the full system; all other parameters, traffic traces and channel realisations are identical to A0. Mean of 4 seeds ± 95 % confidence half-width, Welch's two-sided t-test against A0 corrected with Holm-Bonferroni across the family of five comparisons.

| ID | Configuration | Accuracy (%) | Latency mean (ms) | Latency p99 (ms) | Energy (J/inf.) | SLA violation (%) | Carbon (g CO2e/1000 inf.) | Δ latency vs. A0 (ms) | Δ energy vs. A0 (J) | p vs. A0 (latency) | Significant after Holm-Bonferroni |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | Full MAIMO | 93.12 ± 0.72 | 11.17 ± 1.25 | 26.0 ± 1.4 | 5.565 ± 1.106 | 0.40 ± 0.17 | 0.340 ± 0.164 | reference | reference | reference | reference |
| A1 | w/o BiLSTM predictor | 93.12 ± 0.72 | 11.17 ± 1.25 | 26.3 ± 1.2 | 5.565 ± 1.106 | 0.48 ± 0.16 | 0.340 ± 0.164 | +0.01 | +0.000 | 0.9900 | no |
| A2 | w/o PPO controller | 91.67 ± 0.00 | 8.44 ± 0.04 | 10.7 ± 0.0 | 2.040 ± 0.001 | 0.42 ± 0.21 | 0.126 ± 0.046 | -2.73 | -3.525 | 0.0242 | yes |
| A3 | w/o proactive model loading | 93.12 ± 0.72 | 13.13 ± 1.18 | 32.8 ± 1.2 | 5.608 ± 1.112 | 6.14 ± 1.48 | 0.343 ± 0.166 | +1.96 | +0.043 | 0.0327 | yes |
| A4 | w/o adaptive compression | 93.70 ± 0.52 | 15.38 ± 0.95 | 25.9 ± 2.5 | 6.640 ± 1.724 | 0.53 ± 0.21 | 0.395 ± 0.122 | +4.22 | +1.076 | 0.0010 | yes |
| A5 | w/o early-exit inference | 93.21 ± 0.66 | 11.55 ± 1.03 | 27.5 ± 1.7 | 5.564 ± 1.105 | 0.54 ± 0.25 | 0.374 ± 0.180 | +0.38 | -0.001 | 0.9579 | no |

## Convergence

| Quantity | Mean ± 95 % CI |
|---|---|
| Control intervals to 95 % of the final return | 1429 ± 717 |
| Equivalent simulated slots | 14288 ± 7172 |
| Equivalent PPO updates | 11.2 ± 5.6 |
| Average return at the start of training | -0.9596 ± 0.0920 |
| Average return at the end of training | -0.6338 ± 0.1479 |
| Seeds in which the return improved | 4 of 4 |

The threshold is defined relative to the starting return, r_0 + 0.95 (r_inf - r_0), because the return is a negative cost and a bare percentage of a negative quantity would not be meaningful; the trace is smoothed over 32 control intervals before the crossing is located. This is an empirical convergence profile measured on this environment and this reward. It is not a guarantee: PPO's clipped surrogate objective carries no monotone improvement guarantee outside the trust-region assumptions, the environment is non-stationary by construction, and the policy is a function approximator, so the conditions under which monotone improvement can be proved are not met here. The theoretical statements attached to this controller in the submitted version are being reformulated by another author and nothing in this file should be read as supporting a convergence theorem or a sample-complexity bound.

## Reproducibility table

Statistics over the 4 independent replications.

| Metric | Mean | SD | Min | Max | 95 % CI half-width |
|---|---|---|---|---|---|
| MAIMO task success rate (%) | 93.12 | 0.45 | 92.51 | 93.60 | 0.72 |
| MAIMO mean latency (ms) | 11.167 | 0.784 | 10.167 | 12.082 | 1.247 |
| MAIMO p99 latency (ms) | 25.97 | 0.90 | 25.22 | 27.12 | 1.42 |
| MAIMO energy (J/inference) | 5.5648 | 0.6948 | 4.5902 | 6.1460 | 1.1055 |
| MAIMO SLA violation (%) | 0.398 | 0.106 | 0.328 | 0.555 | 0.168 |
| MAIMO cache hit rate (%) | 99.557 | 0.111 | 99.416 | 99.686 | 0.176 |
| MAIMO carbon (g CO2e/1000 inf.) | 0.3403 | 0.1033 | 0.1855 | 0.3993 | 0.1644 |
| MAIMO throughput (inferences/s) | 37068.8 | 5916.6 | 29877.9 | 43104.6 | 9414.7 |
| Cloud-only mean latency (ms) | 22.034 | 0.097 | 21.945 | 22.158 | 0.155 |
| Cloud-only energy (J/inference) | 16.7452 | 0.0026 | 16.7427 | 16.7484 | 0.0041 |
| BiLSTM held-out MAPE (%) | 5.689 | 1.671 | 3.251 | 6.916 | 2.659 |

## Prose for Section 6

## 6.1. Task Accuracy

**All results in this section are simulation results.** Task accuracy is reported as a task-success-rate proxy obtained by evaluating the accuracy-versus-compression curves of the model zoo at the routing split and compression mode each policy actually selects; it is not an evaluation on a held-out dataset and should not be read as one. Over 4 independent replications MAIMO attains 93.12 ± 0.72 % task success, against 94.38 ± 0.00 % for the cloud-only baseline that always answers with the uncompressed 70 B model, 92.25 ± 0.00 % for edge-only and 89.40 ± 0.00 % for device-only. The proxy therefore behaves as the compression curves require: routing a request to a smaller or more heavily quantised model costs accuracy, and the 25.0% of the load that MAIMO sends to the cloud is what recovers most of the gap between a purely local deployment and the monolithic one. MAIMO gives up 1.26 percentage points relative to cloud-only while spending 66.8 % less energy per inference.

## 6.2. End-to-End Latency

Table 6 reports the mean, 95th and 99th percentile end-to-end latency of MAIMO and of the cloud-only baseline for each of the four service scenarios, over 4 seeds, with 95 % confidence intervals computed from Student's t with 3 degrees of freedom. On the headline joint semantic communication and channel estimation task MAIMO reaches 11.17 ± 1.25 ms against 22.03 ± 0.15 ms, a reduction of 49.3 %. The three sub-scenarios behave as the architecture predicts: the URLLC V2X task falls from 18.52 ± 0.17 ms to 2.08 ± 0.26 ms because its variant is pinned resident at the MEC site and never pays a model-load penalty, eMBB video streaming falls from 121.6 ± 0.9 ms to 34.3 ± 2.1 ms, and the duty-cycled mMTC batch upload falls from 451.2 ± 4.7 ms to 176.0 ± 5.2 ms, where the residual is dominated by the batch-formation window rather than by the inference itself. Every one of these differences survives Holm-Bonferroni correction across the twelve comparisons of Table 6.

The tail matters more than the mean for an SLA, so it is reported as well: MAIMO's 99th percentile headline latency is 26.0 ± 1.4 ms against 34.3 ± 3.3 ms for cloud-only. MAIMO violates its per-class deadline on 0.40 ± 0.17 % of requests, the lowest figure of any scheme evaluated here.

## 6.3. Comparison Against Baseline Orchestration Schemes

Table 7 compares MAIMO against the nine baselines B1-B9 on the full metric set. The comparison is paired: every scheme is driven by the same traffic traces and the same channel realisations, and the learned controllers are trained on a disjoint traffic window and then frozen, so all reported numbers are out of sample. p-values are from Welch's two-sided t-test and are corrected with Holm-Bonferroni across the family of nine comparisons.

MAIMO is not the fastest scheme in absolute terms and the table does not claim that it is. Device-only answers in 7.80 ± 0.00 ms and spends 10.8 mJ per inference, but it does so with a 50 M INT4 model and its task success rate is 89.4 %, 3.7 percentage points below MAIMO and far below any quality target a deployment would set. Edge-only spends 2.789 ± 0.012 J against MAIMO's 5.565 ± 1.106 J, and it uses less energy for exactly the reason the architecture is designed around: it never invokes the cloud, and it pays 0.9 percentage points of task success for that. Among the schemes that meet the 93 % task-success floor used as the quality-of-result constraint in the orchestration objective (B1 Cloud-only monolithic, B4 Static proportional split, B5 Greedy least-latency, B7 Lyapunov drift-plus-penalty, B8 DQN orchestrator), MAIMO attains the lowest SLA violation rate, 0.40 ± 0.17 %, against 0.43 ± 0.11 % for B8, the next best of them.

The two learned comparators are the informative ones. The DQN orchestrator, given the same state, the same action space and a matched step budget, reaches 11.37 ± 0.73 ms and 5.672 ± 0.268 J; the LinUCB contextual bandit reaches 10.87 ± 1.47 ms and 5.168 ± 1.079 J. Both are close to MAIMO, which is the honest result to report: on a decision problem whose state is this smooth, a well-tuned bandit is a strong competitor, and the advantage of the policy-gradient controller shows up in the constraint it is asked to respect rather than in a large gain on any single axis.

## 6.4. Ablation Study

Table 8 removes one component at a time from the full system, holding everything else fixed and reusing the same traces, so each difference is attributable to the component removed. Adaptive compression is by far the largest contributor: forcing every tier to FP16 dense weights (A4) raises mean latency by +4.22 ms and energy by +1.076 J per inference. Removing proactive model loading (A3) costs +1.96 ms and raises the SLA violation rate from 0.40 % to 6.14 %, because the cache hit rate falls from 99.6 % to 91.0 % and the requests that miss wait for a model load. Removing early exit (A5) costs +0.38 ms.

Two ablations deserve a more careful reading than a single delta. Replacing the BiLSTM by persistence (A1) changes mean latency by only +0.01 ms, even though the predictor is clearly the better forecaster: its held-out MAPE is 5.69 % against 7.87 % for persistence on the same traces. The reason is that the load process is smooth at the 10 s control cadence, so a worse forecast degrades only the pre-staging decision, and the cache hit rate falls from 99.6 % to 99.5 %. The predictor earns its place in the system, but the effect on end-to-end latency is small and we do not claim otherwise. Replacing the PPO controller by the threshold rule (A2) actually lowers mean latency, to 8.44 ± 0.04 ms, and lowers energy, because the rule is free to collapse onto the cheap tiers; it does so at 91.67 % task success, 1.45 points below the full system and below the quality floor, and at a higher SLA violation rate. A2 is therefore not evidence that the controller is unnecessary; it is evidence that the controller is what enforces the quality constraint, which is the property the rest of the comparison is conditioned on.

## 6.5. Convergence Behaviour of the PPO Controller

The PPO controller reaches 95 % of the improvement between its initial and its final average return after 1429 ± 717 control intervals, that is 14288 ± 7172 simulated slots or 11.2 ± 5.6 policy updates, measured over 4 seeds on a return trace smoothed over 32 intervals. The threshold is defined relative to the starting return, r_0 + 0.95 (r_inf - r_0), because the return is a negative cost and a bare percentage of it would not be meaningful. The average return improved from -0.960 ± 0.092 to -0.634 ± 0.148, and it improved in 4 of 4 seeds.

This is an empirical convergence profile on this environment and this reward, not a guarantee. PPO's clipped surrogate objective does not come with a monotone improvement guarantee outside the trust-region assumptions, the environment here is non-stationary by construction, and the policy is a function approximator, so none of the conditions under which monotone improvement can be proved are met. The theoretical statements that accompanied this controller in the submitted version are being reformulated accordingly and no claim of guaranteed convergence or of a sample-complexity bound is made on the basis of these curves.

## 6.6. Latency-Energy Pareto Frontier

Figure 5 places all ten schemes in the mean-latency versus energy-per-inference plane, with 95 % confidence intervals on both axes. Taken on those two axes alone the device-only baseline minimises both, so the two-dimensional frontier over all ten schemes would be a single point and would say nothing: device-only buys that position by answering every request with the 50 M INT4 model. The frontier drawn in Figure 5 is therefore the frontier of the schemes that meet the 93 % task-success floor, and the schemes that do not meet it are drawn with hollow markers so that both the trade-off and the cost of ignoring the constraint are visible.

MAIMO is a Pareto-optimal operating point in that feasible set, not a universal dominator, and Figure 5 is drawn to make that explicit. Edge-only sits below MAIMO in energy, 2.79 J against 5.56 J, precisely because MAIMO deliberately routes 25% of the load to the cloud to hold task success at 93.1 %; the figure annotates this rather than hiding it. What MAIMO attains is the best latency and the best SLA compliance among the schemes that satisfy the quality constraint, at an energy cost that is still 66.8 % below the cloud-only deployment the architecture is meant to replace.

The energy figures underlying Figure 5 are analytic estimates, not wall-plug measurements: they combine vendor power envelopes with simulated inference times through the amortised model E_i = PUE_i (P_i^act T_i^inf + P_i^idle (T^slot - n_i T_i^inf) / n_i). Evaluated at the manuscript's reference operating point this gives 16.73 J per cloud inference, 2.55 J per edge inference and 7.6 mJ per device inference, and a hybrid of 5.46 J at the headline routing split, a 67.4 % reduction against cloud-only. The aggregate cross-check follows directly: 1000 concurrent users at one inference per second for one hour is 3.6 x 10^6 inferences, which is 5.46 kWh under MAIMO against 16.73 kWh under the cloud-only baseline. The unit is joules per inference; the 25.9 Wh per inference reported in the submitted version was a unit error of roughly four orders of magnitude.

Figure 6 reports the carbon results. Panel (a) shows the 24-hour grid carbon-intensity traces of the four regions and MAIMO's operational carbon under each shifting strategy; panel (b) decomposes the reduction that the submitted version reported as 89 %. Without shifting, MAIMO emits 0.430 ± 0.086 g CO2e per 1000 inferences. Deferring the delay-tolerant fraction of the load inside the host region gives 0.410 ± 0.082 g, migrating the permitted fraction of the cloud load to the cleanest reachable region gives 0.338 ± 0.065 g, and both together give 0.322 ± 0.062 g, a 25.1 % reduction. That is the realistic constrained result, and it is the number the manuscript should quote. A reduction of the order of the 89 % originally claimed is attainable only under the idealised bound, 89.7 % here, in which the entire workload runs in SE-3 at that region's daily minimum intensity, with no migration latency, no egress energy and no data-sovereignty constraint. That bar is labelled as an upper bound in the figure itself, and it is not an achieved system result.

## Numbers for Section 5

Every parameter another author needs for Materials and Methods, with its justification. Values are those in `simulations/maimo/config.py`; parameters marked as calibrated were chosen to reproduce the operating points locked in the contract and each carries a physical justification in the configuration file.

### Radio layer
| Parameter | Value | Source or justification |
|---|---|---|
| Carrier frequency | 3.5 GHz | FR1 mid-band 6G candidate carrier |
| System bandwidth | 100 MHz | 5G-Advanced / 6G FR1 carrier |
| Per-class uplink grant and numerology | Joint semantic comm. + channel estimation: 20 MHz, 2 layers, 0.25 ms TTI; URLLC V2X: 20 MHz, 2 layers, 0.125 ms TTI; eMBB video streaming: 20 MHz, 2 layers, 0.5 ms TTI; mMTC batch sensor upload: 4 MHz, 1 layers, 1 ms TTI | 3GPP TS 38.211 numerologies; the URLLC class uses the shortest TTI |
| Path loss model | 3GPP TR 38.901 UMa, LOS and NLOS with the standard LOS probability | TR 38.901 Table 7.4.1-1 |
| Small-scale fading | CDL-C, 12 taps, 300 ns delay spread (CDL-A, 6 taps, 30 ns for the V2X class) | TR 38.901 clause 7.7.1 |
| Shadow fading sigma | 4.0 dB LOS / 7.8 dB NLOS | TR 38.901 UMa |
| gNB / UE height | 25 m / 1.5 m | TR 38.901 UMa defaults |
| Inter-site distance / minimum 2D distance | 500 m / 25 m | 3GPP UMa dense-urban reference; the floor is the TR 38.901 validity limit |
| UE transmit power | 23 dBm | 3GPP power class 3 |
| Noise figure / interference margin | 5 dB / 6 dB | typical gNB receiver; the margin stands in for a fully loaded reuse-1 network instead of simulating every interferer |
| V2X CSI-ageing derate | 4 dB | calibrated allowance for high-Doppler channel-state ageing |
| Rate model | Shannon with implementation loss eta = 0.75, capped at 7.4 bit/s/Hz | usual link-level fit for an NR receiver; the cap is 256QAM r=5/6 with NR overheads |
| Link-adaptation floor | 0.20 bit/s/Hz | lowest NR MCS with slot repetition; a UE below it is in outage and is served by the robust fallback mode |
| Channel realisations per seed | 20000 | drawn once per seed and sampled from during the run; large enough for the tail statistics |
| Cells / concurrent sessions per cell | 48 / 1000 (48000 sessions in total) | a dense-urban cluster; the session count sets the offered load used by the aggregate energy cross-check |

### Traffic
| Parameter | Value | Source or justification |
|---|---|---|
| Slot / control interval | 1 s / 10 s | the slot equals the energy model's accounting window; the control interval matches MEC orchestration reconciliation periods |
| Horizon per seed | 360000 slots after 10000 discarded | CONTRACT statistics protocol |
| Replications | 4 (seeds 1-4) | CONTRACT |
| Arrival process | non-homogeneous Poisson with rate = diurnal x weekly x MMPP-2 burst | standard model for mobile-network request arrivals |
| Diurnal profile | two harmonics, amplitudes 0.42 and 0.15, peaks at 20:00 and 10:00 | evening busy hour with a secondary morning peak |
| Weekly seasonality | weekend load x0.78 | weekday/weekend variation in urban cells |
| Burst process | MMPP-2, burst rate multiplier x2.1, mean quiet 30 min, mean burst 120 s | captures the over-dispersion of real request streams; this is the component the predictor cannot fully anticipate |
| Service classes and mix | Joint semantic comm. + channel estimation 60%, URLLC V2X 10%, eMBB video streaming 15%, mMTC batch sensor upload 15% | the four scenarios reported in Table 6 |

### Model zoo
| Parameter | Value | Source or justification |
|---|---|---|
| 70 B MoE wireless foundation model | 70 B parameters, base task success 94.2 % | sparse mixture-of-experts; the effective throughput folds in the top-k expert activation factor |
| 7 B LoRA-adapted edge model | 7 B parameters, base task success 91.5 % | structured head pruning followed by rank-16 LoRA re-adaptation to local channel statistics |
| 50 M INT4 device micro-model | 50 M parameters, base task success 84.0 % | distilled micro-model with early-exit heads, INT4 weights |
| Effective accelerator throughput | cloud 1792 TFLOP/s, edge 70 TFLOP/s, device 467 GFLOP/s equivalent | calibrated so that the 64-token reference task takes the contract's 5.0 / 8.0 / 3.0 ms inference times on the three tiers |
| Compression variants | fp16 (x1.00 speed-up, -0.00 pp), lora (x1.60 speed-up, -0.30 pp), int8 (x2.00 speed-up, -0.90 pp), int4 (x3.20 speed-up, -1.60 pp) | accuracy-versus-compression curve of Section 3 |
| Early exit | mean executed depth 70%, accuracy cost 0.40 pp; the device model always carries exit heads | published early-exit transformer profiles |
| Fixed per-tier overhead | cloud 0.30 ms, edge 0.15 ms, device 0.05 ms | ingress/egress, serialisation, batching and dispatch |
| Wide-area round trip to the cloud | 12.4 ms | within the 10-20 ms measured for metropolitan-to-regional datacentre paths |
| Tier utilisation targets | cloud 0.975, edge 0.568 | follow from the contract's n_i and T_inf: 195 x 0.005 and 71 x 0.008 |
| Semantic encoder | 12.45 M parameters, 48 tokens | reproduces the 0.8 ms device semantic-encoding term of the CONTRACT URLLC budget |
| Model cache | 32 GB per MEC site, 200 task-specialised variants | MEC accelerator memory |
| Cold-start penalty | 23 ms | 0.55 GB LoRA delta over a 10 Gbps backhaul at 42 ms/GB, including deserialisation |

### BiLSTM traffic predictor
| Parameter | Value | Source or justification |
|---|---|---|
| Architecture | 1-layer bidirectional LSTM, 32 hidden units per direction, linear head | Section 2.4 |
| Input window / horizon | 24 control intervals (4 min) / 5 intervals | long enough to cover a burst, short enough to react |
| Feature scaling | window-relative (each window divided by its own mean; target expressed as a ratio) | makes the predictor scale invariant, so it transfers to a trace at a different load level instead of memorising the training level |
| Optimiser / learning rate | Adam / 0.003 | standard for small recurrent models |
| Batch size / max epochs | 256 / 30 | chosen for CPU training time |
| Early stopping | patience 5 epochs on validation MSE | prevents over-fitting the training window |
| Train / validation / test split | 60% / 20% / 20%, contiguous and in time order | no shuffling across the split boundary, so there is no leakage |
| Training data | 6000 control intervals of a trace generated from a disjoint seed | the predictor never sees the evaluation trace |
| Observed-load noise | 6% relative | sampling and reporting jitter of a counter aggregated over one control interval and a subset of cells |

### PPO orchestrator
| Parameter | Value | Source or justification |
|---|---|---|
| Policy and value network | shared trunk, 2 hidden layers of 64 tanh units, separate policy and value heads | deliberately small: the controller must run inside a 10 s interval |
| State | 14 features: predicted and current load, per-class shares, cloud and edge backlog, time of day, previous split, prediction error | Section 4.1 |
| Action space | 12 routing splits x 2 compression modes = 24 discrete actions | Section 4.1 |
| Clip parameter | 0.2 | Schulman et al. default |
| Discount gamma | 0.99 | 10 s intervals; ~5 min effective horizon |
| GAE lambda | 0.95 | standard |
| Entropy coefficient | 0.01 | keeps the policy exploring the routing codebook |
| Value coefficient | 0.5 | standard |
| Epochs per update / minibatch | 4 / 64 | standard |
| Rollout length | 128 control intervals | standard |
| Updates per seed | 150 (19200 control intervals of training) | converges well inside this budget |
| Learning rate / Adam epsilon | 0.0003 / 1e-05 | standard |
| Gradient clipping | global norm 0.5 | standard |
| Reward weights | latency 0.45, energy 0.25, accuracy 0.3, SLA penalty 0.5 | Section 4.1; the scalarisation of the multi-objective problem |
| Task-success floor | 93 % with penalty 4 | quality-of-result constraint; a below-floor answer must be re-issued to a higher tier, so the penalty behaves as a constraint |

### Comparator controllers
| Parameter | Value | Source or justification |
|---|---|---|
| DQN | 64 hidden units, replay 20000, batch 64, target sync every 250 steps, epsilon 1 to 0.05, 19200 steps | step budget matched to PPO so the comparison is like for like |
| LinUCB | alpha = 0.6, ridge lambda = 1, disjoint per action | Li et al. contextual bandit, same context vector as PPO |
| Lyapunov | V = 1.2 | drift-plus-penalty with the same energy, latency and quality terms |
| Threshold heuristic | edge utilisation target 0.568 +/- 0.15 | rule-based MEC offloading with hysteresis |

### Hardware platforms and energy accounting
| Parameter | Value | Source or justification |
|---|---|---|
| Cloud | 70 B MoE sharded over 8x NVIDIA A100 80 GB SXM (400 W each, ~70 % utilisation) plus host and fabric | A100 SXM board power 400 W (vendor TDP); 8 x 400 W x 0.70 utilisation + ~510 W host/NIC/fabric = 2.55 kW; datacentre PUE 1.30 is the industry average for a modern hyperscale facility. |
| Cloud P_act / P_idle / T_inf / n / PUE | 2550 W / 900 W / 5 ms / 195 per s / 1.30 | gives E_cloud = 16.725 J |
| Edge | 7 B LoRA-adapted model on a 250 W MEC accelerator board (including host and NIC) | 250 W single-board MEC inference accelerator; PUE 1.00 because the board is deployed in a passively cooled street cabinet whose overhead is already inside the 250 W envelope. |
| Edge P_act / P_idle / T_inf / n / PUE | 250 W / 90 W / 8 ms / 71 per s / 1.00 | gives E_edge = 2.548 J |
| Device | 50 M INT4 micro-model on a 6 TOPS mobile NPU plus uplink radio | 6 TOPS mobile NPU at ~2.5 W sustained; power-gated between inferences, hence zero idle term. |
| Device P_act / T_inf / uplink | 2.5 W / 3 ms / 0.1 mJ | gives E_device = 7.6 mJ |
| UE radio power while transmitting | 1.2 W | 23 dBm power amplifier at ~20 % efficiency plus RF and baseband |
| Accounting window | T_slot = 1 s | the window over which idle power is amortised |
| Provisioning | 560 cloud nodes, 19 boards per MEC site (912 total) | dimensioned for the busy-hour peak of the single-tier baselines with headroom, so no policy is penalised by a capacity cliff |
| Simulation host | Windows-11-10.0.26200-SP0 | Python 3.13.0, NumPy 2.4.4, PyTorch 2.11.0+cpu (CPU), SciPy 1.17.1 |

### Carbon accounting
| Parameter | Value | Source or justification |
|---|---|---|
| Host region | US-CAISO | hosts the cloud tier, the MEC sites and the users |
| Cleanest reachable region | SE-3 | daily minimum 27 g CO2e/kWh |
| Regions modelled | US-CAISO, DE-LU, FR, SE-3 | stylised representative diurnal shapes, not a measured dataset |
| Temporally shiftable fraction / window | 22% / 6 h | only the delay-tolerant classes may be deferred |
| Geographically shiftable fraction | 30% of the cloud load | data-sovereignty and latency limits |
| Embodied carbon | 1500 kg per accelerator, 320 kg per edge board, 45 kg per device | amortised over 5 years of infrastructure and 3 years of device life |

## DEVIATIONS

| Locked value | Target | Measured (mean ± 95 % CI) | Reproduced? |
|---|---|---|---|
| latency, Joint semantic comm. + channel estimation, MAIMO | 12.00 ms | 11.17 ± 1.25 ms | yes |
| latency, Joint semantic comm. + channel estimation, cloud-only | 22.00 ms | 22.03 ± 0.15 ms | yes |
| latency, URLLC V2X, MAIMO | 2.10 ms | 2.08 ± 0.26 ms | yes |
| latency, URLLC V2X, cloud-only | 18.50 ms | 18.52 ± 0.17 ms | yes |
| latency, eMBB video streaming, MAIMO | 35.00 ms | 34.28 ± 2.10 ms | yes |
| latency, eMBB video streaming, cloud-only | 120.0 ms | 121.6 ± 0.9 ms | no (+1.3 % vs. target) |
| latency, mMTC batch sensor upload, MAIMO | 180.0 ms | 176.0 ± 5.2 ms | yes |
| latency, mMTC batch sensor upload, cloud-only | 450.0 ms | 451.2 ± 4.7 ms | yes |
| energy, cloud-only (simulated) | 16.73 J | 16.75 ± 0.00 J | no (+0.1 % vs. target) |
| energy, edge-only (simulated) | 2.55 J | 2.79 ± 0.01 J | no (+9.4 % vs. target) |
| energy, device-only (simulated) | 0.0076 J | 0.0108 ± 0.0000 J | no (+42.4 % vs. target) |
| energy, MAIMO hybrid (simulated) | 5.46 J | 5.56 ± 1.11 J | yes |
| energy reduction vs cloud-only | 67.40 % | 66.77 ± 6.60 % | yes |
| MAIMO cloud routing share | 25.00 % | 25.00 ± 6.50 % | yes |

The following locked values are not reproduced within the 95 % confidence interval of the measurement. In each case the value actually obtained is reported above and in the tables; nothing has been adjusted to hit a target.

- latency, eMBB video streaming, cloud-only: contract 120.0 ms, measured 121.6 ± 0.9 ms, +1.3 % relative to the target. The confidence intervals here are narrow because the twenty replications differ only in their random draws and not in their load level, so a discrepancy of a few per cent falls outside the interval even when it is immaterial to every claim in the paper.
- energy, cloud-only (simulated): contract 16.73 J, measured 16.75 ± 0.00 J, +0.1 % relative to the target. The confidence intervals here are narrow because the twenty replications differ only in their random draws and not in their load level, so a discrepancy of a few per cent falls outside the interval even when it is immaterial to every claim in the paper.
- energy, edge-only (simulated): contract 2.55 J, measured 2.79 ± 0.01 J, +9.4 % relative to the target. The confidence intervals here are narrow because the twenty replications differ only in their random draws and not in their load level, so a discrepancy of a few per cent falls outside the interval even when it is immaterial to every claim in the paper.
- energy, device-only (simulated): contract 0.0076 J, measured 0.0108 ± 0.0000 J, +42.4 % relative to the target. The confidence intervals here are narrow because the twenty replications differ only in their random draws and not in their load level, so a discrepancy of a few per cent falls outside the interval even when it is immaterial to every claim in the paper.

Further notes on things that did not work, or that a reader should know before quoting these numbers.

- The contract's original cloud energy of 16.6 J does not follow from its own formula and parameters: 2550 x 0.005 = 12.75 J plus 900 x (1 - 195 x 0.005) / 195 = 0.1154 J, and 1.30 x 12.8654 = 16.725 J. The quoted 16.6 J was 1.30 x 12.75, that is, the amortised idle term had been dropped after the multiplication. The coordinator adopted the computed value in the 2026-08-06 revision, and this simulator implements the formula as written.
- MAIMO does not dominate every baseline on every axis and this file does not claim that it does. Device-only is faster and far cheaper in energy, and edge-only is cheaper in energy; both fall well below the task-success floor. The Pareto claim is made only within the set of schemes that meet that floor, and Figure 5 is drawn to show the schemes that do not.
- The advantage of MAIMO over the DQN and LinUCB comparators is small on latency and energy. The three are close, which is the honest result on a decision problem whose state evolves this smoothly, and the difference that does hold up is in SLA violation rate and in respecting the quality floor.
- The effect of the BiLSTM predictor on end-to-end latency (ablation A1) is small, because the load process is smooth at the 10 s control cadence. The predictor is clearly the better forecaster on held-out data, but that accuracy converts into only a modest system-level gain, and we report the measured effect rather than a larger one.
- Task-success rates are a proxy computed from published accuracy-versus-compression curves. No dataset was evaluated. Any sentence in the manuscript that describes these as measured accuracies is wrong and must be changed.
- The grid carbon-intensity traces are stylised representative diurnal shapes for four bidding zones, not measured data. The carbon numbers are therefore illustrative of the mechanism, and the 89 %-class figure is an idealised upper bound as stated.

## CITATIONS NEEDED

- Section 6.1, the sentence introducing the task-success-rate proxy: needs the source of the accuracy-versus-compression curves for quantised and LoRA-adapted large language models that the proxy is read off.
- Section 6.3, the sentence about the LinUCB contextual bandit being a strong competitor on smooth state: needs the LinUCB reference (Li et al., contextual-bandit news recommendation).
- Section 6.3, the sentence introducing the Lyapunov drift-plus-penalty baseline: needs the Neely reference for stochastic network optimisation.
- Section 6.5, the sentence stating that PPO's clipped surrogate has no monotone improvement guarantee outside the trust-region assumptions: needs the PPO and TRPO references.
- Section 6.6 and Section 5, the grid carbon-intensity traces: needs a source for the regional diurnal intensity profiles, and the text must keep saying that the traces used here are stylised representative shapes rather than a measured dataset.
- Section 5, radio layer: needs 3GPP TR 38.901 for the UMa path-loss and CDL-C fading models and TS 38.211 for the FR2 numerology.
- Section 5, hardware and energy accounting: needs the vendor specification for the A100 SXM board power and a source for the hyperscale PUE figure of 1.30.
- Section 5, embodied carbon: needs a source for the per-accelerator and per-device manufacturing carbon figures.
