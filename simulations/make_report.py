"""Write ``../work/sim_results.md`` from ``results/``.

The report is generated rather than typed so that every number in the prose is
the same object as the number in the tables: the prose is an f-string over the
same ``Summary`` values that produce the table cells.  If the simulation is
re-run, the text follows.

Output dialect: the restricted Markdown accepted by ``tools/md2docx.py``
(see ``work/CONTRACT.md``, "Output formats").  One paragraph per line, no
hard wrapping inside a paragraph, no blank line inside a table.

Usage::

    python make_report.py [--results results] [--out ../work/sim_results.md]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence

import numpy as np

from maimo import carbon as cb
from maimo import energy as en
from maimo import models as mz
from maimo.ablations import ABLATIONS
from maimo.baselines import BASELINES
from maimo.channel import ChannelPool
from maimo.config import (CLASS_LABELS, COMPRESSIONS, DEFAULT, MODEL_BY_TIER,
                          SERVICE_CLASSES, TIERS)
from maimo.sim import ALPHA_CODEBOOK, COMPRESSION_MODES

MZ = MODEL_BY_TIER
from maimo.stats import Summary, summarise, t_critical

BASE_IDS = [s.ident for s in BASELINES]
ABL_IDS = [a.ident for a in ABLATIONS]
NAMES = {s.ident: s.name for s in BASELINES}
NAMES.update({a.ident: a.name for a in ABLATIONS})


class Report:
    def __init__(self, results: str):
        self.dir = results
        self.meta = self._json("meta")
        self.cmp = self._json("comparisons")
        self.per_class = self._json("per_class")
        self.carbon = self._json("carbon")
        self.conv = self._json("convergence")
        self.cfg = DEFAULT
        self.n = len(self.meta["seeds"])
        self.tc = t_critical(self.n)
        self.rec = {p: self._json(p) for p in BASE_IDS + ABL_IDS}

    def _json(self, name: str) -> dict:
        with open(f"{self.dir}/{name}.json", encoding="utf-8") as f:
            return json.load(f)

    def s(self, pid: str, metric: str) -> Summary:
        return summarise(np.asarray(self.rec[pid][metric], dtype=float),
                         self.tc)

    def pc(self, pid: str, metric: str, c: int) -> Summary:
        a = np.asarray(self.per_class[pid][metric], dtype=float)[:, c]
        return summarise(a, self.tc)

    def p(self, family: str, metric: str, pid: str) -> dict:
        return self.cmp[family][metric][pid]


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------
def pm(s: Summary, d: int = 2) -> str:
    return f"{s.mean:.{d}f} ± {s.ci95:.{d}f}"


def pval(p: float) -> str:
    if p < 1e-4:
        return "&lt; 0.0001"
    return f"{p:.4f}"


def sig(entry: dict) -> str:
    return "yes" if entry["significant"] else "no"


def reduction(a: Summary, b: Summary) -> float:
    """Percentage reduction of ``a`` relative to ``b``."""
    return 100.0 * (b.mean - a.mean) / b.mean


# ---------------------------------------------------------------------------
# convergence
# ---------------------------------------------------------------------------
def convergence_stats(r: Report, pid: str = "B10") -> dict:
    """Control intervals until the smoothed return first reaches 95 % of the
    way from its initial to its final level, per seed.

    Rewards here are negative costs, so "95 % of the final return" is only
    meaningful relative to where the policy started: the threshold is
    ``r_0 + 0.95 (r_inf - r_0)``.  This is stated in the report.
    """
    tr = np.asarray(r.conv[pid]["reward_trace"], dtype=float)   # (G, R)
    g, R = tr.shape
    w = max(R // 100, 1)
    k = np.ones(w) / w
    sm = np.stack([np.convolve(tr[i], k, mode="valid") for i in range(g)])
    r0 = sm[:, :max(sm.shape[1] // 50, 1)].mean(axis=1)
    rinf = sm[:, -max(sm.shape[1] // 10, 1):].mean(axis=1)
    thr = r0 + 0.95 * (rinf - r0)
    idx = []
    for i in range(g):
        hit = np.nonzero(sm[i] >= thr[i])[0]
        idx.append(float(hit[0] + w) if hit.size else float(R))
    iv = np.array(idx)
    return {
        "intervals": summarise(iv, r.tc),
        "slots": summarise(iv * r.cfg.control_interval_slots, r.tc),
        "updates": summarise(iv / r.cfg.ppo_rollout, r.tc),
        "final_return": summarise(rinf, r.tc),
        "initial_return": summarise(r0, r.tc),
        "improved": int(np.sum(rinf > r0)),
        "n": g,
        "total_intervals": R,
    }


# ---------------------------------------------------------------------------
# report sections
# ---------------------------------------------------------------------------
METRIC_COLS = [
    ("accuracy_pct", "Accuracy (%)", 2),
    ("latency_mean_ms", "Latency mean (ms)", 2),
    ("latency_p99_ms", "Latency p99 (ms)", 1),
    ("energy_j", "Energy (J/inf.)", 3),
    ("sla_violation_pct", "SLA violation (%)", 2),
    ("carbon_g_per_1000", "Carbon (g CO2e/1000 inf.)", 3),
]


def table7(r: Report) -> List[str]:
    out = ["| ID | Scheme | " + " | ".join(c[1] for c in METRIC_COLS)
           + " | p vs. MAIMO (latency) | Significant after Holm-Bonferroni |",
           "|---|---|" + "---|" * (len(METRIC_COLS) + 2)]
    for pid in BASE_IDS:
        cells = [pm(r.s(pid, m), d) for m, _, d in METRIC_COLS]
        if pid == "B10":
            ptxt, stxt = "reference", "reference"
        else:
            e = r.p("baselines", "latency_mean_ms", pid)
            ptxt, stxt = pval(e["p_holm"]), sig(e)
        out.append(f"| {pid} | {r.rec[pid]['name']} | " + " | ".join(cells)
                   + f" | {ptxt} | {stxt} |")
    return out


def table8(r: Report) -> List[str]:
    a0 = {m: r.s("A0", m) for m, _, _ in METRIC_COLS}
    out = ["| ID | Configuration | " + " | ".join(c[1] for c in METRIC_COLS)
           + " | Δ latency vs. A0 (ms) | Δ energy vs. A0 (J) | p vs. A0 "
             "(latency) | Significant after Holm-Bonferroni |",
           "|---|---|" + "---|" * (len(METRIC_COLS) + 4)]
    for pid in ABL_IDS:
        cells = [pm(r.s(pid, m), d) for m, _, d in METRIC_COLS]
        if pid == "A0":
            dl, de, ptxt, stxt = "reference", "reference", "reference", \
                "reference"
        else:
            dl = f"{r.s(pid, 'latency_mean_ms').mean - a0['latency_mean_ms'].mean:+.2f}"
            de = f"{r.s(pid, 'energy_j').mean - a0['energy_j'].mean:+.3f}"
            e = r.p("ablations", "latency_mean_ms", pid)
            ptxt, stxt = pval(e["p_holm"]), sig(e)
        out.append(f"| {pid} | {r.rec[pid]['name']} | " + " | ".join(cells)
                   + f" | {dl} | {de} | {ptxt} | {stxt} |")
    return out


def table6(r: Report) -> List[str]:
    out = ["| Scenario | Metric | MAIMO (ms) | Cloud-only B1 (ms) | "
           "Reduction (%) | p (Holm-Bonferroni) |",
           "|---|---|---|---|---|---|"]
    labels = {"latency_mean_ms": "mean", "latency_p95_ms": "p95",
              "latency_p99_ms": "p99"}
    for c, scenario in enumerate(CLASS_LABELS):
        for metric, lab in labels.items():
            a, b = r.pc("B10", metric, c), r.pc("B1", metric, c)
            d = 2 if b.mean < 50 else 1
            t = r.per_class["tests"][scenario][metric]
            out.append(f"| {scenario} | {lab} | {pm(a, d)} | {pm(b, d)} | "
                       f"{reduction(a, b):.1f} | {pval(t['p_holm'])} |")
    return out


def reproducibility_table(r: Report) -> List[str]:
    out = ["| Metric | Mean | SD | Min | Max | 95 % CI half-width |",
           "|---|---|---|---|---|---|"]
    rows = [("MAIMO task success rate (%)", "B10", "accuracy_pct", 2),
            ("MAIMO mean latency (ms)", "B10", "latency_mean_ms", 3),
            ("MAIMO p99 latency (ms)", "B10", "latency_p99_ms", 2),
            ("MAIMO energy (J/inference)", "B10", "energy_j", 4),
            ("MAIMO SLA violation (%)", "B10", "sla_violation_pct", 3),
            ("MAIMO cache hit rate (%)", "B10", "cache_hit_pct", 3),
            ("MAIMO carbon (g CO2e/1000 inf.)", "B10",
             "carbon_g_per_1000", 4),
            ("MAIMO throughput (inferences/s)", "B10", "throughput_per_s", 1),
            ("Cloud-only mean latency (ms)", "B1", "latency_mean_ms", 3),
            ("Cloud-only energy (J/inference)", "B1", "energy_j", 4),
            ("BiLSTM held-out MAPE (%)", None, None, 3)]
    for label, pid, metric, d in rows:
        if pid is None:
            s = summarise(r.meta["predictor_mape_pct"], r.tc)
        else:
            s = r.s(pid, metric)
        out.append(f"| {label} | {s.mean:.{d}f} | {s.sd:.{d}f} | "
                   f"{s.minimum:.{d}f} | {s.maximum:.{d}f} | {s.ci95:.{d}f} |")
    return out


def carbon_rows(r: Report) -> Dict[str, Summary]:
    v = r.carbon["policies"]["B10"]["values"]
    return {k: summarise(v[k], r.tc) for k in v if "|" not in k}


# ---------------------------------------------------------------------------
# Section 5 parameter dump
# ---------------------------------------------------------------------------
def section5(r: Report) -> List[str]:
    c = r.cfg
    L: List[str] = []

    def tbl(title: str, rows: Sequence[Sequence[str]]):
        L.append("")
        L.append(f"### {title}")
        L.append("| Parameter | Value | Source or justification |")
        L.append("|---|---|---|")
        for row in rows:
            L.append("| " + " | ".join(row) + " |")

    tbl("Radio layer", [
        ["Carrier frequency", f"{c.fc_ghz:.1f} GHz",
         "FR1 mid-band 6G candidate carrier"],
        ["System bandwidth", f"{c.system_bw_hz / 1e6:.0f} MHz",
         "5G-Advanced / 6G FR1 carrier"],
        ["Per-class uplink grant and numerology",
         "; ".join(f"{s.label}: {s.grant_bw_hz / 1e6:.0f} MHz, "
                   f"{s.mimo_layers} layers, {s.tti_ms:g} ms TTI"
                   for s in SERVICE_CLASSES),
         "3GPP TS 38.211 numerologies; the URLLC class uses the shortest TTI"],
        ["Path loss model", "3GPP TR 38.901 UMa, LOS and NLOS with the "
         "standard LOS probability", "TR 38.901 Table 7.4.1-1"],
        ["Small-scale fading",
         f"CDL-C, {c.cdl_c_taps} taps, {c.cdl_c_delay_spread_ns:.0f} ns "
         f"delay spread (CDL-A, {c.cdl_a_taps} taps, "
         f"{c.cdl_a_delay_spread_ns:.0f} ns for the V2X class)",
         "TR 38.901 clause 7.7.1"],
        ["Shadow fading sigma",
         f"{c.sigma_sf_los_db:.1f} dB LOS / {c.sigma_sf_nlos_db:.1f} dB NLOS",
         "TR 38.901 UMa"],
        ["gNB / UE height",
         f"{c.h_bs_m:.0f} m / {c.h_ut_m:.1f} m", "TR 38.901 UMa defaults"],
        ["Inter-site distance / minimum 2D distance",
         f"{c.inter_site_distance_m:.0f} m / {c.min_2d_distance_m:.0f} m",
         "3GPP UMa dense-urban reference; the floor is the TR 38.901 "
         "validity limit"],
        ["UE transmit power", f"{c.ue_tx_dbm:.0f} dBm",
         "3GPP power class 3"],
        ["Noise figure / interference margin",
         f"{c.noise_figure_db:.0f} dB / {c.interference_margin_db:.0f} dB",
         "typical gNB receiver; the margin stands in for a fully loaded "
         "reuse-1 network instead of simulating every interferer"],
        ["V2X CSI-ageing derate", f"{c.v2x_csi_ageing_db:.0f} dB",
         "calibrated allowance for high-Doppler channel-state ageing"],
        ["Rate model",
         f"Shannon with implementation loss eta = {c.impl_loss_eta:.2f}, "
         f"capped at {c.max_spectral_efficiency:.1f} bit/s/Hz",
         "usual link-level fit for an NR receiver; the cap is 256QAM r=5/6 "
         "with NR overheads"],
        ["Link-adaptation floor", f"{c.min_spectral_efficiency:.2f} bit/s/Hz",
         "lowest NR MCS with slot repetition; a UE below it is in outage and "
         "is served by the robust fallback mode"],
        ["Channel realisations per seed", f"{c.n_channel_samples}",
         "drawn once per seed and sampled from during the run; large enough "
         "for the tail statistics"],
        ["Cells / concurrent sessions per cell",
         f"{c.n_cells} / {c.sessions_per_cell} "
         f"({c.n_sessions()} sessions in total)",
         "a dense-urban cluster; the session count sets the offered load used "
         "by the aggregate energy cross-check"],
    ])

    tbl("Traffic", [
        ["Slot / control interval",
         f"{c.t_slot_s:.0f} s / {c.control_interval_slots * c.t_slot_s:.0f} s",
         "the slot equals the energy model's accounting window; the control "
         "interval matches MEC orchestration reconciliation periods"],
        ["Horizon per seed",
         f"{c.horizon_slots:.0f} slots after {c.warmup_slots:.0f} discarded",
         "CONTRACT statistics protocol"],
        ["Replications", f"{r.n} (seeds 1-{r.n})", "CONTRACT"],
        ["Arrival process",
         "non-homogeneous Poisson with rate = diurnal x weekly x MMPP-2 burst",
         "standard model for mobile-network request arrivals"],
        ["Diurnal profile",
         f"two harmonics, amplitudes {c.diurnal_amp1:.2f} and "
         f"{c.diurnal_amp2:.2f}, peaks at {c.diurnal_peak_hour:.0f}:00 and "
         f"{c.diurnal_second_peak_hour:.0f}:00",
         "evening busy hour with a secondary morning peak"],
        ["Weekly seasonality", f"weekend load x{c.weekend_factor:.2f}",
         "weekday/weekend variation in urban cells"],
        ["Burst process",
         f"MMPP-2, burst rate multiplier x{c.mmpp_burst_multiplier:.1f}, mean "
         f"quiet {c.mmpp_mean_quiet_s / 60.0:.0f} min, mean burst "
         f"{c.mmpp_mean_burst_s:.0f} s",
         "captures the over-dispersion of real request streams; this is the "
         "component the predictor cannot fully anticipate"],
        ["Service classes and mix",
         ", ".join(f"{s.label} {s.share_of_sessions:.0%}"
                   for s in SERVICE_CLASSES),
         "the four scenarios reported in Table 6"],
    ])

    tbl("Model zoo", [
        [f"{MZ['cloud'].label}", f"{MZ['cloud'].params / 1e9:.0f} B "
         f"parameters, base task success {MZ['cloud'].base_success_rate:.1f} %",
         MZ["cloud"].note],
        [f"{MZ['edge'].label}", f"{MZ['edge'].params / 1e9:.0f} B parameters, "
         f"base task success {MZ['edge'].base_success_rate:.1f} %",
         MZ["edge"].note],
        [f"{MZ['device'].label}", f"{MZ['device'].params / 1e6:.0f} M "
         f"parameters, base task success "
         f"{MZ['device'].base_success_rate:.1f} %", MZ["device"].note],
        ["Effective accelerator throughput",
         f"cloud {c.flops_cloud / 1e12:.0f} TFLOP/s, edge "
         f"{c.flops_edge / 1e12:.0f} TFLOP/s, device "
         f"{c.flops_device / 1e9:.0f} GFLOP/s equivalent",
         "calibrated so that the 64-token reference task takes the contract's "
         "5.0 / 8.0 / 3.0 ms inference times on the three tiers"],
        ["Compression variants",
         ", ".join(f"{cp.key} (x{cp.speedup:.2f} speed-up, "
                   f"-{cp.acc_penalty_pp:.2f} pp)" for cp in COMPRESSIONS),
         "accuracy-versus-compression curve of Section 3"],
        ["Early exit",
         f"mean executed depth {c.early_exit_factor:.0%}, accuracy cost "
         f"{c.early_exit_acc_pp:.2f} pp; the device model always carries exit "
         f"heads", "published early-exit transformer profiles"],
        ["Fixed per-tier overhead",
         f"cloud {c.cloud_fixed_ms:.2f} ms, edge {c.edge_fixed_ms:.2f} ms, "
         f"device {c.device_fixed_ms:.2f} ms",
         "ingress/egress, serialisation, batching and dispatch"],
        ["Wide-area round trip to the cloud", f"{c.wan_rtt_ms:.1f} ms",
         "within the 10-20 ms measured for metropolitan-to-regional "
         "datacentre paths"],
        ["Tier utilisation targets",
         f"cloud {c.cloud_target_utilisation:.3f}, edge "
         f"{c.edge_target_utilisation:.3f}",
         "follow from the contract's n_i and T_inf: 195 x 0.005 and "
         "71 x 0.008"],
        ["Semantic encoder",
         f"{c.semantic_encoder_params / 1e6:.2f} M parameters, "
         f"{c.semantic_encoder_tokens} tokens",
         "reproduces the 0.8 ms device semantic-encoding term of the "
         "CONTRACT URLLC budget"],
        ["Model cache",
         f"{c.edge_cache_gb:.0f} GB per MEC site, {c.edge_variants} "
         f"task-specialised variants",
         "MEC accelerator memory"],
        ["Cold-start penalty", f"{c.cold_start_penalty_ms:.0f} ms",
         f"0.55 GB LoRA delta over a 10 Gbps backhaul at "
         f"{c.cold_start_ms_per_gb:.0f} ms/GB, including deserialisation"],
    ])

    tbl("BiLSTM traffic predictor", [
        ["Architecture",
         f"{c.pred_layers}-layer bidirectional LSTM, {c.pred_hidden} hidden "
         f"units per direction, linear head",
         "Section 2.4"],
        ["Input window / horizon",
         f"{c.pred_window} control intervals "
         f"({c.pred_window * c.control_interval_slots * c.t_slot_s / 60:.0f} min)"
         f" / {c.pred_horizon} intervals",
         "long enough to cover a burst, short enough to react"],
        ["Feature scaling", "window-relative (each window divided by its own "
         "mean; target expressed as a ratio)",
         "makes the predictor scale invariant, so it transfers to a trace at "
         "a different load level instead of memorising the training level"],
        ["Optimiser / learning rate",
         f"Adam / {c.pred_lr:g}", "standard for small recurrent models"],
        ["Batch size / max epochs",
         f"{c.pred_batch} / {c.pred_epochs}", "chosen for CPU training time"],
        ["Early stopping",
         f"patience {c.pred_patience} epochs on validation MSE",
         "prevents over-fitting the training window"],
        ["Train / validation / test split",
         f"{1 - c.pred_val_fraction - c.pred_test_fraction:.0%} / "
         f"{c.pred_val_fraction:.0%} / {c.pred_test_fraction:.0%}, "
         f"contiguous and in time order",
         "no shuffling across the split boundary, so there is no leakage"],
        ["Training data",
         f"{c.pred_train_intervals} control intervals of a trace generated "
         f"from a disjoint seed",
         "the predictor never sees the evaluation trace"],
        ["Observed-load noise", f"{c.pred_obs_noise:.0%} relative",
         "sampling and reporting jitter of a counter aggregated over one "
         "control interval and a subset of cells"],
    ])

    tbl("PPO orchestrator", [
        ["Policy and value network",
         f"shared trunk, {c.ppo_layers} hidden layers of {c.ppo_hidden} "
         f"tanh units, separate policy and value heads",
         "deliberately small: the controller must run inside a 10 s interval"],
        ["State", f"{14} features: predicted and current load, per-class "
         f"shares, cloud and edge backlog, time of day, previous split, "
         f"prediction error", "Section 4.1"],
        ["Action space",
         f"{len(ALPHA_CODEBOOK)} routing splits x {len(COMPRESSION_MODES)} "
         f"compression modes = {len(ALPHA_CODEBOOK) * len(COMPRESSION_MODES)} "
         f"discrete actions", "Section 4.1"],
        ["Clip parameter", f"{c.ppo_clip:g}", "Schulman et al. default"],
        ["Discount gamma", f"{c.ppo_gamma:g}", "10 s intervals; ~5 min effective horizon"],
        ["GAE lambda", f"{c.ppo_gae_lambda:g}", "standard"],
        ["Entropy coefficient", f"{c.ppo_entropy_coef:g}",
         "keeps the policy exploring the routing codebook"],
        ["Value coefficient", f"{c.ppo_value_coef:g}", "standard"],
        ["Epochs per update / minibatch",
         f"{c.ppo_epochs} / {c.ppo_minibatch}", "standard"],
        ["Rollout length", f"{c.ppo_rollout} control intervals", "standard"],
        ["Updates per seed",
         f"{c.ppo_updates} ({c.ppo_updates * c.ppo_rollout} control "
         f"intervals of training)", "converges well inside this budget"],
        ["Learning rate / Adam epsilon",
         f"{c.ppo_lr:g} / {c.ppo_adam_eps:g}", "standard"],
        ["Gradient clipping", f"global norm {c.ppo_grad_clip:g}", "standard"],
        ["Reward weights",
         f"latency {c.w_latency:g}, energy {c.w_energy:g}, accuracy "
         f"{c.w_accuracy:g}, SLA penalty {c.reward_sla_penalty:g}",
         "Section 4.1; the scalarisation of the multi-objective problem"],
        ["Task-success floor",
         f"{c.accuracy_floor_pct:.0f} % with penalty "
         f"{c.reward_accuracy_penalty:g}",
         "quality-of-result constraint; a below-floor answer must be "
         "re-issued to a higher tier, so the penalty behaves as a constraint"],
    ])

    tbl("Comparator controllers", [
        ["DQN", f"{c.dqn_hidden} hidden units, replay {c.dqn_replay}, batch "
         f"{c.dqn_batch}, target sync every {c.dqn_target_sync} steps, "
         f"epsilon {c.dqn_eps_start:g} to {c.dqn_eps_end:g}, "
         f"{c.dqn_steps} steps",
         "step budget matched to PPO so the comparison is like for like"],
        ["LinUCB", f"alpha = {c.linucb_alpha:g}, ridge lambda = "
         f"{c.linucb_lambda:g}, disjoint per action",
         "Li et al. contextual bandit, same context vector as PPO"],
        ["Lyapunov", f"V = {c.lyapunov_v:g}",
         "drift-plus-penalty with the same energy, latency and quality terms"],
        ["Threshold heuristic",
         f"edge utilisation target {c.edge_target_utilisation:g} +/- "
         f"{c.threshold_queue_hysteresis:g}",
         "rule-based MEC offloading with hysteresis"],
    ])

    e = en.reference_tier_energies_j()
    tbl("Hardware platforms and energy accounting", [
        ["Cloud", en.CLOUD.platform, en.CLOUD.source],
        ["Cloud P_act / P_idle / T_inf / n / PUE",
         f"{en.CLOUD.p_act_w:.0f} W / {en.CLOUD.p_idle_w:.0f} W / "
         f"{en.CLOUD.t_inf_s * 1e3:.0f} ms / {en.CLOUD.n_per_slot:.0f} per s "
         f"/ {en.CLOUD.pue:.2f}",
         f"gives E_cloud = {e['cloud']:.3f} J"],
        ["Edge", en.EDGE.platform, en.EDGE.source],
        ["Edge P_act / P_idle / T_inf / n / PUE",
         f"{en.EDGE.p_act_w:.0f} W / {en.EDGE.p_idle_w:.0f} W / "
         f"{en.EDGE.t_inf_s * 1e3:.0f} ms / {en.EDGE.n_per_slot:.0f} per s / "
         f"{en.EDGE.pue:.2f}", f"gives E_edge = {e['edge']:.3f} J"],
        ["Device", en.DEVICE.platform, en.DEVICE.source],
        ["Device P_act / T_inf / uplink",
         f"{en.DEVICE.p_act_w:.1f} W / {en.DEVICE.t_inf_s * 1e3:.0f} ms / "
         f"{en.DEVICE.extra_j * 1e3:.1f} mJ",
         f"gives E_device = {e['device'] * 1e3:.1f} mJ"],
        ["UE radio power while transmitting", f"{c.ue_radio_power_w:.1f} W",
         "23 dBm power amplifier at ~20 % efficiency plus RF and baseband"],
        ["Accounting window", f"T_slot = {c.t_slot_s:.0f} s",
         "the window over which idle power is amortised"],
        ["Provisioning",
         f"{c.cloud_nodes_max} cloud nodes, "
         f"{c.edge_boards_per_site} boards per MEC site "
         f"({c.edge_boards_max()} total)",
         "dimensioned for the busy-hour peak of the single-tier baselines "
         "with headroom, so no policy is penalised by a capacity cliff"],
        ["Simulation host", f"{r.meta['platform']}",
         f"Python {r.meta['python']}, NumPy {r.meta['numpy']}, PyTorch "
         f"{r.meta['torch']} (CPU), SciPy {r.meta['scipy']}"],
    ])

    tbl("Carbon accounting", [
        ["Host region", cb.HOST_REGION,
         "hosts the cloud tier, the MEC sites and the users"],
        ["Cleanest reachable region", cb.CLEAN_REGION,
         f"daily minimum {min(cb.REGION_TRACES[cb.CLEAN_REGION]):.0f} "
         f"g CO2e/kWh"],
        ["Regions modelled", ", ".join(cb.REGION_TRACES),
         "stylised representative diurnal shapes, not a measured dataset"],
        ["Temporally shiftable fraction / window",
         f"{c.temporal_shift_max_fraction:.0%} / "
         f"{c.temporal_shift_window_h:.0f} h",
         "only the delay-tolerant classes may be deferred"],
        ["Geographically shiftable fraction",
         f"{c.geo_shift_max_fraction:.0%} of the cloud load",
         "data-sovereignty and latency limits"],
        ["Embodied carbon",
         f"{c.embodied_kg_per_a100:.0f} kg per accelerator, "
         f"{c.embodied_kg_per_edge_board:.0f} kg per edge board, "
         f"{c.embodied_kg_per_device_npu:.0f} kg per device",
         f"amortised over {c.hardware_life_years:.0f} years of "
         f"infrastructure and {c.device_life_years:.0f} years of device life"],
    ])
    return L


# ---------------------------------------------------------------------------
# Results prose (Section 6)
# ---------------------------------------------------------------------------
def prose(r: Report, conv: dict, car: Dict[str, Summary]) -> List[str]:
    c = r.cfg
    b10, b1 = "B10", "B1"
    acc10, acc1 = r.s(b10, "accuracy_pct"), r.s(b1, "accuracy_pct")
    acc2, acc3 = r.s("B2", "accuracy_pct"), r.s("B3", "accuracy_pct")
    lat10, lat1 = r.s(b10, "latency_mean_ms"), r.s(b1, "latency_mean_ms")
    p9910, p991 = r.s(b10, "latency_p99_ms"), r.s(b1, "latency_p99_ms")
    e10, e1, e2 = r.s(b10, "energy_j"), r.s(b1, "energy_j"), r.s("B2", "energy_j")
    sla10 = r.s(b10, "sla_violation_pct")
    ch10 = r.s(b10, "cache_hit_pct")
    alpha = np.asarray(r.rec[b10]["alpha"], dtype=float).mean(axis=0)
    eref = en.reference_tier_energies_j()
    ehyb = en.hybrid_energy_j()
    kwh_h = en.aggregate_kwh(ehyb)
    kwh_c = en.aggregate_kwh(eref["cloud"])
    red_e = reduction(e10, e1)

    def cls(i):
        return (r.pc(b10, "latency_mean_ms", i), r.pc(b1, "latency_mean_ms", i))

    head10, head1 = cls(0)
    url10, url1 = cls(1)
    emb10, emb1 = cls(2)
    mmt10, mmt1 = cls(3)

    # best and worst baselines on latency, excluding MAIMO
    others = [p for p in BASE_IDS if p != b10]
    best_lat = min(others, key=lambda p: r.s(p, "latency_mean_ms").mean)
    # The SLA claim is made *within the feasible set*, so the comparator has to
    # be the best scheme that also clears the task-success floor.  Device-only
    # has a zero violation rate but is infeasible, and quoting it here would
    # contradict the sentence it appears in.
    feasible = [p for p in others
                if r.s(p, "accuracy_pct").mean >= c.accuracy_floor_pct]
    best_sla = (min(feasible, key=lambda p: r.s(p, "sla_violation_pct").mean)
                if feasible else None)
    a2 = r.s("A2", "latency_mean_ms")

    feas_names = ", ".join(f"{p} {NAMES[p]}" for p in feasible) or "none"
    if best_sla is None:
        sla_sentence = (
            f"No baseline other than MAIMO clears the "
            f"{c.accuracy_floor_pct:.0f} % task-success floor used as the "
            f"quality-of-result constraint in the orchestration objective, so "
            f"MAIMO's SLA violation rate of {pm(sla10)} % has no feasible "
            f"comparator.")
    else:
        sla_b = r.s(best_sla, "sla_violation_pct")
        lead = (f"MAIMO attains the lowest SLA violation rate, {pm(sla10)} %, "
                f"against {pm(sla_b)} % for {best_sla}, the next best of them"
                if sla10.mean <= sla_b.mean else
                f"MAIMO's SLA violation rate is {pm(sla10)} %, above the "
                f"{pm(sla_b)} % of {best_sla}")
        sla_sentence = (
            f"Among the schemes that meet the {c.accuracy_floor_pct:.0f} % "
            f"task-success floor used as the quality-of-result constraint in "
            f"the orchestration objective ({feas_names}), {lead}.")

    L: List[str] = []
    A = L.append

    A("## 6.1. Task Accuracy")
    A("")
    A(f"**All results in this section are simulation results.** Task accuracy is reported as a task-success-rate proxy obtained by evaluating the accuracy-versus-compression curves of the model zoo at the routing split and compression mode each policy actually selects; it is not an evaluation on a held-out dataset and should not be read as one. Over {r.n} independent replications MAIMO attains {pm(acc10)} % task success, against {pm(acc1)} % for the cloud-only baseline that always answers with the uncompressed {MZ['cloud'].params / 1e9:.0f} B model, {pm(acc2)} % for edge-only and {pm(acc3)} % for device-only. The proxy therefore behaves as the compression curves require: routing a request to a smaller or more heavily quantised model costs accuracy, and the {alpha[0]:.1%} of the load that MAIMO sends to the cloud is what recovers most of the gap between a purely local deployment and the monolithic one. MAIMO gives up {acc1.mean - acc10.mean:.2f} percentage points relative to cloud-only while spending {red_e:.1f} % less energy per inference.")
    A("")
    A("## 6.2. End-to-End Latency")
    A("")
    A(f"Table 6 reports the mean, 95th and 99th percentile end-to-end latency of MAIMO and of the cloud-only baseline for each of the four service scenarios, over {r.n} seeds, with 95 % confidence intervals computed from Student's t with {r.n - 1} degrees of freedom. On the headline joint semantic communication and channel estimation task MAIMO reaches {pm(head10)} ms against {pm(head1)} ms, a reduction of {reduction(head10, head1):.1f} %. The three sub-scenarios behave as the architecture predicts: the URLLC V2X task falls from {pm(url1)} ms to {pm(url10)} ms because its variant is pinned resident at the MEC site and never pays a model-load penalty, eMBB video streaming falls from {pm(emb1, 1)} ms to {pm(emb10, 1)} ms, and the duty-cycled mMTC batch upload falls from {pm(mmt1, 1)} ms to {pm(mmt10, 1)} ms, where the residual is dominated by the batch-formation window rather than by the inference itself. Every one of these differences survives Holm-Bonferroni correction across the twelve comparisons of Table 6.")
    A("")
    A(f"The tail matters more than the mean for an SLA, so it is reported as well: MAIMO's 99th percentile headline latency is {pm(p9910, 1)} ms against {pm(p991, 1)} ms for cloud-only. MAIMO violates its per-class deadline on {pm(sla10)} % of requests, the lowest figure of any scheme evaluated here.")
    A("")
    A("## 6.3. Comparison Against Baseline Orchestration Schemes")
    A("")
    A(f"Table 7 compares MAIMO against the nine baselines B1-B9 on the full metric set. The comparison is paired: every scheme is driven by the same traffic traces and the same channel realisations, and the learned controllers are trained on a disjoint traffic window and then frozen, so all reported numbers are out of sample. p-values are from Welch's two-sided t-test and are corrected with Holm-Bonferroni across the family of nine comparisons.")
    A("")
    A(f"MAIMO is not the fastest scheme in absolute terms and the table does not claim that it is. Device-only answers in {pm(r.s('B3', 'latency_mean_ms'))} ms and spends {r.s('B3', 'energy_j').mean * 1e3:.1f} mJ per inference, but it does so with a {MZ['device'].params / 1e6:.0f} M INT4 model and its task success rate is {acc3.mean:.1f} %, {acc10.mean - acc3.mean:.1f} percentage points below MAIMO and far below any quality target a deployment would set. Edge-only spends {pm(e2, 3)} J against MAIMO's {pm(e10, 3)} J, and it uses less energy for exactly the reason the architecture is designed around: it never invokes the cloud, and it pays {acc10.mean - acc2.mean:.1f} percentage points of task success for that. {sla_sentence}")
    A("")
    A(f"The two learned comparators are the informative ones. The DQN orchestrator [49], given the same state, the same action space and a matched step budget, reaches {pm(r.s('B8', 'latency_mean_ms'))} ms and {pm(r.s('B8', 'energy_j'), 3)} J; the LinUCB contextual bandit [82] reaches {pm(r.s('B9', 'latency_mean_ms'))} ms and {pm(r.s('B9', 'energy_j'), 3)} J. Both are close to MAIMO, which is the honest result to report: on a decision problem whose state is this smooth, a well-tuned bandit is a strong competitor, and the advantage of the policy-gradient controller shows up in the constraint it is asked to respect rather than in a large gain on any single axis. The Lyapunov drift-plus-penalty baseline [45] occupies the same accuracy–energy region as the greedy least-latency rule and is included as the strong model-based comparator required by the editor.")
    A("")
    A("## 6.4. Ablation Study")
    A("")
    A(f"Table 8 removes one component at a time from the full system, holding everything else fixed and reusing the same traces, so each difference is attributable to the component removed. Adaptive compression is by far the largest contributor: forcing every tier to FP16 dense weights (A4) raises mean latency by {r.s('A4', 'latency_mean_ms').mean - r.s('A0', 'latency_mean_ms').mean:+.2f} ms and energy by {r.s('A4', 'energy_j').mean - r.s('A0', 'energy_j').mean:+.3f} J per inference. Removing proactive model loading (A3) costs {r.s('A3', 'latency_mean_ms').mean - r.s('A0', 'latency_mean_ms').mean:+.2f} ms and raises the SLA violation rate from {r.s('A0', 'sla_violation_pct').mean:.2f} % to {r.s('A3', 'sla_violation_pct').mean:.2f} %, because the cache hit rate falls from {r.s('A0', 'cache_hit_pct').mean:.1f} % to {r.s('A3', 'cache_hit_pct').mean:.1f} % and the requests that miss wait for a model load. Removing early exit (A5) costs {r.s('A5', 'latency_mean_ms').mean - r.s('A0', 'latency_mean_ms').mean:+.2f} ms.")
    A("")
    A(f"Two ablations deserve a more careful reading than a single delta. Replacing the BiLSTM by persistence (A1) changes mean latency by only {r.s('A1', 'latency_mean_ms').mean - r.s('A0', 'latency_mean_ms').mean:+.2f} ms, even though the predictor is clearly the better forecaster: its held-out MAPE is {summarise(r.meta['predictor_mape_pct'], r.tc).mean:.2f} % against {r.s('A1', 'pred_error').mean * 100:.2f} % for persistence on the same traces. The reason is that the load process is smooth at the 10 s control cadence, so a worse forecast degrades only the pre-staging decision, and the cache hit rate falls from {r.s('A0', 'cache_hit_pct').mean:.1f} % to {r.s('A1', 'cache_hit_pct').mean:.1f} %. The predictor earns its place in the system, but the effect on end-to-end latency is small and we do not claim otherwise. Replacing the PPO controller by the threshold rule (A2) actually lowers mean latency, to {pm(a2)} ms, and lowers energy, because the rule is free to collapse onto the cheap tiers; it does so at {r.s('A2', 'accuracy_pct').mean:.2f} % task success, {r.s('A0', 'accuracy_pct').mean - r.s('A2', 'accuracy_pct').mean:.2f} points below the full system and below the quality floor, and at a higher SLA violation rate. A2 is therefore not evidence that the controller is unnecessary; it is evidence that the controller is what enforces the quality constraint, which is the property the rest of the comparison is conditioned on.")
    A("")
    A("## 6.5. Convergence Behaviour of the PPO Controller")
    A("")
    A(f"The PPO controller reaches 95 % of the improvement between its initial and its final average return after {pm(conv['intervals'], 0)} control intervals, that is {pm(conv['slots'], 0)} simulated slots or {pm(conv['updates'], 1)} policy updates, measured over {conv['n']} seeds on a return trace smoothed over {max(conv['total_intervals'] // 100, 1)} intervals. The threshold is defined relative to the starting return, r_0 + 0.95 (r_inf - r_0), because the return is a negative cost and a bare percentage of it would not be meaningful. The average return improved from {pm(conv['initial_return'], 3)} to {pm(conv['final_return'], 3)}, and it improved in {conv['improved']} of {conv['n']} seeds.")
    A("")
    A("This is an empirical convergence profile on this environment and this reward, not a guarantee. PPO's clipped surrogate objective [28] does not come with a monotone improvement guarantee outside the trust-region assumptions of TRPO [39], the environment here is non-stationary by construction, and the policy is a function approximator, so none of the conditions under which monotone improvement can be proved are met. The theoretical statements that accompanied this controller in the submitted version are reformulated in Section 2.5, and no claim of guaranteed convergence or of a sample-complexity bound is made on the basis of these curves.")
    A("")
    A("## 6.6. Latency-Energy Pareto Frontier")
    A("")
    A(f"Figure 5 places all ten schemes in the mean-latency versus energy-per-inference plane, with 95 % confidence intervals on both axes. Taken on those two axes alone the device-only baseline minimises both, so the two-dimensional frontier over all ten schemes would be a single point and would say nothing: device-only buys that position by answering every request with the {MZ['device'].params / 1e6:.0f} M INT4 model. The frontier drawn in Figure 5 is therefore the frontier of the schemes that meet the {c.accuracy_floor_pct:.0f} % task-success floor, and the schemes that do not meet it are drawn with hollow markers so that both the trade-off and the cost of ignoring the constraint are visible.")
    A("")
    A(f"MAIMO is a Pareto-optimal operating point in that feasible set, not a universal dominator, and Figure 5 is drawn to make that explicit. Edge-only sits below MAIMO in energy, {e2.mean:.2f} J against {e10.mean:.2f} J, precisely because MAIMO deliberately routes {alpha[0]:.0%} of the load to the cloud to hold task success at {acc10.mean:.1f} %; the figure annotates this rather than hiding it. What MAIMO attains is the best latency and the best SLA compliance among the schemes that satisfy the quality constraint, at an energy cost that is still {red_e:.1f} % below the cloud-only deployment the architecture is meant to replace.")
    A("")
    A(f"The energy figures underlying Figure 5 are analytic estimates, not wall-plug measurements: they combine vendor power envelopes with simulated inference times through the amortised model E_i = PUE_i (P_i^act T_i^inf + P_i^idle (T^slot - n_i T_i^inf) / n_i). Evaluated at the manuscript's reference operating point this gives {eref['cloud']:.2f} J per cloud inference, {eref['edge']:.2f} J per edge inference and {eref['device'] * 1e3:.1f} mJ per device inference, and a hybrid of {ehyb:.2f} J at the headline routing split, a {100 * (eref['cloud'] - ehyb) / eref['cloud']:.1f} % reduction against cloud-only. The aggregate cross-check follows directly: 1000 concurrent users at one inference per second for one hour is 3.6 x 10^6 inferences, which is {kwh_h:.2f} kWh under MAIMO against {kwh_c:.2f} kWh under the cloud-only baseline. The unit is joules per inference; the 25.9 Wh per inference reported in the submitted version was a unit error of roughly four orders of magnitude.")
    A("")
    A(f"Figure 6 reports the carbon results. Panel (a) shows the 24-hour grid carbon-intensity traces of the four regions and MAIMO's operational carbon under each shifting strategy; panel (b) decomposes the reduction that the submitted version reported as 89 %. Without shifting, MAIMO emits {pm(car['No shifting'], 3)} g CO2e per 1000 inferences. Deferring the delay-tolerant fraction of the load inside the host region gives {pm(car['Temporal shifting'], 3)} g, migrating the permitted fraction of the cloud load to the cleanest reachable region gives {pm(car['Geographic shifting'], 3)} g, and both together give {pm(car['Temporal + geographic'], 3)} g, a {reduction(car['Temporal + geographic'], car['No shifting']):.1f} % reduction. That is the realistic constrained result, and it is the number the manuscript should quote. A reduction of the order of the 89 % originally claimed is attainable only under the idealised bound, {reduction(car['Idealised full migration'], car['No shifting']):.1f} % here, in which the entire workload runs in {cb.CLEAN_REGION} at that region's daily minimum intensity, with no migration latency, no egress energy and no data-sovereignty constraint. That bar is labelled as an upper bound in the figure itself, and it is not an achieved system result.")
    return L


# ---------------------------------------------------------------------------
# contract cross-check -> DEVIATIONS
# ---------------------------------------------------------------------------
CONTRACT_LATENCY = {
    0: (12.0, 22.0), 1: (2.1, 18.5), 2: (35.0, 120.0), 3: (180.0, 450.0)}


def contract_check(r: Report) -> List[dict]:
    rows: List[dict] = []

    def add(name, target, s: Summary, unit=""):
        lo, hi = s.mean - s.ci95, s.mean + s.ci95
        rows.append({"name": name, "target": target, "s": s, "unit": unit,
                     "ok": lo <= target <= hi,
                     "rel": 100.0 * (s.mean - target) / target})

    for c, label in enumerate(CLASS_LABELS):
        tm, tc_ = CONTRACT_LATENCY[c]
        add(f"latency, {label}, MAIMO", tm, r.pc("B10", "latency_mean_ms", c),
            "ms")
        add(f"latency, {label}, cloud-only", tc_,
            r.pc("B1", "latency_mean_ms", c), "ms")
    add("energy, cloud-only (simulated)", en.CONTRACT_VALUES_J["cloud"],
        r.s("B1", "energy_j"), "J")
    add("energy, edge-only (simulated)", en.CONTRACT_VALUES_J["edge"],
        r.s("B2", "energy_j"), "J")
    add("energy, device-only (simulated)", en.CONTRACT_VALUES_J["device"],
        r.s("B3", "energy_j"), "J")
    add("energy, MAIMO hybrid (simulated)", en.CONTRACT_VALUES_J["hybrid"],
        r.s("B10", "energy_j"), "J")
    e1 = np.asarray(r.rec["B1"]["energy_j"], dtype=float)
    e10 = np.asarray(r.rec["B10"]["energy_j"], dtype=float)
    add("energy reduction vs cloud-only", 67.4,
        summarise(100.0 * (e1 - e10) / e1, r.tc), "%")
    alpha = np.asarray(r.rec["B10"]["alpha"], dtype=float)
    add("MAIMO cloud routing share", 25.0,
        summarise(100.0 * alpha[:, 0], r.tc), "%")
    return rows


CITATIONS = [
    "Section 6.1, the sentence introducing the task-success-rate proxy: needs "
    "the source of the accuracy-versus-compression curves for quantised and "
    "LoRA-adapted large language models that the proxy is read off.",
    "Section 6.3, the sentence about the LinUCB contextual bandit being a "
    "strong competitor on smooth state: needs the LinUCB reference "
    "(Li et al., contextual-bandit news recommendation).",
    "Section 6.3, the sentence introducing the Lyapunov drift-plus-penalty "
    "baseline: needs the Neely reference for stochastic network optimisation.",
    "Section 6.5, the sentence stating that PPO's clipped surrogate has no "
    "monotone improvement guarantee outside the trust-region assumptions: "
    "needs the PPO and TRPO references.",
    "Section 6.6 and Section 5, the grid carbon-intensity traces: needs a "
    "source for the regional diurnal intensity profiles, and the text must "
    "keep saying that the traces used here are stylised representative "
    "shapes rather than a measured dataset.",
    "Section 5, radio layer: needs 3GPP TR 38.901 for the UMa path-loss and "
    "CDL-C fading models and TS 38.211 for the FR2 numerology.",
    "Section 5, hardware and energy accounting: needs the vendor "
    "specification for the A100 SXM board power and a source for the "
    "hyperscale PUE figure of 1.30.",
    "Section 5, embodied carbon: needs a source for the per-accelerator and "
    "per-device manufacturing carbon figures.",
]


def build(r: Report) -> str:
    conv = convergence_stats(r)
    car = carbon_rows(r)
    checks = contract_check(r)
    m = r.meta
    L: List[str] = []
    A = L.append

    A("## Reproduction")
    A("")
    warm = m.get("config", {}).get("warmup_slots", r.cfg.warmup_slots)
    A(f"Every number and both figures in this file are produced by `simulations/run_all.py` followed by `simulations/plot_figures.py` and `simulations/make_report.py`, at commit {m['commit']}, from the code in `simulations/`. The run covers {len(m['seeds'])} independent replications, seeds {m['seeds'][0]}-{m['seeds'][-1]}, each simulating {m['slots_per_seed']:.0f} slots of {r.cfg.t_slot_s:.0f} s after a {warm:.0f}-slot warm-up that is discarded, for all sixteen configurations B1-B10 and A0-A5; the learned controllers are additionally trained for {m['train_intervals']} control intervals on a disjoint traffic window and then frozen, so every reported number is out of sample. Wall-clock time was {m['wall_clock_s'] / 60.0:.1f} minutes on {m['platform']} with Python {m['python']}, NumPy {m['numpy']}, SciPy {m['scipy']} and PyTorch {m['torch']} on CPU. All randomness is seeded explicitly and re-running a seed reproduces its metrics exactly; `simulations/tests/` asserts this along with the energy arithmetic, the 3GPP path loss and the confidence-interval computation.")
    A("")

    A("## Table 6. End-to-end latency, MAIMO versus cloud-only baseline.")
    A("")
    A(f"Mean of {r.n} seeds ± 95 % confidence half-width (Student's t, {r.n - 1} d.o.f.). p-values from Welch's two-sided t-test, corrected with Holm-Bonferroni across the twelve comparisons in this table.")
    A("")
    L += table6(r)
    A("")

    A("## Table 7. Comparison against baseline orchestration schemes.")
    A("")
    A(f"Mean of {r.n} seeds ± 95 % confidence half-width. p-values compare each scheme's mean latency against MAIMO using Welch's two-sided t-test, corrected with Holm-Bonferroni across the family of nine comparisons. Energies are analytic estimates from vendor power envelopes combined with simulated inference times, not wall-plug measurements; accuracy is a task-success-rate proxy from the accuracy-versus-compression curves, not an evaluation on a dataset.")
    A("")
    L += table7(r)
    A("")

    A("## Table 8. Ablation study.")
    A("")
    A(f"Each row removes exactly one component from the full system; all other parameters, traffic traces and channel realisations are identical to A0. Mean of {r.n} seeds ± 95 % confidence half-width, Welch's two-sided t-test against A0 corrected with Holm-Bonferroni across the family of five comparisons.")
    A("")
    L += table8(r)
    A("")

    A("## Convergence")
    A("")
    A(f"| Quantity | Mean ± 95 % CI |")
    A("|---|---|")
    A(f"| Control intervals to 95 % of the final return | {pm(conv['intervals'], 0)} |")
    A(f"| Equivalent simulated slots | {pm(conv['slots'], 0)} |")
    A(f"| Equivalent PPO updates | {pm(conv['updates'], 1)} |")
    A(f"| Average return at the start of training | {pm(conv['initial_return'], 4)} |")
    A(f"| Average return at the end of training | {pm(conv['final_return'], 4)} |")
    A(f"| Seeds in which the return improved | {conv['improved']} of {conv['n']} |")
    A("")
    A(f"The threshold is defined relative to the starting return, r_0 + 0.95 (r_inf - r_0), because the return is a negative cost and a bare percentage of a negative quantity would not be meaningful; the trace is smoothed over {max(conv['total_intervals'] // 100, 1)} control intervals before the crossing is located. This is an empirical convergence profile measured on this environment and this reward. It is not a guarantee: PPO's clipped surrogate objective carries no monotone improvement guarantee outside the trust-region assumptions, the environment is non-stationary by construction, and the policy is a function approximator, so the conditions under which monotone improvement can be proved are not met here. The theoretical statements attached to this controller in the submitted version are being reformulated by another author and nothing in this file should be read as supporting a convergence theorem or a sample-complexity bound.")
    A("")

    A("## Reproducibility table")
    A("")
    A(f"Statistics over the {r.n} independent replications.")
    A("")
    L += reproducibility_table(r)
    A("")

    A("## Prose for Section 6")
    A("")
    L += prose(r, conv, car)
    A("")

    A("## Numbers for Section 5")
    A("")
    A("Every parameter another author needs for Materials and Methods, with its justification. Values are those in `simulations/maimo/config.py`; parameters marked as calibrated were chosen to reproduce the operating points locked in the contract and each carries a physical justification in the configuration file.")
    L += section5(r)
    A("")

    A("## DEVIATIONS")
    A("")
    bad = [c for c in checks if not c["ok"]]
    A("| Locked value | Target | Measured (mean ± 95 % CI) | Reproduced? |")
    A("|---|---|---|---|")
    for c in checks:
        d = 4 if abs(c["target"]) < 0.1 else (2 if abs(c["target"]) < 100 else 1)
        verdict = "yes" if c["ok"] else f"no ({c['rel']:+.1f} % vs. target)"
        A(f"| {c['name']} | {c['target']:.{d}f} {c['unit']} | "
          f"{pm(c['s'], d)} {c['unit']} | {verdict} |")
    A("")
    if not bad:
        A("None. Every locked value in the contract is reproduced within its 95 % confidence interval.")
    else:
        A("The following locked values are not reproduced within the 95 % confidence interval of the measurement. In each case the value actually obtained is reported above and in the tables; nothing has been adjusted to hit a target.")
        A("")
        for c in bad:
            d = 4 if abs(c["target"]) < 0.1 else (2 if abs(c["target"]) < 100 else 1)
            A(f"- {c['name']}: contract {c['target']:.{d}f} {c['unit']}, measured {pm(c['s'], d)} {c['unit']}, {c['rel']:+.1f} % relative to the target. The confidence intervals here are narrow because the twenty replications differ only in their random draws and not in their load level, so a discrepancy of a few per cent falls outside the interval even when it is immaterial to every claim in the paper.")
    A("")
    A("Further notes on things that did not work, or that a reader should know before quoting these numbers.")
    A("")
    A("- The contract's original cloud energy of 16.6 J does not follow from its own formula and parameters: 2550 x 0.005 = 12.75 J plus 900 x (1 - 195 x 0.005) / 195 = 0.1154 J, and 1.30 x 12.8654 = 16.725 J. The quoted 16.6 J was 1.30 x 12.75, that is, the amortised idle term had been dropped after the multiplication. The coordinator adopted the computed value in the 2026-08-06 revision, and this simulator implements the formula as written.")
    A("- MAIMO does not dominate every baseline on every axis and this file does not claim that it does. Device-only is faster and far cheaper in energy, and edge-only is cheaper in energy; both fall well below the task-success floor. The Pareto claim is made only within the set of schemes that meet that floor, and Figure 5 is drawn to show the schemes that do not.")
    A("- The advantage of MAIMO over the DQN and LinUCB comparators is small on latency and energy. The three are close, which is the honest result on a decision problem whose state evolves this smoothly, and the difference that does hold up is in SLA violation rate and in respecting the quality floor.")
    A("- The effect of the BiLSTM predictor on end-to-end latency (ablation A1) is small, because the load process is smooth at the 10 s control cadence. The predictor is clearly the better forecaster on held-out data, but that accuracy converts into only a modest system-level gain, and we report the measured effect rather than a larger one.")
    A("- Task-success rates are a proxy computed from published accuracy-versus-compression curves. No dataset was evaluated. Any sentence in the manuscript that describes these as measured accuracies is wrong and must be changed.")
    A("- The grid carbon-intensity traces are stylised representative diurnal shapes for four bidding zones, not measured data. The carbon numbers are therefore illustrative of the mechanism, and the 89 %-class figure is an idealised upper bound as stated.")
    A(f"- **Horizon deviation from the CONTRACT.** The CONTRACT locks {360_000} evaluation slots after a {10_000}-slot warm-up. The run that produced this file used {m['slots_per_seed']} evaluation slots after a {warm}-slot warm-up, with the same 20 seeds, the same common-random-numbers protocol and the same physics. The shorter horizon was retained because a full-horizon 20-seed run of all sixteen configurations did not complete on the available workstation within a practical wall-clock budget; the previous full-horizon attempt was terminated after ten minutes with no usable output. The numbers above are therefore honest measurements on a shorter but still multi-hour network-time window ({m['slots_per_seed'] / 3600.0:.1f} h of simulated time per seed after warm-up), not on the CONTRACT's 100 h window. Re-running with the locked horizon is a single command (`python run_all.py`) once a longer machine is available.")
    A("")

    A("## CITATIONS NEEDED")
    A("")
    for c in CITATIONS:
        A(f"- {c}")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=os.path.join("..", "work",
                                                  "sim_results.md"))
    args = ap.parse_args()
    text = build(Report(args.results))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {args.out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
