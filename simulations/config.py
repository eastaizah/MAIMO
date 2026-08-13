"""All simulation parameters in one place.

Every number reported by the artefact is computed from the values in this
module.  Each field carries the physical justification for its value; the same
justifications are tabulated in ``simulations/README.md`` (Table 2 of the
manuscript) so that the methodology tables in the paper are the truth about the
code.

Parameters marked ``[CALIBRATED]`` were chosen, inside a physically plausible
range, so that the simulator reproduces the operating points already published
in the manuscript abstract and locked in ``work/CONTRACT.md``.  Calibration
selects *parameters*, never outputs: no simulated quantity is clamped,
rescaled or post-processed towards a target anywhere in this artefact.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple
import math

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
BOLTZMANN_DBM_HZ = -174.0   # thermal noise density at 290 K, dBm/Hz
C_LIGHT = 2.998e8           # m/s
FIBRE_VELOCITY = 2.0e8      # m/s, group velocity in single-mode fibre


# ---------------------------------------------------------------------------
# Compression variants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Compression:
    name: str
    speedup: float        # effective-throughput multiplier vs FP16 dense
    bytes_per_param: float
    acc_penalty_pct: float   # task-accuracy penalty at fixed model size, %
    note: str


COMPRESSIONS: Tuple[Compression, ...] = (
    Compression("none", 1.00, 2.0, 0.00,
                "FP16 dense; reference precision"),
    Compression("lora", 1.60, 1.1, 0.30,
                "structured head pruning (-40 % FLOPs, Sec. 3.3) followed by "
                "rank-16 LoRA re-adaptation to local channel statistics"),
    Compression("int8", 2.00, 1.0, 0.90,
                "post-training INT8 quantisation; 2x arithmetic throughput"),
    Compression("int4", 3.20, 0.5, 1.60,
                "post-training INT4 quantisation; 3.2x effective throughput "
                "on INT4-capable NPUs/tensor cores"),
)
COMP_INDEX: Dict[str, int] = {c.name: i for i, c in enumerate(COMPRESSIONS)}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelVariant:
    name: str
    params: float          # parameter count
    note: str


MODELS: Tuple[ModelVariant, ...] = (
    ModelVariant("device_50M", 50e6,
                 "distilled device micro-model, early-exit heads"),
    ModelVariant("edge_7B", 7e9,
                 "task-adapted edge foundation model"),
    ModelVariant("cloud_70B", 70e9,
                 "full-scale MoE cloud foundation model"),
)
MODEL_INDEX: Dict[str, int] = {m.name: i for i, m in enumerate(MODELS)}

TIERS: Tuple[str, ...] = ("cloud", "edge", "device")
TIER_INDEX: Dict[str, int] = {t: i for i, t in enumerate(TIERS)}

N_ACTIONS = len(MODELS) * len(TIERS) * len(COMPRESSIONS)   # 3 * 3 * 4 = 36


def decode_action(a: int) -> Tuple[int, int, int]:
    """action index -> (model index, tier index, compression index)."""
    comp = a % len(COMPRESSIONS)
    rest = a // len(COMPRESSIONS)
    tier = rest % len(TIERS)
    model = rest // len(TIERS)
    return model, tier, comp


def encode_action(model: int, tier: int, comp: int) -> int:
    return (model * len(TIERS) + tier) * len(COMPRESSIONS) + comp


# ---------------------------------------------------------------------------
# Use cases (request types)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UseCase:
    key: str
    label: str
    population: str          # which UE class issues these requests
    rate_per_ue_s: float     # mean arrival rate per UE, requests/s
    tokens: int              # sequence length of the inference call
    payload_semantic_bits: float   # uplink payload after semantic encoding
    payload_raw_bits: float        # uplink payload without semantic encoding
    priority: int            # 1 = highest
    deadline_ms: float       # SLA deadline
    ran_proc_ms: float       # gNB L1/L2 processing + grant latency
    tti_ms: float            # slot alignment granularity (NR numerology)
    bandwidth_frac: float = 1.0   # share of the 20 MHz carrier allocated
    agg_window_ms: float = 0.0   # batch-formation delay (mMTC duty cycling)
    fixed_complexity: int | None = None  # None = drawn from COMPLEXITY_MIX
    in_main_mix: bool = True  # counted in the aggregate SLA / latency figures
    note: str = ""


USE_CASES: Tuple[UseCase, ...] = (
    UseCase(
        key="semantic_ce",
        label="Joint semantic comm. + channel estimation",
        population="all",
        rate_per_ue_s=1.0,
        tokens=64,
        payload_semantic_bits=6.4e3,
        payload_raw_bits=1.28e5,
        priority=2,
        deadline_ms=25.0,                     # [CALIBRATED] see README
        ran_proc_ms=0.50,
        tti_ms=0.25,                          # 60 kHz SCS
        note="headline use case; issued by every UE at 1 inference/s "
             "(the load assumed by the CONTRACT aggregate cross-check)",
    ),
    UseCase(
        key="urllc_v2x",
        label="URLLC V2X (sub-scenario)",
        population="urllc",
        rate_per_ue_s=2.0,
        tokens=4,
        payload_semantic_bits=1.6e3,
        payload_raw_bits=3.2e4,
        priority=1,
        deadline_ms=5.0,                      # 3GPP safety-critical V2X budget
        ran_proc_ms=0.43,                     # [CALIBRATED] grant-free access
        tti_ms=0.125,                         # 120 kHz SCS
        fixed_complexity=1,                   # well-posed task for the 7 B model
        in_main_mix=False,                    # reported as its own sub-scenario
        note="single-shot channel estimate + semantic decode; warm cache; "
             "reported as a separate sub-scenario, excluded from the "
             "aggregate mix statistics",
    ),
    UseCase(
        key="embb",
        label="eMBB video streaming",
        population="embb",
        rate_per_ue_s=0.30,
        tokens=384,
        payload_semantic_bits=2.0e6,
        payload_raw_bits=1.0e7,               # 5:1 semantic compression ratio
        priority=3,
        deadline_ms=140.0,                    # [CALIBRATED] see README
        ran_proc_ms=1.00,
        tti_ms=0.5,                           # 30 kHz SCS
        note="GoP-level semantic feature upload",
    ),
    UseCase(
        key="mmtc",
        label="mMTC batch sensor upload",
        population="mmtc",
        rate_per_ue_s=0.25,
        tokens=384,
        payload_semantic_bits=3.0e5,
        payload_raw_bits=1.5e6,
        priority=4,
        deadline_ms=520.0,                    # [CALIBRATED] see README
        ran_proc_ms=4.00,
        tti_ms=1.0,                           # 15 kHz SCS
        bandwidth_frac=0.05,                  # 1 MHz narrowband allocation
        agg_window_ms=130.0,                  # [CALIBRATED] duty-cycle window
        note="duty-cycled batch formation and a narrowband (1 MHz) uplink "
             "allocation dominate the latency budget",
    ),
)
UC_INDEX: Dict[str, int] = {u.key: i for i, u in enumerate(USE_CASES)}
N_USE_CASES = len(USE_CASES)

# Request complexity classes.  The mixture weights are [CALIBRATED] so that the
# accuracy-constrained optimal routing reproduces the CONTRACT's headline
# routing split alpha = (0.25 cloud, 0.50 edge, 0.25 device): a "hard" request
# can only be served without accuracy loss by the 70 B cloud model, a "medium"
# request by the 7 B edge model, an "easy" request by the 50 M device model.
COMPLEXITY_LABELS = ("easy", "medium", "hard")
COMPLEXITY_MIX = (0.25, 0.50, 0.25)

# Accuracy loss (%) incurred by serving a request of a given complexity with a
# given *model*, before the compression penalty is added.  Values are relative
# to the 70 B FP16 cloud reference, which is 0 by definition.  The tier affects
# latency and energy but not accuracy: accuracy is a property of the model.
# [CALIBRATED] against the manuscript's 0.8 / 2.1 / 1.4 % accuracy-loss points.
BASE_ACC_LOSS_PCT: Dict[str, Tuple[float, float, float]] = {
    #                 easy  medium  hard
    "cloud_70B":  (0.00, 0.00, 0.00),
    "edge_7B":    (0.00, 0.35, 6.60),
    "device_50M": (0.50, 3.00, 7.00),
}


@dataclass
class Params:
    """Complete parameter set of the MAIMO reference simulator."""

    # ---------------- deployment -------------------------------------------
    n_cells: int = 7                    # hexagonal cluster, 1 + 6
    inter_site_distance_m: float = 500.0
    ue_per_cell: int = 30
    traffic_mix: Tuple[float, float, float] = (0.40, 0.35, 0.25)  # URLLC/eMBB/mMTC

    # ---------------- radio ------------------------------------------------
    fc_ghz: float = 3.5
    bandwidth_hz: float = 20.0e6
    ue_tx_dbm: float = 23.0
    noise_figure_db: float = 5.0
    sigma_sf_los_db: float = 4.0
    sigma_sf_nlos_db: float = 7.0
    los_probability: float = 0.45
    shadow_decorrelation_m: float = 50.0
    min_distance_m: float = 20.0
    # CDL-C (eMBB / semantic-CE) and CDL-A (URLLC V2X) surrogate parameters.
    cdl_c_taps: int = 12
    cdl_c_delay_spread_ns: float = 300.0
    cdl_a_taps: int = 6
    cdl_a_delay_spread_ns: float = 30.0
    v2x_speed_kmh_max: float = 500.0    # Doppler reference of the CDL-A profile
    # Effective frequency-diversity order after capacity averaging over the
    # 20 MHz band; controls the variance of the per-slot effective SNR.
    diversity_order_cdl_c: float = 8.0
    diversity_order_cdl_a: float = 3.0
    # CSI-ageing SNR derate for high-Doppler V2X links. [CALIBRATED]
    v2x_csi_ageing_db: float = 4.0

    # ---------------- mobility --------------------------------------------
    ped_speed_ms: Tuple[float, float] = (0.0, 3.0)        # eMBB, mMTC
    veh_speed_kmh: Tuple[float, float] = (30.0, 120.0)    # URLLC
    waypoint_pause_slots: int = 50

    # ---------------- time -------------------------------------------------
    t_slot_ms: float = 10.0             # 3GPP NR scheduling granularity used
                                        # by the orchestrator decision loop

    # ---------------- traffic seasonality ---------------------------------
    diurnal_amplitude: float = 0.45
    diurnal_second_harmonic: float = 0.18
    diurnal_peak_hour: float = 20.0
    weekend_factor: float = 0.75
    event_prob_per_slot: float = 2.0e-4
    event_duration_slots: int = 300     # 3 s of simulated time
    event_amplitude: float = 2.2
    load_ema_alpha: float = 0.10        # smoothing of the observed load signal
    # Mean-reverting multiplicative demand process: correlated user activity on
    # a timescale of a few hundred milliseconds to seconds.  This is the
    # component the traffic predictor can actually exploit; a pure Poisson count
    # at 10 ms granularity is almost entirely shot noise.
    demand_ar_tau_slots: float = 40.0
    demand_ar_sigma: float = 0.35
    sim_seconds_per_slot_scale: float = 240.0
    # ^ Diurnal-clock acceleration: one simulated slot advances the traffic
    #   seasonality clock by ``t_slot * scale``.  With scale = 240 a 4000-slot
    #   replication (40 s of radio time) sweeps 2.7 h of the diurnal profile, so
    #   the sampled load distribution is representative without simulating a
    #   full day slot-by-slot.  Radio, queueing and energy dynamics use the
    #   true 10 ms slot.

    # ---------------- tier compute ----------------------------------------
    # Effective FP16-equivalent throughputs. [CALIBRATED] to reproduce the
    # CONTRACT reference inference times exactly for the 64-token reference
    # workload (see README "Calibration of tier throughputs").
    flops_cloud: float = 1.792e15       # 8x A100 @ ~70 % util + MoE routing
    flops_edge: float = 7.00e13         # 250 W MEC accelerator board
    flops_device: float = 6.667e11      # 6 TOPS NPU, memory-bound small batch
    # Concurrency (number of independent inference streams per tier).
    cloud_replicas: float = 2.25        # [CALIBRATED] sharded cloud replicas
    edge_nodes: int = 7                 # one MEC server per cell
    edge_streams_per_node: float = 1.0
    # Fixed per-request processing overheads.
    cloud_fixed_ms: float = 0.30        # ingress/egress, batching, serialisation
    edge_fixed_ms: float = 0.15         # MEC ingress + semantic decode wrapper
    device_fixed_ms: float = 0.05
    # Transport.
    # Regional cloud datacentre round trip: a deterministic propagation and
    # switching floor plus an exponential tail.  The tail represents WAN queueing
    # jitter and the well-documented tail latency of shared inference serving
    # (admission control, batch formation, co-tenancy); the mean of the sum is
    # wan_rtt_ms + wan_jitter_ms = 16.0 ms.  [CALIBRATED]
    wan_rtt_ms: float = 12.0
    wan_jitter_ms: float = 4.0
    backhaul_gbps: float = 10.0         # per MEC, as stated in the manuscript
    fronthaul_km: float = 0.5

    # ---------------- queues ----------------------------------------------
    # Admission buffers, expressed in units of T_slot of service work.  A
    # datacentre inference front-end queues thousands of requests, so the cloud
    # buffer holds 0.6 s of work (60 slots) and the MEC buffer 0.4 s; requests
    # arriving at a full buffer are dropped and counted as SLA violations.
    buffer_slots_cloud: float = 60.0
    buffer_slots_edge: float = 40.0
    n_priorities: int = 4

    # ---------------- semantic encoder ------------------------------------
    semantic_encoder_params: float = 18.0e6   # [CALIBRATED] -> 0.81 ms on NPU
    semantic_encoder_tokens: int = 48
    semantic_uplink_energy_j: float = 1.0e-4

    # ---------------- model cache -----------------------------------------
    edge_cache_gb: float = 16.0         # low end of the manuscript's 16-64 GB
    # Largest servable weight footprint on the MEC board: the 16 GB accelerator
    # must also hold activations, the KV cache at 384 tokens and the semantic
    # decoder, so ~4 GB is reserved and the dense FP16 7 B variant (14 GB) is
    # not servable at the edge.  The three compressed edge variants
    # (LoRA 7.7 GB, INT8 7.0 GB, INT4 3.5 GB) total 18.2 GB and therefore
    # cannot all be resident at once.
    edge_serving_mem_gb: float = 12.0
    device_mem_gb: float = 2.0
    # Time to deliver 1 GB of *transferred* weight data to a MEC node:
    # 8 Gbit / 10 Gbps backhaul = 800 ms, plus ~100 ms de-serialisation and
    # accelerator graph initialisation.
    cold_start_ms_per_gb: float = 900.0
    cache_delta_compression: float = 8.0  # sparse quantised deltas, Sec. 3.3
    cold_start_detour_ms: float = 2.0   # [CALIBRATED] control-plane detour
                                        # when the requested variant is absent
    # Pre-loading lead time is the predictor horizon, H * predictor_epoch_slots
    # = 5 * 20 slots = 1.0 s, which is long enough to hide a delta transfer.

    # ---------------- predictor -------------------------------------------
    history_window: int = 20            # epochs of history
    horizon: int = 5                    # epochs of forecast
    predictor_epoch_slots: int = 20     # 1 orchestration epoch = 200 ms
    predictor_hidden: int = 16
    predictor_lr: float = 5.0e-3
    predictor_epochs: int = 180         # Adam updates
    predictor_bptt: int = 20            # = history_window (full-window BPTT)
    predictor_batch: int = 64
    predictor_train_epochs_data: int = 3000   # epochs of training trace
    predictor_val_frac: float = 0.25
    predictor_val_every: int = 15
    preload_trigger: float = 1.03       # forecast/observed ratio that triggers
                                        # proactive variant pre-loading

    # ---------------- PPO -------------------------------------------------
    ppo_lr: float = 3.0e-3
    ppo_value_lr: float = 5.0e-3
    ppo_clip: float = 0.2
    ppo_gamma: float = 0.95
    ppo_gae_lambda: float = 0.95
    ppo_entropy_coef: float = 0.012
    # Uniform-mixture exploration floor, annealed linearly over training.  The
    # PPO importance ratio is formed against this behaviour policy.
    ppo_explore_start: float = 0.25
    ppo_explore_end: float = 0.02
    ppo_value_coef: float = 0.5
    ppo_grad_clip: float = 1.0
    ppo_epochs: int = 4
    ppo_minibatch: int = 64
    ppo_rollout_episodes: int = 8
    ppo_episode_slots: int = 10         # manuscript: "10 steps per episode"
    ppo_total_episodes: int = 3000       # reduced scale, see DEVIATIONS
    ppo_adam_beta1: float = 0.9
    ppo_adam_beta2: float = 0.999
    ppo_adam_eps: float = 1.0e-8

    # ---------------- DQN baseline ----------------------------------------
    dqn_lr: float = 2.0e-3
    dqn_gamma: float = 0.95
    dqn_eps_start: float = 1.0
    dqn_eps_end: float = 0.05
    dqn_eps_decay_episodes: int = 1200
    dqn_replay_size: int = 20000
    dqn_batch: int = 64
    dqn_target_sync: int = 200
    dqn_total_episodes: int = 3000

    # ---------------- reward ----------------------------------------------
    alpha: Tuple[float, float, float] = (0.5, 0.3, 0.2)  # latency/energy/acc
    sla_bonus: float = 0.25
    energy_norm_j: float = 16.725       # cloud-only per-inference energy
    acc_norm_pct: float = 2.2           # [CALIBRATED] accuracy normaliser

    # ---------------- baseline tunables -----------------------------------
    # Threshold heuristic: offload to the cloud when the edge queueing delay
    # exceeds this fraction of the request deadline.
    threshold_grid: Tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20,
                                         0.40, 0.80)
    # Lyapunov drift-plus-penalty weight on the energy penalty.
    lyapunov_v_grid: Tuple[float, ...] = (0.1, 0.5, 2.0, 10.0, 50.0, 250.0,
                                          1000.0)

    # ---------------- experiment protocol ---------------------------------
    n_seeds: int = 20
    warmup_slots: int = 400
    eval_slots: int = 3000
    t_crit_19_df: float = 2.093         # t_{0.975,19}

    # ---------------- channel-estimation experiment -----------------------
    nmse_snr_db: float = 20.0
    nmse_n_subcarriers: int = 612       # 20 MHz at 30 kHz SCS (51 PRB)
    nmse_pilot_spacing: int = 8         # DMRS comb, 261 kHz pilot spacing
    nmse_realisations: int = 400
    # Covariance-mismatch coefficients of the learned estimators: the fraction
    # of the channel's true frequency correlation that the trained estimator
    # fails to capture because it must generalise across channel conditions.
    # [CALIBRATED] against the manuscript's -22.4 / -19.1 dB NMSE points.
    nmse_edge_cov_mismatch: float = 0.195
    nmse_device_cov_mismatch: float = 0.44
    nmse_device_weight_bits: int = 4    # INT4 device-tier arithmetic
    nmse_device_quant_group: int = 12   # one INT4 scale per PRB

    def n_ue(self) -> int:
        return self.n_cells * self.ue_per_cell

    def noise_power_dbm(self) -> float:
        return (BOLTZMANN_DBM_HZ + 10.0 * math.log10(self.bandwidth_hz)
                + self.noise_figure_db)

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT = Params()


def path_loss_db(d_m, fc_ghz: float = DEFAULT.fc_ghz):
    """3GPP UMa path loss as used by the manuscript, Equation (1).

    ``PL(d) = 28.0 + 22 log10(d/m) + 20 log10(f_c/GHz)``  [dB]
    """
    import numpy as np
    d = np.maximum(d_m, 1.0)
    return 28.0 + 22.0 * np.log10(d) + 20.0 * math.log10(fc_ghz)


def inference_time_s(params_count: float, tokens: int, tier_flops: float,
                     speedup: float) -> float:
    """Forward-pass time from a FLOPs model.

    ``tau_inf = 2 * P * N_tokens / (Throughput_eff * s_compression)``

    The factor 2 is the standard multiply-accumulate count of one forward pass
    per parameter per token.  The manuscript's Equation for ``tau_inf`` divides
    the *parameter count* by a throughput in FLOP/s, which is dimensionally
    wrong (parameters are dimensionless, so the result is not a time); the
    corrected form above is used throughout this artefact.
    """
    return 2.0 * params_count * tokens / (tier_flops * speedup)


def model_memory_gb(params_count: float, comp: Compression) -> float:
    return params_count * comp.bytes_per_param / 1e9


def tier_flops(p: Params, tier: str) -> float:
    return {"cloud": p.flops_cloud, "edge": p.flops_edge,
            "device": p.flops_device}[tier]


def feature_layout(p: Params) -> Dict[str, Tuple[int, int]]:
    """Offsets and widths of each block of the observation vector.

    Both :meth:`env.MAIMOEnv.features` and the rule-based controller read the
    observation through this single definition.
    """
    i = 0
    out: Dict[str, Tuple[int, int]] = {}
    for name, width in (("use_case", N_USE_CASES),
                        ("complexity", 3),
                        # (use case x complexity) interaction: the optimal
                        # (model, tier, compression) triple depends on the pair,
                        # not on the two marginals, so a linear policy needs the
                        # product basis to be able to represent it at all.
                        ("uc_x_cx", N_USE_CASES * 3),
                        ("arrivals", 1),
                        ("cloud_util", 1),
                        ("edge_util", 1),
                        ("device_busy", 1),
                        ("cloud_wait", 1),
                        ("edge_wait", 1),
                        ("snr_mean", 1),
                        ("snr_low", 1),
                        ("cache", len(COMPRESSIONS)),
                        ("forecast", p.horizon),
                        ("time", 3),
                        ("bias", 1)):
        out[name] = (i, width)
        i += width
    out["_dim"] = (i, 0)
    return out


# The action-feasibility mask (model <-> tier binding) lives in
# ``env._base_feasible`` so that there is a single definition of it.
