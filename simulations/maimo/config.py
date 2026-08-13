"""Every parameter of the MAIMO reference simulator, in one place.

No numeric constant that influences a reported result may appear anywhere else
in the package: modules import :data:`DEFAULT` (or receive a :class:`Config`)
and read fields from it.  Each field carries a short justification and a
source; the same justifications are tabulated in ``simulations/README.md`` and
in the ``## Numbers for Section 5`` block of ``work/sim_results.md`` so that
the Materials-and-Methods tables of the manuscript are literally the truth
about this code.

Fields tagged ``[CAL]`` were *calibrated*: their value was chosen inside a
physically defensible range so that the simulator reproduces the operating
points locked in ``work/CONTRACT.md``.  Calibration selects **parameters**,
never outputs.  No simulated quantity is clamped, rescaled or post-processed
towards a target anywhere in this artefact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict, replace
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
THERMAL_NOISE_DBM_HZ = -174.0     # kTB at 290 K, dBm/Hz
C_LIGHT = 2.99792458e8            # m/s
JOULES_PER_KWH = 3.6e6
SECONDS_PER_DAY = 86400.0
SECONDS_PER_WEEK = 7.0 * SECONDS_PER_DAY

# ---------------------------------------------------------------------------
# Tier and service-class identifiers (normative ordering used by every array)
# ---------------------------------------------------------------------------
TIERS: Tuple[str, ...] = ("cloud", "edge", "device")
N_TIER = len(TIERS)
TIER_IDX: Dict[str, int] = {t: i for i, t in enumerate(TIERS)}

CLASSES: Tuple[str, ...] = ("semantic_ce", "urllc_v2x", "embb", "mmtc")
CLASS_LABELS: Tuple[str, ...] = (
    "Joint semantic comm. + channel estimation",
    "URLLC V2X",
    "eMBB video streaming",
    "mMTC batch sensor upload",
)
N_CLASS = len(CLASSES)
CLASS_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
HEADLINE_CLASS = 0                # semantic_ce is the manuscript's flagship task

# Request-complexity strata (used by the accuracy model and by the
# complexity-aware router).  Ordered hardest first so that a routing split
# alpha = (alpha_cloud, alpha_edge, alpha_device) can be applied by filling the
# cloud with the hardest fraction of the offered load.
COMPLEXITY: Tuple[str, ...] = ("hard", "medium", "easy")
N_COMPLEXITY = len(COMPLEXITY)


@dataclass(frozen=True)
class ServiceClass:
    """Per-service-class workload description."""

    key: str
    label: str
    share_of_sessions: float   # fraction of the active-session population
    rate_per_session: float    # mean inference requests per session per second
    tokens: int                # sequence length presented to the model
    raw_bits: float            # uplink payload without semantic encoding
    semantic_ratio: float      # payload compression of the semantic front-end
    result_bits: float         # downlink / result payload
    grant_bw_hz: float         # scheduled uplink bandwidth per grant
    mimo_layers: int           # uplink spatial layers
    tti_ms: float              # slot alignment granularity (NR numerology)
    ran_proc_ms: float         # gNB L1/L2 processing + grant acquisition
    harq_retx_prob: float      # probability of one HARQ retransmission
    aggregation_ms: float      # duty-cycle batch-formation delay
    deadline_ms: float         # SLA deadline
    early_exit: bool           # early-exit heads are useful for this class
    pinned_cache: bool         # variant is pinned resident, never cold-started
    note: str


SERVICE_CLASSES: Tuple[ServiceClass, ...] = (
    ServiceClass(
        key="semantic_ce",
        label="Joint semantic comm. + channel estimation",
        share_of_sessions=0.60,
        rate_per_session=1.00,
        tokens=64,
        raw_bits=7.0e4,
        # raw CSI over the 20 MHz grant (612 subcarriers x 2 layers x I/Q x
        # 12 bit = 29.4 kbit) plus the uncompressed source message (~40 kbit)
        semantic_ratio=20.0,
        result_bits=2.0e3,
        grant_bw_hz=20.0e6,
        mimo_layers=2,
        tti_ms=0.25,            # 60 kHz SCS
        ran_proc_ms=0.50,
        harq_retx_prob=0.05,
        aggregation_ms=0.0,
        deadline_ms=30.0,
        early_exit=False,       # 64-token task; exit heads only pay off on the
                                # device tier, handled by ``device_early_exit``
        pinned_cache=False,
        note="headline task; one inference per active session per second, "
             "which is the load assumed by the CONTRACT aggregate cross-check",
    ),
    ServiceClass(
        key="urllc_v2x",
        label="URLLC V2X",
        share_of_sessions=0.10,
        rate_per_session=2.00,
        tokens=5,
        raw_bits=6.9e4,         # [CAL] raw V2X sensor feature frame
        semantic_ratio=40.0,    # [CAL] aggressive semantic coding of a short
                                # safety message; within the 10-100x range
                                # reported for semantic communication
        result_bits=5.0e2,
        grant_bw_hz=20.0e6,
        mimo_layers=2,
        tti_ms=0.125,           # 120 kHz SCS
        ran_proc_ms=0.40,       # grant-free / configured-grant access
        harq_retx_prob=0.10,    # high Doppler -> higher first-transmission BLER
        aggregation_ms=0.0,
        deadline_ms=5.0,        # 3GPP safety-critical V2X budget
        early_exit=False,
        pinned_cache=True,      # the safety-critical variant is pinned
                                # resident at every MEC site, so a V2X request
                                # is never delayed by a model load
        note="single-shot channel estimate plus semantic decode, warm cache",
    ),
    ServiceClass(
        key="embb",
        label="eMBB video streaming",
        share_of_sessions=0.15,
        rate_per_session=0.30,
        tokens=256,
        raw_bits=1.85e6,        # [CAL] GoP-level raw feature upload
        semantic_ratio=10.0,    # [CAL] semantic video feature coding, at the
                                # conservative end of the 10-100x range
                                # reported for semantic communication
        result_bits=5.0e4,
        grant_bw_hz=20.0e6,
        mimo_layers=2,
        tti_ms=0.5,             # 30 kHz SCS
        ran_proc_ms=1.00,
        harq_retx_prob=0.08,
        aggregation_ms=0.0,
        deadline_ms=140.0,
        early_exit=True,
        pinned_cache=False,
        note="long-sequence task; early-exit heads save ~30 % of the depth",
    ),
    ServiceClass(
        key="mmtc",
        label="mMTC batch sensor upload",
        share_of_sessions=0.15,
        rate_per_session=0.25,
        tokens=128,
        raw_bits=1.08e6,        # [CAL] aggregated batch of sensor records
        semantic_ratio=10.0,
        result_bits=1.0e4,
        grant_bw_hz=3.6e6,      # narrowband allocation for coverage-limited MTC
        mimo_layers=1,
        tti_ms=1.0,             # 15 kHz SCS
        ran_proc_ms=4.00,       # four-step random access
        harq_retx_prob=0.12,
        aggregation_ms=135.0,   # [CAL] duty-cycle batch-formation window
        deadline_ms=520.0,
        early_exit=True,
        pinned_cache=False,
        note="duty-cycled batch formation dominates the latency budget",
    ),
)

# Complexity mixture of the offered load.  [CAL]: chosen so that a router that
# matches request complexity to model capacity settles on the CONTRACT
# headline split alpha = (0.25 cloud, 0.50 edge, 0.25 device).
COMPLEXITY_MIX: Tuple[float, float, float] = (0.25, 0.50, 0.25)  # hard/med/easy


@dataclass(frozen=True)
class ModelSpec:
    """One member of the model zoo."""

    key: str
    label: str
    params: float             # parameter count
    tier: str                 # the tier this variant is designed for
    base_success_rate: float  # task success rate on the reference task, %
    note: str


MODEL_ZOO: Tuple[ModelSpec, ...] = (
    ModelSpec("cloud_70b_moe", "70 B MoE wireless foundation model", 70e9,
              "cloud", 94.2,
              "sparse mixture-of-experts; the effective throughput folds in "
              "the top-k expert activation factor"),
    ModelSpec("edge_7b_lora", "7 B LoRA-adapted edge model", 7e9,
              "edge", 91.5,
              "structured head pruning followed by rank-16 LoRA re-adaptation "
              "to local channel statistics"),
    ModelSpec("device_50m_int4", "50 M INT4 device micro-model", 50e6,
              "device", 84.0,
              "distilled micro-model with early-exit heads, INT4 weights"),
)
MODEL_BY_TIER: Dict[str, ModelSpec] = {m.tier: m for m in MODEL_ZOO}


@dataclass(frozen=True)
class Compression:
    """A weight-compression variant."""

    key: str
    speedup: float            # effective-throughput multiplier over FP16 dense
    bytes_per_param: float
    acc_penalty_pp: float     # task-success-rate penalty, percentage points
    note: str


COMPRESSIONS: Tuple[Compression, ...] = (
    Compression("fp16", 1.00, 2.0, 0.00, "FP16 dense reference precision"),
    Compression("lora", 1.60, 1.1, 0.30,
                "40 % structured head pruning + rank-16 LoRA re-adaptation"),
    Compression("int8", 2.00, 1.0, 0.90, "post-training INT8 quantisation"),
    Compression("int4", 3.20, 0.5, 1.60, "post-training INT4 quantisation"),
)
COMP_IDX: Dict[str, int] = {c.key: i for i, c in enumerate(COMPRESSIONS)}


@dataclass
class Config:
    """Complete parameter set of the MAIMO reference simulator."""

    # ------------------------------------------------------------------
    # 1. Deployment
    # ------------------------------------------------------------------
    n_cells: int = 48
    """Macro cells in the metropolitan cluster.  [CAL] sized so that the cloud
    tier operates with enough parallel serving nodes for the M/G/c waiting time
    to be a small fraction of the wide-area round trip; see README."""

    sessions_per_cell: int = 1000
    """Concurrent active AI sessions per cell.  A 6G dense-urban macro cell is
    dimensioned for 10^6 connected devices/km^2; 1000 *simultaneously
    inferencing* sessions is a conservative fraction of that."""

    inter_site_distance_m: float = 500.0   # 3GPP UMa dense-urban reference
    min_2d_distance_m: float = 25.0        # TR 38.901 validity floor

    # ------------------------------------------------------------------
    # 2. Radio layer (3GPP TR 38.901 UMa)
    # ------------------------------------------------------------------
    fc_ghz: float = 3.5                    # FR1 mid-band 6G candidate carrier
    system_bw_hz: float = 100.0e6          # 5G-Advanced/6G FR1 carrier
    h_bs_m: float = 25.0                   # TR 38.901 UMa base-station height
    h_ut_m: float = 1.5                    # TR 38.901 UE height
    h_e_m: float = 1.0                     # effective environment height, UMa
    ue_tx_dbm: float = 23.0                # 3GPP power class 3
    noise_figure_db: float = 5.0           # gNB receiver noise figure
    interference_margin_db: float = 6.0
    """Uplink inter-cell interference margin.  Standard link-budget practice
    for a fully loaded hexagonal reuse-1 network; avoids simulating every
    interferer explicitly."""
    sigma_sf_los_db: float = 4.0           # TR 38.901 UMa LOS shadowing
    sigma_sf_nlos_db: float = 7.8          # TR 38.901 UMa NLOS shadowing
    impl_loss_eta: float = 0.75
    """Shannon implementation-loss factor: R = eta * W * log2(1+SINR).
    0.75 is the usual link-level fit for an LTE/NR receiver."""
    max_spectral_efficiency: float = 7.4   # 256QAM r=5/6 with NR overheads
    min_spectral_efficiency: float = 0.20
    """Link-adaptation floor: the lowest NR MCS (QPSK, r ~ 1/8, with slot
    repetition) delivers about 0.19 bit/s/Hz.  A UE whose instantaneous SINR
    would give less than this is in outage and is served by the robust
    fallback transmission mode at exactly this efficiency.  Without the floor
    the mean of ``payload/rate`` is dominated by a measure-zero set of deep
    fades and diverges, which is a modelling artefact, not physics."""
    cdl_c_taps: int = 12
    cdl_c_delay_spread_ns: float = 300.0   # TR 38.901 CDL-C "short delay" UMa
    cdl_a_taps: int = 6
    cdl_a_delay_spread_ns: float = 30.0    # near-flat V2X profile
    v2x_csi_ageing_db: float = 4.0
    """[CAL] SNR derate for CSI ageing on high-Doppler V2X links."""
    n_channel_samples: int = 20000
    """Size of the per-seed pool of channel realisations drawn once and then
    sampled from during the run.  Large enough for the tail statistics."""

    # ------------------------------------------------------------------
    # 3. Time base and experiment protocol (CONTRACT-locked)
    # ------------------------------------------------------------------
    t_slot_s: float = 1.0
    """Simulation slot.  Equal to the ``T^slot = 1 s`` accounting window of the
    CONTRACT energy model, so the idle-power amortisation term is evaluated on
    exactly the window the model prescribes."""

    control_interval_slots: int = 10
    """Orchestration decision period, 10 s.  Model (re)loading takes seconds
    and MEC orchestration frameworks reconcile at this cadence; a per-slot
    controller would be unimplementable in a real deployment."""

    horizon_slots: int = 360_000     # CONTRACT: 3.6e5 slots = 100 h
    warmup_slots: int = 10_000       # CONTRACT: discarded
    n_seeds: int = 20                # CONTRACT: seeds 1..20
    t_crit_19df: float = 2.093       # CONTRACT: t_{0.975,19}

    # ------------------------------------------------------------------
    # 4. Traffic: diurnal + weekly + bursty (MMPP-2)
    # ------------------------------------------------------------------
    diurnal_amp1: float = 0.42       # first harmonic of the daily profile
    diurnal_amp2: float = 0.15       # second harmonic (morning/evening peaks)
    diurnal_peak_hour: float = 20.0  # evening busy hour
    diurnal_second_peak_hour: float = 10.0
    weekend_factor: float = 0.78     # weekend load relative to weekday
    mmpp_burst_multiplier: float = 2.1
    mmpp_mean_quiet_s: float = 1800.0
    mmpp_mean_burst_s: float = 120.0
    poisson_arrivals: bool = True

    # ------------------------------------------------------------------
    # 5. Tier compute
    # ------------------------------------------------------------------
    flops_cloud: float = 1.792e15
    """[CAL] Effective FP16 throughput of one cloud serving node (8x A100
    80 GB SXM at ~70 % utilisation, MoE top-k activation folded in).  Chosen so
    that the 64-token reference task takes exactly the CONTRACT's
    T_inf = 5.0 ms."""
    flops_edge: float = 7.00e13
    """[CAL] Effective FP16 throughput of one 250 W MEC accelerator board;
    with the 1.6x LoRA/pruning speed-up the 64-token task takes the CONTRACT's
    T_inf = 8.0 ms."""
    flops_device: float = 4.667e11
    """[CAL] Effective FP16-equivalent throughput of a 6 TOPS mobile NPU on a
    batch-1 memory-bound workload; with the 3.2x INT4 speed-up and the 0.70
    early-exit factor the 64-token task takes the CONTRACT's T_inf = 3.0 ms."""

    cloud_target_utilisation: float = 0.975
    """CONTRACT: n_cloud = 195 inferences/s per node at T_inf = 5 ms, i.e.
    rho = 195 * 0.005 = 0.975.  Autoscaling holds this utilisation."""
    edge_target_utilisation: float = 0.568
    """CONTRACT: n_edge = 71 inferences/s per board at T_inf = 8 ms, i.e.
    rho = 71 * 0.008 = 0.568.  The edge is provisioned with latency headroom."""

    cloud_nodes_max: int = 560
    """Provisioned cloud serving nodes.  Dimensioned for the busy-hour peak of
    the *cloud-only* baseline with ~50 % headroom, so that no policy is
    penalised by an arbitrary capacity cliff and the differences between
    policies come from routing physics rather than from provisioning."""
    edge_boards_per_site: int = 19
    """250 W accelerator boards per MEC site, likewise dimensioned for the
    busy-hour peak of the *edge-only* baseline."""
    cloud_fixed_ms: float = 0.30     # ingress/egress, serialisation, batching
    edge_fixed_ms: float = 0.15      # MEC ingress + semantic decode wrapper
    device_fixed_ms: float = 0.05    # NPU dispatch overhead
    wan_rtt_ms: float = 12.4
    """[CAL] UE-to-regional-cloud round-trip transport delay.  Within the
    10-20 ms range measured for metropolitan-to-regional-datacentre paths."""
    early_exit_factor: float = 0.70  # mean fraction of depth actually executed
    early_exit_acc_pp: float = 0.40  # accuracy cost of exiting early
    device_early_exit: bool = True   # the 50 M model always has exit heads

    # ------------------------------------------------------------------
    # 6. Semantic communication front-end
    # ------------------------------------------------------------------
    semantic_encoder_params: float = 12.45e6
    """[CAL] Device-side semantic encoder; 48 tokens at INT4 on the 6 TOPS NPU
    gives the CONTRACT's 0.8 ms device semantic-encoding term."""
    semantic_encoder_tokens: int = 48
    ue_radio_power_w: float = 1.2
    """UE radio power while transmitting: a 23 dBm power amplifier at ~20 %
    efficiency (1.0 W) plus RF and baseband.  With the 0.089 ms result
    transmission of a device-tier inference this reproduces the CONTRACT's
    0.1 mJ uplink term."""

    # ------------------------------------------------------------------
    # 7. Model cache and proactive loading
    # ------------------------------------------------------------------
    edge_cache_gb: float = 32.0
    """Host memory that one MEC site can devote to resident model variants,
    alongside the base model and the activation working set of a 250 W
    accelerator board."""
    edge_variants: int = 200
    """Task-specialised variants in the edge zoo.  A metropolitan MEC site
    serves many verticals (V2X perception, video analytics, industrial
    telemetry, AR overlay, ...) and each carries several task- and
    domain-specialised adapters; 200 is the working-set size assumed here.
    With ``variant_gb`` this is 110 GB of adapters competing for a 32 GB
    cache, which is what makes model placement a real decision rather than a
    formality."""
    variant_gb: float = 0.55
    """Resident footprint of one specialised variant: a sparse-quantised LoRA
    delta over the shared 7 B base model."""
    variant_zipf_s: float = 1.35
    """[CAL] Skew of variant popularity.  Request popularity in content and
    model serving is well fitted by a Zipf law; the exponent is calibrated
    here so that Che's approximation returns an on-demand LRU miss rate of
    about 11 % for the cache and zoo sizes above, which is the order reported
    for model-serving caches whose working set exceeds the host memory.  It is
    the only free parameter of the cache model: the two miss rates that used
    to be asserted directly are now derived from it in ``maimo.cache``."""
    cold_start_ms_per_gb: float = 42.0   # 10 Gbps backhaul + deserialisation
    cold_start_penalty_ms: float = 23.0
    """Latency added to a request that arrives while its variant is absent:
    a 0.55 GB sparse-quantised LoRA delta fetched over the 10 Gbps backhaul
    and deserialised, 0.55 GB x 42 ms/GB = 23 ms."""

    # The reactive and proactive miss rates are *derived* from the parameters
    # above by ``maimo.cache``; they are not tunable.

    # ------------------------------------------------------------------
    # 8. BiLSTM traffic predictor
    # ------------------------------------------------------------------
    pred_window: int = 24            # control intervals of history (4 min)
    pred_horizon: int = 5            # control intervals ahead (50 s)
    pred_hidden: int = 32
    pred_layers: int = 1
    pred_dropout: float = 0.0
    pred_lr: float = 3.0e-3
    pred_epochs: int = 30
    pred_batch: int = 256
    pred_patience: int = 5           # early stopping on validation MSE
    pred_train_intervals: int = 6000    # 60 000 slots of the training trace
    pred_val_fraction: float = 0.2
    pred_test_fraction: float = 0.2
    pred_obs_noise: float = 0.06
    """Relative measurement noise on the observed aggregate load.  The load
    counter the orchestrator actually sees is aggregated over one 10 s control
    interval and over a sampled subset of cells, so it carries sampling and
    reporting jitter of this order.  Denoising that counter is a large part of
    what a learned predictor buys over reusing the last measurement."""

    # ------------------------------------------------------------------
    # 9. PPO orchestrator
    # ------------------------------------------------------------------
    ppo_hidden: int = 64
    ppo_layers: int = 2
    ppo_lr: float = 3.0e-4
    ppo_clip: float = 0.2
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_entropy_coef: float = 0.01
    ppo_value_coef: float = 0.5
    ppo_grad_clip: float = 0.5
    ppo_epochs: int = 4
    ppo_minibatch: int = 64
    ppo_rollout: int = 128           # control intervals per rollout
    ppo_updates: int = 150           # PPO updates per seed
    ppo_adam_eps: float = 1.0e-5

    # ------------------------------------------------------------------
    # 10. DQN and LinUCB comparators (B8, B9)
    # ------------------------------------------------------------------
    dqn_hidden: int = 64
    dqn_lr: float = 5.0e-4
    dqn_gamma: float = 0.99
    dqn_eps_start: float = 1.0
    dqn_eps_end: float = 0.05
    dqn_eps_decay_steps: int = 6000
    dqn_replay: int = 20000
    dqn_batch: int = 64
    dqn_target_sync: int = 250
    dqn_steps: int = 19200           # = ppo_updates * ppo_rollout, matched
    linucb_alpha: float = 0.6        # exploration width of the UCB bonus
    linucb_lambda: float = 1.0       # ridge regulariser

    # ------------------------------------------------------------------
    # 11. Reward weights
    # ------------------------------------------------------------------
    w_latency: float = 0.45
    w_energy: float = 0.25
    w_accuracy: float = 0.30
    reward_sla_penalty: float = 0.5
    accuracy_floor_pct: float = 93.0
    """Quality-of-result floor in the orchestration objective.  The deployment
    contract specifies a minimum task-success rate; an answer that falls below
    it has to be re-issued to a higher tier, which costs far more than serving
    the request correctly the first time.  The floor is what makes the
    orchestrator keep a cloud share instead of collapsing onto the cheapest
    tier, and it is the mechanism behind the ~25 % cloud share reported in the
    manuscript."""
    reward_accuracy_penalty: float = 4.0
    """Penalty per normalised percentage point below ``accuracy_floor_pct``;
    an order of magnitude above ``w_accuracy`` so that the floor behaves like a
    constraint rather than another term to be traded off."""
    latency_norm_ms: float = 22.0    # cloud-only headline reference
    energy_norm_j: float = 16.725    # cloud-only per-inference reference
    accuracy_norm_pp: float = 5.0    # accuracy-loss normaliser

    # ------------------------------------------------------------------
    # 12. Baseline tunables
    # ------------------------------------------------------------------
    threshold_sinr_db: float = 4.0       # B6 offload threshold
    threshold_queue_hysteresis: float = 0.15
    lyapunov_v: float = 1.2              # B7 drift-plus-penalty control knob
    lyapunov_v_grid: Tuple[float, ...] = (0.3, 0.6, 1.2, 2.4, 4.8)

    # ------------------------------------------------------------------
    # 13. Latency-sampling resolution
    # ------------------------------------------------------------------
    latency_samples_per_group: int = 4
    latency_hist_bins: int = 1200
    latency_hist_min_ms: float = 0.05
    latency_hist_max_ms: float = 5000.0

    # ------------------------------------------------------------------
    # 14. Carbon
    # ------------------------------------------------------------------
    embodied_kg_per_a100: float = 1500.0   # cradle-to-gate estimate per GPU
    a100_per_cloud_node: int = 8
    hardware_life_years: float = 5.0
    embodied_kg_per_edge_board: float = 320.0
    embodied_kg_per_device_npu: float = 45.0
    device_life_years: float = 3.0
    geo_shift_max_fraction: float = 0.30
    """Fraction of cloud work that may legally and practically be migrated to
    another region (data-sovereignty and latency constrained)."""
    temporal_shift_max_fraction: float = 0.22
    """Fraction of the workload that is delay tolerant enough to be deferred
    (mMTC batch upload and the deferrable part of eMBB)."""
    temporal_shift_window_h: float = 6.0

    # ------------------------------------------------------------------
    # derived helpers
    # ------------------------------------------------------------------
    def n_sessions(self) -> int:
        return self.n_cells * self.sessions_per_cell

    def edge_boards_max(self) -> int:
        return self.n_cells * self.edge_boards_per_site

    def control_intervals(self) -> int:
        return (self.horizon_slots + self.warmup_slots) // self.control_interval_slots

    def train_intervals(self) -> int:
        """Length of the disjoint training window, in control intervals.

        Equal to ``ppo_updates * ppo_rollout`` so that PPO consumes it exactly
        once and DQN, whose step budget is matched to PPO's, sees the same
        number of transitions."""
        return self.ppo_updates * self.ppo_rollout

    def warmup_intervals(self) -> int:
        return self.warmup_slots // self.control_interval_slots

    def t_control_s(self) -> float:
        return self.t_slot_s * self.control_interval_slots

    def noise_dbm(self, bw_hz: float) -> float:
        return (THERMAL_NOISE_DBM_HZ + 10.0 * math.log10(bw_hz)
                + self.noise_figure_db + self.interference_margin_db)

    def as_dict(self) -> dict:
        return asdict(self)

    def replace(self, **kw) -> "Config":
        return replace(self, **kw)


DEFAULT = Config()


# ---------------------------------------------------------------------------
# Fast-run configuration used by the tests and by the calibration harness
# ---------------------------------------------------------------------------
def quick(cfg: Config = DEFAULT, horizon_slots: int = 20_000,
          warmup_slots: int = 2_000, n_seeds: int = 4) -> Config:
    """A short configuration with identical physics, for tests/calibration."""
    return cfg.replace(horizon_slots=horizon_slots, warmup_slots=warmup_slots,
                       n_seeds=n_seeds, ppo_updates=25, dqn_steps=3200,
                       pred_epochs=6, pred_train_intervals=1500)
