"""Radio layer: 3GPP TR 38.901 UMa path loss, CDL small-scale fading, uplink
SINR and the Shannon-with-implementation-loss rate.

Two dimensional corrections relative to the submitted manuscript are applied
here and are flagged in the README:

1. The received SNR is formed **in the linear domain**,
   ``gamma = P_tx |h|^2 10^(-PL_dB/10) / (sigma_n^2)``.  The manuscript's
   Equation (2) divides a linear power by a path loss expressed in dB, which
   is dimensionally inconsistent.
2. Path loss uses the full TR 38.901 UMa LOS/NLOS pair with the correct
   breakpoint distance, not the single-slope free-space-like expression.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from .config import Config, DEFAULT, SERVICE_CLASSES, C_LIGHT


# ---------------------------------------------------------------------------
# TR 38.901 UMa path loss (Table 7.4.1-1)
# ---------------------------------------------------------------------------
def breakpoint_distance_m(cfg: Config) -> float:
    """``d'_BP = 4 h'_BS h'_UT f_c / c`` with effective antenna heights."""
    h_bs = cfg.h_bs_m - cfg.h_e_m
    h_ut = cfg.h_ut_m - cfg.h_e_m
    return 4.0 * h_bs * h_ut * (cfg.fc_ghz * 1e9) / C_LIGHT


def uma_los_pathloss_db(d2d_m, cfg: Config = DEFAULT):
    """UMa LOS path loss, dB.  ``d2d_m`` may be an array."""
    d2d = np.maximum(np.asarray(d2d_m, dtype=float), 1.0)
    dh = cfg.h_bs_m - cfg.h_ut_m
    d3d = np.sqrt(d2d ** 2 + dh ** 2)
    fc = cfg.fc_ghz
    pl1 = 28.0 + 22.0 * np.log10(d3d) + 20.0 * math.log10(fc)
    d_bp = breakpoint_distance_m(cfg)
    pl2 = (28.0 + 40.0 * np.log10(d3d) + 20.0 * math.log10(fc)
           - 9.0 * np.log10(d_bp ** 2 + dh ** 2))
    return np.where(d2d <= d_bp, pl1, pl2)


def uma_nlos_pathloss_db(d2d_m, cfg: Config = DEFAULT):
    """UMa NLOS path loss, dB: ``max(PL_LOS, PL'_NLOS)``."""
    d2d = np.maximum(np.asarray(d2d_m, dtype=float), 1.0)
    dh = cfg.h_bs_m - cfg.h_ut_m
    d3d = np.sqrt(d2d ** 2 + dh ** 2)
    pl_nlos = (13.54 + 39.08 * np.log10(d3d) + 20.0 * math.log10(cfg.fc_ghz)
               - 0.6 * (cfg.h_ut_m - 1.5))
    return np.maximum(uma_los_pathloss_db(d2d, cfg), pl_nlos)


def uma_los_probability(d2d_m, cfg: Config = DEFAULT):
    """TR 38.901 Table 7.4.2-1 UMa LOS probability for ``h_UT <= 13 m``."""
    d2d = np.maximum(np.asarray(d2d_m, dtype=float), 1.0)
    return (np.minimum(18.0 / d2d, 1.0) * (1.0 - np.exp(-d2d / 63.0))
            + np.exp(-d2d / 63.0))


# ---------------------------------------------------------------------------
# CDL small-scale fading
# ---------------------------------------------------------------------------
def _exponential_pdp(delay_spread_ns: float, n_taps: int):
    """Exponential power-delay profile: tap delays (s) and normalised powers."""
    tau = delay_spread_ns * 1e-9
    delays = np.linspace(0.0, 8.0 * tau, n_taps)
    powers = np.exp(-delays / tau)
    return delays, powers / powers.sum()


def cdl_band_averaged_gain(rng: np.random.Generator, n: int,
                           delay_spread_ns: float, n_taps: int,
                           bandwidth_hz: float, n_freq: int = 64) -> np.ndarray:
    """Band-averaged small-scale power gain of a CDL tapped-delay channel.

    The uplink grant spans many coherence bandwidths, so the quantity that
    determines the achievable rate is the gain averaged over the allocated
    band rather than a single flat-fading draw.  We synthesise the frequency
    response of an exponential-PDP channel with complex Gaussian taps and
    average ``|H(f)|^2`` over ``n_freq`` equally spaced subcarriers.  The
    result has unit mean by construction; its variance falls as the frequency
    diversity of the profile rises, which is what distinguishes the 300 ns
    CDL-C profile from the near-flat 30 ns CDL-A V2X profile.
    """
    delays, powers = _exponential_pdp(delay_spread_ns, n_taps)
    freqs = np.linspace(0.0, bandwidth_hz, n_freq, endpoint=False)
    phase = np.exp(-2j * math.pi * np.outer(freqs, delays))       # (F, L)
    taps = ((rng.normal(size=(n, len(delays)))
             + 1j * rng.normal(size=(n, len(delays)))) / math.sqrt(2.0))
    taps = taps * np.sqrt(powers)[None, :]
    h = taps @ phase.T                                            # (n, F)
    return np.mean(np.abs(h) ** 2, axis=1)


# ---------------------------------------------------------------------------
# Link budget
# ---------------------------------------------------------------------------
def linear_sinr(tx_dbm: float, pl_db, shadow_db, fading_gain, noise_dbm: float):
    """Uplink SINR as a linear power ratio (all logs converted first)."""
    p_tx_w = 10.0 ** ((tx_dbm - 30.0) / 10.0)
    n_w = 10.0 ** ((noise_dbm - 30.0) / 10.0)
    return p_tx_w * fading_gain * 10.0 ** (-(pl_db + shadow_db) / 10.0) / n_w


def shannon_rate_bps(sinr_lin, bw_hz: float, layers: int, cfg: Config = DEFAULT):
    """``R = layers * eta * W * min(log2(1+SINR), SE_max)``."""
    se = cfg.impl_loss_eta * np.log2(1.0 + np.maximum(sinr_lin, 1e-12))
    se = np.clip(se, cfg.min_spectral_efficiency, cfg.max_spectral_efficiency)
    return layers * se * bw_hz


# ---------------------------------------------------------------------------
# Per-seed pool of uplink rates
# ---------------------------------------------------------------------------
class ChannelPool:
    """A pool of uplink transmission rates, one distribution per class.

    Drawing a UE position, LOS state, shadow realisation and CDL band-averaged
    fading gain is expensive; the offered load is stationary in space, so we
    draw ``cfg.n_channel_samples`` independent realisations once per seed and
    then index into the pool during the run.  This is exactly equivalent to
    drawing i.i.d. samples from the spatial/fading distribution and is what
    makes the 3.6e5-slot horizon affordable.
    """

    def __init__(self, cfg: Config, seed: int):
        self.cfg = cfg
        rng = np.random.default_rng(1_000_000 + seed)
        n = cfg.n_channel_samples
        r_cell = cfg.inter_site_distance_m / 2.0
        d2d = np.maximum(r_cell * np.sqrt(rng.random(n)), cfg.min_2d_distance_m)

        p_los = uma_los_probability(d2d, cfg)
        is_los = rng.random(n) < p_los
        pl = np.where(is_los, uma_los_pathloss_db(d2d, cfg),
                      uma_nlos_pathloss_db(d2d, cfg))
        sigma = np.where(is_los, cfg.sigma_sf_los_db, cfg.sigma_sf_nlos_db)
        shadow = rng.normal(0.0, 1.0, n) * sigma

        self.d2d_m = d2d
        self.los_fraction = float(is_los.mean())
        self.mean_pathloss_db = float(pl.mean())

        self.rate_bps: Dict[str, np.ndarray] = {}
        self.sinr_db: Dict[str, np.ndarray] = {}
        for sc in SERVICE_CLASSES:
            if sc.key == "urllc_v2x":
                gain = cdl_band_averaged_gain(
                    rng, n, cfg.cdl_a_delay_spread_ns, cfg.cdl_a_taps,
                    sc.grant_bw_hz)
                gain = gain * 10.0 ** (-cfg.v2x_csi_ageing_db / 10.0)
            else:
                gain = cdl_band_averaged_gain(
                    rng, n, cfg.cdl_c_delay_spread_ns, cfg.cdl_c_taps,
                    sc.grant_bw_hz)
            sinr = linear_sinr(cfg.ue_tx_dbm, pl, shadow, gain,
                               cfg.noise_dbm(sc.grant_bw_hz))
            self.sinr_db[sc.key] = 10.0 * np.log10(sinr)
            self.rate_bps[sc.key] = shannon_rate_bps(
                sinr, sc.grant_bw_hz, sc.mimo_layers, cfg)

    # -- uplink transmission times -------------------------------------
    def uplink_time_pool_ms(self, class_key: str, payload_bits: float
                            ) -> np.ndarray:
        return 1.0e3 * payload_bits / self.rate_bps[class_key]

    def uplink_time_mean_ms(self, class_key: str, payload_bits: float) -> float:
        return float(np.mean(self.uplink_time_pool_ms(class_key, payload_bits)))

    def summary(self) -> dict:
        out = {
            "los_fraction": self.los_fraction,
            "mean_pathloss_db": self.mean_pathloss_db,
            "mean_2d_distance_m": float(self.d2d_m.mean()),
        }
        for k, v in self.sinr_db.items():
            out[f"mean_sinr_db_{k}"] = float(v.mean())
            out[f"mean_rate_mbps_{k}"] = float(self.rate_bps[k].mean() / 1e6)
        return out
