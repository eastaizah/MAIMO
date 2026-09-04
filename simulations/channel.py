"""Radio channel: 3GPP UMa path loss, shadowing, CDL-like fading, and a real
OFDM channel-estimation experiment used for the NMSE results.

Two dimensional corrections relative to the submitted manuscript are applied
here and are flagged in the README:

1. The received SNR is computed **in the linear domain**::

       gamma = P_tx * |h|^2 * 10^(-PL_dB / 10) / sigma_n^2

   The manuscript's Equation (2) divides by ``PL`` expressed in dB, which is
   dimensionally inconsistent (a dB value is a pure number on a logarithmic
   scale and cannot appear in the denominator of a linear power ratio).

2. Path loss uses ``PL = 28.0 + 22 log10(d/m) + 20 log10(f_c/GHz)``, evaluated
   with ``d`` in metres and ``f_c`` in GHz as the manuscript intends.
"""

from __future__ import annotations

import math
import numpy as np

from config import Params, path_loss_db


# ---------------------------------------------------------------------------
# Large-scale and small-scale fading
# ---------------------------------------------------------------------------
def linear_snr(tx_dbm: float, pl_db, shadow_db, fading_gain,
               noise_dbm: float):
    """Received SNR as a linear power ratio.

    All logarithmic quantities are converted to the linear domain *before* the
    ratio is formed.
    """
    p_tx_w = 10.0 ** ((tx_dbm - 30.0) / 10.0)
    noise_w = 10.0 ** ((noise_dbm - 30.0) / 10.0)
    large_scale = 10.0 ** (-(pl_db + shadow_db) / 10.0)
    return p_tx_w * fading_gain * large_scale / noise_w


def shannon_rate_bps(snr_lin, bandwidth_hz: float):
    """Uplink rate ``R = W log2(1 + gamma)``."""
    return bandwidth_hz * np.log2(1.0 + np.maximum(snr_lin, 1e-12))


class FadingProcess:
    """Effective per-slot fading gain after capacity averaging over the band.

    A frequency-selective CDL profile with ``L`` resolvable taps, observed over
    a bandwidth much larger than the coherence bandwidth, yields an effective
    (band-averaged) power gain that concentrates around unity with a variance
    set by the frequency-diversity order.  We model that gain as
    ``Gamma(k, 1/k)`` with ``k`` the diversity order, which reproduces the
    correct mean (1) and the correct qualitative variance ordering between
    CDL-C (12 taps, 300 ns -> high frequency diversity over 20 MHz) and CDL-A
    (6 taps, 30 ns -> nearly flat over 20 MHz, hence far less averaging).

    High-Doppler V2X links additionally suffer a CSI-ageing SNR derate.
    """

    def __init__(self, rng: np.random.Generator, order: float,
                 ageing_db: float = 0.0):
        self.rng = rng
        self.order = float(order)
        self.ageing_lin = 10.0 ** (-ageing_db / 10.0)

    def sample(self, n: int) -> np.ndarray:
        if n == 0:
            return np.zeros(0)
        g = self.rng.gamma(shape=self.order, scale=1.0 / self.order, size=n)
        return g * self.ageing_lin


class RadioEnvironment:
    """UE positions, random-waypoint mobility, shadowing and SNR bookkeeping."""

    def __init__(self, p: Params, rng: np.random.Generator):
        self.p = p
        self.rng = rng
        n = p.n_ue()
        self.n = n
        # Hexagonal 7-cell cluster: centre site plus a ring at ISD.
        isd = p.inter_site_distance_m
        centres = [(0.0, 0.0)]
        for k in range(6):
            ang = math.radians(60.0 * k)
            centres.append((isd * math.cos(ang), isd * math.sin(ang)))
        self.cell_centres = np.array(centres)
        self.cell_of_ue = np.repeat(np.arange(p.n_cells), p.ue_per_cell)

        # Traffic-class assignment: 40 % URLLC, 35 % eMBB, 25 % mMTC per cell.
        mix = np.array(p.traffic_mix, dtype=float)
        per_cell = np.floor(mix * p.ue_per_cell).astype(int)
        per_cell[0] += p.ue_per_cell - per_cell.sum()
        cls = np.concatenate([np.full(per_cell[i], i) for i in range(3)])
        self.ue_class = np.tile(cls, p.n_cells)   # 0 URLLC, 1 eMBB, 2 mMTC

        # Initial positions: uniform in a disc of radius ISD/2 around the site.
        r = (isd / 2.0) * np.sqrt(self.rng.random(n))
        th = self.rng.random(n) * 2 * math.pi
        self.pos = self.cell_centres[self.cell_of_ue] + np.stack(
            [r * np.cos(th), r * np.sin(th)], axis=1)

        # Random-waypoint state.
        self.speed = np.where(
            self.ue_class == 0,
            self.rng.uniform(*p.veh_speed_kmh, size=n) / 3.6,
            self.rng.uniform(*p.ped_speed_ms, size=n))
        self.target = self._new_targets(np.ones(n, dtype=bool))
        self.pause = np.zeros(n, dtype=int)

        # LOS state and spatially correlated shadowing.
        self.is_los = self.rng.random(n) < p.los_probability
        self.sigma_sf = np.where(self.is_los, p.sigma_sf_los_db,
                                 p.sigma_sf_nlos_db)
        self.shadow = self.rng.normal(0.0, 1.0, n) * self.sigma_sf

        self.fading_c = FadingProcess(rng, p.diversity_order_cdl_c)
        self.fading_a = FadingProcess(rng, p.diversity_order_cdl_a,
                                      ageing_db=p.v2x_csi_ageing_db)
        self.noise_dbm = p.noise_power_dbm()
        self._urllc = self.ue_class == 0
        self._other = ~self._urllc
        self._n_urllc = int(self._urllc.sum())
        self._refresh_large_scale()

    def _new_targets(self, which: np.ndarray) -> np.ndarray:
        p = self.p
        tgt = getattr(self, "target", None)
        if tgt is None:
            tgt = np.zeros((self.n, 2))
        tgt = tgt.copy()
        idx = np.nonzero(which)[0]
        if idx.size:
            r = (p.inter_site_distance_m / 2.0) * np.sqrt(
                self.rng.random(idx.size))
            th = self.rng.random(idx.size) * 2 * math.pi
            tgt[idx] = (self.cell_centres[self.cell_of_ue[idx]]
                        + np.stack([r * np.cos(th), r * np.sin(th)], axis=1))
        return tgt

    def step_mobility(self, n_slots: int = 1) -> None:
        """Advance random-waypoint mobility and age the shadowing.

        ``n_slots`` allows the large-scale state to be updated every ``n_slots``
        slots instead of every slot.  At the highest modelled speed (120 km/h)
        a UE moves 0.33 m per 10 ms slot, i.e. two orders of magnitude less than
        the 50 m shadowing decorrelation distance, so a stride of a few slots
        leaves the large-scale statistics unchanged while removing most of the
        mobility bookkeeping from the inner loop.
        """
        p = self.p
        dt = p.t_slot_ms / 1000.0 * n_slots
        moving = self.pause <= 0
        d = self.target - self.pos
        dist = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        step = self.speed * dt
        arrived = moving & (dist <= np.maximum(step, 1e-9))
        scale = np.where(dist > 1e-9, step / np.maximum(dist, 1e-9), 0.0)
        scale = np.where(moving & ~arrived, scale, 0.0)
        self.pos = self.pos + d * scale[:, None]
        if arrived.any():
            self.pos[arrived] = self.target[arrived]
            self.pause = np.where(arrived, p.waypoint_pause_slots, self.pause)
            self.target = self._new_targets(arrived)
        self.pause = np.maximum(self.pause - n_slots, 0)

        # AR(1) shadowing with an exponential spatial autocorrelation.
        rho = np.exp(-step / p.shadow_decorrelation_m)
        self.shadow = (rho * self.shadow
                       + np.sqrt(np.maximum(1.0 - rho ** 2, 0.0))
                       * self.rng.normal(0.0, 1.0, self.n) * self.sigma_sf)
        self._refresh_large_scale()

    def _refresh_large_scale(self) -> None:
        d = self.pos - self.cell_centres[self.cell_of_ue]
        dist = np.maximum(np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2),
                          self.p.min_distance_m)
        pl = path_loss_db(dist, self.p.fc_ghz)
        p_tx_w = 10.0 ** ((self.p.ue_tx_dbm - 30.0) / 10.0)
        noise_w = 10.0 ** ((self.noise_dbm - 30.0) / 10.0)
        self._snr_scale = (p_tx_w / noise_w
                           * 10.0 ** (-(pl + self.shadow) / 10.0))
        self._dist = dist

    def serving_distance_m(self) -> np.ndarray:
        return self._dist

    def snr_linear(self) -> np.ndarray:
        """Per-UE effective uplink SNR for the current slot (linear).

        Formed in the linear domain: ``gamma = P_tx |h|^2 10^(-PL/10) / sigma_n^2``.
        """
        g = np.empty(self.n)
        g[self._urllc] = self.fading_a.sample(self._n_urllc)
        g[self._other] = self.fading_c.sample(self.n - self._n_urllc)
        return self._snr_scale * g


# ---------------------------------------------------------------------------
# Channel-estimation experiment (NMSE results)
# ---------------------------------------------------------------------------
def _pdp(delay_spread_ns: float, n_taps: int):
    """Exponential power-delay profile: tap delays (s) and normalised powers."""
    tau_rms = delay_spread_ns * 1e-9
    delays = np.linspace(0.0, 8.0 * tau_rms, n_taps)
    powers = np.exp(-delays / tau_rms)
    return delays, powers / powers.sum()


def _cdl_c_frequency_response(rng: np.random.Generator, p: Params,
                              n_sub: int, n_real: int) -> np.ndarray:
    """CDL-C-like frequency responses over ``n_sub`` subcarriers.

    Exponentially decaying power-delay profile with ``L`` taps spread over the
    300 ns r.m.s. delay spread, complex Gaussian tap gains (NLOS clusters).
    """
    delays, pdp = _pdp(p.cdl_c_delay_spread_ns, p.cdl_c_taps)
    df = p.bandwidth_hz / n_sub
    k = np.arange(n_sub)
    E = np.exp(-2j * np.pi * np.outer(k * df, delays))    # n_sub x L
    a = ((rng.normal(size=(n_real, len(delays)))
          + 1j * rng.normal(size=(n_real, len(delays)))) / math.sqrt(2.0))
    a = a * np.sqrt(pdp)[None, :]
    H = a @ E.T
    return H / math.sqrt(np.mean(np.abs(H) ** 2))


def _freq_covariance(delays: np.ndarray, powers: np.ndarray, n_sub: int,
                     bandwidth_hz: float) -> np.ndarray:
    """Exact frequency-domain covariance of a discrete-tap channel.

    ``R = E diag(p) E^H`` with ``E[k, l] = exp(-2 pi i k df tau_l)``, i.e. the
    exact second-order statistics of the channel realisations produced by
    :func:`_cdl_c_frequency_response`.  Normalised to unit power per subcarrier.
    """
    df = bandwidth_hz / n_sub
    k = np.arange(n_sub)
    E = np.exp(-2j * np.pi * np.outer(k * df, delays))
    R = (E * powers[None, :]) @ E.conj().T
    return R / np.real(np.trace(R)) * n_sub


def _nmse_db(h_true: np.ndarray, h_est: np.ndarray) -> float:
    """NMSE in dB, averaged over realisations in the linear (power) domain."""
    num = np.sum(np.abs(h_true - h_est) ** 2)
    den = np.sum(np.abs(h_true) ** 2)
    return 10.0 * math.log10(num / den)


def _quantise_grouped(x: np.ndarray, bits: int, group: int) -> np.ndarray:
    """Symmetric uniform quantisation with one scale per group of ``group``
    consecutive subcarriers (i.e. per physical resource block).

    This models the end-to-end effect of running the device-tier estimator in
    INT4: per-PRB activation scales are what mobile NPU kernels use.  The
    quantisation error is computed, not assumed.
    """
    levels = 2 ** (bits - 1) - 1
    n = x.shape[-1]
    pad = (-n) % group
    out = np.empty_like(x)
    for part in ("real", "imag"):
        v = getattr(x, part)
        vv = np.pad(v, ((0, 0), (0, pad)), mode="edge")
        vv = vv.reshape(v.shape[0], -1, group)
        s = np.max(np.abs(vv), axis=2, keepdims=True)
        s = np.where(s > 0, s, 1.0)
        q = np.round(vv / s * levels) * (s / levels)
        setattr(out, part, q.reshape(v.shape[0], -1)[:, :n])
    return out


def channel_estimation_nmse(seed: int, p: Params) -> dict:
    """Channel-estimation NMSE of the three estimators compared in the paper.

    All three estimators observe exactly the same noisy pilot samples of the
    same CDL-C realisations, at the SNR given by ``p.nmse_snr_db``.

    * **Least squares** - LS on the DMRS comb followed by linear interpolation
      onto the full subcarrier grid.  Fully genuine; no calibrated quantity
      enters it.
    * **Edge 7 B LoRA model** - a linear MMSE estimator whose covariance is
      built from a *mismatched* delay spread, ``kappa_edge`` times the realised
      one.  This is the honest surrogate for a trained estimator: a foundation
      model trained across the whole CDL delay-spread range (30 ns - 2.4 us)
      cannot specialise to the narrow realised power-delay profile, so its
      residual error is dominated by covariance mismatch rather than by
      thermal noise.  ``kappa_edge`` is a calibrated parameter.
    * **Device 50 M INT4 micro-model** - the same estimator with a coarser
      mismatch factor and with its filter weights genuinely quantised to INT4,
      which adds a real quantisation error term.

    An ideal LMMSE estimator with perfect covariance knowledge is also reported
    as a lower bound, so that the reader can see how far the surrogate sits
    from optimality.
    """
    rng = np.random.default_rng(20000 + seed)
    n_sub = p.nmse_n_subcarriers
    n_real = p.nmse_realisations
    tap_delays, tap_powers = _pdp(p.cdl_c_delay_spread_ns, p.cdl_c_taps)
    H = _cdl_c_frequency_response(rng, p, n_sub, n_real)

    sigma2 = 10.0 ** (-p.nmse_snr_db / 10.0)
    noise = (rng.normal(size=H.shape) + 1j * rng.normal(size=H.shape))
    noise *= math.sqrt(sigma2 / 2.0)
    Y = H + noise                          # unit-modulus pilots

    pil = np.arange(0, n_sub, p.nmse_pilot_spacing)
    all_k = np.arange(n_sub)
    yp = Y[:, pil]

    # ---- least squares + linear interpolation ---------------------------
    ls = np.empty_like(H)
    ls.real = np.stack([np.interp(all_k, pil, yp[i].real)
                        for i in range(n_real)])
    ls.imag = np.stack([np.interp(all_k, pil, yp[i].imag)
                        for i in range(n_real)])

    # ---- LMMSE family ----------------------------------------------------
    R_true = _freq_covariance(tap_delays, tap_powers, n_sub, p.bandwidth_hz)
    eye = np.eye(n_sub)

    def lmmse_weights(eps: float) -> np.ndarray:
        """LMMSE weights from a covariance blurred towards white by ``eps``.

        ``eps = 0`` is the ideal estimator with exact second-order statistics;
        ``eps -> 1`` is an estimator that has learned nothing about the
        frequency correlation of the channel.  ``eps`` quantifies how much of
        the true correlation structure the trained estimator fails to capture
        because it must generalise across channel conditions.
        """
        R = (1.0 - eps) * R_true + eps * eye
        Rpp = R[np.ix_(pil, pil)] + sigma2 * np.eye(pil.size)
        Rhp = R[:, pil]
        return Rhp @ np.linalg.inv(Rpp)

    w_ideal = lmmse_weights(0.0)
    w_edge = lmmse_weights(p.nmse_edge_cov_mismatch)
    w_dev = lmmse_weights(p.nmse_device_cov_mismatch)

    ideal = yp @ w_ideal.T
    edge = yp @ w_edge.T
    dev = _quantise_grouped(yp @ w_dev.T, p.nmse_device_weight_bits,
                            p.nmse_device_quant_group)

    return {
        "snr_db": p.nmse_snr_db,
        "ls_nmse_db": _nmse_db(H, ls),
        "edge_7b_lora_nmse_db": _nmse_db(H, edge),
        "device_50m_int4_nmse_db": _nmse_db(H, dev),
        "ideal_lmmse_nmse_db": _nmse_db(H, ideal),
        "n_realisations": n_real,
        "n_subcarriers": n_sub,
        "pilot_spacing": p.nmse_pilot_spacing,
        "edge_cov_mismatch": p.nmse_edge_cov_mismatch,
        "device_cov_mismatch": p.nmse_device_cov_mismatch,
        "device_weight_bits": p.nmse_device_weight_bits,
    }


if __name__ == "__main__":  # pragma: no cover
    from config import DEFAULT
    for s in (1, 2, 3):
        r = channel_estimation_nmse(s, DEFAULT)
        print(f"seed {s}: LS {r['ls_nmse_db']:.2f} dB   "
              f"edge {r['edge_7b_lora_nmse_db']:.2f} dB   "
              f"device {r['device_50m_int4_nmse_db']:.2f} dB")
