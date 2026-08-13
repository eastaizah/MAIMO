"""Traffic predictors implemented from scratch in NumPy.

Three variants are provided, matching the ablation configurations:

* ``BiLSTMPredictor``  - forward and backward LSTM passes with the standard
  gated dynamics of Equations (11)-(13), concatenated hidden state, linear
  head.  This is the proposed predictor.
* ``LSTMPredictor``    - unidirectional LSTM, same capacity budget per
  direction, for ablation (g).
* ``NoForecastPredictor`` - persistence (last observed load repeated over the
  horizon), for ablation (b).

Training is truncated backpropagation through time with Adam.  Gradients are
derived analytically; no autograd framework is used.  A history window of
``W`` slots is mapped to a forecast horizon of ``H`` slots.

Deliberate scope statement (editor requirement R7): this is a *reference
implementation at reduced scale*.  The recurrent units are small (16 hidden
units per direction) because the artefact must run to completion on a laptop
CPU in minutes.
"""

from __future__ import annotations

import numpy as np

from config import Params


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


class _Adam:
    def __init__(self, shapes, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


class _LSTMCell:
    """One LSTM direction, batched over sequences.

    Gate order in the stacked weight matrices is ``[i, f, o, g]``.  Shapes are
    ``x: (B, T, n_in)``, ``h: (T+1, B, n_h)``, which lets the recurrence be a
    single Python loop over time regardless of batch size.
    """

    def __init__(self, n_in: int, n_h: int, rng: np.random.Generator):
        s = 1.0 / np.sqrt(n_h)
        self.n_in, self.n_h = n_in, n_h
        self.Wx = rng.uniform(-s, s, (n_in, 4 * n_h))
        self.Wh = rng.uniform(-s, s, (n_h, 4 * n_h))
        self.b = np.zeros(4 * n_h)
        self.b[n_h:2 * n_h] = 1.0            # forget-gate bias

    def params(self):
        return [self.Wx, self.Wh, self.b]

    def forward(self, x):
        B, T = x.shape[0], x.shape[1]
        H = self.n_h
        h = np.zeros((T + 1, B, H))
        c = np.zeros((T + 1, B, H))
        gates = np.zeros((T, B, 4 * H))
        xw = x @ self.Wx                       # (B, T, 4H)
        for t in range(T):
            z = xw[:, t] + h[t] @ self.Wh + self.b
            z[:, :3 * H] = _sigmoid(z[:, :3 * H])
            z[:, 3 * H:] = np.tanh(z[:, 3 * H:])
            gates[t] = z
            c[t + 1] = z[:, H:2 * H] * c[t] + z[:, :H] * z[:, 3 * H:]
            h[t + 1] = z[:, 2 * H:3 * H] * np.tanh(c[t + 1])
        return h, c, gates

    def backward(self, x, h, c, gates, dh_out):
        """Backpropagate ``dh_out`` (T, B, n_h) through time."""
        B, T = x.shape[0], x.shape[1]
        H = self.n_h
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db = np.zeros_like(self.b)
        dh_next = np.zeros((B, H))
        dc_next = np.zeros((B, H))
        for t in reversed(range(T)):
            g = gates[t]
            i, f, o, gg = (g[:, :H], g[:, H:2 * H], g[:, 2 * H:3 * H],
                           g[:, 3 * H:])
            dh = dh_out[t] + dh_next
            tc = np.tanh(c[t + 1])
            do = dh * tc
            dc = dh * o * (1 - tc ** 2) + dc_next
            di = dc * gg
            dgg = dc * i
            df = dc * c[t]
            dc_next = dc * f
            dz = np.concatenate([di * i * (1 - i), df * f * (1 - f),
                                 do * o * (1 - o), dgg * (1 - gg ** 2)],
                                axis=1)
            dWx += x[:, t].T @ dz
            dWh += h[t].T @ dz
            db += dz.sum(axis=0)
            dh_next = dz @ self.Wh.T
        return [dWx, dWh, db]


class BiLSTMPredictor:
    """Bidirectional LSTM traffic predictor with a linear forecast head."""

    kind = "bilstm"
    bidirectional = True

    def __init__(self, p: Params, seed: int = 0):
        self.p = p
        self.W = p.history_window
        self.H = p.horizon
        self.nh = p.predictor_hidden
        rng = np.random.default_rng(seed + 5501)
        self.n_in = 3                     # load, delta, log1p(load)
        self.fw = _LSTMCell(self.n_in, self.nh, rng)
        self.bw = (_LSTMCell(self.n_in, self.nh, rng)
                   if self.bidirectional else None)
        # Readout: final state and mean-pooled state of each direction.
        d = 2 * self.nh * (2 if self.bidirectional else 1)
        self.Wy = rng.normal(0.0, 0.3 / np.sqrt(d), (d, self.H))
        self.by = np.zeros(self.H)
        self.mu = 1.0
        self.sd = 1.0
        self.trained = False
        self.history_loss: list[float] = []

    # ------------------------------------------------------------------
    def _params(self):
        ps = list(self.fw.params())
        if self.bidirectional:
            ps += list(self.bw.params())
        return ps + [self.Wy, self.by]

    def _featurise(self, seq: np.ndarray) -> np.ndarray:
        """(B, T) load histories -> (B, T, 3) input features."""
        seq = np.atleast_2d(np.asarray(seq, dtype=float))
        z = (seq - self.mu) / self.sd
        d = np.diff(z, prepend=z[:, :1], axis=1)
        lg = np.log1p(np.maximum(seq, 0.0)) - np.log1p(self.mu)
        return np.stack([z, d, lg], axis=2)

    def _forward(self, seq: np.ndarray):
        x = self._featurise(seq)
        hf, cf, gf = self.fw.forward(x)
        parts = [hf[-1], hf[1:].mean(axis=0)]
        if self.bidirectional:
            xr = np.ascontiguousarray(x[:, ::-1])
            hb, cb, gb = self.bw.forward(xr)
            parts += [hb[-1], hb[1:].mean(axis=0)]
        else:
            xr = hb = cb = gb = None
        feat = np.concatenate(parts, axis=1)
        y = feat @ self.Wy + self.by
        return y, (x, hf, cf, gf, xr, hb, cb, gb, feat)

    # ------------------------------------------------------------------
    def fit(self, trace: np.ndarray) -> "BiLSTMPredictor":
        """Truncated BPTT with Adam and validation-based early stopping.

        The last ``predictor_val_frac`` of the trace is held out; the parameter
        snapshot with the lowest held-out forecast NRMSE is kept.  Without this
        the small recurrent net overfits the 6000-slot training trace.
        """
        p = self.p
        n_val = int(len(trace) * p.predictor_val_frac)
        train, val = trace[:len(trace) - n_val], trace[len(trace) - n_val:]
        self.mu = float(np.mean(train)) or 1.0
        self.sd = float(np.std(train)) or 1.0
        W, H = self.W, self.H
        idx = np.arange(W, len(train) - H)
        rng = np.random.default_rng(7)
        params = self._params()
        opt = _Adam([q.shape for q in params], p.predictor_lr,
                    p.ppo_adam_beta1, p.ppo_adam_beta2, p.ppo_adam_eps)
        nh = self.nh
        best = (np.inf, [q.copy() for q in params])
        trace = train
        for ep in range(p.predictor_epochs):
            batch = rng.choice(idx, size=min(p.predictor_batch, idx.size),
                               replace=False)
            B = batch.size
            seq = np.stack([trace[t - W:t] for t in batch])
            tgt = (np.stack([trace[t:t + H] for t in batch]) - self.mu) / self.sd
            y, cache = self._forward(seq)
            x, hf, cf, gf, xr, hb, cb, gb, feat = cache
            err = y - tgt
            loss = float((err ** 2).sum()) / (H * B)
            dy = 2.0 * err / (H * B)
            gWy = feat.T @ dy
            gby = dy.sum(axis=0)
            dfeat = dy @ self.Wy.T
            dhf = np.zeros((W, B, nh))
            dhf[-1] += dfeat[:, :nh]
            dhf += dfeat[None, :, nh:2 * nh] / W
            gf_list = self.fw.backward(x, hf, cf, gf, dhf)
            if self.bidirectional:
                dhb = np.zeros((W, B, nh))
                dhb[-1] += dfeat[:, 2 * nh:3 * nh]
                dhb += dfeat[None, :, 3 * nh:] / W
                gb_list = self.bw.backward(xr, hb, cb, gb, dhb)
                grads = gf_list + gb_list + [gWy, gby]
            else:
                grads = gf_list + [gWy, gby]
            gn = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
            if gn > 5.0:
                grads = [g * (5.0 / gn) for g in grads]
            opt.step(params, grads)
            self.history_loss.append(loss)
            if (ep + 1) % p.predictor_val_every == 0:
                v = self.nrmse(val)
                if v < best[0]:
                    best = (v, [q.copy() for q in params])
        for q, b in zip(params, best[1]):
            q[...] = b
        self.val_nrmse = best[0]
        self.trained = True
        return self

    # ------------------------------------------------------------------
    def predict(self, seq: np.ndarray) -> np.ndarray:
        """Forecast the next ``H`` slot loads, normalised by the mean load."""
        seq = np.asarray(seq, dtype=float)
        if seq.size < self.W:
            seq = np.pad(seq, (self.W - seq.size, 0), mode="edge")
        y, _ = self._forward(seq[-self.W:][None, :])
        return np.maximum(y[0] * self.sd + self.mu, 0.0) / max(self.mu, 1e-9)

    def nrmse(self, trace: np.ndarray) -> float:
        """Normalised RMSE of the H-step forecast on a held-out trace."""
        W, H = self.W, self.H
        ts = np.arange(W, len(trace) - H)
        seq = np.stack([trace[t - W:t] for t in ts])
        tgt = np.stack([trace[t:t + H] for t in ts])
        y, _ = self._forward(seq)
        pred = np.maximum(y * self.sd + self.mu, 0.0)
        return float(np.sqrt(((pred - tgt) ** 2).sum() / max((tgt ** 2).sum(),
                                                             1e-12)))


class LSTMPredictor(BiLSTMPredictor):
    """Unidirectional LSTM variant for ablation (g)."""

    kind = "lstm"
    bidirectional = False


class NoForecastPredictor:
    """Persistence baseline: no forecast at all (ablation (b))."""

    kind = "none"

    def __init__(self, p: Params, seed: int = 0):
        self.p = p
        self.W = p.history_window
        self.H = p.horizon
        self.mu = 1.0
        self.trained = True
        self.history_loss: list[float] = []

    def fit(self, trace: np.ndarray) -> "NoForecastPredictor":
        self.mu = float(np.mean(trace)) or 1.0
        return self

    def predict(self, seq: np.ndarray) -> np.ndarray:
        last = float(seq[-1]) if seq.size else self.mu
        return np.full(self.H, last / max(self.mu, 1e-9))

    def nrmse(self, trace: np.ndarray) -> float:
        W, H = self.W, self.H
        ts = np.arange(W, len(trace) - H)
        pred = np.repeat(trace[ts - 1][:, None], H, axis=1)
        tgt = np.stack([trace[t:t + H] for t in ts])
        return float(np.sqrt(((pred - tgt) ** 2).sum()
                             / max((tgt ** 2).sum(), 1e-12)))


def build_predictor(kind: str, p: Params, seed: int = 0):
    return {"bilstm": BiLSTMPredictor, "lstm": LSTMPredictor,
            "none": NoForecastPredictor}[kind](p, seed)
