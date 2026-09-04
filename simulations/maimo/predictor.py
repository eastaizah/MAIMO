"""Bidirectional LSTM traffic predictor (PyTorch, CPU).

One predictor is trained per replication, on the first part of that
replication's own traffic trace, and is then evaluated over the whole
evaluation trace in a **single batched forward pass**.  Traffic is exogenous,
so nothing about the control policy enters the predictor; running it once
per slot inside the control loop would cost three orders of magnitude more
compute for exactly the same predictions.

The predictor consumes a window of noisy aggregate-load measurements and
predicts the mean offered load over the next ``pred_horizon`` control
intervals.  That prediction drives proactive model loading and enters the
orchestrator's state vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .traffic import TrafficTrace, make_trace


class BiLSTMPredictor(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=cfg.pred_hidden,
                            num_layers=cfg.pred_layers, batch_first=True,
                            bidirectional=True, dropout=cfg.pred_dropout)
        self.head = nn.Linear(2 * cfg.pred_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.head(h[:, -1, :]).squeeze(-1)


def _windows(series: np.ndarray, target: np.ndarray, window: int
             ) -> Tuple[np.ndarray, np.ndarray]:
    n = series.size - window
    idx = np.arange(window)[None, :] + np.arange(n)[:, None]
    return series[idx], target[window - 1:window - 1 + n]


def _future_mean(series: np.ndarray, horizon: int) -> np.ndarray:
    """Mean of the next ``horizon`` values, aligned with the current index."""
    pad = np.concatenate([series, np.repeat(series[-1:], horizon)])
    c = np.cumsum(np.concatenate([[0.0], pad]))
    return (c[horizon + 1:horizon + 1 + series.size] - c[1:1 + series.size]) / horizon


def _scale_invariant(series: np.ndarray, target: np.ndarray, window: int):
    """Window-relative features and target.

    Each input window is divided by its own mean and the target is expressed
    as the ratio of the future mean to that same window mean.  The predictor
    therefore learns the *shape* of the load, not its absolute level, and
    transfers without bias to a trace recorded at a different phase of the
    week or with a different session population.  Without this the network
    memorises the training window's level and its out-of-sample error is
    dominated by that offset.
    """
    x, y = _windows(series, target, window)
    scale = x.mean(axis=1, keepdims=True)
    return x / scale - 1.0, target[window - 1:window - 1 + x.shape[0]] \
        / scale[:, 0] - 1.0


@dataclass
class TrainedPredictor:
    model: BiLSTMPredictor
    history: dict


def train_predictor(cfg: Config, seed: int) -> TrainedPredictor:
    """Train one BiLSTM on a training trace generated for this seed.

    The training trace is a *separate* window of the same stochastic process
    (a different phase of the week), so the evaluation trace is never seen
    during training.  Split: 60 % train, 20 % validation, 20 % test,
    chronological, with early stopping on the validation MSE.
    """
    torch.manual_seed(3_000_000 + seed)
    tr = make_trace(cfg, 500_000 + seed, cfg.pred_train_intervals)
    series = tr.observed
    target = _future_mean(tr.total(), cfg.pred_horizon)
    x, y = _scale_invariant(series, target, cfg.pred_window)

    n = x.shape[0]
    n_test = int(n * cfg.pred_test_fraction)
    n_val = int(n * cfg.pred_val_fraction)
    n_train = n - n_val - n_test
    xt = torch.tensor(x[:n_train, :, None], dtype=torch.float32)
    yt = torch.tensor(y[:n_train], dtype=torch.float32)
    xv = torch.tensor(x[n_train:n_train + n_val, :, None], dtype=torch.float32)
    yv = torch.tensor(y[n_train:n_train + n_val], dtype=torch.float32)
    xs = torch.tensor(x[n_train + n_val:, :, None], dtype=torch.float32)
    ys = torch.tensor(y[n_train + n_val:], dtype=torch.float32)

    model = BiLSTMPredictor(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.pred_lr)
    loss_fn = nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0
    hist = {"train": [], "val": []}
    gen = torch.Generator().manual_seed(4_000_000 + seed)
    for epoch in range(cfg.pred_epochs):
        model.train()
        perm = torch.randperm(n_train, generator=gen)
        tot = 0.0
        for i in range(0, n_train, cfg.pred_batch):
            b = perm[i:i + cfg.pred_batch]
            opt.zero_grad()
            loss = loss_fn(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * b.numel()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(xv), yv))
        hist["train"].append(tot / max(n_train, 1))
        hist["val"].append(vl)
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.pred_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_mse = float(loss_fn(model(xs), ys))
        pred_test = model(xs).numpy()
        true_test = ys.numpy()
    hist["test_mse_normalised"] = test_mse
    hist["test_mape_pct"] = float(100.0 * np.mean(
        np.abs(pred_test - true_test) / np.maximum(1.0 + true_test, 1e-9)))
    hist["epochs_run"] = len(hist["train"])
    hist["n_train"], hist["n_val"], hist["n_test"] = n_train, n_val, xs.shape[0]
    return TrainedPredictor(model=model, history=hist)


def predict_trace(tp: TrainedPredictor, cfg: Config, trace: TrafficTrace
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """One batched forward pass over a whole trace.

    Returns ``(prediction, normalised absolute error)``, both of length ``K``.
    """
    series = trace.observed
    w = cfg.pred_window
    pad = np.concatenate([np.repeat(series[:1], w - 1), series])
    idx = np.arange(w)[None, :] + np.arange(series.size)[:, None]
    win = pad[idx]
    scale = win.mean(axis=1, keepdims=True)
    x = torch.tensor((win / scale - 1.0)[:, :, None], dtype=torch.float32)
    with torch.no_grad():
        out = (tp.model(x).numpy() + 1.0) * scale[:, 0]
    truth = _future_mean(trace.total(), cfg.pred_horizon)
    err = np.abs(out - truth) / np.maximum(truth, 1e-9)
    return out, err


def persistence_predict(cfg: Config, trace: TrafficTrace
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """A1 ablation: prediction replaced by the last observed load."""
    out = np.concatenate([trace.observed[:1], trace.observed[:-1]])
    truth = _future_mean(trace.total(), cfg.pred_horizon)
    err = np.abs(out - truth) / np.maximum(truth, 1e-9)
    return out, err


def oracle_predict(cfg: Config, trace: TrafficTrace
                   ) -> Tuple[np.ndarray, np.ndarray]:
    truth = _future_mean(trace.total(), cfg.pred_horizon)
    return truth, np.zeros_like(truth)
