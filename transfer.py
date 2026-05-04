"""Distributed-lag impulse-response estimation by L1-regularised regression.

The model is

    y_t = c + sum_{k=0..K} b_k x_{t-k} + sum_{j} a_j z_j_t + e_t

with Lasso on the b_k so the impulse response is sparse and a small set
of always-included controls z (calendar, lagged y, demand etc.). We
report the IRF coefficients with bootstrap percentile intervals."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


def _build_lag_matrix(x, max_lag):
    n = len(x)
    M = np.zeros((n, max_lag + 1))
    for k in range(max_lag + 1):
        M[k:, k] = x[:n - k] if k > 0 else x
    return M


def fit_irf(x, y, max_lag=90, controls=None, alphas=None, seed=0):
    """Single-fit Lasso IRF estimate.

    Returns (lags, irf_coef, intercept, control_coefs, model)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    L = _build_lag_matrix(x, max_lag)
    if controls is not None:
        Z = np.asarray(controls, dtype=float)
        if Z.ndim == 1:
            Z = Z[:, None]
        X = np.hstack([L, Z])
    else:
        X = L
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    Xv, yv = X[valid], y[valid]
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(Xv)
    if alphas is None:
        alphas = np.geomspace(1e-3, 5e-1, 25)
    m = LassoCV(alphas=alphas, cv=5, max_iter=20000, random_state=seed)
    m.fit(Xs, yv - yv.mean())
    coef = m.coef_ / scaler.scale_
    intercept = yv.mean() - coef @ scaler.mean_
    lags = np.arange(max_lag + 1)
    irf = coef[:max_lag + 1]
    ctrl = coef[max_lag + 1:]
    return lags, irf, float(intercept), ctrl, m


def bootstrap_irf(x, y, max_lag=90, n_boot=200, block=14, controls=None,
                  seed=0):
    """Block-bootstrap percentile intervals for the IRF."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    n_blocks = int(np.ceil(n / block))
    lags = np.arange(max_lag + 1)
    boots = np.empty((n_boot, max_lag + 1))
    for b in range(n_boot):
        starts = rng.integers(0, n - block, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        ctrl_b = controls[idx] if controls is not None else None
        try:
            _, irf, *_ = fit_irf(x[idx], y[idx], max_lag=max_lag,
                                 controls=ctrl_b, seed=seed + b)
            boots[b] = irf
        except Exception:
            boots[b] = np.nan
    return (lags,
            np.nanmedian(boots, axis=0),
            np.nanpercentile(boots, 5, axis=0),
            np.nanpercentile(boots, 95, axis=0))


def median_lag(lags, irf):
    """The lag k* at which the running |IRF| sum first crosses 50%."""
    w = np.abs(irf)
    if w.sum() == 0:
        return float("nan")
    csum = np.cumsum(w) / w.sum()
    k = np.searchsorted(csum, 0.5)
    return float(lags[min(k, len(lags) - 1)])
