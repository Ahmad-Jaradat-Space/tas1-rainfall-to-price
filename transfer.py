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


def fit_irf(x, y, max_lag=90, controls=None, alphas=None, seed=0,
            standardise_y=True):
    """Single-fit Lasso IRF estimate.

    The response y is z-scored for fitting so the regularisation strength
    is comparable across response variables of different magnitudes (the
    storage-change response is in GWh; the price response is dimensionless
    log). Coefficients are returned on the original units of x and y.
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
    y_mean = yv.mean()
    y_std = yv.std() if standardise_y else 1.0
    y_target = (yv - y_mean) / max(y_std, 1e-9)
    if alphas is None:
        alphas = np.geomspace(1e-4, 1e-1, 30)
    m = LassoCV(alphas=alphas, cv=5, max_iter=30000, random_state=seed)
    m.fit(Xs, y_target)
    coef_native = m.coef_ * y_std / scaler.scale_
    intercept = y_mean - coef_native @ scaler.mean_
    lags = np.arange(max_lag + 1)
    irf = coef_native[:max_lag + 1]
    ctrl = coef_native[max_lag + 1:]
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


def lag_interval(lags, boots, q_lo=5, q_hi=95):
    """Median lag plus its bootstrap interval, computed from the boot matrix."""
    n = boots.shape[0]
    med_lags = np.zeros(n)
    for i in range(n):
        med_lags[i] = median_lag(lags, boots[i])
    finite = med_lags[np.isfinite(med_lags)]
    if len(finite) == 0:
        return float("nan"), float("nan"), float("nan")
    return (float(np.median(finite)),
            float(np.percentile(finite, q_lo)),
            float(np.percentile(finite, q_hi)))


def bootstrap_irf_full(x, y, max_lag=90, n_boot=200, block=14, controls=None,
                       seed=0):
    """Same as bootstrap_irf but also returns the full boot matrix."""
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
            np.nanpercentile(boots, 95, axis=0),
            boots)
