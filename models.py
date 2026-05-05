"""Forecast baselines and metric helpers."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def pinball(y, yhat, q):
    """Quantile (pinball) loss at level q in (0,1)."""
    e = np.asarray(y) - np.asarray(yhat)
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def coverage(y, lo, hi):
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def make_features(df, target="price_aud_mwh", horizon=1,
                  rain_lags=(1, 3, 7, 14, 28),
                  price_lags=(1, 7),
                  storage_lags=(1, 7, 28)):
    """Build a tidy feature matrix for tree-based models.

    Predicts price h days ahead given today's storage, recent rainfall by
    catchment, demand, calendar, Basslink, and lagged prices."""
    out = df.copy()
    rain_cols = [c for c in df.columns if c.startswith("rain_") and c.endswith("_mm")]
    # rain features are rolling cumulative sums ending YESTERDAY (shift 1):
    # "how much rain fell over the last L days, observable at forecast time".
    # The naming convention `_lag{L}` is kept for backwards compat — read it as
    # "L-day cumulative rain, lagged by 1 to be causal".
    for L in rain_lags:
        for c in rain_cols:
            out[f"{c}_sum{L}"] = out[c].rolling(L, min_periods=1).sum().shift(1)
    # Today's price is observable at forecast time (we forecast t+horizon
    # given everything up to and including t), so include it as price_lag0.
    out["price_lag0"] = out[target]
    for L in price_lags:
        out[f"price_lag{L}"] = out[target].shift(L)
    for L in storage_lags:
        out[f"storage_lag{L}"] = out["storage_gwh"].shift(L)
    out["doy_sin"] = np.sin(2 * np.pi * out.index.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out.index.dayofyear / 365.25)
    out["dow"] = out.index.dayofweek
    out["y"] = out[target].shift(-horizon)
    out = out.dropna()
    feature_cols = [c for c in out.columns if c not in (target, "y")
                    and out[c].dtype.kind in "fi"]
    return out[feature_cols].values.astype(np.float32), out["y"].values.astype(np.float32), \
        out.index, feature_cols


def time_split(n, train_frac=0.7, val_frac=0.15):
    n_tr = int(n * train_frac)
    n_va = int(n * val_frac)
    return slice(0, n_tr), slice(n_tr, n_tr + n_va), slice(n_tr + n_va, n)


def fit_gbr(X, y, tr, va, max_iter=400, max_depth=6, lr=0.05, seed=0,
            quantile=None):
    loss = "quantile" if quantile is not None else "squared_error"
    kwargs = dict(loss=loss, learning_rate=lr, max_iter=max_iter,
                  max_depth=max_depth, random_state=seed,
                  early_stopping=True, validation_fraction=0.15)
    if quantile is not None:
        kwargs["quantile"] = quantile
    m = HistGradientBoostingRegressor(**kwargs)
    m.fit(X[tr], y[tr])
    return m


def perfect_foresight(y_true, horizon=1):
    """Trivial benchmark: knowing the future, the best constant-shift
    predictor is the truth itself. Reported as the unattainable lower bound."""
    return np.asarray(y_true)


def naive_lag(y, horizon=1):
    """Yesterday-as-tomorrow benchmark: y_hat_{t+h} = y_t."""
    out = np.empty_like(y, dtype=float)
    out[:] = np.nan
    out[horizon:] = y[:-horizon]
    return out
