"""Plot helpers for the rainfall->price notebook."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ACCENT = "#2c5f8d"
WARN = "#c44e52"
GOOD = "#3a8c5f"
ACCENT2 = "#7e57c2"
GREY = "#7f7f7f"


def apply_style():
    sns.set_theme(style="whitegrid", context="notebook")
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "axes.titleweight": "semibold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    })


def caption(fig, text):
    fig.text(0.5, -0.04, text, ha="center", va="top",
             fontsize=9, color="#555", style="italic", wrap=True)


def four_panel_overview(df, ax=None):
    """Rainfall (catchment-mean) / flow / storage / price stacked over time."""
    if ax is None:
        fig, ax = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    ax[0].fill_between(df.index, 0, df["rain_mean_mm"], color=ACCENT, alpha=0.7)
    ax[0].set_ylabel("rain (mm/day)")
    ax[1].plot(df.index, df["flow_mean_cms"], color=GOOD, lw=0.8)
    ax[1].set_ylabel("flow (m³/s)")
    ax[2].plot(df.index, df["storage_gwh"], color=ACCENT2, lw=1.0)
    ax[2].set_ylabel("storage (GWh)")
    ax[3].plot(df.index, df["price_aud_mwh"], color=WARN, lw=0.6)
    ax[3].set_ylabel("TAS1 price ($/MWh)")
    ax[3].set_xlabel("")
    return ax


def coherence_heatmap(coh, periods_days, times, ax=None, title=""):
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4))
    extent = [0, coh.shape[1], periods_days[0], periods_days[-1]]
    im = ax.imshow(coh, aspect="auto", origin="lower",
                   cmap="magma", vmin=0, vmax=1, extent=extent)
    ax.set_yscale("log")
    ax.set_ylabel("period (days)")
    ax.set_xlabel("time index (days)")
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, label="wavelet coherence")
    return ax


def irf_plot(lags, irf_med, irf_lo, irf_hi, ax=None, color=ACCENT,
             label=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(lags, irf_lo, irf_hi, color=color, alpha=0.2)
    ax.plot(lags, irf_med, color=color, lw=1.4, label=label)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("lag (days)")
    ax.set_ylabel("response")
    if label:
        ax.legend()
    return ax


def lag_forest(rows, ax=None):
    """Forest plot: each row = (label, median lag, lo, hi)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.4 * len(rows) + 1.5))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        ax.plot([r[2], r[3]], [yi, yi], color=ACCENT, lw=2.5)
        ax.plot([r[1]], [yi], "o", color=WARN, ms=7)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("lag (days)")
    ax.axvline(0, color="black", lw=0.5)
    return ax


def horizon_table(results, ax=None):
    """Bar chart of RMSE per (model, horizon)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.5))
    horizons = list(results.keys())
    models = list(results[horizons[0]].keys())
    width = 0.8 / len(models)
    palette = [ACCENT, WARN, GOOD, ACCENT2, GREY]
    for i, m in enumerate(models):
        vals = [results[h][m] for h in horizons]
        x = np.arange(len(horizons)) + i * width
        ax.bar(x, vals, width=width, color=palette[i % len(palette)], label=m)
    ax.set_xticks(np.arange(len(horizons)) + width * (len(models) - 1) / 2)
    ax.set_xticklabels([f"{h}d" for h in horizons])
    ax.set_ylabel("RMSE ($/MWh)")
    ax.set_xlabel("forecast horizon")
    ax.legend()
    return ax


def storage_sensitivity(deciles, response, lo, hi, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(deciles, lo, hi, color=ACCENT, alpha=0.2)
    ax.plot(deciles, response, "-o", color=ACCENT, lw=1.6, ms=6)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("storage decile (% of capacity)")
    ax.set_ylabel("price response to +10mm rainfall ($/MWh)")
    return ax


def counterfactual(dates, actual, scenarios, ax=None):
    """Realised price plus three counterfactual price paths overlaid."""
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4.2))
    ax.plot(dates, actual, color="black", lw=1.0, label="realised")
    palette = {"actual rainfall": ACCENT,
               "dry (2019) rainfall": WARN,
               "wet (2022) rainfall": GOOD}
    for name, path in scenarios.items():
        ax.plot(dates, path, color=palette.get(name, ACCENT2),
                lw=1.0, label=name, alpha=0.85)
    ax.set_ylabel("TAS1 price ($/MWh)")
    ax.legend()
    return ax


def catchment_map(catchments, ax=None):
    """Schematic Tasmania outline + catchment markers + reservoirs."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    tas_x = [144.6, 148.5, 148.4, 145.5, 144.6]
    tas_y = [-40.6, -40.6, -43.7, -43.7, -40.6]
    ax.fill(tas_x, tas_y, color="#e6e9ef", ec=GREY, lw=1.2)
    for c in catchments:
        ax.scatter(c["lon"], c["lat"], s=120, color=ACCENT,
                   edgecolor="black", zorder=3)
        ax.annotate(c["name"], (c["lon"], c["lat"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect("equal")
    return ax


def climatology_grid(rain_by_catchment, ax=None):
    n = len(rain_by_catchment)
    if ax is None:
        fig, ax = plt.subplots(1, n, figsize=(3.2 * n, 3), sharey=True)
    for axi, (name, series) in zip(ax, rain_by_catchment.items()):
        clim = series.groupby(series.index.dayofyear).mean()
        axi.fill_between(clim.index, 0, clim.values, color=ACCENT, alpha=0.7)
        axi.set_title(name, fontsize=10)
        axi.set_xlabel("day of year")
    ax[0].set_ylabel("mean daily rain (mm)")
    return ax


def storage_breakdown(df, ax=None):
    cols = [c for c in df.columns if c.endswith("_gwh") and c != "storage_gwh"]
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 3.8))
    ax.stackplot(df.index, [df[c].values for c in cols],
                 labels=[c.replace("_gwh", "") for c in cols],
                 alpha=0.85)
    ax.set_ylabel("energy in storage (GWh)")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    return ax


def trace_density(samples, names, ax=None):
    n = len(names)
    if ax is None:
        fig, ax = plt.subplots(n, 2, figsize=(11, 1.8 * n))
    for i, name in enumerate(names):
        s = np.asarray(samples[name])
        if s.ndim > 1:
            s = s.reshape(s.shape[0], -1).mean(axis=1)
        ax[i, 0].plot(s, color=ACCENT, lw=0.6)
        ax[i, 0].set_ylabel(name)
        ax[i, 1].hist(s, bins=40, color=ACCENT, alpha=0.85)
        ax[i, 1].set_yticks([])
        ax[i, 1].axvline(np.median(s), color=WARN, lw=1.0)
    ax[0, 0].set_title("trace")
    ax[0, 1].set_title("posterior density")
    ax[-1, 0].set_xlabel("MCMC iteration")
    ax[-1, 1].set_xlabel("value")
    return ax
