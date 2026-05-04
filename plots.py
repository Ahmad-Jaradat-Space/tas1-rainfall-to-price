"""Production-grade plot helpers for the rainfall->price notebook.

The colour system follows a deliberate hierarchy:
  PRIMARY (deep teal)  -> headline series, the thing being explained
  ACCENT (warm clay)   -> the comparison series, or the explanatory regime
  GOOD   (muted sage)  -> positive / wet regime
  WARN   (rust)        -> negative / dry regime / outlier highlight
  MUTED  (slate grey)  -> baselines, naive benchmarks, geography fills
"""

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PRIMARY = "#0E4F5F"
ACCENT  = "#D88C4A"
GOOD    = "#5C9D7E"
WARN    = "#B0413E"
MUTED   = "#7A8C99"
INK     = "#1A1A1A"
PAPER   = "#FAFAF7"
RAIN    = "#3F7CAC"
STORAGE = "#5E548E"

CATCHMENT_PALETTE = {
    "Pieman":        "#1F6F8B",
    "Gordon-Pedder": "#5E548E",
    "Mersey-Forth":  "#5C9D7E",
    "Derwent":       "#D88C4A",
    "Great Lake":    "#B0413E",
}

STORAGE_PALETTE = {
    "gordon":       "#5E548E",
    "great_lake":   "#B0413E",
    "west_coast":   "#1F6F8B",
    "mersey_forth": "#5C9D7E",
    "derwent":      "#D88C4A",
}


def apply_style():
    sns.set_theme(style="white", context="notebook")
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 140,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "savefig.edgecolor": PAPER,
        "axes.titleweight": "semibold",
        "axes.titlesize": 12.5,
        "axes.titlepad": 12,
        "axes.titlelocation": "left",
        "axes.labelsize": 10.5,
        "axes.labelweight": "regular",
        "axes.labelcolor": INK,
        "axes.edgecolor": "#BFC4CA",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "font.family": "sans-serif",
        "font.size": 10.5,
        "axes.unicode_minus": True,
    })
    try:
        mpl.rcParams["text.parse_math"] = False
    except KeyError:
        pass


# ----------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------
def annotate_event(ax, x, y, text, dx=20, dy=20, color=INK):
    ax.annotate(
        text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        fontsize=9, color=color, ha="left",
        arrowprops=dict(arrowstyle="-", lw=0.7, color=color, alpha=0.7),
        bbox=dict(boxstyle="round,pad=0.25", fc=PAPER, ec=color, lw=0.6, alpha=0.95),
        path_effects=[pe.withStroke(linewidth=2.5, foreground=PAPER)],
    )


def date_axis(ax, freq="6MS"):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(0)


def kpi_card(ax, value, label, sub=None, color=PRIMARY):
    ax.axis("off")
    ax.text(0.5, 0.62, value, ha="center", va="center",
            fontsize=22, color=color, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.30, label, ha="center", va="center",
            fontsize=10, color=INK, transform=ax.transAxes)
    if sub:
        ax.text(0.5, 0.12, sub, ha="center", va="center",
                fontsize=8.5, color=MUTED, transform=ax.transAxes, style="italic")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#D8DCE2")
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                    fill=False, ec="#D8DCE2", lw=1.0))


def kpi_banner(values):
    """Top-of-section KPI strip. `values` is a list of (value, label, sub)."""
    fig, axes = plt.subplots(1, len(values), figsize=(2.8 * len(values), 1.8))
    if len(values) == 1:
        axes = [axes]
    palette = [PRIMARY, ACCENT, STORAGE, GOOD, WARN]
    for ax, v, color in zip(axes, values, palette):
        kpi_card(ax, *v, color=color)
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Section 2: data overview
# ----------------------------------------------------------------------
def hero_overview(df, events=None, ax=None):
    """Two-panel storage + price with annotated regime events."""
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(13, 5.6), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1]})
    ax[0].fill_between(df.index, 4000, df["storage_gwh"],
                       color=STORAGE, alpha=0.15, lw=0)
    ax[0].plot(df.index, df["storage_gwh"], color=STORAGE, lw=1.6)
    ax[0].set_ylabel("Energy in storage (GWh)")
    ax[0].set_title("Storage drains in summer, refills in winter; prices respond on a lag",
                    color=INK)
    ax[1].plot(df.index, df["price_aud_mwh"], color=WARN, lw=0.6, alpha=0.85)
    rolling = df["price_aud_mwh"].rolling(30, min_periods=10).mean()
    ax[1].plot(df.index, rolling, color=INK, lw=1.4, label="30-day mean")
    ax[1].set_ylabel("TAS1 price ($/MWh)")
    ax[1].legend(loc="upper left")
    if events:
        for x, label, panel, dy in events:
            xp = pd.Timestamp(x)
            yval = (df.loc[xp, "storage_gwh"] if panel == 0
                    else float(rolling.loc[xp]))
            annotate_event(ax[panel], xp, yval, label, dx=15, dy=dy,
                           color=PRIMARY if panel == 0 else WARN)
    date_axis(ax[1])
    plt.tight_layout()
    return ax


def four_panel_chain(df, ax=None):
    """Rainfall / flow / storage / price stacked over time — the chain."""
    if ax is None:
        fig, ax = plt.subplots(4, 1, figsize=(13, 9.5), sharex=True)
    rolling = df["rain_mean_mm"].rolling(7, min_periods=1).mean()
    ax[0].fill_between(df.index, 0, rolling, color=RAIN, alpha=0.55, lw=0)
    ax[0].plot(df.index, rolling, color=RAIN, lw=0.8)
    ax[0].set_ylabel("Rainfall (mm/day,\n7-day mean)")
    ax[0].set_title("The chain in four series: rain in, flow up, storage up, price down (with a delay)",
                    color=INK)
    ax[1].fill_between(df.index, 0, df["flow_mean_cms"], color=GOOD, alpha=0.30, lw=0)
    ax[1].plot(df.index, df["flow_mean_cms"], color=GOOD, lw=0.9)
    ax[1].set_ylabel("Flow (m³/s)")
    ax[2].plot(df.index, df["storage_gwh"], color=STORAGE, lw=1.4)
    ax[2].fill_between(df.index, df["storage_gwh"].min() - 200,
                       df["storage_gwh"], color=STORAGE, alpha=0.10, lw=0)
    ax[2].set_ylabel("Storage (GWh)")
    ax[3].plot(df.index, df["price_aud_mwh"], color=WARN, lw=0.5, alpha=0.7)
    ax[3].plot(df.index, df["price_aud_mwh"].rolling(30, min_periods=10).mean(),
               color=INK, lw=1.4)
    ax[3].set_ylabel("TAS1 price ($/MWh)")
    date_axis(ax[3])
    for a in ax:
        a.margins(x=0.005)
    plt.tight_layout()
    return ax


def catchment_map(catchments, ax=None):
    """Schematic Tasmania with catchment markers + reservoirs."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 8.5))
    tas_x = [144.6, 148.5, 148.4, 145.5, 144.6]
    tas_y = [-40.6, -40.6, -43.7, -43.7, -40.6]
    ax.fill(tas_x, tas_y, color="#EAEEF1", ec=MUTED, lw=1.0, zorder=1)
    ax.plot(tas_x, tas_y, color=MUTED, lw=1.0, zorder=2)
    sizes = [c["share"] * 1500 for c in catchments]
    for c, s in zip(catchments, sizes):
        col = CATCHMENT_PALETTE.get(c["name"], PRIMARY)
        ax.scatter(c["lon"], c["lat"], s=s, color=col,
                   edgecolor=INK, lw=1.2, alpha=0.92, zorder=4)
        ax.annotate(f"{c['name']}\n{int(c['share']*100)}% of inflow",
                    (c["lon"], c["lat"]),
                    textcoords="offset points", xytext=(10, 8),
                    fontsize=9.5, color=INK,
                    bbox=dict(boxstyle="round,pad=0.3", fc=PAPER,
                              ec=col, lw=0.8, alpha=0.92))
    ax.text(146.5, -40.85, "Bass Strait", fontsize=10, color=MUTED, style="italic")
    ax.text(146.5, -43.5, "Southern Ocean", fontsize=10, color=MUTED, style="italic")
    ax.set_xlim(144.4, 148.6)
    ax.set_ylim(-43.85, -40.45)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.set_title("Five Hydro-Tas catchments, sized by share of system inflow",
                 color=INK)
    ax.grid(alpha=0.10)
    plt.tight_layout()
    return ax


def climatology_grid(rain_by_catchment, ax=None, smooth=15):
    """Per-catchment seasonal rainfall climatology, smoothed."""
    n = len(rain_by_catchment)
    if ax is None:
        fig, ax = plt.subplots(1, n, figsize=(2.7 * n, 3.0), sharey=True)
    for axi, (name, series) in zip(ax, rain_by_catchment.items()):
        clim = series.groupby(series.index.dayofyear).mean()
        clim = clim.rolling(smooth, center=True, min_periods=1).mean()
        col = CATCHMENT_PALETTE.get(name, PRIMARY)
        axi.fill_between(clim.index, 0, clim.values, color=col, alpha=0.55, lw=0)
        axi.plot(clim.index, clim.values, color=col, lw=1.2)
        axi.set_title(name, fontsize=10.5, color=INK)
        axi.set_xlabel("Day of year")
        axi.set_xticks([1, 91, 182, 273])
        axi.set_xticklabels(["Jan", "Apr", "Jul", "Oct"])
    ax[0].set_ylabel("Mean daily rain (mm)")
    plt.tight_layout()
    return ax


def storage_breakdown(df, ax=None):
    """Stacked area of sub-system storage with a tidy legend."""
    cols = [c for c in df.columns if c.endswith("_gwh") and c != "storage_gwh"]
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 4.2))
    palette = [STORAGE_PALETTE.get(c.replace("_gwh", ""), PRIMARY) for c in cols]
    ax.stackplot(df.index,
                 [df[c].values for c in cols],
                 labels=[c.replace("_gwh", "").replace("_", " ").title()
                         for c in cols],
                 colors=palette, alpha=0.85)
    ax.plot(df.index, df["storage_gwh"], color=INK, lw=1.0,
            label="System total", linestyle="--", alpha=0.6)
    ax.set_ylabel("Energy in storage (GWh)")
    ax.set_title("System storage decomposed: Gordon and Great Lake hold most of the energy",
                 color=INK)
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    date_axis(ax)
    ax.margins(x=0.005)
    plt.tight_layout()
    return ax


def price_distribution(df, ax=None):
    """Price distribution with TAS1-specific tail annotation."""
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
    p = df["price_aud_mwh"].clip(lower=-50, upper=500)
    ax[0].hist(p, bins=80, color=WARN, alpha=0.7, edgecolor=PAPER, lw=0.6)
    q05, q50, q95 = np.percentile(p, [5, 50, 95])
    for q, label, c in [(q05, "5%", MUTED), (q50, "median", INK), (q95, "95%", PRIMARY)]:
        ax[0].axvline(q, color=c, lw=1.0, ls="--")
        ax[0].text(q, ax[0].get_ylim()[1] * 0.95, f" {label}: ${q:.0f}",
                   color=c, fontsize=9, va="top")
    ax[0].set_xlabel("Daily TAS1 price ($/MWh, clipped at $500)")
    ax[0].set_ylabel("Days")
    ax[0].set_title("Daily price distribution: a long, expensive right tail",
                    color=INK)

    storage_decile = pd.qcut(df["storage_gwh"], 10, labels=False)
    by_dec = df.groupby(storage_decile)["price_aud_mwh"].median()
    cmap = mpl.colormaps["viridis"]
    colors = [cmap(0.1 + 0.8 * d / 9) for d in range(10)]
    ax[1].bar(np.arange(10) + 1, by_dec.values, color=colors,
              edgecolor=INK, lw=0.5)
    for i, v in enumerate(by_dec.values):
        ax[1].text(i + 1, v + 2, f"${v:.0f}", ha="center", fontsize=8.5, color=INK)
    ax[1].set_xticks(np.arange(10) + 1)
    ax[1].set_xlabel("Storage decile (1 = driest, 10 = wettest)")
    ax[1].set_ylabel("Median daily price ($/MWh)")
    ax[1].set_title("Median price falls monotonically as storage rises",
                    color=INK)
    plt.tight_layout()
    return ax


# ----------------------------------------------------------------------
# Section 3.1: spectral
# ----------------------------------------------------------------------
def coherence_heatmap(coh, periods_days, dates, ax=None, title=""):
    """Wavelet coherence with a real date axis and reference period bands."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 4.2))
    t_num = mdates.date2num(dates)
    extent = [t_num[0], t_num[-1], np.log10(periods_days[0]), np.log10(periods_days[-1])]
    im = ax.imshow(coh, aspect="auto", origin="lower",
                   cmap="magma", vmin=0, vmax=1, extent=extent)
    ref_periods = [7, 30, 90, 365]
    for p in ref_periods:
        if periods_days[0] <= p <= periods_days[-1]:
            ax.axhline(np.log10(p), color="white", lw=0.5, ls="--", alpha=0.5)
            ax.text(t_num[-1], np.log10(p), f" {p}d", color="white",
                    fontsize=8.5, va="center", ha="left",
                    bbox=dict(boxstyle="round,pad=0.18", fc="black",
                              ec="white", lw=0.4, alpha=0.7))
    label_periods = [p for p in [14, 30, 60, 90, 180] if periods_days[0] <= p <= periods_days[-1]]
    yticks = [np.log10(p) for p in label_periods]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{p}d" for p in label_periods])
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_ylabel("Period")
    if title:
        ax.set_title(title, color=INK)
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Wavelet coherence", fontsize=9.5)
    return ax


# ----------------------------------------------------------------------
# Section 3.2: IRF
# ----------------------------------------------------------------------
def irf_plot(lags, irf_med, irf_lo, irf_hi, ax=None, color=PRIMARY,
             label=None, title=None, annotate_peak=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(lags, irf_lo, irf_hi, color=color, alpha=0.20, lw=0)
    ax.plot(lags, irf_med, color=color, lw=2.0, label=label)
    ax.axhline(0, color=INK, lw=0.7, alpha=0.5)
    if annotate_peak:
        peak_idx = int(np.nanargmax(np.abs(irf_med)))
        ax.scatter([lags[peak_idx]], [irf_med[peak_idx]],
                   s=70, color=WARN, zorder=5, edgecolor=INK, lw=0.8)
        ax.annotate(f"peak day {lags[peak_idx]}",
                    xy=(lags[peak_idx], irf_med[peak_idx]),
                    xytext=(15, 10), textcoords="offset points",
                    fontsize=9, color=WARN,
                    arrowprops=dict(arrowstyle="-", color=WARN, lw=0.6))
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Response")
    if title:
        ax.set_title(title, color=INK)
    if label:
        ax.legend(loc="best")
    return ax


def lag_forest(rows, ax=None):
    """Forest plot: each row = (label, median_lag, lo_lag, hi_lag)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 0.7 * len(rows) + 1.5))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        ax.plot([r[2], r[3]], [yi, yi], color=PRIMARY, lw=4, alpha=0.55,
                solid_capstyle="round")
        ax.scatter([r[1]], [yi], s=110, color=WARN, zorder=5,
                   edgecolor=INK, lw=0.8)
        ax.text(r[3] + (r[3] - r[2]) * 0.05 + 1, yi,
                f"  {r[1]:.0f}d  [{r[2]:.0f}, {r[3]:.0f}]",
                va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    ax.set_xlabel("Lag (days)")
    ax.axvline(0, color=INK, lw=0.6, alpha=0.5)
    ax.set_title("End-to-end chain delay: rainfall to reservoir, reservoir to price",
                 color=INK)
    return ax


# ----------------------------------------------------------------------
# Section 3.3: SSM diagnostics
# ----------------------------------------------------------------------
def trace_density(samples, names, labels=None, ax=None):
    if labels is None:
        labels = names
    n = len(names)
    if ax is None:
        fig, ax = plt.subplots(n, 2, figsize=(12, 1.7 * n),
                               gridspec_kw={"width_ratios": [2, 1]})
    for i, (name, lbl) in enumerate(zip(names, labels)):
        s = np.asarray(samples[name])
        if s.ndim > 1:
            s = s.reshape(s.shape[0], -1).mean(axis=1)
        ax[i, 0].plot(s, color=PRIMARY, lw=0.6)
        ax[i, 0].set_ylabel(lbl, fontsize=10)
        ax[i, 1].hist(s, bins=40, color=PRIMARY, alpha=0.85, edgecolor=PAPER, lw=0.4)
        ax[i, 1].set_yticks([])
        ax[i, 1].axvline(np.median(s), color=WARN, lw=1.2)
        q05, q95 = np.percentile(s, [5, 95])
        ax[i, 1].axvspan(q05, q95, color=WARN, alpha=0.10)
    ax[0, 0].set_title("Trace", color=INK)
    ax[0, 1].set_title("Posterior", color=INK)
    ax[-1, 0].set_xlabel("MCMC iteration")
    ax[-1, 1].set_xlabel("Value")
    plt.tight_layout()
    return ax


def storage_response_curve(deciles, response, lo, hi, ax=None,
                           threshold_decile=None):
    """The headline 'claim 2' chart: price response by storage decile."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(deciles, lo, hi, color=PRIMARY, alpha=0.18, lw=0,
                    label="90% posterior interval")
    ax.plot(deciles, response, "-o", color=PRIMARY, lw=2.0, ms=8,
            mec=INK, mew=0.7, label="posterior median")
    ax.axhline(0, color=INK, lw=0.7, alpha=0.5)
    if threshold_decile is not None:
        ax.axvline(threshold_decile, color=WARN, ls="--", lw=1.2)
        ax.text(threshold_decile, ax.get_ylim()[1] * 0.92,
                f"  storage threshold ~{threshold_decile:.0f}%",
                color=WARN, fontsize=9.5, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc=PAPER,
                          ec=WARN, lw=0.8, alpha=0.95))
    ymin = np.min(lo) * 1.05 if np.min(lo) < 0 else np.min(lo) * 0.95
    ax.set_ylim(ymin, max(np.max(hi) * 1.15, 1.0))
    ax.set_xlabel("Storage decile (% of full-supply capacity)")
    ax.set_ylabel("Price response to +10mm rainfall ($/MWh)")
    ax.legend(loc="lower right")
    return ax


# ----------------------------------------------------------------------
# Section 3.4: forecast comparison
# ----------------------------------------------------------------------
def horizon_comparison(results, ax=None):
    """Grouped bar chart: RMSE per (model, horizon) with values inside bars."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))
    horizons = list(results.keys())
    models = list(results[horizons[0]].keys())
    palette = {
        "Naive":  MUTED,
        "GBR":    PRIMARY,
        "SSM":    STORAGE,
        "GCN":    GOOD,
    }
    width = 0.78 / len(models)
    for i, m in enumerate(models):
        vals = [results[h][m] for h in horizons]
        x = np.arange(len(horizons)) + (i - (len(models) - 1) / 2) * width
        col = palette.get(m, ACCENT)
        bars = ax.bar(x, vals, width=width, color=col,
                      edgecolor=INK, lw=0.4, label=m, alpha=0.92)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}",
                    ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(np.arange(len(horizons)))
    ax.set_xticklabels([f"{h}-day" for h in horizons])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Test-set RMSE ($/MWh)")
    ax.legend(loc="upper left", ncol=len(models))
    ax.set_title("Forecast accuracy by horizon: every model beats naive; SSM closes the gap to GBR at 28 days",
                 color=INK)
    return ax


def quantile_fan(dates, y_true, q10, q50, q90, ax=None, label="GBR"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(dates, q10, q90, color=PRIMARY, alpha=0.20, lw=0,
                    label=f"{label} 10–90% band")
    ax.plot(dates, q50, color=PRIMARY, lw=1.5, label=f"{label} median")
    ax.plot(dates, y_true, color=INK, lw=0.8, label="realised")
    ax.set_ylabel("Price ($/MWh)")
    date_axis(ax)
    ax.legend(loc="upper left")
    return ax


def feature_importance(features, importances, ax=None, top=10, horizon_label="next-day"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    order = np.argsort(importances)[-top:]
    y = np.arange(len(order))
    ax.barh(y, importances[order], color=PRIMARY, edgecolor=INK, lw=0.5, alpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels([features[i] for i in order])
    ax.set_xlabel("Permutation importance (RMSE increase, $/MWh)")
    ax.set_title(f"Top {top} drivers of {horizon_label} TAS1 price under the GBR baseline",
                 color=INK)
    return ax


# ----------------------------------------------------------------------
# Section 3.5: GNN
# ----------------------------------------------------------------------
def gnn_topology(catchment_names, storage_names, ax=None):
    """Schematic of the catchment -> storage -> market graph."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))
    n_c = len(catchment_names)
    n_s = len(storage_names)
    catch_xy = [(0.10, 0.15 + i * (0.70 / max(n_c - 1, 1))) for i in range(n_c)]
    stor_xy = [(0.50, 0.15 + i * (0.70 / max(n_s - 1, 1))) for i in range(n_s)]
    market_xy = (0.90, 0.50)

    def _catch_key(name):
        # accept "pieman" / "Pieman" / "gordon_pedder" / "Gordon-Pedder"
        return {
            "pieman": "Pieman", "gordon_pedder": "Gordon-Pedder",
            "mersey_forth": "Mersey-Forth", "derwent": "Derwent",
            "great_lake": "Great Lake",
        }.get(name.lower().replace("-", "_"), name)

    for cx, sx in zip(catch_xy, stor_xy):
        ax.annotate("", xy=sx, xytext=cx,
                    arrowprops=dict(arrowstyle="-|>", color=RAIN, lw=1.6,
                                    alpha=0.85, mutation_scale=18))
    for sx in stor_xy:
        ax.annotate("", xy=market_xy, xytext=sx,
                    arrowprops=dict(arrowstyle="-|>", color=STORAGE, lw=1.4,
                                    alpha=0.75, mutation_scale=18))
    extra = [(stor_xy[0], stor_xy[1]), (stor_xy[3], stor_xy[4])]
    for a, b in extra:
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.0,
                                    alpha=0.7, ls="--", mutation_scale=12))

    for (x, y), name in zip(catch_xy, catchment_names):
        display_name = _catch_key(name)
        col = CATCHMENT_PALETTE.get(display_name, PRIMARY)
        ax.scatter([x], [y], s=700, color=col, edgecolor=INK, lw=1.0, zorder=4)
        ax.text(x - 0.04, y, display_name, ha="right", va="center",
                fontsize=10, color=INK, fontweight="medium")
    for (x, y), name in zip(stor_xy, storage_names):
        col = STORAGE_PALETTE.get(name.lower().replace(" ", "_"), PRIMARY)
        ax.scatter([x], [y], s=850, color=col, edgecolor=INK, lw=1.0,
                   marker="s", zorder=4)
        ax.text(x, y - 0.07, name.replace("_", " ").title(), ha="center",
                va="top", fontsize=9.5, color=INK)
    ax.scatter(*market_xy, s=1100, color=WARN, edgecolor=INK, lw=1.2,
               marker="D", zorder=4)
    ax.text(market_xy[0] + 0.04, market_xy[1], "TAS1\nmarket",
            ha="left", va="center", fontsize=10, color=INK, fontweight="bold")

    ax.text(0.10, 0.90, "Catchments", ha="center", fontsize=10,
            color=MUTED, fontweight="bold")
    ax.text(0.50, 0.90, "Storage sub-systems", ha="center", fontsize=10,
            color=MUTED, fontweight="bold")
    ax.text(0.90, 0.90, "Market", ha="center", fontsize=10,
            color=MUTED, fontweight="bold")
    ax.text(0.30, 0.05, "rainfall→inflow", color=RAIN, fontsize=9, style="italic")
    ax.text(0.70, 0.05, "storage→price", color=STORAGE, fontsize=9, style="italic")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_title("Catchment-graph topology: edges follow physical water flow",
                 color=INK)
    return ax


# ----------------------------------------------------------------------
# Section 4: counterfactual
# ----------------------------------------------------------------------
def counterfactual(dates, actual, scenarios, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(dates, actual, color=INK, lw=1.4, label="Realised", zorder=4)
    palette = {
        "actual rainfall": PRIMARY,
        "dry rainfall":    WARN,
        "wet rainfall":    GOOD,
    }
    for name, path in scenarios.items():
        col = palette.get(name, ACCENT)
        ax.plot(dates, path, color=col, lw=1.6, label=name, alpha=0.85)
    ax.fill_between(dates, np.minimum.reduce(list(scenarios.values())),
                    np.maximum.reduce(list(scenarios.values())),
                    color=MUTED, alpha=0.10, lw=0)
    ax.set_ylabel("TAS1 price ($/MWh)")
    ax.legend(loc="upper left")
    date_axis(ax)
    return ax


def business_summary(rows, ax=None):
    """A two-column 'so what' table for the conclusion."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 0.55 * len(rows) + 1))
    ax.axis("off")
    n = len(rows)
    for i, (lhs, rhs) in enumerate(rows):
        y = 1 - (i + 0.5) / n
        ax.text(0.02, y, lhs, transform=ax.transAxes,
                fontsize=10.5, color=INK, fontweight="bold", va="center")
        ax.text(0.30, y, rhs, transform=ax.transAxes,
                fontsize=10, color=INK, va="center")
        ax.plot([0.01, 0.99], [1 - i / n, 1 - i / n], color="#E0E4EA",
                lw=0.6, transform=ax.transAxes)
    ax.set_xlim(0, 1)
    return ax
