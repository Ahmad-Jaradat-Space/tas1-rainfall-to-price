"""Data loaders for the rainfall->price chain.

Five sources, all aggregated to daily resolution and joined on date:

1. AEMO TAS1 dispatch price + demand + Basslink flow, via NEMOSIS
2. Hydro Tasmania energy-in-storage, weekly XLS (forward-filled to daily)
3. Open-Meteo ERA5 archive — catchment-mean daily rainfall + 2m temperature
4. Tasmanian WIST river flow gauges — daily mean discharge
5. The joined daily frame that every notebook section consumes

Each loader caches its raw pull under data/cache/ as parquet/feather; the
joined frame is also cached so the notebook is fast on every run after
the first.

If a network source is unreachable, load_joined falls back to a
synthetic-but-physically-coherent series — a hydrological toy that
shares the lag structure of the real chain. This is for development;
the executed notebook on GitHub uses real data.
"""

import io
import os
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "cache")
os.makedirs(CACHE, exist_ok=True)

START = "2020-01-01"
END = "2024-12-31"

CATCHMENTS = [
    {"name": "Pieman",        "lat": -41.80, "lon": 145.50,
     "share": 0.30, "gauge": "Pieman_below_dam"},
    {"name": "Gordon-Pedder", "lat": -42.70, "lon": 146.00,
     "share": 0.28, "gauge": "Gordon_at_Strathgordon"},
    {"name": "Mersey-Forth",  "lat": -41.50, "lon": 146.20,
     "share": 0.16, "gauge": "Forth_at_Wilmot"},
    {"name": "Derwent",       "lat": -42.30, "lon": 146.50,
     "share": 0.18, "gauge": "Derwent_above_Tarraleah"},
    {"name": "Great Lake",    "lat": -41.90, "lon": 146.70,
     "share": 0.08, "gauge": "Liawenee_Canal"},
]

STORAGE_SUBSYSTEMS = ["gordon", "west_coast", "mersey_forth", "derwent",
                      "great_lake"]


# ----------------------------------------------------------------------
# Market data (NEMOSIS)
# ----------------------------------------------------------------------
def load_market(start=START, end=END):
    cache_path = os.path.join(CACHE, "market_daily.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    try:
        from nemosis import dynamic_data_compiler
    except ImportError:
        raise RuntimeError("nemosis not installed; pip install nemosis")
    s = pd.Timestamp(start).strftime("%Y/%m/%d %H:%M:%S")
    e = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y/%m/%d %H:%M:%S")

    price = dynamic_data_compiler(
        s, e, "DISPATCHPRICE", CACHE,
        filter_cols=["REGIONID"], filter_values=(["TAS1"],), keep_csv=False)
    price = price[price["INTERVENTION"] == 0][["SETTLEMENTDATE", "RRP"]]
    demand = dynamic_data_compiler(
        s, e, "DISPATCHREGIONSUM", CACHE,
        filter_cols=["REGIONID"], filter_values=(["TAS1"],), keep_csv=False)
    demand = demand[demand["INTERVENTION"] == 0][
        ["SETTLEMENTDATE", "TOTALDEMAND"]]
    bass = dynamic_data_compiler(
        s, e, "DISPATCHINTERCONNECTORRES", CACHE,
        filter_cols=["INTERCONNECTORID"],
        filter_values=(["T-V-MNSP1"],), keep_csv=False)
    bass = bass[bass["INTERVENTION"] == 0][
        ["SETTLEMENTDATE", "METEREDMWFLOW"]]

    df = price.merge(demand, on="SETTLEMENTDATE").merge(
        bass, on="SETTLEMENTDATE")
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])
    df = df.set_index("SETTLEMENTDATE").sort_index()
    daily = df.resample("D").agg({
        "RRP": "mean",
        "TOTALDEMAND": "mean",
        "METEREDMWFLOW": "mean",
    }).rename(columns={
        "RRP": "price_aud_mwh",
        "TOTALDEMAND": "demand_mw",
        "METEREDMWFLOW": "basslink_mw",
    })
    daily.to_parquet(cache_path)
    return daily


# ----------------------------------------------------------------------
# Energy-in-storage (Hydro Tasmania weekly XLS)
# ----------------------------------------------------------------------
HYDRO_XLS = ("https://www.hydro.com.au/docs/energyinstorage/download/"
             "EnergyInStorage-HistoricalData.xls")


def load_storage(start=START, end=END):
    """Parse the Hydro Tas energy-in-storage XLS.

    The XLS uses a banded layout (basin headers on row 2, lake headers on
    rows 4-5, full-supply row 7, weekly data from row 9). Column indices
    below are stable in the published history."""
    cache_path = os.path.join(CACHE, "storage_daily.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    import requests
    r = requests.get(HYDRO_XLS, timeout=60)
    r.raise_for_status()
    sh = pd.read_excel(io.BytesIO(r.content), sheet_name=0, header=None)

    DATE_COL = 0
    SYSTEM_COL = 21  # 'System (excl. Lake Gardiner et al.)' — the headline figure
    SUBSYSTEM_COLS = {
        "gordon_gwh":       [12],          # Lake Gordon
        "great_lake_gwh":   [5],           # Great Lake / Lake Augusta band
        "west_coast_gwh":   [14, 16, 17, 19],   # Burbury + Murchison + Mackintosh + Plimsoll + Margaret
        "mersey_forth_gwh": [8, 9],        # Mackenzie + Rowallan
        "derwent_gwh":      [1, 2, 3, 6],  # St Clair + Echo + Bronte + Arthurs
    }

    data_rows = sh.iloc[9:].copy()
    data_rows[DATE_COL] = pd.to_datetime(data_rows[DATE_COL], errors="coerce")
    data_rows = data_rows.dropna(subset=[DATE_COL]).set_index(DATE_COL).sort_index()

    out = pd.DataFrame(index=data_rows.index)
    out["storage_gwh"] = pd.to_numeric(data_rows[SYSTEM_COL], errors="coerce")
    for name, cols in SUBSYSTEM_COLS.items():
        vals = data_rows[cols].apply(pd.to_numeric, errors="coerce")
        out[name] = vals.sum(axis=1, min_count=1)

    out = out.dropna(subset=["storage_gwh"])
    out.index.name = "date"
    daily = out.resample("D").interpolate(method="linear").ffill().bfill()
    daily = daily.loc[start:end]
    daily.to_parquet(cache_path)
    return daily


# ----------------------------------------------------------------------
# Catchment rainfall (Open-Meteo)
# ----------------------------------------------------------------------
def _openmeteo_one(lat, lon, start, end):
    import requests
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = dict(
        latitude=lat, longitude=lon,
        start_date=start, end_date=end,
        daily=("precipitation_sum,temperature_2m_mean,"
               "et0_fao_evapotranspiration"),
        timezone="Australia/Hobart",
    )
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(j["time"]),
        "rain": j["precipitation_sum"],
        "temp": j["temperature_2m_mean"],
        "et0":  j["et0_fao_evapotranspiration"],
    }).set_index("date")
    return df


def load_rainfall(start=START, end=END, catchments=CATCHMENTS):
    cache_path = os.path.join(CACHE, "rainfall_daily.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    def _slug(name):
        return name.lower().replace("-", "_").replace(" ", "_")

    pieces = []
    for c in catchments:
        df = _openmeteo_one(c["lat"], c["lon"], start, end)
        df = df.add_prefix(f"{_slug(c['name'])}_")
        pieces.append(df)
    out = pd.concat(pieces, axis=1)
    rain_cols = {_slug(c["name"]) + "_rain": c["share"] for c in catchments}
    out["rain_mean_mm"] = sum(out[k] * w for k, w in rain_cols.items()
                              if k in out.columns)
    rename = {}
    for c in catchments:
        nm = _slug(c["name"])
        rename[f"{nm}_rain"] = f"rain_{nm}_mm"
        rename[f"{nm}_temp"] = f"temp_{nm}_c"
        rename[f"{nm}_et0"]  = f"et_{nm}_mm"
    out = out.rename(columns=rename)
    out.to_parquet(cache_path)
    return out


# ----------------------------------------------------------------------
# Streamflow (WIST). The official portal sometimes blocks programmatic
# access; we keep a manually-cached CSV folder as a robust fallback.
# ----------------------------------------------------------------------
def load_flows(start=START, end=END, catchments=CATCHMENTS):
    """Daily mean discharge per catchment.

    The Tasmanian WIST portal (`portal.wrt.tas.gov.au`) is not amenable to
    programmatic pulls without a per-session click-through; the standard
    route is a one-off CSV export per gauge into
    `data/wist_manual/<gauge>.csv` with columns (date, <gauge_name>).
    When that folder is present this function reads it; when it is not,
    we fall back to a rainfall-derived flow proxy (catchment rainfall
    convolved with a 4-day exponential routing kernel, scaled to typical
    Tasmanian discharge magnitudes). The proxy keeps column names
    identical so the rest of the notebook is agnostic, and a
    `flow_is_proxy` flag column lets downstream analyses filter."""
    cache_path = os.path.join(CACHE, "flows_daily.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    def _slug(name):
        return name.lower().replace("-", "_").replace(" ", "_")

    manual_dir = os.path.join(HERE, "data", "wist_manual")
    if os.path.isdir(manual_dir):
        pieces = []
        for c in catchments:
            f = os.path.join(manual_dir, f"{c['gauge']}.csv")
            if not os.path.exists(f):
                continue
            df = pd.read_csv(f, parse_dates=["date"]).set_index("date")
            df.columns = [f"flow_{_slug(c['name'])}_cms"]
            pieces.append(df)
        if pieces:
            out = pd.concat(pieces, axis=1).resample("D").mean()
            cols = [c for c in out.columns if c.startswith("flow_")]
            out["flow_mean_cms"] = out[cols].mean(axis=1)
            out["flow_is_proxy"] = 0
            out = out.loc[start:end]
            out.to_parquet(cache_path)
            return out

    rain = load_rainfall(start, end, catchments)
    kernel = np.exp(-np.arange(20) / 4.0)
    kernel /= kernel.sum()
    out = pd.DataFrame(index=rain.index)
    for c in catchments:
        s = _slug(c["name"])
        r = rain[f"rain_{s}_mm"].values
        flow = np.convolve(r, kernel, mode="full")[:len(r)] * 8.0
        out[f"flow_{s}_cms"] = flow
    cols = [c for c in out.columns if c.startswith("flow_")]
    out["flow_mean_cms"] = out[cols].mean(axis=1)
    out["flow_is_proxy"] = 1
    out = out.loc[start:end]
    out.to_parquet(cache_path)
    return out


# ----------------------------------------------------------------------
# Synthetic fallback — physically-coherent toy chain
# ----------------------------------------------------------------------
def _synthetic_joined(start=START, end=END, seed=0):
    """A toy of the rainfall->storage->price chain with realistic lags.

    Used only when external data is unreachable. It is deterministic given
    `seed`, includes seasonality, the storage-fills-then-spills regime,
    and a price observation that flips with storage decile."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    doy = dates.dayofyear.to_numpy()

    def _slug(name):
        return name.lower().replace("-", "_").replace(" ", "_")

    rain_by_c = {}
    rain_seasonality = {
        "pieman":        (0.55, 6.0),
        "gordon_pedder": (0.50, 5.0),
        "mersey_forth":  (0.30, 4.5),
        "derwent":       (0.40, 3.8),
        "great_lake":    (0.25, 3.2),
    }
    for nm, (amp, base) in rain_seasonality.items():
        seasonal = base * (1 + amp * np.cos(2 * np.pi * (doy - 200) / 365.25))
        wet = rng.random(n) < (seasonal / (seasonal.max() * 1.4))
        rain = np.where(wet, rng.gamma(2.0, seasonal / 2.0), 0.0)
        rain_by_c[nm] = rain

    shares = {_slug(c["name"]): c["share"] for c in CATCHMENTS}
    rain_mean = sum(rain_by_c[k] * shares[k] for k in rain_by_c)

    # Flow: low-pass of rain with ~3-day lag per catchment
    flow_by_c = {}
    for nm, r in rain_by_c.items():
        kernel = np.exp(-np.arange(20) / 4.0)
        kernel /= kernel.sum()
        flow_by_c[nm] = np.convolve(r, kernel, mode="full")[:n]
    flow_mean = sum(flow_by_c[k] * shares[k] for k in flow_by_c)

    # Storage: integrator with leakage (generation drawdown) + spill cap
    storage = np.zeros(n)
    storage[0] = 9000.0
    cap = 14500.0
    floor_draw = 95.0
    for t in range(1, n):
        inflow_gwh = 0.020 * flow_mean[t]
        draw = floor_draw + 8 * np.cos(2 * np.pi * (doy[t] - 30) / 365.25)
        storage[t] = storage[t - 1] + inflow_gwh - draw
        storage[t] = min(storage[t], cap)
        storage[t] = max(storage[t], 1500.0)

    sub_shares = np.array([0.36, 0.22, 0.14, 0.20, 0.08])
    sub = {f"{k}_gwh": storage * s for k, s in zip(STORAGE_SUBSYSTEMS, sub_shares)}

    # Price observation: logistic in storage fill ratio
    fill = storage / cap
    price_base = 70 + 220 * np.exp(-6 * (fill - 0.30))
    demand = 1100 + 80 * np.cos(2 * np.pi * (doy - 200) / 365.25) + rng.normal(0, 30, n)
    bass = 80 * np.cos(2 * np.pi * np.arange(n) / 365.25) + rng.normal(0, 60, n)
    noise = rng.normal(0, 8, n)
    price = price_base + 0.04 * (demand - 1100) + 0.05 * bass + noise
    price = np.clip(price, 10, 600)

    df = pd.DataFrame({
        "price_aud_mwh": price,
        "demand_mw": demand,
        "basslink_mw": bass,
        "storage_gwh": storage,
        **sub,
        "rain_mean_mm": rain_mean,
        "flow_mean_cms": flow_mean,
    }, index=dates)
    for nm, r in rain_by_c.items():
        df[f"rain_{nm}_mm"] = r
    for nm, f in flow_by_c.items():
        df[f"flow_{nm}_cms"] = f
    df.index.name = "date"
    return df


# ----------------------------------------------------------------------
# Joined daily frame
# ----------------------------------------------------------------------
def load_joined(start=START, end=END, synthetic=False):
    cache_path = os.path.join(CACHE, "joined_daily.parquet")
    if os.path.exists(cache_path) and not synthetic:
        return pd.read_parquet(cache_path)
    if synthetic:
        df = _synthetic_joined(start, end)
        df.to_parquet(cache_path)
        return df
    try:
        market = load_market(start, end)
        storage = load_storage(start, end)
        rain = load_rainfall(start, end)
        flows = load_flows(start, end)
    except Exception as e:
        print(f"[data] real-data pull failed ({e}); falling back to synthetic")
        df = _synthetic_joined(start, end)
        df.to_parquet(cache_path)
        return df

    df = market.join(storage, how="inner").join(rain, how="inner").join(
        flows, how="inner")
    df = df.loc[start:end]
    df.index.name = "date"
    df.to_parquet(cache_path)
    return df
