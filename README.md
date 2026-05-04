# From Rainfall to Price: the Causal Chain in TAS1

Tasmania's wholesale electricity price isn't really an electricity price; it's a delayed, smoothed, censored function of West Coast rainfall. Water falls on the catchments, runs into Hydro Tasmania's reservoirs, sits there as energy in storage, and reshapes the marginal cost the operator is willing to bid into the spot market. Most price-forecasting work treats this as a black box. This notebook opens the box: it estimates the lag distribution at every link of the chain, quantifies how the rainfall→price sensitivity interacts with current storage level, and asks whether a physics-coherent state-space / catchment-graph model beats a strong gradient-boosting baseline at the multi-week horizon.

The notebook puts three claims on trial:

1. The rainfall → storage → price chain has a measurable delay distribution — weeks for storage, weeks-to-months for price — estimable from five years of daily data.
2. The price elasticity to a rainfall shock interacts strongly with current storage decile. A wet event when the lakes are 80% full has near-zero price effect (water spills); the same event at 25% has a meaningful one.
3. Physics-coherent models (Bayesian SSM, catchment GNN) beat gradient boosting at long horizons because the chain has slow dynamics that engineered lag features can't fully express. Gradient boosting wins at next-day. The trade-off is the headline.

The deliverable artefacts are the lag distribution and the storage-conditional sensitivity curve — not a forecast number.

## How the notebook is laid out

The notebook reads as a short paper with five sections:

1. **Introduction** — the question framed as causal/structural, three falsifiable claims.
2. **Data** — five-year daily frame from AEMO TAS1 (NEMOSIS), Hydro Tasmania energy-in-storage, Open-Meteo ERA5 catchment rainfall, and Tasmanian WIST river-flow gauges; catchment map; rainfall climatologies; storage decomposition by sub-basin.
3. **Methods** — six chapters, escalating: (1) wavelet coherence between rainfall, storage and price; (2) L1-regularised distributed-lag IRFs for rainfall→storage and storage→price; (3) Bayesian state-space model in numpyro with a non-linear logistic price observation in storage fill; (4) gradient-boosting baseline with quantile heads; (5) PyTorch GCN over an 11-node catchment→storage→market graph with physical water-flow edges; (6) storage-conditional sensitivity from the SSM posterior.
4. **Results** — lag forest plot, multi-horizon RMSE comparison (1d / 7d / 28d / 56d), storage-decile sensitivity curve, and a 2024-H2 counterfactual under three rainfall regimes.
5. **Conclusion** — verdict on each claim, deployment context (hedge-desk quarterly outlook), implications for Marinus Link and climate change.

Every plot is read out loud: a one-line setup before the cell, a finding-style title, and a 2–4 sentence takeaway after.

## Running it

Tested on macOS with Python 3.12.

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

The first run downloads about 5 years of AEMO dispatch CSVs and Open-Meteo daily archive into `data/cache/` (gitignored) — slow the first time, fast on subsequent runs because every loader caches as parquet/feather. The Hydro Tasmania energy-in-storage history is fetched from the public XLS at `hydro.com.au`. Tasmanian WIST stream-flow gauges occasionally require a one-off manual CSV export per gauge into `data/wist_manual/` — `data.py` documents the fallback. The committed `executed.ipynb` is already run, so GitHub renders all outputs and plots inline.

## What's where

- `notebook.ipynb` — the whole story, runs top to bottom
- `data.py` — five loaders (NEMOSIS market, Hydro Tas storage, Open-Meteo rainfall, WIST flows, joined daily frame); per-source caching
- `spectral.py` — continuous Morlet wavelet, STFT, and wavelet coherence on daily series
- `transfer.py` — L1-regularised distributed-lag IRF estimator with block-bootstrap intervals
- `bayes_ssm.py` — numpyro state-space model: latent storage proxy with a logistic price observation in fill ratio
- `models.py` — gradient-boosting (point + quantile) baseline, RMSE/MAE/pinball/coverage helpers, naive lag and perfect-foresight benchmarks
- `catchment_gnn.py` — PyTorch GCN over an 11-node catchment → storage → market graph with physical water-flow adjacency
- `plots.py` — small matplotlib helpers used by the notebook
