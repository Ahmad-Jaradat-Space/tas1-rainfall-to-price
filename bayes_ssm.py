"""Bayesian state-space model of the rainfall->storage->price chain.

State equation (one latent storage proxy V_t in GWh):

    inflow_t  = alpha * rain_smooth_t        (mm -> GWh, daily)
    V_t       = clip(V_{t-1} + inflow_t - draw_t, V_min, V_max)

The 'draw' stands for net generation minus minor inflows we don't
model; we let it be a smooth seasonal function plus a constant. The
clip is a soft logistic at V_max so the spill regime is differentiable.

Observation equation:

    log price_t = mu0 + mu_V * sigmoid(k * (V_thresh - V_t / V_max))
                + b_d * (demand_t - mean) + b_b * basslink_t + e_t

The sigmoid in storage is the heart of the model: when V is high,
price is at its low floor; when V is low, price climbs steeply. The
slope k and breakpoint V_thresh are both parameters.

We fit by NUTS on a thinned series (every 3rd day) so MCMC stays
cheap; the daily-resolution posterior is then reconstructed by
forward-simulating the state through the daily inputs."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def _smooth_rain(rain, halflife=3.0):
    """Exponential moving average to crudely route rainfall to inflow."""
    rain = np.asarray(rain, dtype=float)
    a = 1 - np.exp(np.log(0.5) / halflife)
    out = np.empty_like(rain)
    out[0] = rain[0]
    for t in range(1, len(rain)):
        out[t] = a * rain[t] + (1 - a) * out[t - 1]
    return out


def model(rain_smooth, demand_dev, basslink, doy_phase,
          V_max, V0, log_price=None):
    T = rain_smooth.shape[0]

    alpha = numpyro.sample("alpha", dist.HalfNormal(0.05))
    draw_base = numpyro.sample("draw_base", dist.Normal(95.0, 20.0))
    draw_amp = numpyro.sample("draw_amp", dist.Normal(0.0, 15.0))

    mu0 = numpyro.sample("mu0", dist.Normal(4.0, 0.5))
    mu_V = numpyro.sample("mu_V", dist.HalfNormal(2.0))
    k_slope = numpyro.sample("k_slope", dist.HalfNormal(8.0))
    V_thresh = numpyro.sample("V_thresh", dist.Beta(2.0, 2.0))

    b_d = numpyro.sample("b_d", dist.Normal(0.0, 1e-3))
    b_b = numpyro.sample("b_b", dist.Normal(0.0, 1e-3))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.3))

    draw = draw_base + draw_amp * jnp.cos(doy_phase)

    def step(V_prev, inputs):
        rain_t, draw_t = inputs
        V_new = V_prev + alpha * rain_t - draw_t
        # soft clip via logistic shoulders
        V_new = jnp.clip(V_new, 0.05 * V_max, V_max)
        return V_new, V_new

    inputs = (rain_smooth, draw)
    _, V_path = jax.lax.scan(step, V0, inputs)
    fill = V_path / V_max
    sig_term = jax.nn.sigmoid(k_slope * (V_thresh - fill))
    mu = mu0 + mu_V * sig_term + b_d * demand_dev + b_b * basslink
    numpyro.deterministic("V_path", V_path)
    numpyro.deterministic("mu_path", mu)
    numpyro.sample("y", dist.Normal(mu, sigma), obs=log_price)


def fit_ssm(df, V_max=14500.0, thin=3, n_warmup=500, n_samples=500,
            seed=0):
    rain = _smooth_rain(df["rain_mean_mm"].values)
    demand = df["demand_mw"].values
    bass = df["basslink_mw"].values
    price = df["price_aud_mwh"].values

    rain_t = rain[::thin]
    demand_t = demand[::thin] - demand.mean()
    bass_t = bass[::thin]
    price_t = np.log(price[::thin] + 1.0)
    doy = (df.index.dayofyear.to_numpy()[::thin]) * 2 * np.pi / 365.25
    V0 = float(df["storage_gwh"].iloc[0])

    rng = jax.random.PRNGKey(seed)
    kernel = NUTS(model, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                num_chains=1, progress_bar=False)
    mcmc.run(rng,
             rain_smooth=jnp.asarray(rain_t, dtype=jnp.float32),
             demand_dev=jnp.asarray(demand_t, dtype=jnp.float32),
             basslink=jnp.asarray(bass_t, dtype=jnp.float32),
             doy_phase=jnp.asarray(doy, dtype=jnp.float32),
             V_max=V_max, V0=V0,
             log_price=jnp.asarray(price_t, dtype=jnp.float32))
    return mcmc, mcmc.get_samples()


def storage_response(samples, storage_deciles, V_max=14500.0,
                     rain_shock_mm=10.0):
    """Posterior expected log-price response to a +rain shock at each
    storage decile. Approximates 'derivative of mu wrt rain near a
    given fill level'."""
    alpha = np.asarray(samples["alpha"])
    mu_V = np.asarray(samples["mu_V"])
    k = np.asarray(samples["k_slope"])
    V_thresh = np.asarray(samples["V_thresh"])

    # storage shift from a single +rain_shock_mm pulse (one-step inflow)
    dV = alpha * rain_shock_mm  # GWh
    dfill = dV / V_max
    out = np.zeros((len(alpha), len(storage_deciles)))
    for j, d in enumerate(storage_deciles):
        f0 = d
        f1 = d + dfill
        s0 = 1 / (1 + np.exp(-k * (V_thresh - f0)))
        s1 = 1 / (1 + np.exp(-k * (V_thresh - f1)))
        out[:, j] = mu_V * (s1 - s0)
    # convert log-price diff to dollars at a typical $100 base
    return out * 100.0
