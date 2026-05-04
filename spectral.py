"""Wavelet and STFT helpers for daily series — used in §3.1.

The coherence routine follows Torrence & Compo / Grinsted: square the
cross-wavelet spectrum, normalise by the smoothed product of the
individual power spectra, then smooth in both time and scale.
"""

import numpy as np
import pywt
from scipy.ndimage import uniform_filter


def cwt_morlet(x, dt_days=1.0, n_scales=48,
               period_min_days=2.0, period_max_days=365.0):
    """Continuous Morlet wavelet transform on a daily series."""
    central_freq = pywt.central_frequency("cmor1.5-1.0")
    periods = np.geomspace(period_min_days, period_max_days, n_scales)
    scales = central_freq * periods / dt_days
    x = np.asarray(x, dtype=float) - np.nanmean(x)
    coeffs, _ = pywt.cwt(x, scales, "cmor1.5-1.0",
                         sampling_period=dt_days)
    return coeffs, periods


def stft(x, window_days=120, hop=15):
    n = len(x)
    win = np.hanning(window_days)
    starts = np.arange(0, n - window_days + 1, hop)
    spec = np.empty((len(starts), window_days // 2 + 1))
    for k, s in enumerate(starts):
        chunk = x[s:s + window_days]
        chunk = (chunk - np.nanmean(chunk)) * win
        F = np.fft.rfft(chunk)
        spec[k] = np.abs(F) ** 2
    freqs = np.fft.rfftfreq(window_days, d=1.0)
    with np.errstate(divide="ignore"):
        periods_days = np.where(freqs > 0, 1.0 / freqs, np.nan)
    return spec, starts, periods_days


def coherence(x, y, dt_days=1.0, n_scales=48,
              period_min_days=2.0, period_max_days=365.0,
              smooth_t=45, smooth_s=5):
    """Wavelet coherence in [0, 1] over (scale, time)."""
    cx, periods = cwt_morlet(x, dt_days, n_scales,
                             period_min_days, period_max_days)
    cy, _ = cwt_morlet(y, dt_days, n_scales,
                       period_min_days, period_max_days)
    Sxx = uniform_filter(np.abs(cx) ** 2, size=(smooth_s, smooth_t))
    Syy = uniform_filter(np.abs(cy) ** 2, size=(smooth_s, smooth_t))
    Sxy = uniform_filter(cx * np.conj(cy), size=(smooth_s, smooth_t))
    coh = np.abs(Sxy) ** 2 / (Sxx * Syy + 1e-12)
    return np.clip(np.real(coh), 0, 1), periods
