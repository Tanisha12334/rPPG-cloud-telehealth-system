"""Experimental RR from slow green-channel intensity modulation; not RR ground truth.

A simple RIIV-inspired spectral proxy, not a reproduction of a validated PPG
algorithm. Shared lighting/motion can create peaks indistinguishable from
respiration here. Quality diagnostics are heuristics, never accuracy confidence.
"""
import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt, welch

RR_CONFIG = {
    'version': 'experimental_riiv_green_v1', 'sample_hz': 20,
    'window_samples': 600, 'band_hz': [0.1, 0.5],
    'green_indices': [1, 4, 7], 'filter_order': 2,
    'minimum_relative_std': 1e-6, 'minimum_agreeing_rois': 2,
    'agreement_tolerance_breaths_min': 2.0, 'minimum_peak_concentration': 0.5,
    'frequency_bin_breaths_min': 2.0,
}
RR_LIMITATIONS = (
    'Experimental RGB intensity proxy; no independent RR labels are available. '
    'Motion and illumination can mimic respiration. Quality checks do not establish '
    'that a detected rhythm is breathing. Search range is 6–30 breaths/min; '
    '30-second windows give 2 breaths/min spectral bins. Band-edge, ambiguous, '
    'or disagreeing signals return null. Null does not mean zero breathing.'
)


def rr_signal_spectrum(rgb_window):
    """Return filtered ROI signals and normalized in-band spectra for diagnostics."""
    if not isinstance(rgb_window, np.ndarray) or rgb_window.shape != (600, 9):
        raise ValueError('rgb_window must be a NumPy array with shape (600, 9)')
    values = np.asarray(rgb_window, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 255):
        raise ValueError('Expected finite raw RGB means in [0, 255]')
    green = values[:, RR_CONFIG['green_indices']]
    mean = green.mean(axis=0)
    relative = green / np.maximum(mean, 1e-6) - 1
    relative = detrend(relative, axis=0, type='linear')
    active = (mean > 1e-6) & (relative.std(axis=0) >= RR_CONFIG['minimum_relative_std'])
    sos = butter(RR_CONFIG['filter_order'], RR_CONFIG['band_hz'], btype='bandpass',
                 fs=RR_CONFIG['sample_hz'], output='sos')
    filtered = sosfiltfilt(sos, relative, axis=0)
    frequency, power = welch(filtered, fs=20, window='hann', nperseg=600,
                             noverlap=0, nfft=600, detrend='constant', axis=0)
    band = (frequency >= 0.1 - 1e-10) & (frequency <= 0.5 + 1e-10)
    frequency, power = frequency[band], power[band]
    total = power.sum(axis=0)
    active &= total > 1e-18
    normalized = np.divide(power, total, out=np.zeros_like(power), where=total > 1e-18)
    return filtered, frequency, normalized, active


def estimate_rr(rgb_window):
    """Return RR plus explicit unvalidated status, abstention reasons, diagnostics."""
    _, frequency, spectra, active = rr_signal_spectrum(rgb_window)
    indices = np.flatnonzero(active)
    reasons = []
    candidate = None
    roi_peaks = [None, None, None]
    agreeing = 0
    concentration = None
    if len(indices) < RR_CONFIG['minimum_agreeing_rois']:
        reasons.append('insufficient_varying_green_rois')
    else:
        aggregate = np.median(spectra[:, indices], axis=1)
        peak = int(np.argmax(aggregate))
        candidate = float(frequency[peak] * 60)
        for i in indices:
            roi_peaks[int(i)] = float(frequency[int(np.argmax(spectra[:, i]))] * 60)
        agreeing = sum(abs(value - candidate) <= RR_CONFIG['agreement_tolerance_breaths_min'] + 1e-8
                       for value in roi_peaks if value is not None)
        concentration = float(aggregate[max(0, peak - 1):peak + 2].sum() / aggregate.sum())
        if peak in (0, len(frequency) - 1):
            reasons.append('peak_at_search_band_edge')
        if agreeing < RR_CONFIG['minimum_agreeing_rois']:
            reasons.append('roi_frequency_disagreement')
        if concentration < RR_CONFIG['minimum_peak_concentration']:
            reasons.append('diffuse_spectrum')
        outside = np.ones(len(frequency), dtype=bool)
        outside[max(0, peak - 1):peak + 2] = False
        if np.any(outside) and aggregate[outside].max() >= 0.8 * aggregate[peak]:
            reasons.append('competing_spectral_peaks')
    return {
        'rr_breaths_min': candidate if not reasons else None,
        'status': 'experimental_unvalidated' if not reasons else 'unavailable_signal_quality',
        'validated': False, 'method': RR_CONFIG['version'],
        'candidate_breaths_min': candidate, 'reasons': reasons,
        'quality': {'active_rois': int(active.sum()), 'agreeing_rois': int(agreeing),
                    'roi_peak_breaths_min': roi_peaks, 'peak_concentration': concentration,
                    'confidence_calibrated': False},
        'limitations': RR_LIMITATIONS,
    }


def predict_rr(rgb_window):
    """Return experimental breaths/min, or None when heuristic quality checks fail.

    Prefer estimate_rr() when displaying values, so status/limitations are retained.
    Input is raw nine-channel RGB at 20 Hz; no labels or HR predictions are used.
    """
    return estimate_rr(rgb_window)['rr_breaths_min']
