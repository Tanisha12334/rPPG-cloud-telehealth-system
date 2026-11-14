"""Experimental POS HR estimator for raw 600x9 RGB at 20 Hz.

Reference: Wang et al., Algorithmic Principles of Remote PPG,
https://doi.org/10.1109/TBME.2016.2609282
No trained weights, reference labels, or CNN fallback are used.
"""
import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt
from backend_inference import validate_rgb_window, SignalInputError


class HRSignalUnavailable(SignalInputError):
    """Valid input whose pulse quality is insufficient for a reading."""
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


def estimate_hr_v1(rgb_window):
    validate_rgb_window(rgb_window)
    spectra, peaks = [], []
    frequencies = np.fft.rfftfreq(8192, 1 / 20)
    band = (frequencies >= 0.7) & (frequencies <= 3.5)
    bpm = frequencies[band] * 60
    filt = butter(3, [0.7, 3.5], fs=20, btype='bandpass', output='sos')
    for rgb in rgb_window.astype(np.float64).reshape(600, 3, 3).transpose(1, 0, 2):
        pulse = np.zeros(600)
        for start in range(600 - 32 + 1):
            section = rgb[start:start + 32]
            means = section.mean(axis=0)
            if np.any(means <= 1e-6):
                continue
            color = section / means - 1
            first = color[:, 1] - color[:, 2]
            second = color[:, 1] + color[:, 2] - 2 * color[:, 0]
            if second.std() < 1e-12:
                continue
            segment = first + first.std() / second.std() * second
            pulse[start:start + 32] += segment - segment.mean()
        if pulse.std() < 1e-10:
            continue
        pulse = sosfiltfilt(filt, detrend(pulse))
        power = np.abs(np.fft.rfft(pulse * np.hanning(600), n=8192))[band] ** 2
        if power.sum() <= 1e-20:
            continue
        power /= power.sum()
        peak = float(bpm[np.argmax(power)])
        spectra.append(power)
        peaks.append(peak)
    if len(spectra) < 2:
        raise SignalInputError('HR unavailable: insufficient color pulse signal; capture again')
    combined = np.mean(spectra, axis=0)
    candidate = float(bpm[np.argmax(combined)])
    agreeing = sum(abs(peak - candidate) <= 6 for peak in peaks)
    concentration = float(combined[np.abs(bpm - candidate) <= 3].sum())
    # These are heuristic gates, not calibrated physiological confidence.
    if agreeing < 2 or concentration < 0.25 or candidate < 45 or candidate > 207:
        raise SignalInputError('HR unavailable: weak or inconsistent pulse spectrum; capture again')
    return {'hr_bpm': candidate, 'method': 'pos_spectral_v1',
            'roi_peak_bpm': peaks, 'agreeing_rois': agreeing,
            'peak_concentration': concentration, 'validated': False}


def _pos_pulses(rgb_window):
    """Separate color pulse traces for each ROI; retain amplitude for validity checks."""
    result = []
    for rgb in rgb_window.astype(np.float64).reshape(600, 3, 3).transpose(1, 0, 2):
        pulse = np.zeros(600)
        for start in range(569):
            section = rgb[start:start + 32]
            means = section.mean(axis=0)
            if np.any(means <= 1e-6):
                continue
            color = section / means - 1
            first = color[:, 1] - color[:, 2]
            second = color[:, 1] + color[:, 2] - 2 * color[:, 0]
            scale = second.std()
            if scale < 1e-12:
                continue
            segment = first + first.std() / scale * second
            pulse[start:start + 32] += segment - segment.mean()
        result.append(pulse)
    return result


def estimate_hr(rgb_window):
    """Average local pulse rates instead of the strongest rhythm over 30 seconds.

    Five-second spectra overlap by 50%. Zero padding interpolates peaks; it does
    not improve the underlying ~12 BPM frequency resolution of each segment.
    Gates require ROI agreement and coverage; they are not calibrated confidence.
    """
    validate_rgb_window(rgb_window)
    filt = butter(3, [0.7, 3.5], fs=20, btype='bandpass', output='sos')
    pulses = [sosfiltfilt(filt, detrend(pulse)) for pulse in _pos_pulses(rgb_window)
              if pulse.std() >= 1e-10]
    if len(pulses) < 2:
        raise HRSignalUnavailable('HR unavailable: insufficient color pulse signal; capture again',
                                  {'method': 'pos_segment_mean_v2', 'validated': False,
                                   'reasons': ['insufficient_color_pulse'], 'accepted_segments': 0,
                                   'required_segments': 8, 'total_segments': 11})
    frequencies = np.fft.rfftfreq(8192, 1 / 20)
    band = (frequencies >= 0.7) & (frequencies <= 3.5)
    bpm = frequencies[band] * 60
    rates, segments = [], []
    for start in range(0, 501, 50):
        spectra = []
        for pulse in pulses:
            section = detrend(pulse[start:start + 100])
            power = np.abs(np.fft.rfft(section * np.hanning(100), n=8192))[band] ** 2
            if power.sum() > 1e-20:
                spectra.append(power / power.sum())
        if len(spectra) < 2:
            segments.append({'start_seconds': start / 20, 'accepted': False})
            continue
        combined = np.mean(spectra, axis=0)
        rate = float(bpm[np.argmax(combined)])
        roi_peaks = [float(bpm[np.argmax(power)]) for power in spectra]
        concentration = float(combined[np.abs(bpm - rate) <= 12].sum())
        agreeing = sum(abs(peak - rate) <= 12 for peak in roi_peaks)
        accepted = agreeing >= 2 and concentration >= 0.5 and 45 <= rate <= 207
        segments.append({'start_seconds': start / 20, 'candidate_bpm': rate,
                         'accepted': bool(accepted), 'agreeing_rois': agreeing,
                         'roi_peak_bpm': roi_peaks, 'peak_concentration': concentration})
        if accepted:
            rates.append(rate)
    # A few clean intervals cannot stand in for an otherwise unusable window.
    if len(rates) < 8:
        raise HRSignalUnavailable('HR unavailable: insufficient reliable pulse intervals; capture again',
                                  {'method': 'pos_segment_mean_v2', 'validated': False,
                                   'reasons': ['insufficient_reliable_intervals'],
                                   'accepted_segments': len(rates), 'required_segments': 8,
                                   'total_segments': len(segments), 'segments': segments})
    return {'hr_bpm': float(np.mean(rates)), 'method': 'pos_segment_mean_v2',
            'segment_seconds': 5, 'segment_step_seconds': 2.5,
            'accepted_segments': len(rates), 'total_segments': len(segments),
            'segment_rate_std_bpm': float(np.std(rates)), 'segments': segments,
            'validated': False}


def pulse_candidate(rgb_window):
    """Unverified full-window spectral candidate for an explicitly labelled demo.

    This bypasses the acceptance decision only for a diagnostic field. It must
    never be used as accepted HR, confidence, or evidence of biological validity.
    """
    validate_rgb_window(rgb_window)
    filt = butter(3, [.7, 3.5], fs=20, btype='bandpass', output='sos')
    frequencies = np.fft.rfftfreq(8192, 1 / 20)
    band = (frequencies >= .7) & (frequencies <= 3.5)
    bpm = frequencies[band] * 60
    spectra = []
    for pulse in _pos_pulses(rgb_window):
        if pulse.std() < 1e-10:
            continue
        pulse = sosfiltfilt(filt, detrend(pulse))
        power = abs(np.fft.rfft(pulse * np.hanning(600), n=8192))[band] ** 2
        if power.sum() > 1e-20:
            spectra.append(power / power.sum())
    if len(spectra) < 2:
        return None
    combined = np.mean(spectra, axis=0)
    candidate = float(bpm[np.argmax(combined)])
    if not 45 <= candidate <= 207:
        return None
    return {'bpm': candidate, 'method': 'full_window_pos_spectral_candidate',
            'status': 'low_quality_unverified', 'validated': False,
            'roi_peak_bpm': [float(bpm[np.argmax(power)]) for power in spectra],
            'message': 'Diagnostic rhythm only: failed HR quality checks; may be noise or motion.'}
