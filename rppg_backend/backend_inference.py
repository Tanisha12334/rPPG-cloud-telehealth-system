"""Checked inference facade; preserves original modules and checkpoint behavior."""
import numpy as np
from hr_pipeline import RGB_COLUMNS
from inference import predict_hr as legacy_predict_hr, DEFAULT_CHECKPOINT
from rr_inference_optimized import estimate_rr


class SignalInputError(ValueError):
    """Invalid or insufficient caller-supplied signal."""


class InferenceFailure(RuntimeError):
    """Internal model/estimator failure; never supply a fallback vital sign."""


def validate_rgb_window(rgb_window):
    if not isinstance(rgb_window, np.ndarray) or rgb_window.shape != (600, 9):
        raise SignalInputError('Expected a NumPy array with shape (600, 9)')
    if not np.issubdtype(rgb_window.dtype, np.number) or np.iscomplexobj(rgb_window):
        raise SignalInputError('RGB values must be real numbers')
    if not np.isfinite(rgb_window).all() or np.any(rgb_window < 0) or np.any(rgb_window > 255):
        raise SignalInputError('Expected finite raw RGB means in [0, 255]')
    if np.max(np.std(rgb_window.astype(np.float64), axis=0)) < 1e-6:
        raise SignalInputError('RGB window has no temporal variation')
    return rgb_window


def predict_hr(rgb_window, checkpoint_path=DEFAULT_CHECKPOINT, device='cpu'):
    validate_rgb_window(rgb_window)
    try:
        value = float(legacy_predict_hr(rgb_window, checkpoint_path, device))
        if not np.isfinite(value) or value <= 0:
            raise InferenceFailure('Model returned an invalid HR prediction')
        return value
    except InferenceFailure:
        raise
    except Exception as exc:
        raise InferenceFailure('Heart-rate inference unavailable') from exc


def predict_vitals(rgb_window, checkpoint_path=DEFAULT_CHECKPOINT, device='cpu'):
    validate_rgb_window(rgb_window)
    hr = predict_hr(rgb_window, checkpoint_path, device)
    try:
        rr = estimate_rr(rgb_window)
    except Exception as exc:
        raise InferenceFailure('Respiratory estimator unavailable') from exc
    return {'hr_bpm': hr, 'hr_status': 'preliminary_cnn',
            'rr_breaths_min': rr['rr_breaths_min'], 'rr_status': rr['status'], 'rr_details': rr,
            'input': {'sample_hz': 20, 'window_seconds': 30, 'samples': 600, 'channels': RGB_COLUMNS},
            'limitations': [
                'HR evaluated on only one held-out subject/window; baseline performed slightly better.',
                'RR has no independent reference evaluation; an experimental rhythm is not verified breathing.',
            ]}
