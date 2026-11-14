"""Reusable combined HR and experimental RR inference without labels."""
import numpy as np
from hr_pipeline import RGB_COLUMNS
from inference import predict_hr, DEFAULT_CHECKPOINT
from rr_inference import estimate_rr


def predict_vitals(rgb_window, checkpoint_path=DEFAULT_CHECKPOINT, device='cpu'):
    if not isinstance(rgb_window, np.ndarray) or rgb_window.shape != (600, 9):
        raise ValueError('Expected a NumPy array with shape (600, 9)')
    if not np.isfinite(rgb_window).all() or np.any(rgb_window < 0) or np.any(rgb_window > 255):
        raise ValueError('Expected finite raw RGB means in [0, 255]')
    if np.max(np.std(rgb_window.astype(np.float64), axis=0)) < 1e-6:
        raise ValueError('RGB window has no temporal variation')
    rr = estimate_rr(rgb_window)
    return {
        'hr_bpm': predict_hr(rgb_window, checkpoint_path, device),
        'hr_status': 'preliminary_cnn',
        'rr_breaths_min': rr['rr_breaths_min'], 'rr_status': rr['status'],
        'rr_details': rr,
        'input': {'sample_hz': 20, 'window_seconds': 30, 'samples': 600, 'channels': RGB_COLUMNS},
        'limitations': [
            'HR evaluated on only one held-out subject/window; baseline performed slightly better.',
            'RR has no independent reference evaluation; an experimental rhythm is not verified breathing.',
        ],
    }
