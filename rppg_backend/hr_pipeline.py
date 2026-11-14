"""Shared preprocessing contract for real CNN training and inference."""
import numpy as np

RGB_COLUMNS = [f'{region}_{channel}' for region in ('forehead', 'left_cheek', 'right_cheek') for channel in ('r', 'g', 'b')]
NORMALIZATION = {'method': 'per_window_channel_zscore', 'epsilon': 1e-6,
                 'axis': 'time', 'ddof': 0}


def normalize_windows(windows):
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (600, 9):
        raise ValueError('Expected shape (N, 600, 9)')
    if not np.isfinite(values).all():
        raise ValueError('RGB windows must contain only finite samples')
    return (values - values.mean(axis=1, keepdims=True)) / np.maximum(
        values.std(axis=1, keepdims=True), NORMALIZATION['epsilon'])
