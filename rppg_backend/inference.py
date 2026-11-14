"""Predict BPM from raw, resampled RGB. No ground truth is needed.

Channel order: forehead R/G/B, left cheek R/G/B, right cheek R/G/B.
Input: 600 samples at 20 Hz (30 seconds), nine raw mean RGB channels.
"""
import argparse
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from model import HeartRateCNN
from hr_pipeline import normalize_windows, RGB_COLUMNS, NORMALIZATION

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / 'outputs/real_cnn_v1/best_hr_model.pth'


@lru_cache(maxsize=4)
def _load_model(checkpoint_path, device):
    selected = torch.device('mps' if device in ('auto', 'mps') and torch.backends.mps.is_available() else 'cpu')
    if device not in ('cpu', 'auto', 'mps'):
        raise ValueError('device must be cpu, auto, or mps')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (checkpoint['channels'] != RGB_COLUMNS or checkpoint['normalization'] != NORMALIZATION
            or checkpoint['window_samples'] != 600 or checkpoint['sample_hz'] != 20
            or checkpoint['architecture'] != 'HeartRateCNN'):
        raise ValueError('Checkpoint preprocessing or architecture mismatch')
    model = HeartRateCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(selected).eval()
    return model, checkpoint, selected


def predict_hr(rgb_window, checkpoint_path=DEFAULT_CHECKPOINT, device='cpu'):
    """Return predicted heart rate in BPM from a finite NumPy (600, 9) array.

    Apply training's per-window, per-channel z-score. Do not normalize the input
    beforehand. CPU is default; 'auto' or 'mps' uses MPS when available.
    """
    if not isinstance(rgb_window, np.ndarray) or rgb_window.shape != (600, 9):
        raise ValueError('rgb_window must be a NumPy array with shape (600, 9)')
    normalized = normalize_windows(rgb_window[None])
    model, checkpoint, selected = _load_model(str(Path(checkpoint_path).resolve()), device)
    inputs = torch.from_numpy(normalized.transpose(0, 2, 1).copy()).to(selected)
    with torch.inference_mode():
        value = model(inputs).cpu().item() * checkpoint['label_scale'] + checkpoint['label_mean']
    if not np.isfinite(value):
        raise ValueError('Model returned a nonfinite prediction')
    return float(value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows', type=Path, default=Path('outputs/multisubject_v1/X_raw.npy'))
    parser.add_argument('--index', type=int, required=True, help='Window row in windows.csv (test row recommended)')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', choices=['cpu', 'auto', 'mps'], default='cpu')
    args = parser.parse_args()
    windows = np.load(args.windows, allow_pickle=False)
    print(f'Predicted heart rate: {predict_hr(windows[args.index], args.checkpoint, args.device):.3f} BPM')
