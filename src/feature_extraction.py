"""Window-based feature extraction for unlabeled PHM gearbox sensor signals."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

FEATURE_KEYS = [
    "mean",
    "std",
    "variance",
    "rms",
    "min",
    "max",
    "peak_to_peak",
    "kurtosis",
    "skewness",
]


def extract_signal_features(signal: Sequence[float]) -> dict[str, float]:
    """Calculate standard time-domain features for a single signal window."""
    arr = np.asarray(signal, dtype=float)
    if arr.size == 0:
        raise ValueError("Signal window is empty; cannot extract features.")
    if arr.ndim != 1:
        arr = arr.reshape(-1)

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    variance = float(np.var(arr, ddof=0))
    rms = float(np.sqrt(np.mean(np.square(arr))))
    minimum = float(np.min(arr))
    maximum = float(np.max(arr))
    peak_to_peak = float(maximum - minimum)

    series = pd.Series(arr)
    skewness = float(series.skew()) if arr.size > 2 else 0.0
    kurtosis = float(series.kurt()) if arr.size > 3 else 0.0

    return {
        "mean": mean,
        "std": std,
        "variance": variance,
        "rms": rms,
        "min": minimum,
        "max": maximum,
        "peak_to_peak": peak_to_peak,
        "kurtosis": kurtosis,
        "skewness": skewness,
    }


def window_signal(signal: Sequence[float], window_size: int = 1024, stride: int | None = None) -> list[np.ndarray]:
    """Split a 1D numeric signal into fixed-size windows with a configurable stride."""
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if stride is None:
        stride = window_size
    if stride <= 0:
        raise ValueError("stride must be positive.")

    arr = np.asarray(signal, dtype=float).reshape(-1)
    windows: list[np.ndarray] = []
    for start in range(0, arr.size, stride):
        end = start + window_size
        chunk = arr[start:end]
        if chunk.size < max(16, window_size // 4):
            break
        windows.append(chunk)
    return windows


def extract_feature_row(signal_columns: Iterable[np.ndarray], signal_names: Sequence[str] | None = None) -> dict[str, float]:
    """Build a feature dictionary from multiple signal channels using the actual signal names."""
    features: dict[str, float] = {}
    signal_list = list(signal_columns)
    labels = signal_names if signal_names is not None else [f"Signal_{index}" for index in range(1, len(signal_list) + 1)]
    for signal_name, signal in zip(labels, signal_list):
        channel_features = extract_signal_features(signal)
        for key, value in channel_features.items():
            features[f"{key}_{signal_name}"] = float(value)
    return features


def _natural_run_sort_key(path: Path) -> tuple[int, str]:
    """Sort PHM runs by numeric suffix, not by lexicographic string order."""
    match = re.search(r"Run_(\d+)", path.name)
    if match is None:
        return (float("inf"), path.name)
    return (int(match.group(1)), path.name)


def build_windowed_feature_dataset(
    data_dir: str | Path,
    window_size: int = 1024,
    stride: int | None = None,
    max_runs: int | None = None,
    run_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read PHM raw sensor runs sequentially, extract per-window features, and return a compact DataFrame."""
    run_dir = Path(data_dir)
    run_files = sorted(run_dir.glob("Run_*.csv"), key=_natural_run_sort_key)

    if run_ids is not None:
        run_ids_set = {str(run_id) for run_id in run_ids}
        run_files = [path for path in run_files if path.stem in run_ids_set]
    if max_runs is not None:
        run_files = run_files[:max_runs]
    if not run_files:
        raise FileNotFoundError(f"No Run_*.csv files were found in {run_dir}")

    sample_rows: list[dict[str, object]] = []
    for file_path in run_files:
        df = pd.read_csv(file_path, header=None)
        if df.shape[1] < 3:
            raise ValueError(f"Unexpected column count in {file_path.name}: expected at least 3 but found {df.shape[1]}")
        signals = [df.iloc[:, index].to_numpy(dtype=float) for index in range(min(3, df.shape[1]))]
        signal_names = ["Signal_1", "Signal_2", "Signal_3"]

        effective_stride = window_size if stride is None else stride
        for window_index, start in enumerate(range(0, len(signals[0]) - window_size + 1, effective_stride)):
            window_signals = [signal[start:start + window_size] for signal in signals]
            if any(window.size != window_size for window in window_signals):
                continue
            feature_row = extract_feature_row(window_signals, signal_names)
            sample_rows.append({
                "run_id": file_path.stem,
                "window_id": window_index,
                "window_size": window_size,
                "signal_length": len(window_signals[0]),
                **feature_row,
            })

    if not sample_rows:
        raise ValueError(f"No valid windows could be created from the runs in {run_dir}")

    feature_frame = pd.DataFrame(sample_rows)
    return feature_frame


def get_feature_columns(feature_frame: pd.DataFrame) -> list[str]:
    """Return all engineered feature columns excluding metadata keys."""
    metadata_columns = {"run_id", "window_id", "window_size", "signal_length"}
    return [column for column in feature_frame.columns if column not in metadata_columns]
