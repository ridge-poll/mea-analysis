"""Threshold-based recruitment detection."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np


Polarity = Literal["positive", "negative", "absolute"]
ThresholdMode = Literal["baseline", "absolute", "relative"]


def detect_activation_times(
    traces: np.ndarray,
    sampling_rate: float,
    threshold_mode: ThresholdMode = "baseline",
    threshold: Optional[float] = None,
    n_std: float = 5.0,
    baseline_window_s: Optional[Tuple[float, float]] = None,
    min_duration_ms: float = 20.0,
    polarity: Polarity = "absolute",
) -> np.ndarray:
    """Return first valid activation time for each electrode, in seconds.

    A valid activation is the first threshold crossing that stays above
    threshold for at least ``min_duration_ms``. Missing activations are ``NaN``.
    """

    traces = np.asarray(traces, dtype=float)
    if traces.ndim != 2:
        raise ValueError("traces must have shape (n_samples, n_electrodes).")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive.")

    n_samples, n_electrodes = traces.shape
    min_samples = max(1, int(np.ceil(min_duration_ms * sampling_rate / 1000.0)))

    signal = _polarity_signal(traces, polarity)
    thresholds = _compute_thresholds(
        signal=signal,
        sampling_rate=sampling_rate,
        threshold_mode=threshold_mode,
        threshold=threshold,
        n_std=n_std,
        baseline_window_s=baseline_window_s,
    )

    active = signal > thresholds.reshape(1, n_electrodes)
    activation_times = np.full(n_electrodes, np.nan, dtype=float)

    for electrode_idx in range(n_electrodes):
        start_idx = _first_run_start(active[:, electrode_idx], min_samples)
        if start_idx is not None:
            activation_times[electrode_idx] = start_idx / sampling_rate

    return activation_times


def _polarity_signal(traces: np.ndarray, polarity: Polarity) -> np.ndarray:
    if polarity == "positive":
        return traces
    if polarity == "negative":
        return -traces
    if polarity == "absolute":
        return np.abs(traces)
    raise ValueError("polarity must be 'positive', 'negative', or 'absolute'.")


def _compute_thresholds(
    signal: np.ndarray,
    sampling_rate: float,
    threshold_mode: ThresholdMode,
    threshold: Optional[float],
    n_std: float,
    baseline_window_s: Optional[Tuple[float, float]],
) -> np.ndarray:
    if threshold_mode == "absolute":
        if threshold is None:
            raise ValueError("threshold is required when threshold_mode='absolute'.")
        return np.full(signal.shape[1], float(threshold), dtype=float)

    if threshold_mode not in ("baseline", "relative"):
        raise ValueError(
            "threshold_mode must be 'baseline', 'relative', or 'absolute'."
        )

    start, end = _baseline_slice(signal.shape[0], sampling_rate, baseline_window_s)
    baseline = signal[start:end, :]
    if baseline.size == 0:
        raise ValueError("baseline window is empty.")

    mean = np.nanmean(baseline, axis=0)
    std = np.nanstd(baseline, axis=0)
    return mean + float(n_std) * std


def _baseline_slice(
    n_samples: int,
    sampling_rate: float,
    baseline_window_s: Optional[Tuple[float, float]],
) -> Tuple[int, int]:
    if baseline_window_s is None:
        return 0, n_samples

    start_s, end_s = baseline_window_s
    start = max(0, int(np.floor(start_s * sampling_rate)))
    end = min(n_samples, int(np.ceil(end_s * sampling_rate)))
    if end <= start:
        raise ValueError("baseline_window_s must contain at least one sample.")
    return start, end


def _first_run_start(mask: np.ndarray, min_samples: int) -> Optional[int]:
    run_start = None
    run_length = 0

    for idx, is_active in enumerate(mask):
        if is_active:
            if run_start is None:
                run_start = idx
            run_length += 1
            if run_length >= min_samples:
                return run_start
        else:
            run_start = None
            run_length = 0

    return None
