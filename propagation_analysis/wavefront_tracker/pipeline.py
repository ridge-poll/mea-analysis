"""End-to-end MVP analysis pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from .detection import detect_activation_times
from .io import load_brw, open_recording
from .plots import save_basic_plots
from .propagation import estimate_local_velocity_field, fit_plane_wave
from .types import WavefrontResult


def run_analysis(
    filepath: str,
    channel_indices: Optional[Sequence[int]] = None,
    start: int = 0,
    end: Optional[int] = None,
    start_time_s: Optional[float] = None,
    end_time_s: Optional[float] = None,
    electrode_pitch_um: float = 1.0,
    threshold_mode: str = "baseline",
    threshold: Optional[float] = None,
    n_std: float = 5.0,
    baseline_window_s: Optional[Tuple[float, float]] = None,
    min_duration_ms: float = 20.0,
    polarity: str = "absolute",
    local_velocity_neighbors: int = 8,
    output_prefix: Optional[str] = None,
) -> WavefrontResult:
    """Load a .brw file, detect recruitment, fit a plane, and optionally plot."""

    start, end, time_offset_s = _resolve_analysis_window(
        filepath=filepath,
        start=start,
        end=end,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
    )
    array = load_brw(
        filepath,
        channel_indices=channel_indices,
        start=start,
        end=end,
        electrode_pitch_um=electrode_pitch_um,
    )
    activation_times = detect_activation_times(
        array.traces,
        array.sampling_rate,
        threshold_mode=threshold_mode,
        threshold=threshold,
        n_std=n_std,
        baseline_window_s=baseline_window_s,
        min_duration_ms=min_duration_ms,
        polarity=polarity,
    )
    activation_times = activation_times + time_offset_s
    fit = fit_plane_wave(array.positions, activation_times)
    local_velocity = estimate_local_velocity_field(
        array.positions,
        activation_times,
        n_neighbors=local_velocity_neighbors,
    )

    if output_prefix is not None:
        prefix_path = Path(output_prefix)
        if prefix_path.parent:
            prefix_path.parent.mkdir(parents=True, exist_ok=True)
        prefix = str(prefix_path)
        save_basic_plots(
            array.positions,
            activation_times,
            fit["coefficients"],
            output_prefix=prefix,
            local_velocity_vectors=local_velocity["velocity_vectors"],
            local_speed_um_per_s=local_velocity["speed_um_per_s"],
        )
        np.save(f"{prefix}_activation_times.npy", activation_times)

    return WavefrontResult(
        activation_times=activation_times,
        direction_deg=fit["direction_deg"],
        speed_um_per_s=fit["speed_um_per_s"],
        plane_fit_r2=fit["r2"],
        plane_coefficients=fit["coefficients"],
        n_recruited=fit["n_recruited"],
        local_velocity_vectors=local_velocity["velocity_vectors"],
        local_speed_um_per_s=local_velocity["speed_um_per_s"],
    )


def _resolve_analysis_window(
    filepath: str,
    start: int,
    end: Optional[int],
    start_time_s: Optional[float],
    end_time_s: Optional[float],
) -> Tuple[int, Optional[int], float]:
    """Resolve optional second-based time window to sample indices."""

    if start_time_s is None and end_time_s is None:
        return start, end, start / _sampling_rate_for_offset(filepath, start)

    if start != 0 or end is not None:
        raise ValueError(
            "Use either sample indices (start/end) or seconds "
            "(start_time_s/end_time_s), not both."
        )

    meta = open_recording(filepath)
    sampling_rate = meta["sampling_rate"]
    if sampling_rate is None:
        raise ValueError("Sampling rate is required to use time-window arguments.")

    if start_time_s is None:
        start_time_s = 0.0
    if start_time_s < 0:
        raise ValueError("start_time_s must be non-negative.")
    if end_time_s is not None and end_time_s <= start_time_s:
        raise ValueError("end_time_s must be greater than start_time_s.")

    start_sample = int(np.floor(start_time_s * sampling_rate))
    end_sample = None
    if end_time_s is not None:
        end_sample = int(np.ceil(end_time_s * sampling_rate))

    return start_sample, end_sample, float(start_sample / sampling_rate)


def _sampling_rate_for_offset(filepath: str, start: int) -> float:
    if start == 0:
        return 1.0
    meta = open_recording(filepath)
    sampling_rate = meta["sampling_rate"]
    if sampling_rate is None:
        raise ValueError("Sampling rate is required when start is non-zero.")
    return float(sampling_rate)
