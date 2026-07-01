"""Simple plane-wave propagation analysis."""

from __future__ import annotations

import numpy as np


def fit_plane_wave(
    positions: np.ndarray,
    activation_times: np.ndarray,
    min_points: int = 3,
) -> dict:
    """Fit ``t(x, y) = ax + by + c`` using recruited electrodes.

    Returns direction in degrees for the gradient vector ``(a, b)``, which points
    from earlier recruitment toward later recruitment. Speed is ``1 / |grad t|``.
    """

    positions = np.asarray(positions, dtype=float)
    activation_times = np.asarray(activation_times, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (n_electrodes, 2).")
    if activation_times.shape[0] != positions.shape[0]:
        raise ValueError("activation_times length must match positions.")

    valid = np.isfinite(activation_times)
    n_recruited = int(np.count_nonzero(valid))
    if n_recruited < min_points:
        return _empty_fit(n_recruited)

    x = positions[valid, 0]
    y = positions[valid, 1]
    t = activation_times[valid]
    design = np.column_stack((x, y, np.ones_like(x)))
    coefficients, *_ = np.linalg.lstsq(design, t, rcond=None)

    predicted = design @ coefficients
    residual_ss = float(np.sum((t - predicted) ** 2))
    total_ss = float(np.sum((t - np.mean(t)) ** 2))
    r2 = np.nan if total_ss == 0 else 1.0 - residual_ss / total_ss

    a, b, _ = coefficients
    grad_norm = float(np.hypot(a, b))
    if grad_norm == 0:
        direction_deg = np.nan
        speed_um_per_s = np.inf
    else:
        direction_deg = float(np.degrees(np.arctan2(b, a)) % 360.0)
        if np.isclose(direction_deg, 360.0):
            direction_deg = 0.0
        speed_um_per_s = float(1.0 / grad_norm)

    return {
        "coefficients": coefficients,
        "direction_deg": direction_deg,
        "speed_um_per_s": speed_um_per_s,
        "r2": float(r2),
        "n_recruited": n_recruited,
        "valid_mask": valid,
    }


def predict_plane_times(positions: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Evaluate a fitted activation-time plane at electrode positions."""

    positions = np.asarray(positions, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    return (
        coefficients[0] * positions[:, 0]
        + coefficients[1] * positions[:, 1]
        + coefficients[2]
    )


def estimate_local_velocity_field(
    positions: np.ndarray,
    activation_times: np.ndarray,
    n_neighbors: int = 8,
    min_points: int = 4,
) -> dict:
    """Estimate local propagation velocity vectors from neighborhood plane fits.

    For each recruited electrode, fit ``t(x, y) = ax + by + c`` using its nearest
    recruited neighbors. The velocity vector is ``grad(t) / |grad(t)|^2`` and
    points from earlier activation toward later activation.
    """

    positions = np.asarray(positions, dtype=float)
    activation_times = np.asarray(activation_times, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (n_electrodes, 2).")
    if activation_times.shape[0] != positions.shape[0]:
        raise ValueError("activation_times length must match positions.")
    if n_neighbors < min_points:
        raise ValueError("n_neighbors must be at least min_points.")

    valid = np.isfinite(activation_times)
    valid_indices = np.flatnonzero(valid)
    velocity_vectors = np.full_like(positions, np.nan, dtype=float)
    local_speeds = np.full(positions.shape[0], np.nan, dtype=float)
    local_r2 = np.full(positions.shape[0], np.nan, dtype=float)

    if valid_indices.size < min_points:
        return {
            "velocity_vectors": velocity_vectors,
            "speed_um_per_s": local_speeds,
            "r2": local_r2,
            "valid_mask": np.isfinite(local_speeds),
        }

    k = min(int(n_neighbors), valid_indices.size)
    valid_positions = positions[valid_indices]

    for center_idx in valid_indices:
        distances = np.linalg.norm(valid_positions - positions[center_idx], axis=1)
        neighbor_valid_order = np.argsort(distances)[:k]
        neighbor_indices = valid_indices[neighbor_valid_order]

        fit = fit_plane_wave(
            positions[neighbor_indices],
            activation_times[neighbor_indices],
            min_points=min_points,
        )
        coefficients = fit["coefficients"]
        a, b = coefficients[:2]
        grad_norm_sq = float(a * a + b * b)
        if not np.isfinite(grad_norm_sq) or grad_norm_sq == 0:
            continue

        velocity_vectors[center_idx] = np.array([a, b], dtype=float) / grad_norm_sq
        local_speeds[center_idx] = float(np.sqrt(grad_norm_sq) / grad_norm_sq)
        local_r2[center_idx] = fit["r2"]

    return {
        "velocity_vectors": velocity_vectors,
        "speed_um_per_s": local_speeds,
        "r2": local_r2,
        "valid_mask": np.isfinite(local_speeds),
    }


def _empty_fit(n_recruited: int) -> dict:
    return {
        "coefficients": np.array([np.nan, np.nan, np.nan], dtype=float),
        "direction_deg": np.nan,
        "speed_um_per_s": np.nan,
        "r2": np.nan,
        "n_recruited": n_recruited,
        "valid_mask": None,
    }
