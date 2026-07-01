"""Small shared data containers for the wavefront tracker."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ElectrodeArray:
    """MEA traces and electrode positions.

    Attributes
    ----------
    positions:
        Electrode coordinates with shape ``(n_electrodes, 2)`` in micrometers.
    traces:
        Signal matrix with shape ``(n_samples, n_electrodes)``.
    sampling_rate:
        Sampling rate in Hz.
    electrode_ids:
        Optional source channel indices from the recording.
    """

    positions: np.ndarray
    traces: np.ndarray
    sampling_rate: float
    electrode_ids: Optional[np.ndarray] = None


@dataclass
class WavefrontResult:
    """Activation timing and global plane-wave summary."""

    activation_times: np.ndarray
    direction_deg: float
    speed_um_per_s: float
    plane_fit_r2: float
    plane_coefficients: np.ndarray
    n_recruited: int
    local_velocity_vectors: Optional[np.ndarray] = None
    local_speed_um_per_s: Optional[np.ndarray] = None
