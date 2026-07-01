"""Input helpers for HD-MEA HDF5 / .brw recordings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from .types import ElectrodeArray

DATASET_PATH = "3BData/Raw"
CHS_PATH = "3BRecInfo/3BMeaStreams/Raw/Chs"
SR_PATH = "3BRecInfo/3BRecVars/SamplingRate"


def open_recording(filepath: Union[str, Path]) -> dict:
    """Read .brw metadata without loading signal data."""

    h5py = _import_h5py()
    filepath = str(filepath)
    with h5py.File(filepath, "r") as f:
        _require(f, DATASET_PATH, filepath)
        dset = f[DATASET_PATH]

        _require(f, CHS_PATH, filepath)
        chs = f[CHS_PATH][:]
        n_channels = chs.shape[0]

        if dset.ndim == 2:
            n_frames = dset.shape[0]
            layout = "2D"
        elif dset.ndim == 1:
            n_total = dset.shape[0]
            if n_total % n_channels != 0:
                raise ValueError(
                    f"Flat dataset length {n_total} is not divisible by "
                    f"channel count {n_channels}."
                )
            n_frames = n_total // n_channels
            layout = "1D"
        else:
            raise ValueError(f"Unexpected dataset dimensionality: {dset.ndim}D")

        sampling_rate = None
        if SR_PATH in f:
            try:
                sampling_rate = float(np.asarray(f[SR_PATH])[0])
            except Exception:
                sampling_rate = None

    return {
        "filepath": filepath,
        "n_channels": n_channels,
        "n_frames": n_frames,
        "sampling_rate": sampling_rate,
        "chs": chs,
        "flat_index": np.arange(n_channels, dtype=int),
        "layout": layout,
    }


def load_traces(
    filepath: Union[str, Path],
    channel_indices: Sequence[int],
    start: int = 0,
    end: Optional[int] = None,
) -> np.ndarray:
    """Load signal data as ``(n_samples, n_channels)`` for selected channels."""

    h5py = _import_h5py()
    filepath = str(filepath)
    channel_indices = list(channel_indices)
    if not channel_indices:
        raise ValueError("channel_indices must not be empty.")
    if start < 0:
        raise ValueError("start must be non-negative.")

    with h5py.File(filepath, "r") as f:
        _require(f, DATASET_PATH, filepath)
        dset = f[DATASET_PATH]

        if dset.ndim == 2:
            n_frames, n_ch_total = dset.shape
            end_idx = _resolve_end(end, n_frames)
            _check_channels(channel_indices, n_ch_total)
            traces = dset[start:end_idx, :][:, channel_indices]
        elif dset.ndim == 1:
            _require(f, CHS_PATH, filepath)
            n_ch_total = f[CHS_PATH].shape[0]
            n_frames = dset.shape[0] // n_ch_total
            end_idx = _resolve_end(end, n_frames)
            _check_channels(channel_indices, n_ch_total)
            flat = dset[start * n_ch_total : end_idx * n_ch_total]
            traces = flat.reshape(end_idx - start, n_ch_total)[:, channel_indices]
        else:
            raise ValueError(f"Unexpected dataset dimensionality: {dset.ndim}D")

    return traces.astype(np.float64, copy=False)


def load_brw(
    filepath: Union[str, Path],
    channel_indices: Optional[Sequence[int]] = None,
    start: int = 0,
    end: Optional[int] = None,
    electrode_pitch_um: float = 1.0,
) -> ElectrodeArray:
    """Load a .brw recording into an :class:`ElectrodeArray`.

    ``Row`` and ``Col`` coordinates from the recording are converted to
    ``(x, y) = (Col, Row) * electrode_pitch_um``.
    """

    meta = open_recording(filepath)
    if meta["sampling_rate"] is None:
        raise ValueError("Sampling rate is missing or unreadable in the .brw file.")

    if channel_indices is None:
        channel_indices = meta["flat_index"]
    channel_indices = np.asarray(channel_indices, dtype=int)

    traces = load_traces(filepath, channel_indices, start=start, end=end)
    chs = meta["chs"][channel_indices]
    positions = np.column_stack((chs["Col"], chs["Row"])).astype(float)
    positions *= float(electrode_pitch_um)

    return ElectrodeArray(
        positions=positions,
        traces=traces,
        sampling_rate=float(meta["sampling_rate"]),
        electrode_ids=channel_indices,
    )


def _import_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required to read .brw files. Install h5py in the Python "
            "environment used for real recordings."
        ) from exc
    return h5py


def _require(f, path: str, filepath: str) -> None:
    if path not in f:
        available = []
        f.visit(lambda name: available.append(name))
        raise KeyError(
            f"Dataset '{path}' not found in '{filepath}'.\n"
            "Available datasets:\n" + "\n".join(f"  {name}" for name in available)
        )


def _resolve_end(end: Optional[int], n_frames: int) -> int:
    end_idx = n_frames if end is None else min(int(end), n_frames)
    if end_idx < 0:
        raise ValueError("end must be non-negative or None.")
    return end_idx


def _check_channels(indices: Sequence[int], n_ch_total: int) -> None:
    bad = [int(i) for i in indices if i < 0 or i >= n_ch_total]
    if bad:
        raise IndexError(
            f"Channel index/indices {bad} out of range "
            f"(file has {n_ch_total} channels, indices 0-{n_ch_total - 1})."
        )
