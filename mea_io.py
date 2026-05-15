#!/usr/bin/env python3
"""
mea_io.py — Data layer for HD-MEA HDF5 / .brw recordings.

Public API
----------
open_recording(filepath) -> dict
    Read file metadata without loading signal data.

load_traces(filepath, channel_indices, start, end) -> np.ndarray
    Load a signal slice for the requested flat channel indices.

Standalone usage (inspect a file):
    python mea_io.py recording.brw
"""

import sys
import h5py
import numpy as np

# ── HDF5 path constants ────────────────────────────────────────────────────────
DATASET_PATH  = "3BData/Raw"
CHS_PATH      = "3BRecInfo/3BMeaStreams/Raw/Chs"
SR_PATH       = "3BRecInfo/3BRecVars/SamplingRate"
NFRAMES_PATH  = "3BRecInfo/3BRecVars/NRecFrames"


# ── Public API ─────────────────────────────────────────────────────────────────

def open_recording(filepath):
    """
    Open an HDF5 / .brw file and return a metadata dict without loading signal data.

    Returns
    -------
    dict with keys:
        filepath        str
        n_channels      int   — number of active electrodes
        n_frames        int   — total number of time samples
        sampling_rate   float — Hz (None if unreadable)
        chs             np.ndarray shape (n_channels,)
                        structured dtype [('Row', i2), ('Col', i2)]
        flat_index      np.ndarray shape (n_channels,) int
                        chs[i] corresponds to signal column flat_index[i]
                        (always just np.arange(n_channels) for BW4, kept explicit)
        layout          '1D' or '2D'
    """
    with h5py.File(filepath, "r") as f:
        _require(f, DATASET_PATH, filepath)
        dset = f[DATASET_PATH]

        # Channel list & spatial positions
        _require(f, CHS_PATH, filepath)
        chs = f[CHS_PATH][:]                     # structured array (Row, Col)
        n_channels = chs.shape[0]

        # Layout detection
        if dset.ndim == 2:
            n_frames = dset.shape[0]
            layout = "2D"
        elif dset.ndim == 1:
            n_total = dset.shape[0]
            if n_total % n_channels != 0:
                raise ValueError(
                    f"Flat dataset length {n_total} not divisible by "
                    f"channel count {n_channels}. File may be corrupt."
                )
            n_frames = n_total // n_channels
            layout = "1D"
        else:
            raise ValueError(f"Unexpected dataset dimensionality: {dset.ndim}D")

        # Sampling rate
        sampling_rate = None
        if SR_PATH in f:
            try:
                sampling_rate = float(f[SR_PATH][0])
            except Exception:
                pass

    return {
        "filepath":      filepath,
        "n_channels":    n_channels,
        "n_frames":      n_frames,
        "sampling_rate": sampling_rate,
        "chs":           chs,
        "flat_index":    np.arange(n_channels, dtype=int),
        "layout":        layout,
    }


def load_traces(filepath, channel_indices, start, end):
    """
    Load signal data for the specified flat channel indices and time range.

    Parameters
    ----------
    filepath        : str
    channel_indices : list[int]  flat column indices into the signal matrix
    start           : int        first sample (inclusive)
    end             : int | None last sample (exclusive); None → end of file

    Returns
    -------
    np.ndarray, shape (n_samples, len(channel_indices)), dtype float64
    """
    if not channel_indices:
        raise ValueError("channel_indices must not be empty.")

    with h5py.File(filepath, "r") as f:
        _require(f, DATASET_PATH, filepath)
        dset = f[DATASET_PATH]

        if dset.ndim == 2:
            n_frames, n_ch_total = dset.shape
            end_idx = _resolve_end(end, n_frames)
            _check_channels(channel_indices, n_ch_total)
            block  = dset[start:end_idx, :]
            traces = block[:, channel_indices]

        elif dset.ndim == 1:
            _require(f, CHS_PATH, filepath)
            n_ch_total   = f[CHS_PATH].shape[0]
            n_total_elem = dset.shape[0]
            n_frames     = n_total_elem // n_ch_total
            end_idx      = _resolve_end(end, n_frames)
            _check_channels(channel_indices, n_ch_total)
            flat   = dset[start * n_ch_total : end_idx * n_ch_total]
            block  = flat.reshape(end_idx - start, n_ch_total)
            traces = block[:, channel_indices]

        else:
            raise ValueError(f"Unexpected dataset dimensionality: {dset.ndim}D")

    return traces.astype(np.float64)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _require(f, path, filepath):
    if path not in f:
        available = []
        f.visit(lambda name: available.append(name))
        raise KeyError(
            f"Dataset '{path}' not found in '{filepath}'.\n"
            "Available datasets:\n" + "\n".join(f"  {n}" for n in available)
        )


def _resolve_end(end, n_frames):
    return min(end, n_frames) if end is not None else n_frames


def _check_channels(indices, n_ch_total):
    bad = [i for i in indices if i < 0 or i >= n_ch_total]
    if bad:
        raise IndexError(
            f"Channel index/indices {bad} out of range "
            f"(file has {n_ch_total} channels, indices 0–{n_ch_total - 1})."
        )


# ── Standalone inspection ──────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <recording.brw>")
        sys.exit(1)

    path = sys.argv[1]
    try:
        meta = open_recording(path)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"File         : {meta['filepath']}")
    print(f"Layout       : {meta['layout']}")
    print(f"Channels     : {meta['n_channels']}")
    print(f"Frames       : {meta['n_frames']:,}")
    sr = meta['sampling_rate']
    if sr:
        duration = meta['n_frames'] / sr
        print(f"Sampling rate: {sr} Hz")
        print(f"Duration     : {duration:.2f} s")
    else:
        print("Sampling rate: unknown")
    print(f"Row range    : {meta['chs']['Row'].min()}–{meta['chs']['Row'].max()}")
    print(f"Col range    : {meta['chs']['Col'].min()}–{meta['chs']['Col'].max()}")


if __name__ == "__main__":
    main()