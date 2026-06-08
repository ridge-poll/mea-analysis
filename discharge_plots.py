#!/usr/bin/env python3
"""
discharge_plots.py — Visualize ictal discharge center paths against MEA traces.

For each discharge in the CSV, produces a two-panel figure:
  TOP    : Traces of all electrodes that were ever nearest to the ictal center,
           plotted over the full discharge duration.  A black dotted overlay
           jumps between whichever electrode is currently nearest.
  BOTTOM : Same traces time-shifted so that each electrode's "nearest moment"
           aligns to t = 0.  X-axis spans one full discharge duration.

Usage
-----
    python discharge_plots.py discharges.csv recording.brw [--out figures/]
"""

import argparse
import ast
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Window extension parameters ───────────────────────────────────────────────
# Extra seconds added to the left/right of the discharge window in both panels.
# Set to 0.0 for no extension.
LEFT_WINDOW_EXTEND  = 0.05   # seconds to add before discharge start
RIGHT_WINDOW_EXTEND = 0.2   # seconds to add after discharge end


# mea_io must be importable (same directory or PYTHONPATH)
try:
    import mea_io
except ImportError:
    print("ERROR: mea_io.py must be in the same directory or on PYTHONPATH.")
    sys.exit(1)


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_list(val):
    """Parse a stringified list/list-of-lists from a CSV cell."""
    if isinstance(val, (list, np.ndarray)):
        return val
    return ast.literal_eval(val)


def _nearest_electrode(point, chs):
    """Return flat index of electrode closest (Euclidean) to point [row, col]."""
    rows = chs["Row"].astype(float)
    cols = chs["Col"].astype(float)
    dists = np.sqrt((rows - point[0]) ** 2 + (cols - point[1]) ** 2)
    return int(np.argmin(dists))


def _load_discharge_row(row):
    """Parse one CSV row into typed fields."""
    points     = np.array(_parse_list(row["points"]),     dtype=float)   # (N, 2)
    timestamps = np.array(_parse_list(row["timestamps"]), dtype=float)   # (N,)
    return points, timestamps


# ── per-discharge figure ──────────────────────────────────────────────────────

def plot_discharge(discharge_row, meta, filepath, out_dir):
    points, timestamps = _load_discharge_row(discharge_row)
    chs = meta["chs"]
    sr  = meta["sampling_rate"]

    discharge_id  = discharge_row["discharge_id"]
    t_start       = timestamps[0]
    t_end         = timestamps[-1]
    duration_s    = t_end - t_start

    # ── 1. Find nearest electrode at every path point ──────────────────────
    nearest_seq = [_nearest_electrode(pt, chs) for pt in points]   # one per path point

    # Unique active electrodes (preserving first-appearance order)
    seen = set()
    active_flat = []
    for fi in nearest_seq:
        if fi not in seen:
            seen.add(fi)
            active_flat.append(fi)

    n_active = len(active_flat)

    # For each active electrode: the timestamp when it was nearest
    nearest_ts = {}   # flat_idx -> timestamp (first time it was nearest)
    for fi, ts in zip(nearest_seq, timestamps):
        if fi not in nearest_ts:
            nearest_ts[fi] = ts

    # ── 2. Build segment list for the black dotted line (top panel) ────────
    # Each segment: (flat_idx, t_from, t_to)
    segments = []
    seg_start_ts = timestamps[0]
    seg_flat     = nearest_seq[0]
    for i in range(1, len(nearest_seq)):
        if nearest_seq[i] != seg_flat:
            segments.append((seg_flat, seg_start_ts, timestamps[i]))
            seg_flat     = nearest_seq[i]
            seg_start_ts = timestamps[i]
    segments.append((seg_flat, seg_start_ts, timestamps[-1]))

    # ── 3. Load traces ──────────────────────────────────────────────────────
    # Top panel: full discharge window + user extensions
    top_t_start = t_start - LEFT_WINDOW_EXTEND
    top_t_end   = t_end   + RIGHT_WINDOW_EXTEND
    frame_start = max(0, int(top_t_start * sr))
    frame_end   = min(meta["n_frames"], int(top_t_end * sr) + 1)

    traces_top = mea_io.load_traces(filepath, active_flat, frame_start, frame_end)
    n_samples_top = traces_top.shape[0]
    t_top = np.linspace(top_t_start, top_t_end, n_samples_top)   # absolute time (s)

    # Bottom panel: per-electrode, shifted window of same (extended) duration
    half_left  = duration_s / 2.0 + LEFT_WINDOW_EXTEND
    half_right = duration_s / 2.0 + RIGHT_WINDOW_EXTEND
    traces_bot  = {}   # flat_idx -> (t_rel, trace)
    for fi in active_flat:
        t_center  = nearest_ts[fi]
        win_start = t_center - half_left
        win_end   = t_center + half_right
        fs = max(0, int(win_start * sr))
        fe = min(meta["n_frames"], int(win_end * sr) + 1)
        tr = mea_io.load_traces(filepath, [fi], fs, fe)
        # Build relative time axis aligned to t_center
        t_abs = np.linspace(fs / sr, fe / sr, tr.shape[0])
        t_rel = t_abs - t_center
        traces_bot[fi] = (t_rel, tr[:, 0])

    # ── 4. Colour map (tab20, same as mea_plot) ─────────────────────────────
    cmap   = plt.get_cmap("tab20")
    colors = {fi: cmap(i % 20) for i, fi in enumerate(active_flat)}

    # ── 5. Draw figure ──────────────────────────────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"hspace": 0.45},
    )

    # ── TOP panel ───────────────────────────────────────────────────────────
    for i, fi in enumerate(active_flat):
        ax_top.plot(
            t_top, traces_top[:, i],
            color=colors[fi], linewidth=0.8,
            label=f"ch {fi}  (r{chs['Row'][fi]},c{chs['Col'][fi]})",
        )

    # Black dotted overlay: jump per segment
    for (fi, ts_from, ts_to) in segments:
        mask = (t_top >= ts_from) & (t_top <= ts_to)
        col_idx = active_flat.index(fi)
        ax_top.plot(
            t_top[mask], traces_top[mask, col_idx],
            color="black", linewidth=0.9
        )

    ax_top.set_xlabel("Time (s)", fontsize=9)
    ax_top.set_ylabel("Amplitude", fontsize=9)
    ax_top.set_title(
        f"{discharge_id}  —  Nearest-electrode overlay  "
        f"({n_active} active channels)",
        fontsize=9,
    )
    ax_top.margins(x=0)
    ax_top.tick_params(labelsize=8)
    ax_top.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1),
        borderaxespad=0, fontsize=7, framealpha=0.7,
    )

    # ── BOTTOM panel ────────────────────────────────────────────────────────
    for fi in active_flat:
        t_rel, tr = traces_bot[fi]
        ax_bot.plot(t_rel, tr, color=colors[fi], linewidth=0.8)

    ax_bot.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_bot.set_xlabel("Time shifted to align with moment  (s)", fontsize=9)
    ax_bot.set_ylabel("Amplitude", fontsize=9)
    ax_bot.set_title(
        f"{discharge_id}  —  Time-aligned to nearest-center moment",
        fontsize=9,
    )
    ax_bot.set_xlim(-half_left, half_right)
    ax_bot.margins(x=0)
    ax_bot.tick_params(labelsize=8)
    ax_bot.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    fig.tight_layout(rect=[0, 0, 0.88, 1])   # leave room for legend

    # ── 6. Save ─────────────────────────────────────────────────────────────
    out_path = Path(out_dir) / f"{discharge_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate per-discharge trace figures from a CSV + .brw file."
    )
    parser.add_argument("csv",      help="Path to discharges CSV")
    parser.add_argument("brw",      help="Path to .brw / HDF5 recording")
    parser.add_argument("--out",    default=".", metavar="DIR",
                        help="Output directory for figures (default: current dir)")
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    df   = pd.read_csv(args.csv)
    meta = mea_io.open_recording(args.brw)

    print(f"Recording : {meta['n_channels']} channels, "
          f"{meta['n_frames']:,} frames, {meta['sampling_rate']} Hz")
    print(f"Discharges: {len(df)}")

    for _, row in df.iterrows():
        try:
            plot_discharge(row, meta, args.brw, args.out)
        except Exception as e:
            print(f"  ERROR on {row.get('discharge_id', '?')}: {e}")


if __name__ == "__main__":
    main()