#!/usr/bin/env python3
"""
mea_plot.py — Plotting layer for HD-MEA traces.

Public API
----------
render_traces(ax, traces, channel_indices, x, xlabel)
    Draw traces into a caller-owned Axes object.

make_x_axis(n_samples, start_sample, sampling_rate_hz) -> (x, xlabel)
    Build the x-axis array and label string.

Standalone usage (CLI, wraps mea_io):
    python mea_plot.py recording.brw \
        --channels 0 1 4 7 --start 0 --end 3000 [--save out.png] [--show]
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np


# ── Public API ─────────────────────────────────────────────────────────────────

def make_x_axis(n_samples, start_sample, sampling_rate_hz):
    """
    Build an x-axis vector and label for a trace plot.

    Parameters
    ----------
    n_samples        : int
    start_sample     : int   — offset of the first sample in the recording
    sampling_rate_hz : float | None

    Returns
    -------
    x      : np.ndarray, length n_samples
    xlabel : str
    """
    if sampling_rate_hz is not None:
        x      = (np.arange(n_samples) + start_sample) / sampling_rate_hz
        xlabel = "Time (s)"
    else:
        x      = np.arange(n_samples) + start_sample
        xlabel = "Sample index"
    return x, xlabel


def render_traces(ax, traces, channel_indices, x, xlabel):
    """
    Draw MEA traces into an existing Axes object.

    Clears the axes before drawing so the function is safe to call repeatedly
    (e.g. on slider update) without accumulating artists.

    Parameters
    ----------
    ax              : matplotlib.axes.Axes   — caller owns the figure/canvas
    traces          : np.ndarray (n_samples, n_channels)
    channel_indices : list[int]              — flat channel indices (for labels)
    x               : np.ndarray (n_samples,)
    xlabel          : str
    """
    ax.cla()

    n_samples, n_ch = traces.shape
    cmap   = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(n_ch)]

    for i, (ch_idx, color) in enumerate(zip(channel_indices, colors)):
        ax.plot(x, traces[:, i], color=color, linewidth=0.7, label=f"ch {ch_idx}")

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.margins(x=0)
    ax.tick_params(labelsize=8)

    if n_ch <= 20:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0,
            fontsize=7,
            framealpha=0.7,
        )


# ── Standalone CLI ─────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot MEA traces from an HDF5 / .brw file."
    )
    parser.add_argument("input_file")
    parser.add_argument("--channels", nargs="+", type=int, required=True, metavar="CH")
    parser.add_argument("--start",   type=int, default=0)
    parser.add_argument("--end",     type=int, default=None)
    parser.add_argument("--save",    type=str, default=None, metavar="PATH")
    parser.add_argument("--show",    action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()

    # Import here so mea_plot has no hard dependency on mea_io at module level.
    try:
        import mea_io
    except ImportError:
        print("ERROR: mea_io.py must be in the same directory (or on PYTHONPATH).")
        sys.exit(1)

    meta   = mea_io.open_recording(args.input_file)
    traces = mea_io.load_traces(
        filepath        = args.input_file,
        channel_indices = args.channels,
        start           = args.start,
        end             = args.end,
    )

    x, xlabel = make_x_axis(
        n_samples        = traces.shape[0],
        start_sample     = args.start,
        sampling_rate_hz = meta["sampling_rate"],
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    render_traces(ax, traces, args.channels, x, xlabel)
    ax.set_title(
        f"MEA traces — {len(args.channels)} channel(s)  |  {traces.shape[0]:,} samples\n"
        f"{args.input_file}",
        fontsize=9,
    )
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.save}")

    if args.show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()