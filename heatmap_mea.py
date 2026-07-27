"""
mea_heatmap.py — Electrode activity scatter map for HD-MEA recordings.

Each electrode is a simple colored circle at its (x, y) position; color
tracks amplitude at the current sample. Drag the slider to scrub through
time; Play/Pause auto-advances it if you want, but scrubbing is the main
way to use this.

DEPENDENCIES
    pip install numpy scipy matplotlib h5py

USAGE
    python mea_heatmap.py
"""
import tkinter as tk
from tkinter import filedialog

import numpy as np
from scipy.signal import iirnotch, filtfilt
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from mea_io import open_recording, load_traces

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════

START_SAMPLE = 0
END_SAMPLE = 100000            # None = whole file

CMAP = 'coolwarm'
MARKER_SIZE = 40
EDGE_PADDING = 1.0            # margin around the electrode layout, data units

OUTLIER_MAD_THRESH = 5
OUTLIER_NEIGHBOR_K = 6

APPLY_NOTCH_FILTER = True
NOTCH_FREQ_HZ = 60            # use 50 outside 60 Hz-mains regions
NOTCH_Q = 30

PLAYBACK_STEP_SAMPLES = 10
PLAYBACK_FPS = 30
ARROW_KEY_STEP_SAMPLES = 1

SPIKE_STD_THRESHOLD = 7.0
SPIKE_FILL_GAP_SAMPLES = 2

# ══════════════════════════════════════════════════════════════════════════
# SIGNAL / CLEANING HELPERS
# ══════════════════════════════════════════════════════════════════════════

def apply_notch_filter(traces, fs, f0, q):
    if f0 <= 0 or f0 >= fs / 2:
        print(f'  Notch frequency {f0} Hz is outside the valid range for '
              f'fs={fs} Hz — skipping.')
        return traces
    w0 = f0 / (fs / 2)
    b, a = iirnotch(w0, w0 / q)
    return filtfilt(b, a, traces, axis=0)


def reject_outliers(traces, ex, ey, mad_thresh, n_neighbors):
    ch_var = traces.var(axis=0)
    med_var = np.median(ch_var)
    mad_var = np.median(np.abs(ch_var - med_var))
    if mad_var == 0:
        print('  No outlier electrodes detected.')
        return traces

    outlier_mask = ch_var > (med_var + mad_thresh * mad_var)
    n_out = int(outlier_mask.sum())
    if n_out == 0:
        print('  No outlier electrodes detected.')
        return traces

    print(f'  Outlier electrodes detected ({n_out}): channels '
          f'{np.where(outlier_mask)[0].tolist()}')
    print(f'  Replacing with IDW of {n_neighbors} nearest spatial neighbours ...')

    cleaned = traces.copy()
    good_idx = np.where(~outlier_mask)[0]
    good_pts = np.column_stack([ex[good_idx], ey[good_idx]])
    tree = cKDTree(good_pts)

    for bad_i in np.where(outlier_mask)[0]:
        pt = np.array([ex[bad_i], ey[bad_i]])
        k = min(n_neighbors, len(good_idx))
        dist, nbr_local = tree.query(pt, k=k)
        dist = np.atleast_1d(dist)
        nbr_local = np.atleast_1d(nbr_local)
        nbrs = good_idx[nbr_local]
        w = 1.0 / np.maximum(dist, 1e-9)
        w /= w.sum()
        cleaned[:, bad_i] = cleaned[:, nbrs] @ w

    return cleaned


def electrode_snapshot(traces, idx):
    idx = max(0, min(idx, traces.shape[0] - 1))
    return traces[idx, :]


def detect_spike_events(traces, std_thresh, fill_gap_samples):
    """Return sample indices corresponding to the start of each detected event."""

    mean = traces.mean(axis=0)
    std = traces.std(axis=0)
    std = np.maximum(std, 1e-12)

    active = np.any(
        np.abs(traces - mean) > (std_thresh * std),
        axis=1
    )

    # Fill short False gaps between active regions.
    if fill_gap_samples > 0:
        inactive = ~active
        starts = np.where(np.diff(np.r_[0, inactive.astype(int)]) == 1)[0]
        ends = np.where(np.diff(np.r_[inactive.astype(int), 0]) == -1)[0]

        for s, e in zip(starts, ends):
            if (e - s + 1) <= fill_gap_samples:
                active[s:e + 1] = True

    starts = np.where(np.diff(np.r_[0, active.astype(int)]) == 1)[0]

    print(
        f'  Detected {len(starts)} spike events '
        f'({std_thresh:.1f}σ, gap={fill_gap_samples} samples).'
    )

    return starts


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title='Open MEA recording',
        filetypes=[('MEA recordings', '*.brw *.h5 *.hdf5'), ('All files', '*.*')])
    if not filepath:
        print('No file selected — aborting.')
        return

    print(f'Loading {filepath} ...')
    meta = open_recording(filepath)
    n_channels = meta['n_channels']
    sampling_rate = meta['sampling_rate']
    if sampling_rate is None:
        print('  Warning: sampling rate unreadable — defaulting to 1 Hz for the time axis.')
        sampling_rate = 1.0
    ex = meta['chs']['Col'].astype(float)
    ey = meta['chs']['Row'].astype(float)

    traces = load_traces(filepath, list(range(n_channels)), START_SAMPLE, END_SAMPLE)
    print(f'  {n_channels} channels  |  {traces.shape[0]} samples  |  '
          f'layout: {meta["layout"]}')

    if APPLY_NOTCH_FILTER:
        print(f'  Applying {NOTCH_FREQ_HZ} Hz notch filter (Q={NOTCH_Q}) ...')
        traces = apply_notch_filter(traces, sampling_rate, NOTCH_FREQ_HZ, NOTCH_Q)

    traces = reject_outliers(traces, ex, ey, OUTLIER_MAD_THRESH, OUTLIER_NEIGHBOR_K)

    spike_samples = detect_spike_events(
        traces,
        SPIKE_STD_THRESHOLD,
        SPIKE_FILL_GAP_SAMPLES
    )

    n_samples = traces.shape[0]
    ds = max(1, n_samples // 200)
    z_global_min = float(traces[::ds].min())
    z_global_max = float(traces[::ds].max())
    if z_global_min == z_global_max:
        z_global_max = z_global_min + 1

    # ── Plot: simple colored circles at electrode positions ────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    plt.subplots_adjust(bottom=0.22)

    z0 = electrode_snapshot(traces, 0)
    sc = ax.scatter(ex, ey, c=z0, cmap=CMAP, vmin=z_global_min, vmax=z_global_max,
                     s=MARKER_SIZE, edgecolors='white', linewidths=0.5)
    fig.colorbar(sc, ax=ax, label='Amplitude')

    ax.set_xlim(ex.min() - EDGE_PADDING, ex.max() + EDGE_PADDING)
    ax.set_ylim(ey.min() - EDGE_PADDING, ey.max() + EDGE_PADDING)
    ax.set_xlabel('x (electrode column)')
    ax.set_ylabel('y (electrode row)')
    ax.set_aspect('equal', adjustable='box')
    title = ax.set_title('')

    # ── Controls: slider is the primary control, Play is optional ──────────
    ax_slider = plt.axes([0.18, 0.08, 0.62, 0.04])
    slider = Slider(ax_slider, 'Sample', 0, n_samples - 1, valinit=0, valstep=1)

    # ax_play = plt.axes([0.82, 0.075, 0.10, 0.05])
    # ax_prev = plt.axes([0.02, 0.075, 0.10, 0.05])
    # ax_next = plt.axes([0.13, 0.075, 0.10, 0.05])

    ax_play = plt.axes([0.84, 0.025, 0.08, 0.035])
    ax_prev = plt.axes([0.04, 0.025, 0.12, 0.035])
    ax_next = plt.axes([0.18, 0.025, 0.12, 0.035])

    btn_play = Button(ax_play, "Play")
    btn_prev = Button(ax_prev, "Prev Spike")
    btn_next = Button(ax_next, "Next Spike")

    state = {'playing': False, 'timer': None}


    def draw_frame(idx):
        idx = int(idx)

        # Find current event index
        event_idx = np.searchsorted(spike_samples, idx, side="right")

        # Convert to 1-based display indexing
        if event_idx > 0:
            current_event = event_idx
        else:
            current_event = 0

        z_vals = electrode_snapshot(traces, idx)
        sc.set_array(z_vals)

        t_s = (START_SAMPLE + idx) / sampling_rate

        if current_event > 0:
            event_text = f'Event {current_event}/{len(spike_samples)}'
        else:
            event_text = 'No event'

        title.set_text(
            f'MEA | {event_text} | '
            f't = {t_s:.4f} s '
            f'(sample {START_SAMPLE + idx} / {START_SAMPLE + n_samples - 1})'
        )

        fig.canvas.draw_idle()

    def on_slider_changed(val):
        # matplotlib's Slider fires this continuously while dragging, so
        # scrubbing works here with no extra plumbing (unlike MATLAB's
        # uicontrol slider, which only fires on mouse release by default).
        draw_frame(slider.val)

    slider.on_changed(on_slider_changed)

    def step(_=None):
        if not state['playing']:
            return
        idx = slider.val + PLAYBACK_STEP_SAMPLES
        if idx > n_samples - 1:
            idx = 0
        slider.set_val(idx)   # triggers on_slider_changed -> draw_frame

    def toggle_play(_event):
        state['playing'] = not state['playing']
        if state['playing']:
            btn_play.label.set_text('Pause')
            if state['timer'] is None:
                state['timer'] = fig.canvas.new_timer(interval=int(1000 / PLAYBACK_FPS))
                state['timer'].add_callback(step)
            state['timer'].start()
        else:
            btn_play.label.set_text('Play')
            if state['timer'] is not None:
                state['timer'].stop()   # fully stopped -> pause is instant
        fig.canvas.draw_idle()
    
    def next_spike(_event):
        if len(spike_samples) == 0:
            return

        i = np.searchsorted(spike_samples, int(slider.val) + 1)
        if i < len(spike_samples):
            slider.set_val(spike_samples[i])


    def prev_spike(_event):
        if len(spike_samples) == 0:
            return

        i = np.searchsorted(spike_samples, int(slider.val)) - 1
        if i >= 0:
            slider.set_val(spike_samples[i])


    def on_key(event):
        current = int(slider.val)

        if event.key == 'right':
            new_idx = min(current + ARROW_KEY_STEP_SAMPLES, n_samples - 1)
            slider.set_val(new_idx)

        elif event.key == 'left':
            new_idx = max(current - ARROW_KEY_STEP_SAMPLES, 0)
            slider.set_val(new_idx)


    btn_play.on_clicked(toggle_play)
    btn_next.on_clicked(next_spike)
    btn_prev.on_clicked(prev_spike)

    fig.canvas.mpl_connect('key_press_event', on_key)

    draw_frame(0)
    print('Drag the slider to scrub through time. Click Play to auto-advance.')
    plt.show()


if __name__ == '__main__':
    main()