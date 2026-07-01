# Simple MEA Wavefront Tracker

Small MVP for detecting electrode recruitment times and fitting a global
plane-wave summary in HD-MEA `.brw` recordings.

## What It Does

1. Loads `.brw` traces and electrode coordinates.
2. Detects each electrode's first sustained threshold crossing.
3. Stores `activation_times[n_electrodes]` in seconds.
4. Fits `t(x, y) = ax + by + c` to recruited electrodes.
5. Produces activation map, raster, and plane-fit plots.

## Quick Use

From the command line:

```bash
python -m wavefront_tracker recording.brw \
  --electrode_pitch_um 42.0 \
  --threshold_mode baseline \
  --n_std 5.0 \
  --baseline_window_s 0 2 \
  --start_time_s 10 \
  --end_time_s 30 \
  --min_duration_ms 20 \
  --polarity absolute \
  --local_velocity_neighbors 8 \
  --output_prefix outputs/example
```

The `--start_time_s` and `--end_time_s` options isolate a recording window.
Activation times are reported in absolute seconds from the original recording,
while `--baseline_window_s` is relative to the selected window.

From Python:

```python
from wavefront_tracker import run_analysis

result = run_analysis(
    "recording.brw",
    electrode_pitch_um=42.0,
    threshold_mode="baseline",
    n_std=5.0,
    baseline_window_s=(0.0, 2.0),
    start_time_s=10.0,
    end_time_s=30.0,
    min_duration_ms=20.0,
    polarity="absolute",
    local_velocity_neighbors=8,
    output_prefix="outputs/example",
)

print(result.direction_deg)
print(result.speed_um_per_s)
print(result.plane_fit_r2)
```

## Batch Event Mode

Create an event CSV:

```csv
start_time_s,end_time_s
120.5,135.2
302.0,319.8
521.1,545.7
```

Run:

```bash
python -m wavefront_tracker recording.brw \
  --event_times_s events.csv \
  --electrode_pitch_um 42.0 \
  --threshold_mode baseline \
  --n_std 5.0 \
  --baseline_window_s 0 2 \
  --min_duration_ms 20 \
  --polarity absolute \
  --output_prefix outputs/batch
```

Batch mode treats `--output_prefix` as the batch output directory:

```text
outputs/batch/
├── summary.csv
├── summary_report.pdf
├── event_0001/
├── event_0002/
└── event_0003/
```

Each event folder contains:

```text
activation_map.png
activation_raster.png
plane_fit.png
local_velocity_field.png
recruitment_times_map.png
activation_times.npy
```

The CLI raises an error if `--event_times_s` is used together with
`--start_time_s` or `--end_time_s`.

For normalized recordings, use an absolute threshold:

```python
result = run_analysis(
    "recording.brw",
    threshold_mode="absolute",
    threshold=4.0,
    min_duration_ms=20.0,
)
```

## Notes

- Traces are represented as `(n_samples, n_electrodes)`.
- Positions are represented as `(x, y)` in micrometers.
- Plot outputs include:
  - `*_recruitment_times_map.png`
  - `*_activation_map.png`
  - `*_activation_raster.png`
  - `*_plane_fit.png`
  - `*_local_velocity_field.png`
- The fitted direction is the direction of increasing activation time, from
  earlier recruited electrodes toward later recruited electrodes.
- Speed is `1 / hypot(a, b)`, so positions must be in micrometers and activation
  times in seconds to get `um/s`.
- Local velocity arrows come from nearest-neighbor plane fits. The arrow direction
  points from earlier activation toward later activation, and arrow color shows
  local speed in `um/s`.
