"""Command-line interface for the MEA wavefront tracker."""

from __future__ import annotations

import argparse

from . import run_analysis
from .batch import run_batch_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run wavefront propagation analysis on an MEA recording."
    )

    parser.add_argument("input_file", type=str, help="Path to the .brw recording file")
    parser.add_argument(
        "--electrode_pitch_um",
        type=float,
        default=42.0,
        help="Electrode pitch in micrometers (default: 42.0)",
    )
    parser.add_argument(
        "--threshold_mode",
        type=str,
        default="baseline",
        choices=["baseline", "absolute", "relative"],
        help="Thresholding mode for recruitment detection",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Absolute threshold value, required when threshold_mode=absolute",
    )
    parser.add_argument(
        "--n_std",
        type=float,
        default=5.0,
        help="Number of standard deviations above baseline",
    )
    parser.add_argument(
        "--baseline_window_s",
        type=float,
        nargs=2,
        default=(0.0, 2.0),
        metavar=("START", "END"),
        help="Start and end time (s) for baseline estimation within the analysis window",
    )
    parser.add_argument(
        "--min_duration_ms",
        type=float,
        default=20.0,
        help="Minimum duration (ms) for threshold crossing",
    )
    parser.add_argument(
        "--polarity",
        type=str,
        default="absolute",
        choices=["positive", "negative", "absolute"],
        help="Signal polarity to use for detection",
    )
    parser.add_argument(
        "--start_time_s",
        type=float,
        default=None,
        help="Start time (s) of the recording window to analyze",
    )
    parser.add_argument(
        "--end_time_s",
        type=float,
        default=None,
        help="End time (s) of the recording window to analyze",
    )
    parser.add_argument(
        "--event_times_s",
        type=str,
        default=None,
        help="CSV of event start_time_s,end_time_s windows for batch analysis",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="outputs/example",
        help="Output prefix for single-event mode, or output directory for batch mode",
    )
    parser.add_argument(
        "--local_velocity_neighbors",
        type=int,
        default=8,
        help="Nearest recruited electrodes used for each local velocity fit",
    )

    args = parser.parse_args()

    if args.event_times_s is not None and (
        args.start_time_s is not None or args.end_time_s is not None
    ):
        parser.error("Use either --event_times_s or --start_time_s/--end_time_s, not both.")

    if args.event_times_s is not None:
        result = run_batch_analysis(
            args.input_file,
            event_times_s=args.event_times_s,
            electrode_pitch_um=args.electrode_pitch_um,
            threshold_mode=args.threshold_mode,
            threshold=args.threshold,
            n_std=args.n_std,
            baseline_window_s=tuple(args.baseline_window_s),
            min_duration_ms=args.min_duration_ms,
            polarity=args.polarity,
            local_velocity_neighbors=args.local_velocity_neighbors,
            output_prefix=args.output_prefix,
        )
        print("Batch output directory:", result.output_dir)
        print("Summary CSV:", result.summary_csv)
        print("Summary report:", result.gallery_pdf)
        print("Events analyzed:", len(result.event_rows))
        return

    result = run_analysis(
        args.input_file,
        electrode_pitch_um=args.electrode_pitch_um,
        threshold_mode=args.threshold_mode,
        threshold=args.threshold,
        n_std=args.n_std,
        baseline_window_s=tuple(args.baseline_window_s),
        min_duration_ms=args.min_duration_ms,
        polarity=args.polarity,
        start_time_s=args.start_time_s,
        end_time_s=args.end_time_s,
        local_velocity_neighbors=args.local_velocity_neighbors,
        output_prefix=args.output_prefix,
    )

    print("Direction (deg):", result.direction_deg)
    print("Speed (um/s):", result.speed_um_per_s)
    print("Plane fit R^2:", result.plane_fit_r2)
    print("Recruited electrodes:", result.n_recruited)


if __name__ == "__main__":
    main()
