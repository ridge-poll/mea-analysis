"""Batch event analysis for MEA wavefront tracking."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .pipeline import run_analysis
from .types import WavefrontResult


@dataclass
class BatchAnalysisResult:
    """Summary of a batch analysis run."""

    output_dir: Path
    summary_csv: Path
    gallery_pdf: Optional[Path]
    event_rows: List[dict]


def run_batch_analysis(
    filepath: str,
    event_times_s: str,
    output_prefix: str,
    channel_indices: Optional[Sequence[int]] = None,
    electrode_pitch_um: float = 1.0,
    threshold_mode: str = "baseline",
    threshold: Optional[float] = None,
    n_std: float = 5.0,
    baseline_window_s: Optional[Tuple[float, float]] = None,
    min_duration_ms: float = 20.0,
    polarity: str = "absolute",
    local_velocity_neighbors: int = 8,
) -> BatchAnalysisResult:
    """Run wavefront analysis for every event listed in a CSV file."""

    events = read_event_times_csv(event_times_s)
    output_dir = Path(output_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    gallery_items = []
    for event_number, (start_time_s, end_time_s) in enumerate(events, start=1):
        event_id = f"event_{event_number:04d}"
        event_dir = output_dir / event_id
        event_dir.mkdir(parents=True, exist_ok=True)

        result = run_analysis(
            filepath,
            channel_indices=channel_indices,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            electrode_pitch_um=electrode_pitch_um,
            threshold_mode=threshold_mode,
            threshold=threshold,
            n_std=n_std,
            baseline_window_s=baseline_window_s,
            min_duration_ms=min_duration_ms,
            polarity=polarity,
            local_velocity_neighbors=local_velocity_neighbors,
            output_prefix=str(event_dir / "event"),
        )
        _normalize_event_outputs(event_dir)

        row = event_metrics_row(
            event_id=event_id,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            result=result,
        )
        rows.append(row)
        gallery_items.append(
            (
                event_id,
                event_dir / "activation_map.png",
                event_dir / "activation_raster.png",
            )
        )

    summary_csv = output_dir / "summary.csv"
    write_summary_csv(summary_csv, rows)
    gallery_pdf = output_dir / "summary_report.pdf"
    write_summary_report(gallery_pdf, gallery_items)

    return BatchAnalysisResult(
        output_dir=output_dir,
        summary_csv=summary_csv,
        gallery_pdf=gallery_pdf,
        event_rows=rows,
    )


def read_event_times_csv(path: str) -> List[Tuple[float, float]]:
    """Read event start/end times from a CSV with required columns."""

    events = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"start_time_s", "end_time_s"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Event CSV must contain columns: start_time_s,end_time_s"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                start_time_s = float(row["start_time_s"])
                end_time_s = float(row["end_time_s"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid event times on CSV line {line_number}."
                ) from exc

            if start_time_s < 0:
                raise ValueError(f"start_time_s must be non-negative on line {line_number}.")
            if end_time_s <= start_time_s:
                raise ValueError(
                    f"end_time_s must be greater than start_time_s on line {line_number}."
                )
            events.append((start_time_s, end_time_s))

    if not events:
        raise ValueError("Event CSV does not contain any events.")
    return events


def event_metrics_row(
    event_id: str,
    start_time_s: float,
    end_time_s: float,
    result: WavefrontResult,
) -> dict:
    """Compute one summary CSV row for an event."""

    activation_times = np.asarray(result.activation_times, dtype=float)
    recruited = np.isfinite(activation_times)
    n_electrodes = activation_times.size
    n_recruited = int(np.count_nonzero(recruited))

    if n_recruited:
        activation_time_spread_s = float(
            np.nanmax(activation_times[recruited])
            - np.nanmin(activation_times[recruited])
        )
    else:
        activation_time_spread_s = np.nan

    return {
        "event_id": event_id,
        "start_time_s": float(start_time_s),
        "end_time_s": float(end_time_s),
        "n_recruited": n_recruited,
        "fraction_recruited": float(n_recruited / n_electrodes) if n_electrodes else np.nan,
        "activation_time_spread_s": activation_time_spread_s,
        "direction_deg": float(result.direction_deg),
        "speed_um_per_s": float(result.speed_um_per_s),
        "plane_fit_r2": float(result.plane_fit_r2),
    }


def write_summary_csv(path: Path, rows: List[dict]) -> None:
    """Write the master batch summary CSV."""

    fieldnames = [
        "event_id",
        "start_time_s",
        "end_time_s",
        "n_recruited",
        "fraction_recruited",
        "activation_time_spread_s",
        "direction_deg",
        "speed_um_per_s",
        "plane_fit_r2",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_report(
    path: Path,
    gallery_items: List[Tuple[str, Path, Path]],
) -> None:
    """Write a PDF gallery with activation map and raster for each event."""

    if not gallery_items:
        return

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(path) as pdf:
        for event_id, activation_map_path, raster_path in gallery_items:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            fig.suptitle(event_id)

            for ax, image_path, title in (
                (axes[0], activation_map_path, "Activation map"),
                (axes[1], raster_path, "Activation raster"),
            ):
                image = plt.imread(image_path)
                ax.imshow(image)
                ax.set_title(title)
                ax.axis("off")

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def _normalize_event_outputs(event_dir: Path) -> None:
    """Rename prefixed single-event outputs to standard batch filenames."""

    renames = {
        "event_activation_map.png": "activation_map.png",
        "event_activation_raster.png": "activation_raster.png",
        "event_plane_fit.png": "plane_fit.png",
        "event_local_velocity_field.png": "local_velocity_field.png",
        "event_recruitment_times_map.png": "recruitment_times_map.png",
        "event_activation_times.npy": "activation_times.npy",
    }
    for source_name, target_name in renames.items():
        source = event_dir / source_name
        target = event_dir / target_name
        if source.exists():
            source.replace(target)
