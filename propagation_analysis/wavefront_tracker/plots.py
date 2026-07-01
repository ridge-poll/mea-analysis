"""Basic visualization helpers for recruitment timing."""

from __future__ import annotations

import numpy as np

from .propagation import estimate_local_velocity_field


def activation_map(
    positions: np.ndarray,
    activation_times: np.ndarray,
    ax=None,
    title: str = "Activation map",
    cmap: str = "viridis",
):
    """Scatter electrode positions colored by activation time."""

    import matplotlib.pyplot as plt

    ax = ax or plt.subplots()[1]
    positions = np.asarray(positions, dtype=float)
    activation_times = np.asarray(activation_times, dtype=float)
    valid = np.isfinite(activation_times)

    ax.scatter(
        positions[~valid, 0],
        positions[~valid, 1],
        c="0.8",
        s=18,
        label="not recruited",
    )
    sc = ax.scatter(
        positions[valid, 0],
        positions[valid, 1],
        c=activation_times[valid],
        s=28,
        cmap=cmap,
        label="recruited",
    )
    if np.any(valid):
        plt.colorbar(sc, ax=ax, label="Activation time (s)")
    ax.set_title(title)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_aspect("equal", adjustable="box")
    return ax


def activation_raster(
    activation_times: np.ndarray,
    ax=None,
    title: str = "Activation raster",
):
    """Plot electrodes sorted by activation time."""

    import matplotlib.pyplot as plt

    ax = ax or plt.subplots()[1]
    activation_times = np.asarray(activation_times, dtype=float)
    valid = np.isfinite(activation_times)
    order = np.argsort(activation_times[valid])
    times = activation_times[valid][order]

    ax.scatter(times, np.arange(times.size), s=14, c="black")
    ax.set_title(title)
    ax.set_xlabel("Activation time (s)")
    ax.set_ylabel("Electrode order")
    return ax


def plot_plane_fit(
    positions: np.ndarray,
    activation_times: np.ndarray,
    plane_coefficients: np.ndarray,
    ax=None,
    title: str = "Plane-wave fit",
    contour_count: int = 8,
):
    """Show activation map with fitted isochrone contours."""

    import matplotlib.pyplot as plt

    ax = activation_map(positions, activation_times, ax=ax, title=title)
    coefficients = np.asarray(plane_coefficients, dtype=float)
    if not np.all(np.isfinite(coefficients)):
        return ax

    positions = np.asarray(positions, dtype=float)
    x_min, y_min = np.nanmin(positions, axis=0)
    x_max, y_max = np.nanmax(positions, axis=0)
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 80),
        np.linspace(y_min, y_max, 80),
    )
    zz = coefficients[0] * xx + coefficients[1] * yy + coefficients[2]
    ax.contour(xx, yy, zz, levels=contour_count, colors="white", linewidths=0.8)
    ax.contour(xx, yy, zz, levels=contour_count, colors="black", linewidths=0.25)
    return ax


def local_velocity_field(
    positions: np.ndarray,
    activation_times: np.ndarray,
    velocity_vectors: np.ndarray,
    local_speeds: np.ndarray,
    ax=None,
    title: str = "Local velocity field",
):
    """Plot local propagation velocity vectors over electrode positions."""

    import matplotlib.pyplot as plt

    ax = ax or plt.subplots()[1]
    positions = np.asarray(positions, dtype=float)
    activation_times = np.asarray(activation_times, dtype=float)
    velocity_vectors = np.asarray(velocity_vectors, dtype=float)
    local_speeds = np.asarray(local_speeds, dtype=float)

    recruited = np.isfinite(activation_times)
    has_velocity = np.isfinite(local_speeds) & np.all(np.isfinite(velocity_vectors), axis=1)

    ax.scatter(
        positions[~recruited, 0],
        positions[~recruited, 1],
        c="0.85",
        s=16,
        label="not recruited",
    )
    ax.scatter(
        positions[recruited & ~has_velocity, 0],
        positions[recruited & ~has_velocity, 1],
        c="0.55",
        s=18,
        label="no local fit",
    )

    if np.any(has_velocity):
        sc = ax.scatter(
            positions[has_velocity, 0],
            positions[has_velocity, 1],
            c=local_speeds[has_velocity],
            s=24,
            cmap="plasma",
        )
        ax.quiver(
            positions[has_velocity, 0],
            positions[has_velocity, 1],
            velocity_vectors[has_velocity, 0],
            velocity_vectors[has_velocity, 1],
            angles="xy",
            scale_units="xy",
            scale=None,
            width=0.004,
            color="black",
        )
        plt.colorbar(sc, ax=ax, label="Local speed (um/s)")

    ax.set_title(title)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_aspect("equal", adjustable="box")
    return ax


def save_basic_plots(
    positions: np.ndarray,
    activation_times: np.ndarray,
    plane_coefficients: np.ndarray,
    output_prefix: str,
    local_velocity_vectors: np.ndarray = None,
    local_speed_um_per_s: np.ndarray = None,
    dpi: int = 160,
) -> None:
    """Save recruitment, raster, plane-fit, and local-velocity PNGs."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    activation_map(positions, activation_times, ax=ax, title="Recruitment times")
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_recruitment_times_map.png", dpi=dpi)
    fig.savefig(f"{output_prefix}_activation_map.png", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    activation_raster(activation_times, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_activation_raster.png", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_plane_fit(positions, activation_times, plane_coefficients, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_plane_fit.png", dpi=dpi)
    plt.close(fig)

    if local_velocity_vectors is None or local_speed_um_per_s is None:
        local = estimate_local_velocity_field(positions, activation_times)
        local_velocity_vectors = local["velocity_vectors"]
        local_speed_um_per_s = local["speed_um_per_s"]

    fig, ax = plt.subplots(figsize=(6, 5))
    local_velocity_field(
        positions,
        activation_times,
        local_velocity_vectors,
        local_speed_um_per_s,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_local_velocity_field.png", dpi=dpi)
    plt.close(fig)
