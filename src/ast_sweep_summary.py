"""Summaries of an ``experiments/ast_sweep.py`` parameter sweep.

All functions take the sweep's per-sample results as a DataFrame:

* :func:`outcome_table` counts discarded, failed and safe samples, and
  :func:`save_outcome_table` renders that table as a PNG.
* :func:`save_ttf_vs_mape` plots time to failure against a nominal MAPE
  (velocity or density) for the samples tested under VSL (failed under VSL or
  safe).
* :func:`best_safe_sample` picks the safe sample with the lowest average of
  velocity and density MAPE,
  and :func:`save_time_space_comparison` plots its simulation against the data
  (``experiments/ast_sweep.py`` reruns the simulation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Outcome labels written by ast_sweep.py
DISCARDED = "discarded"
FAILED_NOMINAL = "failed_nominal"
FAILED_VSL = "failed_vsl"
SAFE = "safe"
ERROR = "error"

# Reference categorical accent for "failed under VSL" plus a recessive gray for "safe".
COLOR_FAILED_VSL = "#eb6834"
COLOR_SAFE = "#8a8984"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def outcome_table(results: pd.DataFrame) -> pd.DataFrame:
    """Return counts and shares of each sample outcome.

    Rows: discarded (MAPE above threshold, split by which MAPE), failed (split
    into "without VSL" and "under VSL", plus a total), safe, and errors if any
    occurred.
    """

    counts = results["outcome"].value_counts()
    total = int(len(results))
    failed_nominal = int(counts.get(FAILED_NOMINAL, 0))
    failed_vsl = int(counts.get(FAILED_VSL, 0))
    rows = [("Discarded (MAPE above threshold)", int(counts.get(DISCARDED, 0)))]
    if "discard_reason" in results:  # absent in CSVs from older versions
        reasons = results.loc[results["outcome"] == DISCARDED, "discard_reason"]
        reason_counts = reasons.value_counts()
        for reason, label in (
            ("velocity", "  velocity MAPE only"),
            ("density", "  density MAPE only"),
            ("velocity+density", "  both"),
        ):
            if reason_counts.get(reason, 0):
                rows.append((label, int(reason_counts[reason])))
    rows += [
        ("Failed", failed_nominal + failed_vsl),
        ("  without VSL (nominal)", failed_nominal),
        ("  under VSL (found by AST)", failed_vsl),
        ("Safe (no failure found)", int(counts.get(SAFE, 0))),
    ]
    if counts.get(ERROR, 0):
        rows.append(("Errors", int(counts[ERROR])))
    rows.append(("Total samples", total))
    table = pd.DataFrame(rows, columns=["outcome", "samples"])
    table["percent_of_total"] = (100.0 * table["samples"] / max(total, 1)).round(1)
    return table


def save_outcome_table(
    table: pd.DataFrame, output_path: str | Path, *, title: Optional[str] = None
) -> Path:
    """Render the outcome table from :func:`outcome_table` as a PNG."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    headers = ["Outcome", "Samples", "% of total"]
    cells = [
        [str(row.outcome), f"{int(row.samples)}", f"{row.percent_of_total:.1f}"]
        for row in table.itertuples(index=False)
    ]

    fig, ax = plt.subplots(figsize=(5.6, 0.3 * (len(cells) + 1)), dpi=200)
    ax.axis("off")
    grid = ax.table(
        cellText=cells,
        colLabels=headers,
        colWidths=[0.62, 0.17, 0.21],
        cellLoc="right",
        colLoc="right",
        loc="center",
    )
    grid.auto_set_font_size(False)
    grid.set_fontsize(9)
    grid.scale(1, 1.35)

    last_row = len(cells)
    for (r, c), cell in grid.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.visible_edges = "B"
        if c == 0:
            cell.set_text_props(ha="left")
            cell._loc = "left"
        if r == 0:
            cell.set_text_props(weight="bold", color=TEXT_SECONDARY)
            cell.set_edgecolor(TEXT_SECONDARY)
        elif r == last_row:
            cell.set_text_props(weight="bold")
            cell.visible_edges = "T"
            cell.set_edgecolor(TEXT_SECONDARY)
        elif cells[r - 1][0].startswith("  "):
            cell.set_text_props(color=TEXT_SECONDARY)

    if title:
        ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def save_ttf_vs_mape(
    results: pd.DataFrame,
    output_path: str | Path,
    *,
    window_minutes: Optional[float] = None,
    variable: str = "velocity",
) -> Path:
    """Scatter time to failure (x) against nominal ``variable`` MAPE (y) under VSL.

    Only samples that reached the AST search are plotted: those that failed
    under VSL, at their time to failure, and safe ones. Samples that already
    failed without VSL are left out. Safe samples have no failure
    time, so they are drawn as hollow gray markers at the end of the window,
    to the right of a dashed line.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Samples that already fail without VSL are left out; only those that
    # reach the AST search (failed under VSL or safe) are plotted.
    kept = results[results["outcome"].isin([FAILED_VSL, SAFE])]
    mape_column = f"nominal_{variable}_mape"
    if window_minutes is None:
        window_minutes = float(kept["time_to_failure_min"].max() or 60.0)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    series = [(FAILED_VSL, "Failed under VSL", COLOR_FAILED_VSL)]
    for outcome, label, color in series:
        subset = kept[kept["outcome"] == outcome]
        ax.scatter(
            subset["time_to_failure_min"],
            subset[mape_column],
            s=14,
            color=color,
            alpha=0.6,
            linewidths=0,
            label=f"{label} (n={len(subset)})",
        )

    safe = kept[kept["outcome"] == SAFE]
    if len(safe):
        safe_x = window_minutes * 1.04
        ax.axvline(window_minutes, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
        ax.scatter(
            [safe_x] * len(safe),
            safe[mape_column],
            s=14,
            facecolors="none",
            edgecolors=COLOR_SAFE,
            linewidths=0.8,
            alpha=0.7,
            label=f"Safe, no failure in {window_minutes:g} min (n={len(safe)})",
        )
        ax.set_xlim(left=0, right=window_minutes * 1.08)
    else:
        ax.set_xlim(left=0, right=window_minutes)

    ax.set_xlabel("Time until failure (min)")
    ax.set_ylabel(f"Nominal {variable} MAPE (%)")
    ax.set_ylim(bottom=0)
    ax.set_title(
        f"Time to failure vs. nominal {variable} MAPE "
        f"({len(kept)} samples tested under VSL)",
        loc="left",
        fontsize=11,
    )
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def best_safe_sample(
    results: pd.DataFrame, exclude_ids: Sequence[int] = ()
) -> Optional[pd.Series]:
    """Return the safe sample with the lowest average of velocity and density
    MAPE, ignoring ``exclude_ids``, or None if no such sample is safe."""

    safe = results[
        (results["outcome"] == SAFE) & ~results["sample_id"].isin(list(exclude_ids))
    ]
    if safe.empty:
        return None
    average = (safe["nominal_velocity_mape"] + safe["nominal_density_mape"]) / 2.0
    return safe.loc[average.idxmin()]


def save_time_space_comparison(
    truth: dict,
    simulated: dict,
    T: float,
    segment_length: float,
    output_path: str | Path,
    *,
    title: Optional[str] = None,
) -> Path:
    """Time-space diagrams of ground truth (left) vs. simulation (right).

    ``truth`` and ``simulated`` map "velocity" and "density" to arrays of shape
    (time steps, segments). Each row shares one color scale.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        ("velocity", "Velocity (km/h)", "RdYlGn"),
        ("density", "Density (veh/km)", "RdYlGn_r"),
    ]
    fig, axes = plt.subplots(
        len(rows), 2, figsize=(10, 6.4), dpi=150, sharex=True, sharey=True,
        constrained_layout=True,
    )
    for (key, label, cmap), row_axes in zip(rows, axes):
        gt = np.asarray(truth[key], dtype=float)
        sim = np.asarray(simulated[key], dtype=float)
        n_steps, n_segments = gt.shape
        finite = np.concatenate([gt[np.isfinite(gt)], sim[np.isfinite(sim)]])
        vmin, vmax = np.percentile(finite, [1, 99]) if finite.size else (0, 1)
        extent = [0, n_steps * T * 60.0, 0, n_segments * segment_length]
        for ax, data, name in zip(row_axes, (gt, sim), ("Ground truth", "Simulated")):
            image = ax.imshow(
                data.T,
                aspect="auto",
                origin="lower",
                cmap=cmap,
                interpolation="none",
                vmin=vmin,
                vmax=vmax,
                extent=extent,
            )
            ax.set_title(f"{name} {key}", loc="left", fontsize=10)
            ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(TEXT_SECONDARY)
        colorbar = fig.colorbar(image, ax=list(row_axes), shrink=0.9, pad=0.01)
        colorbar.set_label(label, fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
        row_axes[0].set_ylabel("Distance (km)")
    for ax in axes[-1]:
        ax.set_xlabel("Time (min)")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=10)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)
    return output_path


def output_path(prefix: str | Path, name: str) -> Path:
    """``<prefix>_<name>.png``; a prefix ending in a separator gives ``<dir>/<name>.png``."""

    prefix = str(prefix)
    if prefix.endswith(("/", "\\")):
        return Path(prefix) / f"{name}.png"
    return Path(f"{prefix}_{name}.png")


def write_summary(
    results: pd.DataFrame,
    fig_prefix: str | Path,
    window_minutes: Optional[float] = None,
    threshold_text: Optional[str] = None,
) -> dict:
    """Save the outcome table and the two scatter plots as PNGs.

    Files are ``<fig_prefix>_outcomes.png``, ``<fig_prefix>_ttf_vs_velocity_mape.png``
    and ``<fig_prefix>_ttf_vs_density_mape.png``. ``threshold_text`` (e.g.
    "kept if velocity MAPE <= 70%") is shown as the table's title.
    """

    table = outcome_table(results)
    table_path = save_outcome_table(
        table,
        output_path(fig_prefix, "outcomes"),
        title=threshold_text[0].upper() + threshold_text[1:] if threshold_text else None,
    )
    plot_paths = [
        save_ttf_vs_mape(
            results,
            output_path(fig_prefix, f"ttf_vs_{variable}_mape"),
            window_minutes=window_minutes,
            variable=variable,
        )
        for variable in ("velocity", "density")
    ]
    return {
        "table": table,
        "table_path": table_path,
        "plot_paths": plot_paths,
    }
