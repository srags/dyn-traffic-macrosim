"""Focused visual summaries for the Adaptive Stress Testing proof of concept.

The plotting functions consume the JSON-compatible dictionary produced by
``experiments/ast_poc.py``. They are intentionally separate from simulation so
saved results can be rendered again without rerunning MCTS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


MODE_ORDER = ("vsl-ast", "vsl-min", "inflow-ast")
MODE_LABELS = {
    "nominal": "Nominal",
    "vsl-ast": "VSL · standard AST",
    "vsl-min": "VSL · minimum intervention",
    "inflow-ast": "Inflow · standard AST",
}
CALIBRATION_COLORS = {
    "static": "#5B55A5",
    "dynamic": "#168A8A",
}
# Fixed-order categorical palette for any other calibration keys (assigned by
# position in the results, so a calibration keeps its color across figures).
EXTRA_CALIBRATION_COLORS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
)


def _calibration_colors(
    calibrations: Sequence[Tuple[str, Mapping[str, Any]]],
) -> dict:
    colors = {}
    extra = 0
    for key, _batch in calibrations:
        if key in CALIBRATION_COLORS:
            colors[key] = CALIBRATION_COLORS[key]
        else:
            colors[key] = EXTRA_CALIBRATION_COLORS[extra % len(EXTRA_CALIBRATION_COLORS)]
            extra += 1
    return colors


def _comparison_title(calibrations: Sequence[Tuple[str, Mapping[str, Any]]]) -> str:
    keys = [key for key, _batch in calibrations]
    if keys == ["static", "dynamic"]:
        return "static vs frozen dynamic"
    if len(keys) == 1:
        return _short_calibration_label(*calibrations[0])
    return "across calibrations"


def _bar_height(count: int) -> float:
    return 0.27 if count <= 2 else 0.8 / count * 0.8


def _calibrations(
    results: Mapping[str, Any],
) -> List[Tuple[str, Mapping[str, Any]]]:
    """Return calibration result batches, including legacy single-batch JSON."""

    if "calibrations" in results:
        return list(results["calibrations"].items())
    return [
        (
            "static",
            {
                "label": "Static calibration",
                "nominal": results["nominal"],
                "searches": results.get("searches", {}),
            },
        )
    ]


def _episodes(batch: Mapping[str, Any]) -> List[Tuple[str, Mapping[str, Any]]]:
    """Return nominal and available search episodes in a stable order."""

    episodes: List[Tuple[str, Mapping[str, Any]]] = [("nominal", batch["nominal"])]
    searches = batch.get("searches", {})
    for mode in MODE_ORDER:
        if mode in searches:
            episodes.append((mode, searches[mode]["best_episode"]))
    return episodes


def _available_modes(calibrations: Sequence[Tuple[str, Mapping[str, Any]]]) -> List[str]:
    """Return the union of plotted modes in their intended display order."""

    present = {mode for _key, batch in calibrations for mode, _episode in _episodes(batch)}
    return [mode for mode in ("nominal", *MODE_ORDER) if mode in present]


def _short_calibration_label(key: str, batch: Mapping[str, Any]) -> str:
    if batch.get("short_label"):
        return str(batch["short_label"])
    if key == "static":
        return "Static"
    if key == "dynamic":
        horizon = batch.get("control_horizon_steps")
        return f"Frozen dynamic (h={horizon})" if horizon is not None else "Frozen dynamic"
    return str(batch.get("label", key))


def _row_offsets(count: int) -> List[float]:
    """Center one or more calibration bars around each mode row."""

    if count == 1:
        return [0.0]
    spacing = 0.34 if count <= 2 else 0.8 / count
    midpoint = (count - 1) / 2.0
    return [(index - midpoint) * spacing for index in range(count)]


def _sequence_label(actions: Sequence[str]) -> str:
    """Format an action sequence compactly while preserving its order."""

    if not actions:
        return "—"
    compact = [
        action.replace("+0 sigma", "0σ")
        .replace("+1 sigma", "+1σ")
        .replace("+2 sigma", "+2σ")
        .replace("+3 sigma", "+3σ")
        .replace("-1 sigma", "−1σ")
        .replace("-2 sigma", "−2σ")
        .replace("-3 sigma", "−3σ")
        for action in actions
    ]
    if len(set(compact)) == 1 and len(compact) > 1:
        return f"{compact[0]} × {len(compact)}"
    return " → ".join(compact)


def _critical_raw_value(episode: Mapping[str, Any]) -> Tuple[float, str]:
    prefix = "failure" if episode.get("failed") else "closest_nonfailure"
    return (
        float(episode[f"{prefix}_raw_value"]),
        str(episode[f"{prefix}_subtype"]),
    )


def plot_time_to_first_invalid_state(results: Mapping[str, Any]) -> Figure:
    """Compare time to failure across static and frozen dynamic calibration."""

    scenario = results["scenario"]
    horizon = float(scenario["window_minutes"])
    calibrations = _calibrations(results)
    modes = _available_modes(calibrations)
    rows = list(range(len(modes)))
    offsets = _row_offsets(len(calibrations))
    colors = _calibration_colors(calibrations)
    bar_height = _bar_height(len(calibrations))

    figure, axis = plt.subplots(figsize=(11.5, 6.0), constrained_layout=True)
    for (key, batch), offset in zip(calibrations, offsets):
        episodes = dict(_episodes(batch))
        color = colors[key]
        calibration_label = _short_calibration_label(key, batch)
        for row, mode in enumerate(modes):
            if mode not in episodes:
                continue
            episode = episodes[mode]
            duration = (
                horizon
                if not episode.get("failed")
                else float(episode["time_to_failure_minutes"])
            )
            y_position = row + offset
            axis.barh(
                y_position,
                duration,
                color=color,
                alpha=0.82,
                height=bar_height,
                label=calibration_label if row == 0 else None,
            )
            if episode.get("failed"):
                axis.scatter(
                    duration,
                    y_position,
                    marker="x",
                    s=62,
                    linewidth=2.2,
                    color="#8B1E1E",
                    zorder=3,
                )
                annotation = f"{duration:.2f} min"
            else:
                axis.scatter(
                    duration,
                    y_position,
                    marker="o",
                    s=42,
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=3,
                )
                annotation = f"safe {duration:g} min"
            label_inside = duration > horizon * 0.7
            axis.annotate(
                annotation,
                (duration, y_position),
                xytext=(-8 if label_inside else 8, 0),
                textcoords="offset points",
                ha="right" if label_inside else "left",
                va="center",
                fontsize=9,
            )

    axis.set_yticks(rows, [MODE_LABELS.get(mode, mode) for mode in modes])
    axis.invert_yaxis()
    axis.set_xlim(0.0, horizon * 1.04)
    axis.set_xlabel("Simulated time (minutes)")
    axis.set_title(
        f"Time to first invalid state: {_comparison_title(calibrations)}",
        loc="left",
        fontsize=17,
        fontweight="bold",
        pad=12,
    )
    axis.grid(axis="x", alpha=0.25)
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=max(1, len(calibrations)),
        fontsize=9,
    )
    return figure


def plot_boundary_severity(results: Mapping[str, Any]) -> Figure:
    """Compare raw pre-clipping boundary severity across calibrations."""

    calibrations = _calibrations(results)
    modes = _available_modes(calibrations)
    critical = [
        _critical_raw_value(episode)
        for _key, batch in calibrations
        for _mode, episode in _episodes(batch)
    ]
    subtypes = [subtype for _value, subtype in critical]
    if not all("velocity" in subtype for subtype in subtypes):
        raise ValueError(
            "The boundary-severity plot currently requires every critical "
            "boundary to be a velocity boundary."
        )

    rows = list(range(len(modes)))
    offsets = _row_offsets(len(calibrations))
    colors = _calibration_colors(calibrations)
    bar_height = _bar_height(len(calibrations))
    figure, axis = plt.subplots(figsize=(11.5, 6.0), constrained_layout=True)
    boundary = float(results.get("scenario", {}).get("min_velocity", 0.0))
    axis.axvline(
        boundary,
        color="#8B1E1E",
        linewidth=1.7,
        label=f"failure boundary ({boundary:g} km/h)",
        zorder=1,
    )
    for (key, batch), offset in zip(calibrations, offsets):
        episodes = dict(_episodes(batch))
        color = colors[key]
        calibration_label = _short_calibration_label(key, batch)
        for row, mode in enumerate(modes):
            if mode not in episodes:
                continue
            value, _subtype = _critical_raw_value(episodes[mode])
            y_position = row + offset
            place_label_left = value > boundary - 0.4
            axis.barh(
                y_position,
                value,
                color=color,
                alpha=0.82,
                height=bar_height,
                zorder=2,
                label=calibration_label if row == 0 else None,
            )
            axis.scatter(value, y_position, color=color, s=48, zorder=3)
            axis.annotate(
                f"{value:+.3f} km/h",
                (value, y_position),
                xytext=(-8 if place_label_left else 8, 0),
                textcoords="offset points",
                ha="right" if place_label_left else "left",
                va="center",
                fontsize=9,
            )

    axis.set_yticks(rows, [MODE_LABELS.get(mode, mode) for mode in modes])
    axis.invert_yaxis()
    axis.set_xlabel("Raw velocity at critical point before clipping (km/h)")
    axis.set_title(
        f"Boundary severity before clipping: {_comparison_title(calibrations)}",
        loc="left",
        fontsize=17,
        fontweight="bold",
        pad=12,
    )
    axis.grid(axis="x", alpha=0.25)
    axis.legend(frameon=False, loc="lower left", fontsize=9)
    return figure


def plot_discovered_sequences(results: Mapping[str, Any]) -> Figure:
    """Render the discovered action sequences and their search metrics."""

    scenario = results["scenario"]
    calibrations = _calibrations(results)
    rows = []
    for key, batch in calibrations:
        calibration_label = _short_calibration_label(key, batch)
        for mode, episode in _episodes(batch):
            if mode == "nominal" and episode.get("failed"):
                sequence = "none"
                metric = f"failed {scenario['window_minutes']}-min gate"
                outcome = f"{episode['time_to_failure_minutes']:.2f} min"
            elif mode == "nominal":
                sequence = "none"
                metric = f"passed {scenario['window_minutes']}-min gate"
                outcome = "No failure"
            elif mode == "vsl-min":
                sequence = _sequence_label(episode.get("actions", []))
                metric = f"intervention cost = {episode['intervention_cost']:.2f}"
                outcome = (
                    f"{episode['time_to_failure_minutes']:.2f} min"
                    if episode.get("failed")
                    else "No failure"
                )
            else:
                sequence = _sequence_label(episode.get("actions", []))
                metric = f"NLL = {episode['negative_log_likelihood']:.2f}"
                outcome = (
                    f"{episode['time_to_failure_minutes']:.2f} min"
                    if episode.get("failed")
                    else "No failure"
                )

            segment = episode.get("failure_segment")
            if segment is None:
                location = "—"
            else:
                location = f"segment {int(segment)} (zero-based)"
            rows.append(
                [
                    calibration_label,
                    MODE_LABELS.get(mode, mode),
                    sequence,
                    metric,
                    outcome,
                    location,
                ]
            )

    figure_height = max(4.0, 1.0 + 0.55 * len(rows))
    figure, axis = plt.subplots(figsize=(15, figure_height), constrained_layout=True)
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=(
            "Calibration",
            "Run",
            "Applied sequence",
            "Search metric",
            "Outcome",
            "Location",
        ),
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=(0.16, 0.21, 0.17, 0.18, 0.12, 0.16),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.9)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#D2D2D2")
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#EAEAEA")
        else:
            cell.set_facecolor("white")

    axis.set_title(
        "Discovered AST sequences and objectives",
        loc="left",
        fontsize=17,
        fontweight="bold",
        pad=16,
    )
    axis.text(
        0.0,
        0.94,
        (
            "NLL applies only to standard AST; intervention cost is a separate "
            "minimum-intervention objective."
        ),
        transform=axis.transAxes,
        color="#555555",
        fontsize=10,
    )
    return figure


def _save_figure(figure: Figure, output_path: Path | str, dpi: int) -> Path:
    path = Path(output_path)
    if path.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ValueError("Figure output must end in .png, .pdf, or .svg.")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path.resolve()


def save_time_to_first_invalid_state(
    results: Mapping[str, Any], output_path: Path | str, *, dpi: int = 200
) -> Path:
    """Save the standalone time-to-failure bar chart."""

    return _save_figure(plot_time_to_first_invalid_state(results), output_path, dpi)


def save_boundary_severity(
    results: Mapping[str, Any], output_path: Path | str, *, dpi: int = 200
) -> Path:
    """Save the standalone raw boundary-severity bar chart."""

    return _save_figure(plot_boundary_severity(results), output_path, dpi)


def save_discovered_sequences(
    results: Mapping[str, Any], output_path: Path | str, *, dpi: int = 200
) -> Path:
    """Save the standalone discovered-sequences table."""

    return _save_figure(plot_discovered_sequences(results), output_path, dpi)
