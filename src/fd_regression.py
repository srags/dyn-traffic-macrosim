"""Fundamental diagram (FD) regression on observed (rho, q) data, and the plots that go with it.

Three ways of fitting, all built on optimization_utils.fit_fd:
  - Rolling FD:       a static FD fit to each sliding time window, in sequence.
  - Drifting FD:      one FD fit to the whole period whose parameters change over time.
  - Cross-validation: static and drifting FDs fit to different subsets of the data, scored on the
                      held-out chunks.

Inputs are per-lane arrays of shape (num_timesteps, num_segments). Times are in minutes.
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from optimization_utils import FD_KEYS, fd_capacity, fd_curve, fit_fd


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

COLOR, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
FD_LABELS = {"v_free": r"$v_{free}$", "p_crit": r"$\rho_{crit}$", "a": r"$a$", "q_cap": r"$q_{cap}$"}


def _style_axes(ax):
    """Recessive axes: hairline y grid, no top/right spines, muted ticks."""
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def _segment_grid(num_panels, ncols, panel_size, **kwargs):
    """One panel per segment in rows of ncols; unused panels are hidden."""
    ncols = min(ncols, num_panels)
    nrows = math.ceil(num_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, layout="constrained",
                             figsize=(panel_size[0] * ncols, panel_size[1] * nrows), **kwargs)
    for ax in axes.flat[num_panels:]:
        ax.set_visible(False)
    return fig, axes


def _finish(fig, save_path):
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def _rmse(pred, obs):
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


# ---------------------------------------------------------------------------
# Rolling FD: a static FD fit to each sliding window, in sequence
# ---------------------------------------------------------------------------

def rolling_fd_fit(rho, q, H, k, segments=None, bounds=None):
    """Fit a static FD to each sliding window [s, s + H), s = 0, k, 2k, ..., for each segment.

    H and k are in timesteps. Returns (starts, fits) where fits[key] has shape
    (num_windows, len(segments)) for key in ("v_free", "p_crit", "a", "q_cap").
    """
    num_timesteps, num_segments = rho.shape
    segments = list(range(num_segments)) if segments is None else list(segments)
    starts = np.arange(0, num_timesteps - H + 1, k)
    fits = {key: np.full((len(starts), len(segments)), np.nan) for key in FD_KEYS + ("q_cap",)}
    for j, i in enumerate(segments):
        for w, s in enumerate(starts):
            values = fit_fd(rho[s:s + H, i], q[s:s + H, i], bounds=bounds)["params_at"]()
            for key in FD_KEYS:
                fits[key][w, j] = values[key]
            fits["q_cap"][w, j] = fd_capacity(**values)
    return starts, fits


def _ref_value(ref_params, key, i):
    """Value of `key` for segment i in a static METANET param dict; q_cap is computed from the others."""
    if key == "q_cap":
        return fd_capacity(*(_ref_value(ref_params, k, i) for k in FD_KEYS))
    p = np.asarray(ref_params[key])
    return float(p[0, i]) if p.ndim == 2 else float(p.reshape(-1)[i])


def plot_rolling_fd_params(starts, fits, H, k, dt_min=10 / 60, segments=None, ref_params=None,
                           ref_label="Static calibration", panel_size=(2.4, 2.0), title=None, save_path=None):
    """One row per FD parameter (and capacity), one column per segment; x is the window center
    in minutes from the start. dt_min: minutes per timestep. segments: labels for the columns of
    fits, i.e. the segments passed to rolling_fd_fit (default 0, 1, ...).

    ref_params: optional static METANET param dict drawn as a dashed reference line.
    """
    keys = [("v_free", r"$v_{free}$"), ("p_crit", r"$\rho_{crit}$"),
            ("a", r"$a$"), ("q_cap", r"$q_{cap}$ ")]
    num_segments = fits["v_free"].shape[1]
    segments = list(range(num_segments)) if segments is None else list(segments)
    t_center = (starts + H / 2) * dt_min

    fig, axes = plt.subplots(len(keys), len(segments), sharex=True, sharey="row", squeeze=False,
                             layout="constrained",
                             figsize=(panel_size[0] * len(segments), panel_size[1] * len(keys)))
    for r, (key, ylabel) in enumerate(keys):
        for j, i in enumerate(segments):
            ax = axes[r, j]
            ax.plot(t_center, fits[key][:, j], color="k", lw=1.5, marker="o", ms=2)
            if ref_params is not None:
                ax.axhline(_ref_value(ref_params, key, i), color="tab:red", ls="--", lw=1, label=ref_label)
            if r == 0:
                ax.set_title(f"S{i}", fontsize=20)
        axes[r, 0].set_ylabel(ylabel, fontsize=20)
    if ref_params is not None:
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper right", fontsize=9)
    fig.supxlabel("Window center (min from start)")
    fig.suptitle(title if title is not None else
                 f"Rolling FD fit (H = {H * dt_min:g} min, k = {k * dt_min:g} min)")
    _finish(fig, save_path)
    return fig, axes


def plot_rolling_fd_curves(starts, fits, H, dt_min=10 / 60, segments=None, ncols=7, rho_max=80, q_max=None,
                           cmap="viridis", panel_size=(2.8, 2.4), title=None, save_path=None):
    """One panel per segment with every rolling-window FD overlaid, colored by window center time.
    dt_min: minutes per timestep. segments: labels for the columns of fits, i.e. the segments
    passed to rolling_fd_fit (default 0, 1, ...)."""
    num_segments = fits["v_free"].shape[1]
    segments = list(range(num_segments)) if segments is None else list(segments)
    t_center = (starts + H / 2) * dt_min
    norm = plt.Normalize(t_center.min(), t_center.max())
    colors = plt.get_cmap(cmap)(norm(t_center))
    rho_grid = np.linspace(0, rho_max, 400)

    fig, axes = _segment_grid(len(segments), ncols, panel_size, sharex=True, sharey=True)
    max_capacity = 0.0
    for j, i in enumerate(segments):   # column j of fits, labeled with segment i
        ax = axes.flat[j]
        for w in range(len(starts)):
            q_fd = fd_curve(rho_grid, fits["v_free"][w, j], fits["p_crit"][w, j], fits["a"][w, j])
            max_capacity = max(max_capacity, q_fd.max())
            ax.plot(rho_grid, q_fd, color=colors[w], lw=1)
        ax.set_title(f"S{i}", fontsize=9)
        ax.set_xlim(0, rho_max)
    axes.flat[0].set_ylim(0, 1.05 * max_capacity if q_max is None else q_max)

    fig.supxlabel(r"$\rho$ (veh/km/lane)")
    fig.supylabel("q (veh/h/lane)")
    fig.suptitle(title if title is not None else f"Rolling FD fits (H = {H * dt_min:g} min)")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(),
                 label="Window center (min from start)", fraction=0.015, pad=0.01)
    _finish(fig, save_path)
    return fig, axes


# ---------------------------------------------------------------------------
# Drifting FD: one FD fit to the whole period, with parameters that change over time
# ---------------------------------------------------------------------------

def drifting_fd_table(rho, q, t, drift, segments=None):
    """Fit a static FD and a drifting FD (fit_fd with `drift`) to each segment.

    drift: {parameter: polynomial degree}, as in fit_fd.
    Returns (coef_table, fits, fit_stats):
      coef_table: polynomial coefficients of each parameter's path, p(t) = c0 + c1 t + c2 t^2 + ...
                  with t in hours from the start (c0 = value at the start). Constant parameters only have c0.
      fits:       {segment: (static fit, drifting fit)}, fit_fd results, for plot_drifting_fd.
      fit_stats:  % reduction in fit error vs the static FD, F and p_naive (which treats every
                  observation as independent, so it overstates confidence).
    """
    segments = range(rho.shape[1]) if segments is None else segments
    n = len(t)
    coef_rows, stat_rows, fits = {}, {}, {}
    for i in segments:
        static = fit_fd(rho[:, i], q[:, i])
        drifting = fit_fd(rho[:, i], q[:, i], t, drift)
        fits[i] = (static, drifting)
        coef_rows[i] = {(key, f"c{k}"): c for key in FD_KEYS for k, c in enumerate(drifting["coefs"][key])}
        extra = drifting["n_params"] - static["n_params"]
        dof = n - drifting["n_params"]
        F = ((static["rss"] - drifting["rss"]) / extra) / (drifting["rss"] / dof) if extra else np.nan
        stat_rows[i] = {"RSS_reduction_%": 100 * (static["rss"] - drifting["rss"]) / static["rss"],
                        "F": F, "p_naive": stats.f.sf(F, extra, dof) if extra else np.nan}
    coef_table = pd.DataFrame.from_dict(coef_rows, orient="index")
    coef_table.columns = pd.MultiIndex.from_tuples(coef_table.columns)
    fit_stats = pd.DataFrame.from_dict(stat_rows, orient="index")
    for table in (coef_table, fit_stats):
        table.index.name = "segment"
    return coef_table, fits, fit_stats


def plot_drifting_fd(rho, q, t, fits, segments=None, num_curves=None, ncols=7, rho_max=80, cmap="viridis",
                     panel_size=(2.8, 2.4), title=None, save_path=None):
    """Data colored by time, with the drifting FD drawn at num_curves evenly spaced times (same
    colormap) and the static FD (dashed black) for comparison. fits: from drifting_fd_table.
    num_curves defaults to the highest drift degree + 2."""
    segments = list(fits) if segments is None else list(segments)
    drifting0 = fits[segments[0]][1]
    if num_curves is None:
        num_curves = max(2, max(drifting0["drift"].values(), default=0) + 2)
    t_curves = np.linspace(*(drifting0["t_span"] or (t.min(), t.max())), num_curves)
    norm = plt.Normalize(t.min(), t.max())
    cm = plt.get_cmap(cmap)
    rho_grid = np.linspace(0, rho_max, 400)

    fig, axes = _segment_grid(len(segments), ncols, panel_size, sharex=True, sharey=True)
    for p, i in enumerate(segments):
        ax = axes.flat[p]
        static, drifting = fits[i]
        ax.scatter(rho[:, i], q[:, i], c=t, cmap=cmap, norm=norm, s=6, alpha=0.5)
        ax.plot(rho_grid, static["predict"](rho_grid), "k--", lw=1.2)
        for t_c in t_curves:
            ax.plot(rho_grid, drifting["predict"](rho_grid, np.full_like(rho_grid, t_c)), color=cm(norm(t_c)), lw=2)
        ax.set_title(f"S{i}", fontsize=9)
        ax.set_xlim(0, rho_max)
        ax.set_ylim(bottom=0)

    handles = [Line2D([0], [0], color="k", ls="--", lw=1.2, label="Static FD"),
               Line2D([0], [0], color=cm(0.5), lw=2, label="Drifting FD (colored by time)")]
    fig.legend(handles=handles, loc="outside upper right", fontsize=9, ncol=2)
    fig.supxlabel(r"$\rho$ (veh/km/lane)")
    fig.supylabel("q (veh/h/lane)")
    drift_desc = ", ".join(f"{key}: {d}" for key, d in drifting0["drift"].items())
    fig.suptitle(title if title is not None else f"Drifting FD fit ({drift_desc})")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(),
                 label="Time (min from start)", fraction=0.015, pad=0.01)
    _finish(fig, save_path)
    return fig, axes


# ---------------------------------------------------------------------------
# Cross-validation: static and drifting FDs fit to different subsets of the data
# ---------------------------------------------------------------------------

def cv_folds(n, block_len, gap):
    """Blocked CV folds: [(train_idx, test_idx), ...], one per contiguous block of block_len.
    Training excludes the test block plus `gap` indices on each side."""
    folds = []
    for s in range(0, n, block_len):
        test = np.arange(s, min(s + block_len, n))
        keep = np.ones(n, bool)
        keep[max(0, s - gap):min(n, s + block_len + gap)] = False
        folds.append((np.flatnonzero(keep), test))
    return folds


def cross_validate_fd(rho, q, t, models, block_min=5, gap_min=1, segments=None):
    """Blocked cross-validation of FD models (t in minutes).

    models: {label: drift dict}, e.g. {"Static FD": {}, "v_free": {"v_free": 1}}. Each is fit with
            fit_fd. The model with an empty drift dict is the static reference; one labeled
            "Static FD" is added first if none is given.
    The hour is split into chunks of block_min minutes. Each chunk is held out once as the test
    set, with gap_min minutes on each side also dropped from training.

    Returns a dict with:
      errors:  long DataFrame (segment, fold, model, train_rmse, test_rmse)
      fits:    {label: [[fit_fd result per fold] per segment]}
      full:    {label: [fit_fd result fit to all data, per segment]}
      models, static, segments, folds, t_span, block_min, gap_min.
    """
    models = dict(models)
    static_labels = [label for label, drift in models.items() if not drift]
    if not static_labels:
        models = {"Static FD": {}, **models}
        static_labels = ["Static FD"]
    segments = list(range(rho.shape[1])) if segments is None else list(segments)
    dt = t[1] - t[0]
    folds = cv_folds(len(t), max(1, int(round(block_min / dt))), int(round(gap_min / dt)))
    t_span = (float(t.min()), float(t.max()))

    rows = []
    fits = {label: [] for label in models}
    full = {label: [] for label in models}
    for i in segments:
        r, qq = rho[:, i], q[:, i]
        for label, drift in models.items():
            full[label].append(fit_fd(r, qq, t, drift, t_span))
            fits[label].append([])
        for f, (train, test) in enumerate(folds):
            for label, drift in models.items():
                fit = fit_fd(r[train], qq[train], t[train], drift, t_span)
                fits[label][-1].append(fit)
                rows.append({"segment": i, "fold": f, "model": label,
                             "train_rmse": _rmse(fit["predict"](r[train], t[train]), qq[train]),
                             "test_rmse": _rmse(fit["predict"](r[test], t[test]), qq[test])})
    return {"errors": pd.DataFrame(rows), "fits": fits, "full": full, "models": models,
            "static": static_labels[0], "segments": segments, "folds": folds, "t_span": t_span,
            "block_min": block_min, "gap_min": gap_min}


def cv_error_table(cv, segments=None):
    """Average training and test RMSE (veh/h/lane) over folds (and the chosen segments), one row per model.
    The improvement columns are how much lower each RMSE is than the static FD's, in %
    (positive = better than static, negative = worse)."""
    err = cv["errors"] if segments is None else cv["errors"][cv["errors"]["segment"].isin(segments)]
    table = err.groupby("model")[["train_rmse", "test_rmse"]].mean().reindex(list(cv["models"]))
    table.columns = ["Avg train RMSE", "Avg test RMSE"]
    for split in ("train", "test"):
        col = f"Avg {split} RMSE"
        static = table.loc[cv["static"], col]
        table[f"{split.capitalize()} improvement vs static (%)"] = 100 * (static - table[col]) / static
    return table[["Avg train RMSE", "Train improvement vs static (%)",
                  "Avg test RMSE", "Test improvement vs static (%)"]]


def cv_segment_table(cv, split="test", relative=True, decimals=1, highlight=True):
    """Per-segment comparison of the models: one row per segment, one column per model.

    relative=True: how much lower each model's average RMSE is than the static FD's, in %
    (positive = better than static, negative = worse; the static column is 0), as in cv_error_table.
    relative=False: average RMSE (veh/h/lane) over folds.
    split: "test" or "train". With highlight=True, returns a pandas Styler with each segment's
    best model highlighted; otherwise the plain DataFrame."""
    table = (cv["errors"].pivot_table(index="segment", columns="model", values=f"{split}_rmse")
                .reindex(columns=list(cv["models"])))
    table.columns.name = None
    if relative:
        static = table[cv["static"]]
        table = table.rsub(static, axis=0).div(static, axis=0) * 100
    if not highlight:
        return table.round(decimals)
    props = f"background-color: #dbe8f8; color: {INK}; font-weight: bold;"
    styled = table.style.format(f"{{:+.{decimals}f}}%" if relative else f"{{:.{decimals}f}}")
    return styled.highlight_max(axis=1, props=props) if relative else styled.highlight_min(axis=1, props=props)


def plot_cv_param_distributions(cv, model, segments=None, relative=False, panel_size=(4.5, 2.2),
                                title=None, save_path=None):
    """Spread across CV folds of every polynomial coefficient of `model`, one panel per coefficient
    (rows: v_free, rho_crit, a; columns: c0 .. c_degree; t in hours from the start). Constant
    parameters only have c0.
    For each segment: the fit to all data (solid dot), the range over the folds (line), and each
    fold (faint dots). relative=True divides by the full-data static FD value (in %), so c1 is
    % change per hour."""
    seg_list = cv["segments"]
    segments = seg_list if segments is None else list(segments)
    si = [seg_list.index(i) for i in segments]
    num_folds = len(cv["folds"])
    ncols = max(len(cv["full"][model][0]["coefs"][key]) for key in FD_KEYS)

    fig, axes = plt.subplots(3, ncols, sharex=True, squeeze=False, layout="constrained",
                             figsize=(panel_size[0] * ncols, panel_size[1] * 3))
    x = np.arange(len(segments))
    for j, key in enumerate(FD_KEYS):
        n_coefs = len(cv["full"][model][0]["coefs"][key])
        for k in range(ncols):
            ax = axes[j, k]
            if k >= n_coefs:   # this parameter has no c_k (lower degree or constant)
                ax.set_visible(False)
                continue
            vals = np.array([[fit["coefs"][key][k] for fit in cv["fits"][model][s]] for s in si])
            full = np.array([cv["full"][model][s]["coefs"][key][k] for s in si])
            if relative:
                scale = 100 / np.array([cv["full"][cv["static"]][s]["params"][key][0] for s in si])
                vals, full = vals * scale[:, None], full * scale
            ax.vlines(x, vals.min(axis=1), vals.max(axis=1), color=COLOR, alpha=0.35, lw=2, capstyle="round", zorder=2)
            ax.scatter(np.repeat(x, vals.shape[1]), vals.ravel(), s=12, color=COLOR, alpha=0.3, lw=0, zorder=3)
            ax.scatter(x, full, s=42, color=COLOR, edgecolor="white", linewidth=1.5, zorder=4)
            if k >= 1:
                ax.axhline(0, color=MUTED, lw=1, zorder=1)
            _style_axes(ax)
            unit = ""
            if relative:
                unit = " (% of static)" if k == 0 else (" (% per hour)" if k == 1 else f" (% per hour$^{k}$)")
            ax.set_title(f"{FD_LABELS[key]}: c{k}{unit}", fontsize=10, color=INK)
            ax.set_xticks(x, [f"S{i}" for i in segments])
            ax.tick_params(labelbottom=True)
    handles = [Line2D([0], [0], marker="o", ls="none", color=COLOR, mec="white", mew=1.5, ms=7,
                      label="Fit to all data"),
               Line2D([0], [0], color=COLOR, alpha=0.35, lw=2, marker="o", ms=4, mew=0,
                      label=f"Range over {num_folds} CV folds")]
    fig.legend(handles=handles, loc="outside lower center", fontsize=9, ncol=2, frameon=False, labelcolor=MUTED)
    fig.suptitle(title if title is not None else f"Coefficients across {num_folds} CV folds ({model})", color=INK)
    _finish(fig, save_path)
    return fig, axes


def plot_cv_capacity(cv, t, model, segments=None, ncols=7, panel_size=(2.8, 2.4), title=None, save_path=None):
    """Capacity rho_crit * v_free * exp(-1/a) over time for `model`, one panel per segment (t in minutes).
    Line: fit to all data. Band: range over the CV folds. Dashed: static FD capacity."""
    seg_list = cv["segments"]
    segments = seg_list if segments is None else list(segments)
    si = [seg_list.index(i) for i in segments]

    fig, axes = _segment_grid(len(segments), ncols, panel_size, sharex=True, sharey=True)
    for p, s in enumerate(si):
        ax = axes.flat[p]
        folds = np.array([fit["capacity"](t) for fit in cv["fits"][model][s]])
        ax.fill_between(t, folds.min(axis=0), folds.max(axis=0), color=COLOR, alpha=0.15, lw=0)
        ax.axhline(cv["full"][cv["static"]][s]["capacity"](), color=MUTED, ls="--", lw=1)
        ax.plot(t, cv["full"][model][s]["capacity"](t), color=COLOR, lw=2, solid_capstyle="round")
        ax.set_title(f"S{segments[p]}", fontsize=9, color=INK)
        _style_axes(ax)
        if p + axes.shape[1] >= len(segments):   # x label on the lowest panel of each column
            ax.set_xlabel("Time (min from start)", color=INK, fontsize=9)

    handles = [Line2D([0], [0], color=COLOR, lw=2, label=f"{model}, fit to all data"),
               Patch(facecolor=COLOR, alpha=0.15, label=f"Range over {len(cv['folds'])} CV folds"),
               Line2D([0], [0], color=MUTED, ls="--", lw=1, label=cv["static"])]
    fig.legend(handles=handles, loc="outside lower center", fontsize=9, ncol=3, frameon=False, labelcolor=MUTED)
    fig.supylabel(r"Capacity $q_{cap}$ (veh/h/lane)", color=INK)
    fig.suptitle(title if title is not None else f"FD capacity over time ({model})", color=INK)
    _finish(fig, save_path)
    return fig, axes
