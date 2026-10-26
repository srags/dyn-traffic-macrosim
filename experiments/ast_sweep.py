"""Sweep VSL Adaptive Stress Testing over uniformly sampled METANET parameters.

Two ways to draw the six parameters (tau, eta_high, K, p_crit, v_free, a):

* ``--sampling uniform`` (default): one value per parameter, used on every
  segment, uniform within the calibration bounds pulled inward by ``--inset``
  (a fraction of each range) so samples never sit exactly on a bound.
* ``--sampling calibration``: around the calibration in ``--calibration-dir``.
  Each sample draws one factor per parameter from U[1 - spread, 1 + spread]
  (``--spread``) and multiplies that parameter's calibrated value on every
  segment, keeping the calibrated differences between segments. Values outside
  the calibration bounds are clipped to them (no inset). Sample 0 uses factors
  of 1, i.e. the calibration itself.

Ramp terms (r, beta) always keep their calibrated per-segment values.

For every sample the script:

1. Simulates the undisturbed (nominal) trajectory over the whole window and
   records whether and when it failed, its distance to failure, and its MAPE
   against the I-24 data.
2. Discards the sample if its velocity MAPE is above ``--velocity-threshold``
   or its density MAPE is above ``--density-threshold`` (or the MAPE is not
   finite). Each threshold is optional; with neither, no sample is discarded.
   Discarded samples still count towards ``--n-samples``.
3. If the sample is kept and the nominal run did not fail, runs the same
   ``vsl-ast`` MCTS search as ``ast_poc.py`` and records the best episode.

Each sample gets one ``outcome``: discarded, failed_nominal, failed_vsl, safe
or error. Results are kept in memory and the outcome table is printed. Figures
are always saved, using ``--fig-output`` as a filename prefix:

* ``<prefix>_outcomes.png``: table of counts of discarded, failed and safe samples.
* ``<prefix>_ttf_vs_velocity_mape.png`` and ``<prefix>_ttf_vs_density_mape.png``:
  time to failure vs. each MAPE for samples tested under VSL.
* ``<prefix>_best_safe_time_space.png``: time-space diagrams of velocity and
  density, ground truth vs. the nominal (no VSL) simulation of the safe sample
  with the lowest average of velocity and density MAPE (excluding sample 0, the
  unperturbed centre, in calibration sampling).

``--json-output PATH`` additionally saves a JSON laid out like ``ast_poc.py``'s:
``scenario``, ``action_models``, ``sampling``, ``outcomes``, a compact
``samples`` list (parameters or factors, outcome, time to failure, MAPEs), and
``highlights`` (the best safe sample and the fastest VSL failures, rerun to
record their ``nominal`` and ``searches`` in the same format as ``ast_poc.py``).

Example (quick test)::

    python experiments/ast_sweep.py --n-samples 200 \
        --fig-output artifacts/ast_sweep/test_200 \
        --json-output artifacts/ast_sweep/test_200.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
import time
import traceback
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Make src/ and experiments/ importable, including in spawned worker processes.
REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ast_poc import (  # noqa: E402
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_SYNTHETIC_DATA_DIR,
    build_static_environment,
    load_scenario_data,
)
from ast_utils import (  # noqa: E402
    ASTEnvironment,
    EpisodeResult,
    MCTSConfig,
    SearchObjective,
    VSLActionModel,
    run_mcts,
)
from eval_metrics import mape  # noqa: E402


# Calibration bounds from optimization_utils.py. tau is listed in seconds here
# for readability and converted to hours before simulation.
PARAMETER_BOUNDS: Dict[str, tuple] = {
    "tau_s": (10.0, 60.0),
    "eta_high": (10.0, 90.0),
    "K": (5.0, 60.0),
    "p_crit": (15.0, 100.0),
    "v_free": (70.0, 150.0),
    "a": (0.5, 5.0),
}
PARAMETER_NAMES = tuple(PARAMETER_BOUNDS)
# Simulator parameter key for each sampled name (tau is sampled in seconds).
SIM_PARAMETER = {"tau_s": "tau", "eta_high": "eta_high", "K": "K",
                 "p_crit": "p_crit", "v_free": "v_free", "a": "a"}
FACTOR_NAMES = tuple(f"{SIM_PARAMETER[name]}_factor" for name in PARAMETER_NAMES)
SAMPLING_MODES = ("uniform", "calibration")

# Columns of the in-memory per-sample results table.
RESULT_COLUMNS = [
    "sample_id",
    "mcts_seed",
    # uniform sampling fills the values; calibration sampling fills the
    # factors and how many segment values had to be clipped to the bounds.
    *PARAMETER_NAMES,
    *FACTOR_NAMES,
    "clipped_values",
    "outcome",
    "time_to_failure_min",
    "status",
    "error",
    "runtime_s",
    # Nominal trajectory vs. data (whole window, even after a failure)
    "discard_reason",
    "nominal_velocity_mape",
    "nominal_density_mape",
    # Undisturbed baseline
    "nominal_failed",
    "nominal_time_to_failure_min",
    "nominal_failure_subtype",
    "nominal_failure_segment",
    "nominal_failure_raw_value",
    "nominal_distance",
    "nominal_closest_subtype",
    "nominal_closest_segment",
    "nominal_closest_time_min",
    "nominal_min_raw_velocity",
    "nominal_min_raw_density",
    "nominal_max_density",
    "nominal_max_velocity",
    # VSL AST search (only for kept samples whose baseline is valid)
    "search_run",
    "search_found_failure",
    "search_time_to_failure_min",
    "search_failure_subtype",
    "search_failure_segment",
    "search_failure_raw_value",
    "search_failure_signed_margin",
    "search_log_likelihood",
    "search_intervention_cost",
    "search_actions",
    "search_distance",
    "search_closest_subtype",
    "search_min_raw_velocity",
    "search_min_raw_density",
    "search_max_density",
    "search_max_velocity",
]


def sample_parameters(
    n_samples: int,
    seed: int,
    inset: float,
    sampling: str = "uniform",
    spread: float = 0.1,
) -> np.ndarray:
    """Return an (n_samples, 6) array of draws, in ``PARAMETER_NAMES`` order.

    ``uniform``: parameter values, uniform inside the inset bounds.
    ``calibration``: multiplicative factors in [1 - spread, 1 + spread], with
    sample 0 fixed at 1 (the unperturbed calibration).

    All draws are generated up front from one seed, so sample ``i`` is the same
    regardless of how many workers run or whether the sweep was resumed.
    """

    if sampling == "calibration":
        if not 0.0 < spread < 1.0:
            raise ValueError("spread must be in (0, 1).")
        rng = np.random.default_rng(seed)
        factors = rng.uniform(
            1.0 - spread, 1.0 + spread, size=(n_samples, len(PARAMETER_NAMES))
        )
        if n_samples:
            factors[0] = 1.0
        return factors
    if sampling != "uniform":
        raise ValueError(f"Unknown sampling mode: {sampling}")
    if not 0.0 <= inset < 0.5:
        raise ValueError("inset must be in [0, 0.5).")
    rng = np.random.default_rng(seed)
    lows, highs = [], []
    for low, high in PARAMETER_BOUNDS.values():
        margin = inset * (high - low)
        lows.append(low + margin)
        highs.append(high - margin)
    return rng.uniform(lows, highs, size=(n_samples, len(PARAMETER_BOUNDS)))


def _extremes(episode: EpisodeResult, prefix: str) -> Dict[str, Any]:
    """Physical-unit extremes up to (and including) the first failure, if any."""

    if episode.failed:
        end = episode.failure.time_index  # raw history row k is step k + 1
        raw_v = episode.raw_velocity_history[:end]
        raw_rho = episode.raw_density_history[:end]
        states_v = episode.velocity_history[: end + 1]
        states_rho = episode.density_history[: end + 1]
    else:
        raw_v = episode.raw_velocity_history
        raw_rho = episode.raw_density_history
        states_v = episode.velocity_history
        states_rho = episode.density_history
    return {
        f"{prefix}_min_raw_velocity": float(np.min(raw_v)) if raw_v.size else None,
        f"{prefix}_min_raw_density": float(np.min(raw_rho)) if raw_rho.size else None,
        f"{prefix}_max_density": float(np.max(states_rho)),
        f"{prefix}_max_velocity": float(np.max(states_v)),
    }


def _finite_or_inf(value: float) -> float:
    return float(value) if np.isfinite(value) else float("inf")


# ---------------------------------------------------------------------------
# Worker process state and per-sample evaluation
# ---------------------------------------------------------------------------

_ENV: Optional[ASTEnvironment] = None
_VSL_MODEL: Optional[VSLActionModel] = None
_SETTINGS: Dict[str, Any] = {}
_TRUTH: Dict[str, np.ndarray] = {}
_CENTER: Dict[str, np.ndarray] = {}  # calibrated per-segment values


def _load_ground_truth(settings: Dict[str, Any], num_steps: int) -> Dict[str, np.ndarray]:
    """Observed velocity and total density for the simulated window.

    Uses the same data as the simulation (see ``ast_poc.load_scenario_data``):
    for I-24 this matches Optimization.ipynb (zeros replaced by 1e-3,
    v = q / rho, interior segments only); for synthetic data it is every
    segment, with velocity from v_hat.npy.
    """

    data = load_scenario_data(
        Path(settings["data_dir"]), synthetic=settings.get("synthetic", False)
    )
    start = settings["start_step"]
    window = slice(start, start + num_steps)
    return {
        "velocity": data["velocity"][window],
        "density": data["density_total"][window],
        "lanes": data["lane_map"],
    }


def _init_worker(settings: Dict[str, Any]) -> None:
    """Load data and calibration once per worker."""

    global _ENV, _VSL_MODEL, _SETTINGS, _TRUTH, _CENTER
    _SETTINGS = settings
    _ENV = build_static_environment(
        Path(settings["data_dir"]),
        Path(settings["calibration_dir"]),
        start_step=settings["start_step"],
        window_minutes=settings["window_minutes"],
        action_minutes=settings["action_minutes"],
        max_density=settings["max_density"],
        max_velocity=settings["max_velocity"],
        min_density=settings["min_density"],
        min_velocity=settings["min_velocity"],
        synthetic=settings.get("synthetic", False),
    )
    _VSL_MODEL = VSLActionModel.proof_of_concept(
        min_speed_mph=settings["vsl_min_mph"],
        max_speed_mph=settings["vsl_max_mph"],
        step_mph=settings["vsl_step_mph"],
        probability_decay_mph=settings["vsl_probability_decay_mph"],
    )
    _TRUTH = _load_ground_truth(settings, _ENV.num_steps)
    _CENTER = {}
    for key in SIM_PARAMETER.values():
        center = np.asarray(_ENV.params[key], dtype=float)
        if center.shape != (_ENV.num_segments,):
            raise ValueError(
                f"Calibrated '{key}' has shape {center.shape}; calibration sampling "
                "needs one static value per segment."
            )
        _CENTER[key] = center.copy()


def _sample_columns(sample_values: Sequence[float]) -> Dict[str, float]:
    """Result columns holding one sample's draw (values or factors, by mode)."""

    names = FACTOR_NAMES if _SETTINGS["sampling"] == "calibration" else PARAMETER_NAMES
    return dict(zip(names, (float(v) for v in sample_values)))


def _environment_for(sample: Dict[str, float]) -> Tuple[ASTEnvironment, int]:
    """Copy the base environment with one sample's parameters.

    ``sample`` is the dict from :func:`_sample_columns`. Returns the environment
    and how many segment values were clipped to the calibration bounds (always
    0 for uniform sampling).
    """

    assert _ENV is not None
    n = _ENV.num_segments
    params = dict(_ENV.params)  # keeps calibrated r, beta, gamma, q_capacity
    clipped = 0
    for name in PARAMETER_NAMES:
        key = SIM_PARAMETER[name]
        to_sim = 1.0 / 3600.0 if name == "tau_s" else 1.0  # tau: s -> h
        if _SETTINGS["sampling"] == "calibration":
            low, high = (bound * to_sim for bound in PARAMETER_BOUNDS[name])
            center = _CENTER[key]
            scaled = center * sample[f"{key}_factor"]
            # Calibrated values can sit a hair outside a bound (solver
            # tolerance / rounding); widen per segment so factor 1 is exact.
            values = np.clip(scaled, np.minimum(low, center), np.maximum(high, center))
            clipped += int(np.count_nonzero(values != scaled))
            params[key] = values
        else:
            params[key] = np.full(n, sample[name] * to_sim)
    return dataclasses.replace(_ENV, params=params), clipped


def _nominal_mapes(nominal: EpisodeResult) -> Dict[str, float]:
    """MAPE (%) of the full nominal trajectory against the data.

    Like Optimization.ipynb, simulated step k is compared with observation k,
    and per-lane simulated density is scaled by lanes to compare with total
    density. Non-finite trajectories get an infinite MAPE.
    """

    n = _TRUTH["velocity"].shape[0]
    v_sim = nominal.velocity_history[:n]
    rho_sim = nominal.density_history[:n] * _TRUTH["lanes"]
    return {
        "velocity": _finite_or_inf(mape(_TRUTH["velocity"], v_sim)),
        "density": _finite_or_inf(mape(_TRUTH["density"], rho_sim)),
    }


def _discard_reasons(mapes: Dict[str, float], settings: Dict[str, Any]) -> List[str]:
    """Names of the MAPEs that exceed their threshold (unset thresholds are skipped).

    A non-finite MAPE counts as exceeding any threshold that is set.
    """

    reasons = []
    for variable in ("velocity", "density"):
        threshold = settings.get(f"{variable}_threshold")
        if threshold is not None and not mapes[variable] <= threshold:
            reasons.append(variable)
    return reasons


def _threshold_text(settings: Dict[str, Any]) -> str:
    """Describe the discard rule."""

    parts = [
        f"{variable} MAPE <= {settings[f'{variable}_threshold']:g}%"
        for variable in ("velocity", "density")
        if settings.get(f"{variable}_threshold") is not None
    ]
    return "kept if " + " and ".join(parts) if parts else "no MAPE threshold"


def _mcts_config(seed: int) -> MCTSConfig:
    return MCTSConfig(
        iterations=_SETTINGS["iterations"],
        exploration_constant=_SETTINGS["exploration_constant"],
        failure_bonus=_SETTINGS["failure_bonus"],
        miss_distance_weight=_SETTINGS["miss_distance_weight"],
        seed=seed,
    )


def evaluate_sample(task: tuple) -> Dict[str, Any]:
    """MAPE check, then (if kept) the baseline and VSL AST search for one sample."""

    sample_id, sample_values = task
    values = _sample_columns(sample_values)
    mcts_seed = _SETTINGS["mcts_seed"] + sample_id
    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "mcts_seed": mcts_seed,
        **values,
        "status": "ok",
        "error": None,
        "search_run": False,
    }
    start = time.perf_counter()
    try:
        # Overflow/invalid values become inf/nan, which the failure detector
        # reports as numerical failures, so silence NumPy's warnings.
        with np.errstate(all="ignore"):
            env, row["clipped_values"] = _environment_for(values)
            minutes_per_step = env.T * 60.0

            # 1. Full nominal trajectory: failure check and fit to the data.
            nominal = env.run_nominal(stop_on_failure=False)
            mapes = _nominal_mapes(nominal)
            row["nominal_velocity_mape"] = mapes["velocity"]
            row["nominal_density_mape"] = mapes["density"]
            row["nominal_failed"] = nominal.failed
            row.update(_extremes(nominal, "nominal"))
            if nominal.failed:
                failure = nominal.failure
                row["nominal_time_to_failure_min"] = nominal.time_to_failure_minutes
                row["nominal_failure_subtype"] = failure.subtype
                row["nominal_failure_segment"] = failure.segment
                row["nominal_failure_raw_value"] = failure.raw_value
                row["nominal_distance"] = 0.0
            else:
                closest = nominal.closest_assessment
                row["nominal_distance"] = closest.distance
                row["nominal_closest_subtype"] = closest.subtype
                row["nominal_closest_segment"] = closest.segment
                row["nominal_closest_time_min"] = closest.time_index * minutes_per_step

            # 2. Discard samples that fit the data too poorly.
            reasons = _discard_reasons(mapes, _SETTINGS)
            if reasons:
                row["outcome"] = "discarded"
                row["discard_reason"] = "+".join(reasons)
            elif nominal.failed:
                row["outcome"] = "failed_nominal"
                row["time_to_failure_min"] = nominal.time_to_failure_minutes
            else:
                # 3. VSL AST search on kept samples with a valid baseline.
                result = run_mcts(
                    env, _VSL_MODEL, SearchObjective.STANDARD_AST, _mcts_config(mcts_seed)
                )
                best = result.best_episode  # a failure whenever one was found
                row["search_run"] = True
                row["search_found_failure"] = best.failed
                row["search_log_likelihood"] = best.log_likelihood
                row["search_intervention_cost"] = best.intervention_cost
                row["search_actions"] = "|".join(best.action_labels)
                row.update(_extremes(best, "search"))
                if best.failed:
                    failure = best.failure
                    row["outcome"] = "failed_vsl"
                    row["time_to_failure_min"] = best.time_to_failure_minutes
                    row["search_time_to_failure_min"] = best.time_to_failure_minutes
                    row["search_failure_subtype"] = failure.subtype
                    row["search_failure_segment"] = failure.segment
                    row["search_failure_raw_value"] = failure.raw_value
                    row["search_failure_signed_margin"] = failure.signed_margin
                    row["search_distance"] = 0.0
                else:
                    row["outcome"] = "safe"
                    row["search_distance"] = best.closest_nonfailure_distance
                    row["search_closest_subtype"] = best.closest_assessment.subtype
    except Exception as exc:  # keep the sweep going; record what broke
        row["status"] = "error"
        row["outcome"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc} | {traceback.format_exc(limit=2)!r}"
    row["runtime_s"] = time.perf_counter() - start
    return row


# ---------------------------------------------------------------------------
# Driver: parallel execution, figures, optional JSON
# ---------------------------------------------------------------------------

HIGHLIGHT_FAILURES = 5  # fastest VSL failures recorded in full in the JSON


def _sample_names(settings: Dict[str, Any]) -> Tuple[str, ...]:
    return FACTOR_NAMES if settings["sampling"] == "calibration" else PARAMETER_NAMES


def _results_frame(rows: List[Dict[str, Any]]) -> Any:
    import pandas as pd

    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    return frame.sort_values("sample_id").reset_index(drop=True)


def _plain(value: Any) -> Any:
    """JSON-friendly scalar: NumPy types to Python, NaN (missing) to None."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _excluded_from_best(settings: Dict[str, Any]) -> Tuple[int, ...]:
    """Sample ids left out of the "best safe sample" choice.

    In calibration sampling, sample 0 is the unperturbed centre, so it is left
    out to show the best *perturbed* sample instead.
    """

    return (0,) if settings["sampling"] == "calibration" else ()


def _rerun(row: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    """Rerun one sample (deterministic) to recover its full episodes."""

    values = {name: float(row[name]) for name in _sample_names(settings)}
    with np.errstate(all="ignore"):
        env, _ = _environment_for(values)
        rerun = {
            "values": values,
            "env": env,
            "nominal": env.run_nominal(),  # stops at failure, as in ast_poc.py
            "nominal_full": env.run_nominal(stop_on_failure=False),
            "search": None,
        }
        if bool(row["search_run"]):
            rerun["search"] = run_mcts(
                env,
                _VSL_MODEL,
                SearchObjective.STANDARD_AST,
                _mcts_config(int(row["mcts_seed"])),
            )
    return rerun


def _save_best_safe_time_space(
    results: Any, settings: Dict[str, Any], fig_prefix: str
) -> Optional[Path]:
    """Rerun the most accurate safe sample and plot it against the data.

    "Most accurate" means the lowest average of nominal velocity and density
    MAPE, excluding sample 0 (the unperturbed centre) in calibration sampling.
    The simulation is the undisturbed (no VSL) run over the whole window.
    """

    from ast_sweep_summary import best_safe_sample, output_path, save_time_space_comparison

    best = best_safe_sample(results, _excluded_from_best(settings))
    if best is None:
        print("No safe samples; skipping the time-space plot.")
        return None
    rerun = _rerun(best.to_dict(), settings)
    nominal = rerun["nominal_full"]
    n = _TRUTH["velocity"].shape[0]
    simulated = {
        "velocity": nominal.velocity_history[:n],
        "density": nominal.density_history[:n] * _TRUTH["lanes"],
    }
    truth = {"velocity": _TRUTH["velocity"], "density": _TRUTH["density"]}
    mapes = _nominal_mapes(nominal)
    values = rerun["values"]
    params_text = ", ".join(f"{name}={values[name]:.3g}" for name in values)
    title = (
        f"Most accurate safe sample #{int(best['sample_id'])}: "
        f"velocity MAPE {mapes['velocity']:.1f}%, density MAPE {mapes['density']:.1f}%"
        f"\n{params_text}"
    )
    return save_time_space_comparison(
        truth,
        simulated,
        rerun["env"].T,
        rerun["env"].l,
        output_path(fig_prefix, "best_safe_time_space"),
        title=title,
    )


def _vsl_action_models() -> Dict[str, Any]:
    """The VSL action model, recorded exactly as ast_poc.py records it."""

    assert _VSL_MODEL is not None
    return {
        "vsl": {
            "note": "Synthetic proof-of-concept assumptions pending domain review.",
            "actions": [
                {
                    "label": action.label,
                    "speed_kmh": action.speed,
                    "speed_mph": (
                        None
                        if action.label == "unrestricted"
                        else action.speed / VSLActionModel.MPH_TO_KMH
                    ),
                    "probability": _VSL_MODEL.action_probability(action),
                    "intervention_cost": _VSL_MODEL.intervention_cost(action),
                }
                for action in _VSL_MODEL.actions
            ],
        }
    }


def _highlight(row: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    """Full record of one sample, with ``nominal`` and ``searches`` as in ast_poc.py."""

    rerun = _rerun(row, settings)
    entry = {
        "sample_id": int(row["sample_id"]),
        **{name: _plain(row[name]) for name in _sample_names(settings)},
        "outcome": row["outcome"],
        "nominal_velocity_mape": _plain(row["nominal_velocity_mape"]),
        "nominal_density_mape": _plain(row["nominal_density_mape"]),
        "nominal": rerun["nominal"].summary(),
        "searches": {},
    }
    if rerun["search"] is not None:
        entry["searches"]["vsl-ast"] = rerun["search"].summary()
    return entry


def _build_json(
    results: Any, table: Any, settings: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    from ast_sweep_summary import FAILED_VSL, best_safe_sample

    assert _ENV is not None
    names = _sample_names(settings)
    calibration = settings["sampling"] == "calibration"

    samples = []
    for row in results.to_dict("records"):
        entry = {
            "sample_id": int(row["sample_id"]),
            **{name: _plain(row[name]) for name in names},
            "outcome": row["outcome"],
            "discard_reason": _plain(row["discard_reason"]),
            "time_to_failure_min": _plain(row["time_to_failure_min"]),
            "nominal_velocity_mape": _plain(row["nominal_velocity_mape"]),
            "nominal_density_mape": _plain(row["nominal_density_mape"]),
        }
        if calibration:
            entry["clipped_values"] = _plain(row["clipped_values"])
        if row["status"] == "error":
            entry["error"] = row["error"]
        samples.append(entry)

    best = best_safe_sample(results, _excluded_from_best(settings))
    fastest = (
        results[results["outcome"] == FAILED_VSL]
        .sort_values(["time_to_failure_min", "sample_id"])
        .head(HIGHLIGHT_FAILURES)
    )
    sampling: Dict[str, Any] = {
        "mode": settings["sampling"],
        "seed": args.seed,
        "parameter_bounds": {name: list(bounds) for name, bounds in PARAMETER_BOUNDS.items()},
    }
    if calibration:
        sampling["spread"] = args.spread
        sampling["center"] = {key: value.tolist() for key, value in _CENTER.items()}
    else:
        sampling["inset"] = args.inset

    return {
        "scenario": {
            "data_dir": settings["data_dir"],
            "synthetic": args.synthetic,
            "calibration_dir": None if args.synthetic else settings["calibration_dir"],
            "start_step": args.start_step,
            "window_minutes": args.window_minutes,
            "action_minutes": args.action_minutes,
            "decision_horizon": _ENV.decision_horizon,
            "max_density": args.max_density,
            "max_velocity": args.max_velocity,
            "min_density": args.min_density,
            "min_velocity": args.min_velocity,
            "velocity_threshold": args.velocity_threshold,
            "density_threshold": args.density_threshold,
            "n_samples": args.n_samples,
            "iterations": args.iterations,
            "exploration_constant": args.exploration_constant,
            "failure_bonus": args.failure_bonus,
            "miss_distance_weight": args.miss_distance_weight,
            "mcts_seed": args.mcts_seed,
        },
        "action_models": _vsl_action_models(),
        "sampling": sampling,
        "outcomes": table.to_dict("records"),
        "samples": samples,
        "highlights": {
            "best_safe": None if best is None else _highlight(best.to_dict(), settings),
            "fastest_vsl_failures": [
                _highlight(row, settings) for row in fastest.to_dict("records")
            ],
        },
    }


def _settings_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    settings = {
        key: value
        for key, value in vars(args).items()
        if key not in ("fig_output", "json_output")
    }
    settings["data_dir"] = str(Path(args.data_dir).resolve())
    settings["calibration_dir"] = str(Path(args.calibration_dir).resolve())
    return settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10_000,
        help="Total samples drawn, including discarded ones.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for parameter draws.")
    parser.add_argument(
        "--mcts-seed",
        type=int,
        default=1,
        help="Base MCTS seed; sample i uses mcts_seed + i.",
    )
    parser.add_argument(
        "--sampling",
        choices=SAMPLING_MODES,
        default="uniform",
        help="uniform: same value on every segment, uniform within the bounds. "
        "calibration: scale the --calibration-dir values by random factors.",
    )
    parser.add_argument(
        "--spread",
        type=float,
        default=0.1,
        help="Calibration sampling: factors are uniform in [1 - spread, 1 + spread].",
    )
    parser.add_argument(
        "--inset",
        type=float,
        default=0.05,
        help="Uniform sampling: fraction of each bound's range kept clear at both ends.",
    )
    parser.add_argument(
        "--velocity-threshold",
        type=float,
        default=None,
        help="Discard samples whose nominal velocity MAPE (%%) is above this. "
        "Default: no velocity threshold.",
    )
    parser.add_argument(
        "--density-threshold",
        type=float,
        default=None,
        help="Discard samples whose nominal density MAPE (%%) is above this. "
        "Default: no density threshold.",
    )
    parser.add_argument(
        "--fig-output",
        default=str(REPO_ROOT / "artifacts" / "ast_sweep" / "sweep"),
        help=(
            "Filename prefix for the figures, which are always saved as "
            "<prefix>_outcomes.png etc. (a prefix ending in / saves into that folder)."
        ),
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional JSON output path. No JSON is written when omitted.",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help=(
            "Use synthetic data: every segment is simulated, the boundaries come "
            "from inflow.npy and downstream_density.npy in --data-dir, the "
            "default param_loader parameters are used (the centre for "
            "--sampling calibration; --calibration-dir is ignored), and T = 10 s."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            f"Data folder (default: {DEFAULT_DATA_DIR.relative_to(REPO_ROOT)}, "
            f"or {DEFAULT_SYNTHETIC_DATA_DIR.relative_to(REPO_ROOT)} with --synthetic)."
        ),
    )
    parser.add_argument("--calibration-dir", default=str(DEFAULT_CALIBRATION_DIR))
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--window-minutes", type=int, default=15)
    parser.add_argument("--action-minutes", type=int, default=1)
    parser.add_argument("--max-density", type=float, default=190.0)
    parser.add_argument("--max-velocity", type=float, default=250.0)
    parser.add_argument(
        "--min-density",
        type=float,
        default=0.0,
        help="Raw density (veh/km/lane) below this counts as unsafe (default 0).",
    )
    parser.add_argument(
        "--min-velocity",
        type=float,
        default=0.0,
        help="Raw velocity (km/h) below this counts as unsafe (default 0).",
    )
    parser.add_argument("--vsl-min-mph", type=float, default=40.0)
    parser.add_argument("--vsl-max-mph", type=float, default=75.0)
    parser.add_argument("--vsl-step-mph", type=float, default=5.0)
    parser.add_argument("--vsl-probability-decay-mph", type=float, default=10.0)
    parser.add_argument("--iterations", type=int, default=100, help="MCTS iterations per sample.")
    parser.add_argument("--exploration-constant", type=float, default=float(np.sqrt(2.0)))
    parser.add_argument("--failure-bonus", type=float, default=1000.0)
    parser.add_argument("--miss-distance-weight", type=float, default=100.0)
    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = str(
            DEFAULT_SYNTHETIC_DATA_DIR if args.synthetic else DEFAULT_DATA_DIR
        )
    return args


def main() -> None:
    args = parse_args()
    data_root = (REPO_ROOT / "data").resolve()
    for option, path in (("--fig-output", args.fig_output), ("--json-output", args.json_output)):
        if path is not None and data_root in Path(path).resolve().parents:
            raise SystemExit(f"Refusing to write inside data/; choose another {option}.")

    settings = _settings_from_args(args)
    samples = sample_parameters(
        args.n_samples, args.seed, args.inset, args.sampling, args.spread
    )
    tasks = [(i, samples[i]) for i in range(args.n_samples)]
    print(f"Running {len(tasks)} samples with {args.workers} workers.")

    rows: List[Dict[str, Any]] = []
    start = time.time()
    report_every = max(1, len(tasks) // 20)
    with Pool(args.workers, initializer=_init_worker, initargs=(settings,)) as pool:
        for row in pool.imap_unordered(evaluate_sample, tasks, chunksize=4):
            rows.append(row)
            if len(rows) % report_every == 0 or len(rows) == len(tasks):
                elapsed = time.time() - start
                remaining = elapsed / len(rows) * (len(tasks) - len(rows))
                print(
                    f"{len(rows)}/{len(tasks)} done | {elapsed / 60:.1f} min elapsed "
                    f"| ~{remaining / 60:.1f} min left",
                    flush=True,
                )

    from ast_sweep_summary import write_summary

    results = _results_frame(rows)
    threshold_text = _threshold_text(settings)
    _init_worker(settings)  # this process reruns highlighted samples
    summary = write_summary(
        results,
        args.fig_output,
        window_minutes=args.window_minutes,
        threshold_text=threshold_text,
    )
    print(f"\nOutcomes ({threshold_text}):")
    print(summary["table"].to_string(index=False))
    print(f"\nSaved {summary['table_path']}")
    for path in summary["plot_paths"]:
        print(f"Saved {path}")
    time_space_path = _save_best_safe_time_space(results, settings, args.fig_output)
    if time_space_path is not None:
        print(f"Saved {time_space_path}")

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(
                _build_json(results, summary["table"], settings, args),
                output_file,
                indent=2,
                allow_nan=True,
            )
        print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
