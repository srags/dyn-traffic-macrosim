"""Run the three agreed Adaptive Stress Testing proof-of-concept searches.

The script does not write results unless an output option is supplied. This
keeps exploratory runs from modifying the repository's tracked calibration
artifacts.

Examples
--------
Run all three searches for both static and frozen dynamic calibration::

    python experiments/ast_poc.py --mode all --iterations 2000

Run a quick smoke test of standard AST over upstream-inflow noise::

    python experiments/ast_poc.py --mode inflow-ast --iterations 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, Optional

import numpy as np


# Make src/ importable when this file is run directly from any working folder.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ast_utils import (  # noqa: E402
    ASTEnvironment,
    GaussianInflowActionModel,
    MCTSConfig,
    PhysicalFailureConfig,
    SearchObjective,
    TrafficState,
    VSLActionModel,
    run_mcts,
)
from optimization_utils import smooth_inflow  # noqa: E402
from param_loader import METANET_Params  # noqa: E402


DEFAULT_DATA_DIR = REPO_ROOT / "data" / "i24" / "i24_11_30"
DEFAULT_SYNTHETIC_DATA_DIR = REPO_ROOT / "data" / "synthetic"
DEFAULT_CALIBRATION_DIR = (
    REPO_ROOT
    / "calibration"
    / "i24"
    / "i24_11_30"
    / "calibration_static"
    / "fixed_ramping"
)
DEFAULT_DYNAMIC_CALIBRATION_DIR = (
    REPO_ROOT
    / "calibration"
    / "i24"
    / "i24_11_30"
    / "calibration_dynamic_smooth"
)


DATA_STEP_SECONDS = 10.0  # resolution of the I-24 data


def load_scenario_data(
    data_dir: Path, *, synthetic: bool = False, upsample: int = 1
) -> Dict[str, np.ndarray]:
    """Load the traffic data and boundary conditions for the simulated segments.

    Returns per-lane ``density``, ``velocity``, total ``density_total`` (all
    shaped (steps, segments)), upstream ``inflow``, per-lane ``downstream``
    density and the simulated segments' ``lane_map``. The data length is taken
    from the downstream density.

    I-24 (default), matching Optimization.ipynb: the first and last segments are
    boundaries and are not simulated; inflow and downstream density are their
    smoothed flow and density; ``rho_hat`` is total density.

    Synthetic (separate boundary conditions): every segment is simulated, and
    the boundaries come from ``inflow.npy`` and ``downstream_density.npy`` as
    saved (no smoothing). ``rho_hat`` is total density (divided by lanes for the
    simulator, as for I-24), and velocity is read
    from ``v_hat.npy``.
    """

    if synthetic:
        if upsample != 1:
            raise ValueError("Synthetic data runs at the 10 s data step only.")
        downstream = np.load(data_dir / "downstream_density.npy").astype(float)
        length = downstream.shape[0]
        lane_map = np.load(data_dir / "lane_mapping.npy").astype(float)
        # rho_hat is total density, like the I-24 data; the simulator uses per lane.
        density_total = np.load(data_dir / "rho_hat.npy").astype(float)[:length]
        return {
            "density": density_total / lane_map,
            "velocity": np.load(data_dir / "v_hat.npy").astype(float)[:length],
            "density_total": density_total,
            "inflow": np.load(data_dir / "inflow.npy").astype(float)[:length],
            "downstream": downstream,
            "lane_map": lane_map,
        }

    q_hat = np.load(data_dir / "q_hat.npy")
    rho_hat = np.load(data_dir / "rho_hat.npy")
    lane_map = np.load(data_dir / "lane_mapping.npy")[1:-1]

    # Match Optimization.ipynb's divide-by-zero guard before deriving speed.
    q_safe = np.where(q_hat == 0.0, 1e-3, q_hat)
    rho_safe = np.where(rho_hat == 0.0, 1e-3, rho_hat)
    if upsample > 1:
        q_safe = np.repeat(q_safe, upsample, axis=0)
        rho_safe = np.repeat(rho_safe, upsample, axis=0)
    v_hat = q_safe / rho_safe

    # The simulator evolves per-lane density, matching Optimization.ipynb.
    downstream = smooth_inflow(rho_safe[:, -1]) / lane_map[-1]
    length = downstream.shape[0]
    return {
        "density": (rho_safe[:, 1:-1] / lane_map)[:length],
        "velocity": v_hat[:length, 1:-1],
        "density_total": rho_safe[:length, 1:-1],
        "inflow": smooth_inflow(q_safe[:, 0])[:length],
        "downstream": downstream,
        "lane_map": lane_map,
    }


def build_static_environment(
    data_dir: Path,
    calibration_dir: Optional[Path],
    *,
    start_step: int = 0,
    window_minutes: int = 15,
    action_minutes: int = 1,
    max_density: float = 190.0,
    max_velocity: float = 250.0,
    min_density: float = 0.0,
    min_velocity: float = 0.0,
    step_seconds: float = DATA_STEP_SECONDS,
    synthetic: bool = False,
) -> ASTEnvironment:
    """Load the same static I-24 scenario used by Optimization.ipynb.

    ``step_seconds`` is the simulation time step. It must divide the 10 s data
    step evenly and should match the T used for this calibration. For shorter
    steps each 10 s observation is repeated (held), then smoothed, the same way
    Optimization.ipynb upsamples the data before calibrating. ``start_step`` is
    always counted in 10 s data steps.

    With ``synthetic=True`` the data and boundary conditions come from
    ``data_dir`` as described in :func:`load_scenario_data`, the default
    parameters from ``param_loader.METANET_Params`` are used (``calibration_dir``
    is ignored), and the time step must be 10 s.
    """

    if window_minutes <= 0 or action_minutes <= 0:
        raise ValueError("Window and action durations must be positive.")
    if max_density <= 0 or max_velocity <= 0:
        raise ValueError("Physical failure thresholds must be positive.")
    upsample = int(round(DATA_STEP_SECONDS / step_seconds))
    if step_seconds <= 0 or not np.isclose(upsample * step_seconds, DATA_STEP_SECONDS):
        raise ValueError(
            f"step_seconds={step_seconds:g} must evenly divide the "
            f"{DATA_STEP_SECONDS:g} s data step."
        )

    T = step_seconds / 3600.0  # simulation step, expressed in hours
    segment_length = 0.4  # km
    simulation_steps = int(round(window_minutes * 60 / step_seconds))
    action_interval_steps = int(round(action_minutes * 60 / step_seconds))
    start_step = start_step * upsample  # data steps -> simulation steps
    end_step = start_step + simulation_steps

    data = load_scenario_data(data_dir, synthetic=synthetic, upsample=upsample)
    length = data["downstream"].shape[0]
    if start_step < 0 or end_step > length:
        raise ValueError(
            f"Requested steps [{start_step}, {end_step}) exceed data length "
            f"{length} (at {step_seconds:g} s steps)."
        )
    if simulation_steps % action_interval_steps:
        raise ValueError("The simulation window must be divisible by the action interval.")

    initial_state = TrafficState(
        density=data["density"][start_step],
        velocity=data["velocity"][start_step],
        flow_origin=float(data["inflow"][start_step]),
        queue=0.0,
        time_index=0,
    )

    num_segments = initial_state.density.size
    params = METANET_Params(
        path=None if synthetic else str(calibration_dir), num_segments=num_segments
    ).get_params()
    lane_map = data["lane_map"]
    lanes = {i: float(lane_map[i]) for i in range(num_segments)}

    return ASTEnvironment(
        T=T,
        l=segment_length,
        initial_state=initial_state,
        nominal_demand=data["inflow"][start_step:end_step],
        downstream_density=data["downstream"][start_step:end_step],
        params=params,
        lanes=lanes,
        failure_config=PhysicalFailureConfig(
            max_density=max_density,
            max_velocity=max_velocity,
            min_density=min_density,
            min_velocity=min_velocity,
        ),
        action_interval_steps=action_interval_steps,
    )


def build_dynamic_environment(
    data_dir: Path,
    dynamic_calibration_dir: Path,
    *,
    control_horizon: int = 6,
    start_step: int = 0,
    window_minutes: int = 15,
    action_minutes: int = 1,
    max_density: float = 190.0,
    max_velocity: float = 250.0,
    min_density: float = 0.0,
    min_velocity: float = 0.0,
) -> ASTEnvironment:
    """Build an AST environment with a frozen rolling-calibration schedule.

    "Frozen" means the previously saved parameter schedule is replayed during
    every rollout. AST disturbances do not trigger new calibration solves.
    """

    if control_horizon <= 0:
        raise ValueError("The dynamic control horizon must be positive.")
    first_parameter_dir = (
        dynamic_calibration_dir
        / f"control_h_{control_horizon}"
        / "params_1"
    )
    if not first_parameter_dir.is_dir():
        raise FileNotFoundError(
            "Dynamic calibration parameters not found at "
            f"{first_parameter_dir}."
        )

    # Reuse the common scenario construction, then replace the temporary first
    # parameter set with the complete time-varying schedule.
    environment = build_static_environment(
        data_dir,
        first_parameter_dir,
        start_step=start_step,
        window_minutes=window_minutes,
        action_minutes=action_minutes,
        max_density=max_density,
        max_velocity=max_velocity,
        min_density=min_density,
        min_velocity=min_velocity,
    )
    # Data length, read from the downstream boundary like the static builder.
    total_steps = int(load_scenario_data(data_dir)["downstream"].shape[0])
    if total_steps % control_horizon:
        raise ValueError(
            f"Data length {total_steps} is not divisible by dynamic control "
            f"horizon {control_horizon}."
        )
    full_schedule = METANET_Params(
        path=str(dynamic_calibration_dir),
        control_h=control_horizon,
        num_timesteps=total_steps,
        num_segments=environment.num_segments,
    ).get_params()
    end_step = start_step + environment.num_steps
    environment.params = {
        key: np.asarray(value)[start_step:end_step].copy()
        for key, value in full_schedule.items()
    }
    for key, value in environment.params.items():
        if value.shape != (environment.num_steps, environment.num_segments):
            raise ValueError(
                f"Dynamic parameter '{key}' has sliced shape {value.shape}; "
                f"expected {(environment.num_steps, environment.num_segments)}."
            )
    return environment


def _print_episode(title: str, summary: Dict[str, object]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(json.dumps(summary, indent=2, allow_nan=True))


def _selected_modes(mode: str) -> Iterable[str]:
    if mode == "all":
        return ("vsl-ast", "vsl-min", "inflow-ast")
    return (mode,)


def _static_calibration_specs(args: argparse.Namespace) -> list:
    """Pair each --calibration-dir with its time step and label.

    A single folder keeps the original "static" key so older results and plots
    are unchanged. Several folders get keys "static:<label>".
    """

    dirs = [Path(d) for d in args.calibration_dir]
    steps = list(args.calibration_step_seconds)
    if len(steps) == 1:
        steps = steps * len(dirs)
    if len(steps) != len(dirs):
        raise SystemExit(
            "--calibration-step-seconds needs one value, or one per --calibration-dir."
        )
    labels = list(args.calibration_labels or [d.name for d in dirs])
    if len(labels) != len(dirs):
        raise SystemExit("--calibration-labels needs one label per --calibration-dir.")
    if len(set(labels)) != len(labels):
        raise SystemExit(
            "Calibration labels must be unique; pass --calibration-labels to name them."
        )

    specs = []
    for directory, step, label in zip(dirs, steps, labels):
        if len(dirs) == 1 and args.calibration_labels is None:
            key, short, long = "static", "Static", "Static calibration"
        else:
            key, short, long = f"static:{label}", label, f"Static calibration ({label})"
        if step != DATA_STEP_SECONDS:
            short, long = f"{short} (T={step:g} s)", f"{long}, T={step:g} s"
        specs.append((key, directory, step, short, long))
    return specs


def run_experiments(args: argparse.Namespace) -> Dict[str, object]:
    data_dir = Path(args.data_dir)
    environments: Dict[str, tuple[str, ASTEnvironment, Dict[str, object]]] = {}
    if args.synthetic:
        # Synthetic data: default param_loader parameters at the 10 s step;
        # calibration folders, step sizes and --calibration-mode are ignored.
        environment = build_static_environment(
            data_dir,
            None,
            start_step=args.start_step,
            window_minutes=args.window_minutes,
            action_minutes=args.action_minutes,
            max_density=args.max_density,
            max_velocity=args.max_velocity,
            min_density=args.min_density,
            min_velocity=args.min_velocity,
            synthetic=True,
        )
        environments["synthetic"] = (
            "Synthetic data, default parameters",
            environment,
            {
                "type": "synthetic",
                "short_label": "Synthetic (default params)",
                "step_seconds": DATA_STEP_SECONDS,
            },
        )
    if not args.synthetic and args.calibration_mode in ("static", "both"):
        for key, directory, step, short, long in _static_calibration_specs(args):
            environment = build_static_environment(
                data_dir,
                directory,
                start_step=args.start_step,
                window_minutes=args.window_minutes,
                action_minutes=args.action_minutes,
                max_density=args.max_density,
                max_velocity=args.max_velocity,
                min_density=args.min_density,
                min_velocity=args.min_velocity,
                step_seconds=step,
            )
            environments[key] = (
                long,
                environment,
                {
                    "type": "static",
                    "short_label": short,
                    "calibration_dir": str(directory.resolve()),
                    "step_seconds": step,
                },
            )
    if not args.synthetic and args.calibration_mode in ("dynamic", "both"):
        environment = build_dynamic_environment(
            data_dir,
            Path(args.dynamic_calibration_dir),
            control_horizon=args.dynamic_control_horizon,
            start_step=args.start_step,
            window_minutes=args.window_minutes,
            action_minutes=args.action_minutes,
            max_density=args.max_density,
            max_velocity=args.max_velocity,
            min_density=args.min_density,
            min_velocity=args.min_velocity,
        )
        update_minutes = args.dynamic_control_horizon * environment.T * 60.0
        environments["dynamic"] = (
            f"Frozen dynamic calibration (h={args.dynamic_control_horizon}, "
            f"{update_minutes:g} min)",
            environment,
            {
                "type": "frozen_dynamic_schedule",
                "step_seconds": DATA_STEP_SECONDS,
                "calibration_dir": str(
                    Path(args.dynamic_calibration_dir).resolve()
                ),
                "control_horizon_steps": args.dynamic_control_horizon,
                "parameter_update_minutes": update_minutes,
            },
        )

    config = MCTSConfig(
        iterations=args.iterations,
        exploration_constant=args.exploration_constant,
        failure_bonus=args.failure_bonus,
        miss_distance_weight=args.miss_distance_weight,
        seed=args.seed,
    )

    # One common VSL is applied to all segments in this first POC. The action
    # labels use mph, while their values are converted to METANET's km/h units.
    vsl_model = VSLActionModel.proof_of_concept(
        min_speed_mph=args.vsl_min_mph,
        max_speed_mph=args.vsl_max_mph,
        step_mph=args.vsl_step_mph,
        probability_decay_mph=args.vsl_probability_decay_mph,
    )

    # This exactly follows the existing repository convention: a requested
    # percent of the nominal inflow's standard deviation.
    reference_environment = next(iter(environments.values()))[1]
    inflow_sigma = (
        args.inflow_noise_percent
        * np.std(reference_environment.nominal_demand)
        / 100.0
    )
    inflow_model = GaussianInflowActionModel(sigma=inflow_sigma)

    results: Dict[str, object] = {
        "scenario": {
            "data_dir": str(Path(args.data_dir).resolve()),
            "start_step": args.start_step,
            "window_minutes": args.window_minutes,
            "action_minutes": args.action_minutes,
            "decision_horizon": reference_environment.decision_horizon,
            "max_density": args.max_density,
            "max_velocity": args.max_velocity,
            "min_density": args.min_density,
            "min_velocity": args.min_velocity,
            "inflow_sigma": inflow_sigma,
        },
        "action_models": {
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
                        "probability": vsl_model.action_probability(action),
                        "intervention_cost": vsl_model.intervention_cost(action),
                    }
                    for action in vsl_model.actions
                ],
            },
            "gaussian_inflow": {
                "sigma": inflow_model.sigma,
                "actions": [
                    {
                        "label": inflow_model.action_label(action),
                        "z": action.z,
                        "probability": inflow_model.action_probability(action),
                    }
                    for action in inflow_model.actions
                ],
            },
        },
        "calibrations": {},
    }

    for calibration_key, (label, environment, metadata) in environments.items():
        # Required precondition: each undisturbed calibration must be valid
        # before searching for a disturbance that causes failure.
        nominal = environment.run_nominal()
        _print_episode(f"{label}: undisturbed baseline", nominal.summary())

        calibration_results: Dict[str, object] = {
            "label": label,
            **metadata,
            "nominal": nominal.summary(),
            "searches": {},
        }
        results["calibrations"][calibration_key] = calibration_results
        if nominal.failed:
            # Searching for a disturbance is meaningless if the undisturbed run
            # already fails; record it and move on to the next calibration.
            print(
                f"\nWARNING: the undisturbed {label} simulation already fails "
                f"({nominal.failure.subtype} at "
                f"{nominal.time_to_failure_minutes:.2f} min). Skipping its searches."
            )
            continue

        for mode in _selected_modes(args.mode):
            if mode == "vsl-ast":
                model = vsl_model
                objective = SearchObjective.STANDARD_AST
                title = "Standard AST over VSL"
            elif mode == "vsl-min":
                model = vsl_model
                objective = SearchObjective.MIN_INTERVENTION
                title = "Minimum-intervention VSL falsification"
            elif mode == "inflow-ast":
                model = inflow_model
                objective = SearchObjective.STANDARD_AST
                title = "Standard AST over Gaussian inflow perturbations"
            else:  # argparse prevents this, but keep the function safe to call.
                raise ValueError(f"Unknown AST mode: {mode}")

            result = run_mcts(environment, model, objective, config)
            result_summary = result.summary()
            calibration_results["searches"][mode] = result_summary
            _print_episode(f"{label}: {title}", result_summary)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=("all", "vsl-ast", "vsl-min", "inflow-ast"),
        default="all",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help=(
            "Use synthetic data: every segment is simulated, the boundaries come "
            "from inflow.npy and downstream_density.npy in --data-dir, the "
            "default param_loader parameters are used, and T = 10 s. "
            "Calibration options are ignored."
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
    parser.add_argument(
        "--calibration-dir",
        nargs="+",
        default=[str(DEFAULT_CALIBRATION_DIR)],
        help="One or more static calibration folders, each run separately.",
    )
    parser.add_argument(
        "--calibration-step-seconds",
        nargs="+",
        type=float,
        default=[DATA_STEP_SECONDS],
        help=(
            "Simulation time step (s) for each --calibration-dir, matching the T "
            "it was calibrated with. One value applies to all folders."
        ),
    )
    parser.add_argument(
        "--calibration-labels",
        nargs="+",
        default=None,
        help="Short names for each --calibration-dir (default: folder names).",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=("static", "dynamic", "both"),
        default="both",
        help="Run static calibration, a frozen dynamic schedule, or both.",
    )
    parser.add_argument(
        "--dynamic-calibration-dir",
        default=str(DEFAULT_DYNAMIC_CALIBRATION_DIR),
    )
    parser.add_argument(
        "--dynamic-control-horizon",
        type=int,
        default=6,
        help="Saved dynamic parameter update interval in 10-second steps.",
    )
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
    parser.add_argument(
        "--vsl-min-mph",
        type=float,
        default=40.0,
        help="Lowest controlled VSL action in mph (approximately 64 km/h).",
    )
    parser.add_argument(
        "--vsl-max-mph",
        type=float,
        default=75.0,
        help="Highest controlled VSL action in mph (approximately 120 km/h).",
    )
    parser.add_argument(
        "--vsl-step-mph",
        type=float,
        default=5.0,
        help="Spacing between controlled VSL actions in mph.",
    )
    parser.add_argument(
        "--vsl-probability-decay-mph",
        type=float,
        default=10.0,
        help="Exponential scale governing how VSL probability falls with restriction.",
    )
    parser.add_argument(
        "--inflow-noise-percent",
        type=float,
        default=5.0,
        help="Gaussian sigma as a percentage of nominal inflow standard deviation.",
    )
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--exploration-constant", type=float, default=np.sqrt(2.0))
    parser.add_argument("--failure-bonus", type=float, default=1000.0)
    parser.add_argument("--miss-distance-weight", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional JSON output path. No files are written when omitted.",
    )
    parser.add_argument(
        "--time-output",
        default=None,
        help="Optional time-to-failure bar chart path (.png, .pdf, or .svg).",
    )
    parser.add_argument(
        "--severity-output",
        default=None,
        help="Optional boundary-severity bar chart path (.png, .pdf, or .svg).",
    )
    parser.add_argument(
        "--table-output",
        default=None,
        help="Optional discovered-sequences table path (.png, .pdf, or .svg).",
    )
    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = str(
            DEFAULT_SYNTHETIC_DATA_DIR if args.synthetic else DEFAULT_DATA_DIR
        )
    return args


def main() -> None:
    args = parse_args()
    results = run_experiments(args)
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=2, allow_nan=True)
        print(f"\nSaved summary to {output_path}")
    if args.time_output or args.severity_output or args.table_output:
        # Keep Matplotlib optional for runs that only print or save JSON.
        from ast_plotting import (
            save_boundary_severity,
            save_discovered_sequences,
            save_time_to_first_invalid_state,
        )
    if args.time_output:
        output_path = save_time_to_first_invalid_state(results, args.time_output)
        print(f"Saved time-to-failure chart to {output_path}")
    if args.severity_output:
        try:
            output_path = save_boundary_severity(results, args.severity_output)
            print(f"Saved boundary-severity chart to {output_path}")
        except ValueError as exc:
            # This chart only supports velocity boundaries; keep the other outputs.
            print(f"Skipped boundary-severity chart: {exc}")
    if args.table_output:
        output_path = save_discovered_sequences(results, args.table_output)
        print(f"Saved discovered-sequences table to {output_path}")


if __name__ == "__main__":
    main()
