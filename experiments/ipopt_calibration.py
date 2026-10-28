"""
Static METANET calibration with IPOPT for several data folders
================================================================
Runs the same static calibration as experiments/Optimization.ipynb (cells 4,
9-12, 19, 23, 24) once per data folder, one folder at a time.

Each data folder must contain q_hat.npy, rho_hat.npy, v_hat.npy,
lane_mapping.npy, on_ramp_mapping.npy and off_ramp_mapping.npy.

Results go to the matching calibration folder: "data/" at the start of the
path is replaced with "calibration/", then calibration_static/<run-name>/ is
added. For example:
    data/i24_4hrs/i24_11_23  ->  calibration/i24_4hrs/i24_11_23/calibration_static/test/

Usage:
    python experiments/ipopt_calibration.py data/i24_4hrs/i24_11_21 data/i24_4hrs/i24_11_22 --run-name test
    python experiments/ipopt_calibration.py data/i24_4hrs/i24_* --run-name test --overwrite
"""

import os
import sys

# Put src/ on the import path and run from the repo root so relative data/ and
# calibration/ paths resolve no matter where this script is launched from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
os.chdir(_REPO_ROOT)

import argparse
import json
import random
import time
import traceback

import matplotlib

matplotlib.use("Agg")  # no plot windows when running as a script

import numpy as np

from optimization_utils import run_calibration, smooth_inflow
from param_loader import METANET_Params
from simulation_utils import run_metanet_sim
from eval_metrics import mape, rmse

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS (same as Optimization.ipynb cells 9-10)
# ─────────────────────────────────────────────────────────────────────────────
T = 10 / 3600  # time step [h]
l = 0.4  # segment length [km]
INCLUDE_RAMPING = True
VARYLANES = False
SMOOTHING = True
CONSTRAINT_TOL = 1e-12
TIME_VARYING_RAMPS = False
SEED = 1
MAX_DV = 35.0  # km/h: largest allowed speed change between consecutive steps (None = off)

# A finished run is detected by this file existing in the output folder.
DONE_MARKER = "metrics.json"


def calibration_dir(data_path, run_name):
    """data/<...>/<day> -> calibration/<...>/<day>/calibration_static/<run_name>"""
    rel = os.path.relpath(os.path.abspath(data_path), _REPO_ROOT)
    parts = rel.split(os.sep)
    if parts[0] != "data":
        raise ValueError(f"Data path must be inside data/, got {data_path}")
    return os.path.join("calibration", *parts[1:], "calibration_static", run_name)


def load_day(data_path):
    """Load one day's arrays and build boundary conditions (notebook cell 4)."""
    q_hat = np.load(f"{data_path}/q_hat.npy")
    rho_hat = np.load(f"{data_path}/rho_hat.npy")
    v_hat = np.load(f"{data_path}/v_hat.npy")
    for name, arr in [("q_hat", q_hat), ("rho_hat", rho_hat), ("v_hat", v_hat)]:
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} in {data_path} has NaN/inf values")

    lane_map = np.load(f"{data_path}/lane_mapping.npy")[1:-1]
    lane_mapping = {i: float(lane_map[i]) for i in range(len(lane_map))}
    ramp_mapping = {
        "on_ramps": np.load(f"{data_path}/on_ramp_mapping.npy")[1:-1],
        "off_ramps": np.load(f"{data_path}/off_ramp_mapping.npy")[1:-1],
    }

    downstream_density = smooth_inflow(rho_hat[:, -1])
    data_inflow = smooth_inflow(q_hat[:, 0])
    init_traffic_state = (rho_hat[0, 1:-1] / lane_map, v_hat[0, 1:-1], data_inflow[0], 0)

    return {
        "q_hat": q_hat,
        "rho_hat": rho_hat,
        "v_hat": v_hat,
        "lane_map": lane_map,
        "lane_mapping": lane_mapping,
        "ramp_mapping": ramp_mapping,
        "downstream_density": downstream_density,
        "data_inflow": data_inflow,
        "init_traffic_state": init_traffic_state,
        "num_calibrated_segments": v_hat.shape[1] - 2,
    }


def calibrate_day(data):
    """Run the IPOPT calibration (notebook cell 10)."""
    random.seed(SEED)
    np.random.seed(SEED)
    return run_calibration(
        data["rho_hat"],
        data["q_hat"],
        T,
        l,
        num_calibrated_segments=data["num_calibrated_segments"],
        varylanes=VARYLANES,
        include_ramping=INCLUDE_RAMPING,
        lane_mapping=data["lane_mapping"],
        ramp_mapping=data["ramp_mapping"],
        smoothing=SMOOTHING,
        constraint_tol=CONSTRAINT_TOL,
        warmstart=None,
        tee=False,
        robust_opt=None,
        time_varying_ramps=TIME_VARYING_RAMPS,
        fixed_inflows=None,
        max_dv=MAX_DV,
    )


def save_results(results, data, out_dir):
    """Save boundary conditions, parameters and predictions (notebook cells 11-12)."""
    os.makedirs(out_dir, exist_ok=True)
    q_hat, rho_hat = data["q_hat"], data["rho_hat"]

    np.save(f"{out_dir}/upstream_flow.npy", smooth_inflow(q_hat[:, 0]) if SMOOTHING else q_hat[:, 0])
    np.save(
        f"{out_dir}/downstream_density.npy",
        smooth_inflow(rho_hat[:, -1]) if SMOOTHING else rho_hat[:, -1],
    )

    np.save(f"{out_dir}/tau.npy", results["tau"])
    np.save(f"{out_dir}/K.npy", results["K"])
    np.save(f"{out_dir}/eta_high.npy", results["eta_high"])
    np.save(f"{out_dir}/rho_crit.npy", results["rho_crit"])
    np.save(f"{out_dir}/v_free.npy", results["v_free"])
    np.save(f"{out_dir}/a.npy", results["a"])
    np.save(f"{out_dir}/num_lanes.npy", results["num_lanes"])
    np.save(f"{out_dir}/q_pred.npy", results["v_pred"] * results["rho_pred"])
    np.save(f"{out_dir}/v_pred.npy", results["v_pred"])
    np.save(f"{out_dir}/rho_pred.npy", results["rho_pred"])
    np.save(f"{out_dir}/beta_array.npy", results["beta"])
    np.save(f"{out_dir}/r_inflow_array.npy", results["r_inflow"])


def simulate_and_score(results, data, out_dir):
    """Re-simulate with the saved parameters and compute errors (cells 18, 19, 23, 24)."""
    q_hat, rho_hat, v_hat = data["q_hat"], data["rho_hat"], data["v_hat"]
    lane_map = data["lane_map"]
    n_seg = data["num_calibrated_segments"]

    params = METANET_Params(path=out_dir, num_segments=n_seg).get_params()
    rho_sim, v_sim, _, _ = run_metanet_sim(
        T,
        l,
        data["init_traffic_state"],
        data["data_inflow"],
        data["downstream_density"] / lane_map[-1],
        params,
        vsl_speeds=None,
        lanes=data["lane_mapping"],
        plotting=True,  # this flag selects the (rho, v, queue, tts) return format
        real_data=True,
    )
    np.save(f"{out_dir}/v_sim.npy", v_sim)

    rho_pred, v_pred = results["rho_pred"], results["v_pred"]
    q_pred = rho_pred * v_pred
    rho_sim_total = rho_sim[0:-1, :] * np.array(lane_map)
    return {
        "calibration": {
            "rho_mape": float(mape(rho_hat[:, 1:-1], rho_pred)),
            "q_mape": float(mape(q_hat[:, 1:-1], q_pred)),
            "v_mape": float(mape(v_hat[:, 1:-1], v_pred)),
            "rho_rmse": float(rmse(rho_hat[:, 1:-1], rho_pred)),
            "q_rmse": float(rmse(q_hat[:, 1:-1], q_pred)),
            "v_rmse": float(rmse(v_hat[:, 1:-1], v_pred)),
        },
        "simulation": {
            "rho_mape": float(mape(rho_hat[:, 1:-1], rho_sim_total)),
            "v_mape": float(mape(v_hat[:, 1:-1], v_sim[0:-1, :])),
            "rho_rmse": float(rmse(rho_hat[:, 1:-1], rho_sim_total)),
            "v_rmse": float(rmse(v_hat[:, 1:-1], v_sim[0:-1, :])),
        },
        # Model mismatch: IPOPT prediction vs forward simulation with the same parameters.
        # MAPE divides by the data (not the sim), since sim speeds can be ~0.
        "mismatch": {
            "rho_mape": float(100 * np.mean(np.abs(rho_pred - rho_sim_total) / np.abs(rho_hat[:, 1:-1]))),
            "v_mape": float(100 * np.mean(np.abs(v_pred - v_sim[0:-1, :]) / np.abs(v_hat[:, 1:-1]))),
            "rho_rmse": float(rmse(rho_sim_total, rho_pred)),
            "v_rmse": float(rmse(v_sim[0:-1, :], v_pred)),
        },
    }


def run_one(data_path, run_name, overwrite):
    out_dir = calibration_dir(data_path, run_name)
    if not overwrite and os.path.exists(os.path.join(out_dir, DONE_MARKER)):
        print(f"[skip] {data_path}: results already in {out_dir} (use --overwrite)")
        return {"data_path": data_path, "out_dir": out_dir, "status": "skipped"}

    print(f"[start] {data_path} -> {out_dir}")
    data = load_day(data_path)
    print(f"        {data['rho_hat'].shape[0]} time steps, {data['num_calibrated_segments']} segments")

    start = time.time()
    results = calibrate_day(data)
    calib_seconds = time.time() - start
    print(
        f"        IPOPT finished in {calib_seconds:.0f}s: "
        f"{results.get('termination_condition')} (objective {results.get('obj_val')})"
    )

    save_results(results, data, out_dir)
    metrics = simulate_and_score(results, data, out_dir)

    summary = {
        "data_path": data_path,
        "out_dir": out_dir,
        "status": "done",
        "calibration_seconds": calib_seconds,
        "solver_status": results.get("solver_status"),
        "termination_condition": results.get("termination_condition"),
        "objective": results.get("obj_val"),
        "metrics": metrics,
        "settings": {
            "T": T,
            "l": l,
            "include_ramping": INCLUDE_RAMPING,
            "varylanes": VARYLANES,
            "smoothing": SMOOTHING,
            "constraint_tol": CONSTRAINT_TOL,
            "time_varying_ramps": TIME_VARYING_RAMPS,
            "seed": SEED,
            "max_dv": MAX_DV,
        },
    }
    # Written last, so its presence means the run finished.
    with open(os.path.join(out_dir, DONE_MARKER), "w") as f:
        json.dump(summary, f, indent=2)
    sim, mis = metrics["simulation"], metrics["mismatch"]
    print(
        f"        Simulation error (sim vs data): v MAPE {sim['v_mape']:.3f}, v RMSE {sim['v_rmse']:.2f}, "
        f"rho MAPE {sim['rho_mape']:.3f}, rho RMSE {sim['rho_rmse']:.2f}"
    )
    print(
        f"        Model mismatch (pred vs sim): v MAPE {mis['v_mape']:.3f}, v RMSE {mis['v_rmse']:.2f}, "
        f"rho MAPE {mis['rho_mape']:.3f}, rho RMSE {mis['rho_rmse']:.2f}"
    )
    print(
        f"[done] {data_path}: v MAPE calib {metrics['calibration']['v_mape']:.3f}, "
        f"sim {metrics['simulation']['v_mape']:.3f}"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_paths", nargs="+", help="Data folders, e.g. data/i24_4hrs/i24_11_23")
    parser.add_argument("--run-name", default="test", help="Subfolder under calibration_static/ (default: test)")
    parser.add_argument("--overwrite", action="store_true", help="Re-run days that already have results")
    args = parser.parse_args()

    summaries = []
    for data_path in args.data_paths:
        try:
            summaries.append(run_one(data_path.rstrip("/"), args.run_name, args.overwrite))
        except Exception as e:
            print(f"[error] {data_path}: {e}")
            traceback.print_exc()
            summaries.append({"data_path": data_path, "status": f"error: {e}"})

    print("\nSummary")
    print(f"{'data path':40s} {'status':10s} {'time (s)':>9s} {'v MAPE calib':>13s} {'v MAPE sim':>11s}")
    for s in summaries:
        m = s.get("metrics")
        print(
            f"{s['data_path']:40s} {s['status'][:10]:10s} "
            f"{s.get('calibration_seconds', float('nan')):9.0f} "
            f"{m['calibration']['v_mape'] if m else float('nan'):13.3f} "
            f"{m['simulation']['v_mape'] if m else float('nan'):11.3f}"
        )
    if any(s["status"].startswith("error") for s in summaries):
        sys.exit(1)


if __name__ == "__main__":
    main()
