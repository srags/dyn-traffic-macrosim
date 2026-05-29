import numpy as np
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from simulation_utils import run_metanet_sim
import optimization_utils as opt
from param_loader import METANET_Params
from typing import Callable, Dict, List, Optional, Any, Tuple

def mpc_results_dir(root, control_horizon):
    return f"{root}/control_h_{control_horizon}"

def plot_mpc_acc_robustness(root_dir, control_horizons, percent_noise=[0.001], metric="mean", g_fontsize=23):
    import os
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "mathtext.fontset": "cm",   # Computer Modern for math
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "text.usetex": False,
        "font.size": g_fontsize,
        "axes.labelsize": g_fontsize,
        "xtick.labelsize": g_fontsize - 2,
        "ytick.labelsize": g_fontsize - 2,
        "legend.fontsize": g_fontsize - 1,
    })

    mape = []
    acc_at_noise = {pn: [] for pn in percent_noise}
    valid_control_horizons = []

    # Read in results files for each control horizon
    for control_h in control_horizons:
        results_dir = mpc_results_dir(root_dir, control_h) + "/noise_robustness_results.csv"
        if not os.path.exists(results_dir):
            print(f"Results file not found for control horizon {control_h} at {results_dir}")
            continue

        df = pd.read_csv(results_dir)

        # true_mape taken from first requested noise level
        row_mape = df[df['noise'] == percent_noise[0]]
        if row_mape.empty:
            print(f"No true_mape row found for noise level {percent_noise[0]} in control horizon {control_h}")
            continue

        valid_control_horizons.append(control_h)
        mape.append(row_mape['true_mape'].values[0])

        for pn in percent_noise:
            row = df[df['noise'] == pn]
            if row.empty:
                print(f"No results found for noise level {pn} in control horizon {control_h}")
                acc_at_noise[pn].append(np.nan)
                continue
            acc_at_noise[pn].append(row[metric].values[0])

    # Plotting
    fig, ax = plt.subplots(figsize=(13, 7))

    line_info = []

    color_map = {0.01: "darkred", 1: "indianred", 5: "coral"}

    # Plot noise curves
    for noise_key in percent_noise:
        yvals = acc_at_noise[noise_key]
        line, = ax.plot(
            valid_control_horizons[:len(yvals)],
            yvals,
            linewidth=4.0,
            marker='o',
            label=f'{metric} MAPE at {noise_key}% noise',
            color=color_map.get(noise_key, None)
        )
        final_val = yvals[-1] if len(yvals) > 0 and not np.isnan(yvals[-1]) else -np.inf
        line_info.append((line, f'Avg MAPE at {noise_key}% noise', noise_key, final_val))

    # Plot no-noise baseline
    baseline_line, = ax.plot(
        valid_control_horizons[:len(mape)],
        mape,
        marker='o',
        linewidth=3.0,
        linestyle="dashed",
        color="gray",
        label='MAPE with no noise'
    )
    baseline_final = mape[-1] if len(mape) > 0 else -np.inf
    line_info.append((baseline_line, 'MAPE with no noise', -np.inf, baseline_final))

    # Axis labels
    ax.set_xlabel('Horizon length (min)', fontsize=g_fontsize)
    ax.set_ylabel('MAPE on ground truth vel (%)', fontsize=g_fontsize)

    # Convert x ticks to minutes by dividing by 6
    min_h = min(valid_control_horizons) / 6
    max_h = max(valid_control_horizons) / 6

    tick_minutes = np.arange(5 * np.floor(min_h / 5), 5 * np.ceil(max_h / 5) + 5, 5)
    tick_positions = tick_minutes * 6  # convert minutes back to horizon units

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{int(t)}" for t in tick_minutes], fontsize=g_fontsize - 2)

    ax.tick_params(axis='y', which='major', labelsize=g_fontsize - 2)

    # Legend order:
    # 1. decreasing noise
    # 2. for same noise, decreasing final MAPE
    # baseline/no-noise last
    sorted_line_info = sorted(
        line_info,
        key=lambda x: (
            x[2] == -np.inf,   # baseline last
            -x[2] if x[2] != -np.inf else np.inf,
            -x[3] if x[3] != -np.inf else np.inf
        )
    )

    handles = [x[0] for x in sorted_line_info]
    labels = [x[1] for x in sorted_line_info]

    ax.legend(handles, labels, fontsize=g_fontsize - 1)
    plt.tight_layout()
    plt.savefig("dippy_plot.pdf", bbox_inches='tight', pad_inches=0)

    plt.show()



def save_results(results, RESULTS_DIR, control_horizon=None, from_param_loader=False):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Save everything
    tau_array = results["tau"]
    K_array = results["K"]
    eta_high_array = results["eta_high"]
    v_free_array = results["v_free"]
    a_array = results["a"]
    num_lanes_array = results["num_lanes"]
    beta_array = results["beta"]

    if from_param_loader:
        r_inflow_array = results["r"]
        rho_crit_array = results["p_crit"]
    else:
        r_inflow_array = results["r_inflow"]
        rho_crit_array = results["rho_crit"]
        v_pred_array = results["v_pred"][0:control_horizon, :]
        rho_pred_array = results["rho_pred"][0:control_horizon, :]
        q_array = v_pred_array * rho_pred_array

    np.save(f"{RESULTS_DIR}/tau.npy", tau_array)
    np.save(f"{RESULTS_DIR}/K.npy", K_array)
    np.save(f"{RESULTS_DIR}/eta_high.npy", eta_high_array)
    np.save(f"{RESULTS_DIR}/rho_crit.npy", rho_crit_array)
    np.save(f"{RESULTS_DIR}/v_free.npy", v_free_array)
    np.save(f"{RESULTS_DIR}/a.npy", a_array)
    np.save(f"{RESULTS_DIR}/num_lanes.npy", num_lanes_array)
    np.save(f"{RESULTS_DIR}/beta_array.npy", beta_array)
    np.save(f"{RESULTS_DIR}/r_inflow_array.npy", r_inflow_array)

    if not from_param_loader:
        np.save(f"{RESULTS_DIR}/q_pred.npy", q_array)
        np.save(f"{RESULTS_DIR}/v_pred.npy", v_pred_array)
        np.save(f"{RESULTS_DIR}/rho_pred.npy", rho_pred_array)


def calibrate_params_mpc(
    rho_hat: np.ndarray,
    q_hat: np.ndarray,
    T: float,
    l: float,
    control_h: int,
    *,
    pred_h: Optional[int] = None,
    num_calibrated_segments: int,
    varylanes: bool = False,
    include_ramping: bool = False,
    smoothing: bool = True,
    constraint_tol: float = 1e-12,
    verbose: bool = False,
    results_root_dir: Optional[str] = None,
    robust_opt: Optional[opt.RobustOptConfig] = None,
    tighter_bounds: bool = False,
    enforce_ramps: bool = False,
    warmstart: Optional[str] = None,
    lane_mapping: Optional[Dict[int, float]] = None,
    stable_params: bool = False,
    ramp_mapping: Optional[Dict[int, float]] = None,
    separate_boundary_conditions: Optional[Dict[int, float]] = None,
    use_A_regularizer: bool = False,
    lambda_reg: float = 1.0
) -> None:
    if pred_h is None:
        pred_h = control_h
    if control_h <= 0 or pred_h <= 0:
        raise ValueError("control_h and pred_h must be positive integers.")

    K = rho_hat.shape[0]
    if K != q_hat.shape[0]:
        raise ValueError("rho_hat and q_hat must have the same time dimension.")
    if rho_hat.shape[1] != q_hat.shape[1]:
        raise ValueError("rho_hat and q_hat must have the same number of columns/segments.")
    
    if K % control_h != 0:
        raise ValueError("Total timesteps K must be divisible by control_h.")

    # Processing boundary and initial conditions

    if separate_boundary_conditions is not None:
        downstream_density = separate_boundary_conditions["downstream_density"]
        data_inflow = separate_boundary_conditions["initial_flow"]
    else:
        downstream_density = rho_hat[:, -1]
        data_inflow = q_hat[:, 0]

    if smoothing:
        downstream_density = opt.smooth_inflow(downstream_density)
        data_inflow = opt.smooth_inflow(data_inflow)


    init_rho = rho_hat[0, 1:-1] if separate_boundary_conditions is None else rho_hat[0, :]
    init_q = q_hat[0, 1:-1] if separate_boundary_conditions is None else q_hat[0, :]
    init_v = init_q / init_rho

    ind_start = 0
    total_steps = K // control_h
    # need at least 2 timesteps in a window for dynamics
    while ind_start < K:
        # print(os.path.exists(os.getcwd() + "/" + warmstart + f"/control_h_{control_h}"))
        # print(os.getcwd() + "/" + warmstart + f"/control_h_{control_h}")

        if warmstart is None:
            warmstart_path = None
        elif warmstart == "use_standard_mpc":
            warmstart_path = f"mpc_calibration_results/control_h_{control_h}/params_{ind_start // control_h + 1}"
            # Check if warmstart file has control_h folders in it
        elif os.path.exists(warmstart + f"/control_h_{control_h}"):
            warmstart_path = f"{warmstart}/control_h_{control_h}/params_{ind_start // control_h + 1}"
        else: 
            warmstart_path = warmstart

        ind_end = min(K, ind_start + pred_h)

        trunc_rho = np.copy(rho_hat[ind_start:ind_end, 1:-1]) if separate_boundary_conditions is None else np.copy(rho_hat[ind_start:ind_end, :])
        trunc_q = np.copy(q_hat[ind_start:ind_end, 1:-1]) if separate_boundary_conditions is None else np.copy(q_hat[ind_start:ind_end, :])

        # If you want each window to start from measured initial state at its start (common):
        trunc_rho[0, :] = np.copy(init_rho)
        trunc_q[0, :] = np.copy(init_q)

        calib_res = opt.run_calibration(
            trunc_rho,
            trunc_q,
            T,
            l,
            num_calibrated_segments=num_calibrated_segments,
            sep_boundary_conditions={"downstream_density": downstream_density[ind_start:ind_end], 
                                     "initial_flow": data_inflow[ind_start:ind_end]},
            varylanes=varylanes,
            include_ramping=include_ramping,
            smoothing=False,
            constraint_tol=constraint_tol,
            robust_opt=robust_opt,
            warmstart=warmstart_path, #"calibration_results/robust_S5_0001" if ind_start == 0 else f"{results_root_dir}/control_h_{control_h}/params_{ind_start // control_h}",
            lane_mapping=lane_mapping,
            prev_param_path = f"{mpc_results_dir(results_root_dir, control_h)}/params_{ind_start // control_h}" if stable_params and ind_start > 0 else None,
            ramp_mapping=ramp_mapping,
            use_A_regularizer=use_A_regularizer,
            lambda_reg=lambda_reg,
        )

        RESULTS_DIR = f"{mpc_results_dir(results_root_dir, control_h)}/params_{ind_start // control_h + 1}"
        save_results(calib_res, RESULTS_DIR, control_horizon=control_h)
        num_lanes_array = np.array(calib_res["num_lanes"], dtype=float)
        v_pred = np.array(calib_res["v_pred"])
        rho_pred = np.array(calib_res["rho_pred"])

        params = METANET_Params(path=RESULTS_DIR, num_segments=num_calibrated_segments).get_params()

        # print("--------Simulation boundary conditions--------")
        # print("Initial flow:", data_inflow[0:10])
        # print("Downstream density:", downstream_density[0:10] * np.array(num_lanes_array[-1]))
        # print("Initial velocity:", init_traffic_state[1])
        # print("Initial density:", init_traffic_state[0] * np.array(num_lanes_array))

        dd_sim = downstream_density[ind_start:ind_end] / num_lanes_array[-1]
        inflow_sim = data_inflow[ind_start:ind_end]

        init_traffic_state = (init_rho/np.array(num_lanes_array), init_v, data_inflow[ind_start], 0)

        # print(params)

        # print(f"Inflow sim (first 10 values) at step {ind_start}: {inflow_sim[0:10]}")
        # print(f"Downstream density sim (first 10 values) at step {ind_start}: {dd_sim[0:10]}")
        # print(init_traffic_state)

        rho_sim, v_sim, _, tts_sim = run_metanet_sim(
            T, 
            l, 
            init_traffic_state,
            inflow_sim,
            dd_sim,
            params,
            vsl_speeds=None,
            lanes={i: num_lanes_array[i] for i in range(num_calibrated_segments)},
            plotting=True,
            real_data=True
        )

        ind_start += control_h

        print(v_sim.shape)
        init_rho = np.copy(rho_sim[control_h, :]) * np.array(num_lanes_array)
        init_q = init_rho * np.copy(v_sim[control_h, :])
        init_v = np.copy(v_sim[control_h, :])



        for i in range(control_h):
            init_rho_test = rho_pred[i, :] 
            init_v_test = v_pred[i, :]

            init_rho_sim = rho_sim[i, :] * np.array(num_lanes_array)
            init_v_sim = v_sim[i, :]

            if not np.allclose(init_rho_test, init_rho_sim, atol=1):
                print(f"Density mismatch at step {i} of {np.sum(init_rho_test-init_rho_sim)/np.size(init_rho_test)}")
                break
        


        # print("--------NEXT STATE FROM SIM:")
        # print(init_v)
        # print(init_rho)
        # print("--------TEST FROM PREDICTION:")
        # print(init_v_test)
        # print(init_rho_test)

        if verbose:
            print(f"Results saved to {RESULTS_DIR}/")
            print(f"New velocities: {v_sim[control_h, :]}")
            print(f"New densities: {rho_sim[control_h, :]}")

    print(f"Calibration over {K} time steps with control horizon {control_h} completed.")


def simulate_multiple_params(
    init_traffic_state: Tuple[np.ndarray, np.ndarray, float, float],
    downstream_density: np.ndarray,
    data_inflow: np.ndarray,
    T: float,
    l: float,
        control_h: int,
    root_results_dir: str,
):
    time_steps = data_inflow.shape[0]
    num_horizons = time_steps // control_h
    num_calibrated_segments = init_traffic_state[0].shape[0]

    if time_steps % control_h != 0:
        raise ValueError("Total time steps must be divisible by control_h.")
    
    v_pred_total = None
    v_sim_total = None
    
    for h in range(num_horizons):
        start_time = h * control_h
        end_time = start_time + control_h

        RESULTS_DIR = f"{mpc_results_dir(root_results_dir, control_h)}/params_{start_time // control_h + 1}"
        #RESULTS_DIR = f'{root_results_dir}/{h+1}_of_{num_horizons}'
        params = METANET_Params(path=RESULTS_DIR, num_segments=num_calibrated_segments).get_params()
        v_plot_pred = np.load(f"{RESULTS_DIR}/v_pred.npy")
        v_pred_total = (v_plot_pred if v_pred_total is None else np.vstack((v_pred_total, v_plot_pred)))

        num_lanes_array = np.load(f"{RESULTS_DIR}/num_lanes.npy")

        # if h == 1:
        #     print(f"Inflow sim (first 10 values) at step {start_time}: {data_inflow[start_time:start_time+10]}")
        #     print(f"Downstream density sim (first 10 values) at step {start_time}: {downstream_density[start_time:start_time+10]}")
        #     print(init_traffic_state)

        rho_sim, v_sim, _, tts_sim = run_metanet_sim(
            T, 
            l, 
            init_traffic_state,
            data_inflow[start_time:end_time],
            downstream_density[start_time:end_time],
            params,
            vsl_speeds=None,
            lanes={i: num_lanes_array[i] for i in range(num_calibrated_segments)},
            plotting=True,
            real_data=True
        )

        v_sim_trim = v_sim[:-1, :]
        v_sim_total = (v_sim_trim if v_sim_total is None else np.vstack((v_sim_total, v_sim_trim)))

        if h < num_horizons - 1:
            init_traffic_state = (rho_sim[-1, :], v_sim[-1, :], data_inflow[end_time], 0)
    
    return v_pred_total, v_sim_total

