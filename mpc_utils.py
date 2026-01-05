import numpy as np
import os
import random
from simulation_utils import run_metanet_sim
from optimization_utils import run_calibration, smooth_inflow
from param_loader import METANET_Params
from typing import Callable, Dict, List, Optional, Any, Tuple

def save_results(results, RESULTS_DIR, control_horizon=None):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Save everything
    tau_array = results["tau"]
    K_array = results["K"]
    eta_high_array = results["eta_high"]
    rho_crit_array = results["rho_crit"]
    v_free_array = results["v_free"]
    a_array = results["a"]
    num_lanes_array = results["num_lanes"]
    beta_array = results["beta"]
    r_inflow_array = results["r_inflow"]
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
    np.save(f"{RESULTS_DIR}/q_pred.npy", q_array)
    np.save(f"{RESULTS_DIR}/v_pred.npy", v_pred_array)
    np.save(f"{RESULTS_DIR}/rho_pred.npy", rho_pred_array)
    np.save(f"{RESULTS_DIR}/beta_array.npy", beta_array)
    np.save(f"{RESULTS_DIR}/r_inflow_array.npy", r_inflow_array)


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
    if smoothing:
        downstream_density = smooth_inflow(rho_hat[:, -1])
        data_inflow = smooth_inflow(q_hat[:, 0])
    else:
        downstream_density = rho_hat[:, -1]
        data_inflow = q_hat[:, 0]

    init_rho = rho_hat[0, 1:-1]
    init_q = q_hat[0, 1:-1]
    init_v = init_q / init_rho

    ind_start = 0
    total_steps = K // control_h
    # need at least 2 timesteps in a window for dynamics
    while ind_start < K:
        ind_end = min(K, ind_start + pred_h)

        trunc_rho = np.copy(rho_hat[ind_start:ind_end, 1:-1])
        trunc_q = np.copy(q_hat[ind_start:ind_end, 1:-1])

        # If you want each window to start from measured initial state at its start (common):
        trunc_rho[0, :] = np.copy(init_rho)
        trunc_q[0, :] = np.copy(init_q)

        calib_res = run_calibration(
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
        )

        RESULTS_DIR = f"{results_root_dir}/{ind_start // control_h + 1}_of_{total_steps}"
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

        dd_sim = downstream_density[ind_start:ind_start+control_h] / num_lanes_array[-1]
        inflow_sim = data_inflow[ind_start:ind_start+control_h]

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
        init_rho = np.copy(rho_sim[-1, :]) * np.array(num_lanes_array)
        init_q = init_rho * np.copy(v_sim[-1, :])
        init_v = np.copy(v_sim[-1, :])



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

        RESULTS_DIR = f'{root_results_dir}/{h+1}_of_{num_horizons}'
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

