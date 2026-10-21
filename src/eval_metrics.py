import numpy as np
import pandas as pd
import simulation_utils as sim
from param_loader import METANET_Params
import csv
import paths


def mape(flow_hat, flow_pred):
    """
    Compute the Mean Absolute Percentage Error (MAPE) between ground truth and prediction.

    Parameters:
        flow_hat (np.ndarray): Ground truth array of shape [t, i]
        flow_pred (np.ndarray): Predicted array of shape [t, i]

    Returns:
        float: The mean absolute percentage error (in percent)
    """
    # Avoid division by zero by masking out zero ground truth values
    mask = flow_hat != 0
    # print(flow_hat.shape)
    # print(flow_pred.shape)
    error = np.abs((flow_pred[mask] - flow_hat[mask]) / flow_hat[mask])
    return np.mean(error) * 100


def rmse(flow_hat, flow_pred):
    """
    Compute the Root Mean Squared Error (RMSE) between ground truth and prediction.

    Parameters:
        flow_hat (np.ndarray): Ground truth array of shape [t, i]
        flow_pred (np.ndarray): Predicted array of shape [t, i]

    Returns:
        float: The root mean squared error
    """
    error = flow_pred - flow_hat
    mse = np.mean(np.square(error))
    return np.sqrt(mse)

def J_cost(v_sim, v_hat, p_sim, p_hat, T):
    """
    v_sim : array of shape (T_steps, N) - simulated speed trajectory
    v_hat : array of shape (T_steps, N) - observed speed trajectory
    p_sim : array of shape (T_steps, N) - simulated density trajectory
    p_hat : array of shape (T_steps, N) - observed density trajectory
    """
    # Combine v_sim and p_sim into a single array of shape (T_steps, 2N) for comparison with v_hat and p_hat
    x_sim = np.hstack((v_sim[0:T, :], p_sim[0:T, :]))  # shape (T_steps, 2N)
    x_hat = np.hstack((v_hat[0:T, :], p_hat[0:T, :]))
    return np.sum((x_sim - x_hat)**2)

def generate_perturbations(signal, percent_noise=0.1, seed=0, num=100):
    # Add noise_std percent gaussian noise to the whole signal

    # x = np.asarray(signal, dtype=float)

    rng = np.random.default_rng(seed)

    # noise = np.random.normal(0, signal.std(), signal.size)
    sigma = percent_noise * np.std(signal) / 100
    noise = rng.normal(loc=0.0, scale=sigma, size=(num, *signal.shape))
    perturbed = signal[None, ...] + noise  # broadcast signal across `num`
    return perturbed


def eval_robustness_static(v_gt, params, data_inflow, downstream_density, init_state, lanes, 
                           T=10/3600, l=0.4, percent_noises=[0.1], plotting_dir=None, save_results=False, rho_gt=None):
    import plotting  # imported here to avoid a circular import with optimization_utils
    noise_results = []
    for percent_noise in percent_noises:
        true_rho_sim, true_v_sim, _, _ = sim.run_metanet_sim(T,
                                        l,
                                        init_state,
                                        data_inflow,
                                        downstream_density,
                                        params, 
                                        vsl_speeds=None,
                                        lanes=lanes,
                                        plotting=True,
                                        real_data=True)
        true_error = mape(v_gt, true_v_sim[0:-1, :])

        errors = []
        plotting_v_sim = None
        worst_v_sim = None

        if rho_gt is not None:
            J = J_cost(true_v_sim[0:-1, :], v_gt, true_rho_sim[0:-1, :], rho_gt, v_gt.shape[0])



        for perturbed_conditions in generate_perturbations(data_inflow, percent_noise=percent_noise):
            rho_sim, v_sim, _, _ = sim.run_metanet_sim(T,
                                        l,
                                        init_state,
                                        perturbed_conditions,
                                        downstream_density,
                                        params, 
                                        vsl_speeds=None,
                                        lanes=lanes,
                                        plotting=True,
                                        real_data=True)
            
            error = mape(v_gt, v_sim[0:-1, :])
            errors.append(error)

            if worst_v_sim is None or error > worst_v_sim:
                plotting_v_sim = v_sim
                worst_v_sim = error
        
        if plotting_dir is not None:
            plotting.plot_sim_vs_gt(v_gt, plotting_v_sim, T, l, percent_noise, save_dir=plotting_dir + "/figs")      

        eval_results = {"noise": percent_noise, "true_mape": true_error,"max": max(errors), "mean": np.mean(errors)}

        errors = np.array(errors)
        eval_results["percentile_5"] = np.percentile(errors, 5)
        eval_results["percentile_95"] = np.percentile(errors, 95)

        eval_results["J_cost"] = J
        print(f"J_cost for true simulation: {J}")

        noise_results.append(eval_results)
        print(f"Robustness error with {percent_noise}% noise level on inflows: worst-case = {eval_results['max']} / mean = {eval_results['mean']}")

    #Save noise results to a file
    if plotting_dir is not None and save_results:
        # The keys from the first dictionary define the column headers
        fieldnames = noise_results[0].keys()
        filename = f'{plotting_dir}/noise_robustness_results.csv'

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()  # Write the header
            writer.writerows(noise_results) # Write all the rows at once

        print(f"Results saved to {filename}")
    
    return noise_results

def eval_robustness_dynamic(v_gt, control_len, params_dir, data_inflow, downstream_density, init_traffic_state, lanes,
                            T=10/3600, l=0.4, percent_noises=[0.1], save_results=False, rho_gt=None):
    import mpc_utils as mpc
    import plotting  # imported here to avoid a circular import with optimization_utils

    noise_results = []

    params = METANET_Params(params_dir, control_h=control_len, 
                            num_timesteps=v_gt.shape[0], num_segments=v_gt.shape[1]).get_params()
    for percent_noise in percent_noises:
        # true_rho_sim, true_v_sim = mpc.simulate_multiple_params(init_traffic_state,
        #                                         downstream_density,
        #                                         data_inflow,
        #                                         T,
        #                                         l,
        #                                         control_len,
        #                                         params_dir)
        

        
        true_rho_sim, true_v_sim, _, _ = sim.run_metanet_sim(T,
                                        l,
                                        init_traffic_state,
                                        data_inflow,
                                        downstream_density,
                                        params, 
                                        vsl_speeds=None,
                                        lanes=lanes,
                                        plotting=True,
                                        real_data=True)
        true_rho_sim = true_rho_sim[0:-1, :]
        true_v_sim = true_v_sim[0:-1, :]
        
        true_error = mape(v_gt, true_v_sim)

        errors = []
        plotting_v_sim = None
        worst_v_sim = None

        if rho_gt is not None:
            # print(np.max(rho_gt), np.max(true_rho_sim))
            # print(mape(rho_gt, true_rho_sim))
            # print(mape(v_gt, true_v_sim))
            J = J_cost(true_v_sim, v_gt, true_rho_sim, rho_gt, v_gt.shape[0])

        for perturbed_conditions in generate_perturbations(data_inflow, percent_noise=percent_noise):
            _, v_sim = mpc.simulate_multiple_params(init_traffic_state,
                                            downstream_density,
                                            perturbed_conditions,
                                            T,
                                            l,
                                            control_len,
                                            params_dir)
            
            error = mape(v_gt, v_sim)
            errors.append(error)

            if worst_v_sim is None or error > worst_v_sim:
                plotting_v_sim = v_sim
                worst_v_sim = error
        
        plotting.plot_sim_vs_gt(v_gt, plotting_v_sim, T, l, percent_noise, save_dir=mpc.mpc_results_dir(params_dir, control_len) + "/figs")                     

        eval_results = {"noise": percent_noise, "true_mape": true_error, "max": max(errors), "mean": np.mean(errors)}
        if rho_gt is not None:
            eval_results["J_cost"] = J
            print(f"J_cost for true simulation: {J}")

        errors = np.array(errors)
        eval_results["percentile_5"] = np.percentile(errors, 5)
        eval_results["percentile_95"] = np.percentile(errors, 95)

        noise_results.append(eval_results)
        print(f"Robustness error with {percent_noise}% noise level on inflows: worst-case = {eval_results['max']} / mean = {eval_results['mean']}")


    # The keys from the first dictionary define the column headers
    if save_results:
        fieldnames = noise_results[0].keys()
        filename = f'{mpc.mpc_results_dir(params_dir, control_len)}/noise_robustness_results.csv'

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()  # Write the header
            writer.writerows(noise_results) # Write all the rows at once

        print(f"Results saved to {filename}")

    return noise_results

# Plot time-space diagrams of the worst-case MAPE for different noise perturbations given parameters
def plot_worst_case_simulations(v_gt, control_len, params_dir, data_inflow, downstream_density, init_traffic_state,
                            T=10/3600, l=0.4, percent_noises=[0.1]):
    import mpc_utils as mpc
    import plotting  # imported here to avoid a circular import with optimization_utils
    
    for percent_noise in percent_noises:
        _, true_v_sim = mpc.simulate_multiple_params(init_traffic_state,
                                                downstream_density,
                                                data_inflow,
                                                T,
                                                l,
                                                control_len,
                                                params_dir)
        true_error = mape(v_gt, true_v_sim)

        worst_v_sim = None
        worst_error = -1

        for perturbed_conditions in generate_perturbations(data_inflow, percent_noise=percent_noise):
            _, v_sim = mpc.simulate_multiple_params(init_traffic_state,
                                            downstream_density,
                                            perturbed_conditions,
                                            T,
                                            l,
                                            control_len,
                                            params_dir)
            
            error = mape(v_gt, v_sim)

            if error > worst_error:
                worst_v_sim = v_sim
                worst_error = error
        
        plotting.plot_sim_vs_gt(v_gt, worst_v_sim, T, l, percent_noise, save_dir=mpc.mpc_results_dir(params_dir, control_len) + "/figs")


I24_HOLIDAYS = ("11_24", "11_25")   # Thanksgiving Thursday and Friday; excluded from cross-day evaluation


def _day_id(day):
    """Accept "11_30" or "i24_11_30" and return "11_30"."""
    return day[len("i24_"):] if day.startswith("i24_") else day


ERROR_METRICS = {"mape": mape, "rmse": rmse}   # MAPE in %, RMSE in the quantity's units


def _simulate_day(params, day, T, l):
    """Run METANET over one I-24 day with that day's boundary conditions and return the errors of the
    calibrated segments against the day's data: {"v_mape", "v_rmse" (km/h), "rho_mape", "rho_rmse"
    (veh/km/lane)}. Density is compared per lane."""
    from optimization_utils import smooth_inflow  # imported here to avoid a circular import

    data_path = paths.i24_data(day)
    lanes = np.load(f"{data_path}/lane_mapping.npy")
    rho_hat = np.load(f"{data_path}/rho_hat.npy")   # total density, includes boundary cells
    q_hat = np.load(f"{data_path}/q_hat.npy")       # total flow, includes boundary cells
    rho_hat = np.where(rho_hat == 0.0, 1e-3, rho_hat)
    q_hat = np.where(q_hat == 0.0, 1e-3, q_hat)
    v_hat = q_hat / rho_hat

    seg_lanes = lanes[1:-1]
    data_inflow = smooth_inflow(q_hat[:, 0])
    downstream_density = smooth_inflow(rho_hat[:, -1] / lanes[-1])
    init_traffic_state = (rho_hat[0, 1:-1] / seg_lanes, v_hat[0, 1:-1], data_inflow[0], 0)

    rho_sim, v_sim, _, _ = sim.run_metanet_sim(T, l, init_traffic_state, data_inflow, downstream_density, params,
                                               vsl_speeds=None,
                                               lanes={i: seg_lanes[i] for i in range(len(seg_lanes))},
                                               plotting=True, real_data=True)
    observed = {"v": v_hat[:, 1:-1], "rho": rho_hat[:, 1:-1] / seg_lanes}
    simulated = {"v": v_sim[:-1, :], "rho": rho_sim[:-1, :]}
    return {f"{quantity}_{name}": fn(observed[quantity], simulated[quantity])
            for quantity in ("v", "rho") for name, fn in ERROR_METRICS.items()}


def evaluate_cross_day(calibrations, train_day, days=None, exclude=I24_HOLIDAYS, T=10/3600, l=0.4):
    """
    Evaluate calibrations on the day they were calibrated on and on every other I-24 day.

    Each calibration is simulated over the whole period with each day's boundary conditions (inflow,
    downstream density, initial state) and compared to that day's data by MAPE and RMSE of speed and
    (per-lane) density. The train
    error is computed the same way on train_day. Time-varying (dynamic) parameters are applied by
    time of day, and calibrated ramp flows are reused as is on every day.

    Parameters:
        calibrations (dict): {method name: METANET_Params (or its get_params() dict)}
        train_day (str): Day the calibrations were fit on, e.g. "11_30" or "i24_11_30"
        days (list or None): Days to evaluate; defaults to every day in data/i24/
        exclude (tuple): Days to leave out (default: the 11/24 and 11/25 holidays)

    Returns:
        pd.DataFrame: One row per (method, day) with columns method, day, split ("train"/"test"),
        v_mape (%), v_rmse (km/h), rho_mape (%) and rho_rmse (veh/km/lane)
    """
    train_day = _day_id(train_day)
    days = [_day_id(d) for d in (paths.i24_dates() if days is None else days)]
    days = [d for d in days if d not in exclude]
    if train_day not in days:
        raise ValueError(f"train_day {train_day} is not among the evaluated days {days}")

    rows = []
    for method, params in calibrations.items():
        params = params.get_params() if hasattr(params, "get_params") else params
        for day in days:
            rows.append({"method": method, "day": day, "split": "train" if day == train_day else "test",
                         **_simulate_day(params, day, T, l)})
    return pd.DataFrame(rows)
