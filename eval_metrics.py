import numpy as np
from optimization_utils import mape
from simulation_utils import run_metanet_sim
from mpc_utils import simulate_multiple_params

def generate_perturbations(signal, percent_noise=0.1, seed=0, num=500):
    # Add noise_std percent gaussian noise to the whole signal

    # x = np.asarray(signal, dtype=float)

    rng = np.random.default_rng(seed)

    # noise = np.random.normal(0, signal.std(), signal.size)
    sigma = percent_noise * np.std(signal) / 100
    noise = rng.normal(loc=0.0, scale=sigma, size=(num, *signal.shape))
    perturbed = signal[None, ...] + noise  # broadcast signal across `num`
    return perturbed


def eval_robustness_static(v_gt, params, data_inflow, downstream_density, init_state, lanes, 
                           T=10/3600, l=0.4, percent_noise=0.1):
    
    _, true_v_sim, _, _ = run_metanet_sim(T,
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

    for perturbed_conditions in generate_perturbations(data_inflow, percent_noise=percent_noise):
        rho_sim, v_sim, _, _ = run_metanet_sim(T,
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
    
    return true_error, max(errors)

def eval_robustness_dynamic(v_gt, control_len, params_dir, data_inflow, downstream_density, init_traffic_state, 
                            T=10/3600, l=0.4, percent_noise=0.1):
    
    _, true_v_sim = simulate_multiple_params(init_traffic_state,
                                                downstream_density,
                                                data_inflow,
                                                T,
                                                l,
                                                control_len,
                                                params_dir)
    
    true_error = mape(v_gt, true_v_sim)


    errors = []

    for perturbed_conditions in generate_perturbations(data_inflow, percent_noise=percent_noise):
        _, v_sim = simulate_multiple_params(init_traffic_state,
                                            downstream_density,
                                            perturbed_conditions,
                                            T,
                                            l,
                                            control_len,
                                            params_dir)
        
        error = mape(v_gt, v_sim)
        errors.append(error)
    
    return true_error, max(errors)
