import numpy as np
import matplotlib.pyplot as plt
from eval_metrics import generate_perturbations, mape, rmse
from pyomo.environ import (
    ConcreteModel,
    RangeSet,
    Param,
    Var,
    Expression,
    Objective,
    SolverFactory,
    minimize,
    value,
    Constraint,
    ConstraintList,
)
from param_loader import METANET_Params
import pyomo.environ as pyo
from dataclasses import dataclass, asdict, field

from simulation_utils import run_metanet_sim

CALIB_PARAM_NAMES = ["eta_high", "tau", "K", "rho_crit", "v_free", "a"]   # <<<
 
 #['eta_high', 'v_free'] 

CALIB_PARAM_BOUNDS = {                                                       # <<<
    "eta_high": (10.0,        90.0),                                         # <<<
    "tau":      (10.0/3600,   60.0/3600),                                    # <<<
    "K":        (5.0,         60.0),                                         # <<<
    "rho_crit": (15.0,        75.0),                                         # <<<
    "v_free":   (70.0,        150.0),                                        # <<<
    "a":        (0.5,         5.0),                                          # <<<
}        
# CALIB_PARAM_BOUNDS = {                                                       # <<<
#     "eta_high": (30.0,        31.0),                                         # <<<
#     "tau":      (15.0/3600,   16.0/3600),                                    # <<<
#     "K":        (39.0,         40.0),                                         # <<<
#     "rho_crit": (34.0,        35.0),                                         # <<<
#     "v_free":   (90.0,        91.0),                                        # <<<
#     "a":        (1.1,         1.2),                                          # <<<
# }                                                                            # <<<
 

@dataclass
class RobustOptConfig:
    """
    Configuration for scenario-based robust METANET calibration.

    Will only use lam_worst if objective_mode=="mean_plus_worst".

    This is intended to be passed into metanet_param_fit_robust(..., cfg=RobustOptConfig(...))
    or unpacked via cfg.to_kwargs().
    """

    # --- Robust scenario settings ---
    S: int = 25                              # number of scenarios
    bc_noise_percent: float = 10.0           # passed to generate_perturbations(percent_noise=...)
    seed: int = 0                            # RNG seed for perturbations

    # --- Robust objective selection ---
    objective_mode: str = "minmax"
    lam_worst: float = 0.2   


# def smooth_inflow(inflow, window_size=2):
#     kernel = np.ones(window_size) / window_size
#     smoothed = np.apply_along_axis(
#         lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=inflow
#     )
#     return smoothed

def smooth_inflow(inflow, window_size=2):
    # Create averaging kernel
    kernel = np.ones(window_size) / window_size
    
    # Compute asymmetric padding for even window sizes
    pad_left = window_size // 2
    pad_right = window_size - pad_left - 1

    # Pad using boundary values (edge padding)
    if inflow.ndim == 1:
        padded = np.pad(inflow, (pad_left, pad_right), mode='edge')
    else:
        padded = np.pad(inflow, ((pad_left, pad_right), (0, 0)), mode='edge')

    # Convolve along time dimension
    smoothed = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="valid"), axis=0, arr=padded
    )
    return smoothed


def fit_fd1(
    flattened_rho_hat,
    flattened_q_hat,
    C_i=None,
    V_free_init=60,
    a_init=1.0,
    solver_name="ipopt",
    plot=True,
    top_k_for_C=5,
):
    """
    Fit a smooth FD1 fundamental diagram to (rho_hat, q_hat) data using Pyomo.

    Parameters
    ----------
    flattened_rho_hat : array-like
        Density measurements (veh/km).
    flattened_q_hat : array-like
        Flow measurements (veh/h).
    C_i : float or None
        Fixed capacity value. If None, capacity is estimated from the top K flows.
    V_free_init : float
        Initial guess for free-flow speed.
    a_init : float
        Initial guess for shape parameter a.
    solver_name : str
        Solver to use (default: 'ipopt').
    plot : bool
        Whether to plot the fitted curve and data.
    top_k_for_C : int
        Number of top flow values to average for capacity estimate if C_i is None.

    Returns
    -------
    dict
        Optimized parameters {'rho_crit', 'V_free', 'a', 'C'}.
    """

    # Ensure arrays
    flattened_rho_hat = np.array(flattened_rho_hat)
    flattened_q_hat = np.array(flattened_q_hat)
    K = len(flattened_rho_hat)

    # Estimate capacity if not provided
    if C_i is None:
        C_i = np.mean(sorted(flattened_q_hat)[-top_k_for_C:])

    # Build Pyomo model
    model = ConcreteModel()
    model.k = RangeSet(0, K - 1)

    # Parameters
    model.rho_hat = Param(
        model.k, initialize={k: flattened_rho_hat[k] for k in range(K)}
    )
    model.q_hat = Param(model.k, initialize={k: flattened_q_hat[k] for k in range(K)})
    model.C = Param(initialize=C_i)

    # Variables
    model.rho_crit = Var(
        bounds=(1e-2, max(flattened_rho_hat)), initialize=np.median(flattened_rho_hat)
    )
    model.V_free = Var(bounds=(10, 150), initialize=V_free_init)
    model.a = Var(bounds=(0.01, 10), initialize=a_init)

    # Smoothed flow function
    def q_pred_expr(model, k):
        rho = model.rho_hat[k]
        rho_crit = model.rho_crit
        V_free = model.V_free
        a = model.a
        Q = rho * V_free * pyo.exp(-1 / a * (rho / rho_crit) ** a)
        return Q

    model.q_pred = Expression(model.k, rule=q_pred_expr)

    # Objective function (least squares)
    def obj_rule(model):
        return sum((model.q_pred[k] - model.q_hat[k]) ** 2 for k in model.k)

    model.obj = Objective(rule=obj_rule, sense=minimize)

    # Solve
    solver = SolverFactory(solver_name)
    solver.solve(model, tee=False)

    # Extract parameters
    rho_crit_opt = value(model.rho_crit)
    V_free_opt = value(model.V_free)
    a_opt = value(model.a)
    C_opt = value(model.C)

    # Define fitted FD1 function
    def Q_fd1(rho):
        rho = np.array(rho)
        Q_free = V_free_opt * rho * np.exp(-1 / a_opt * (rho / rho_crit_opt) ** a_opt)
        return Q_free

    # Plot if requested
    if plot:
        rho_range = np.linspace(0, max(flattened_rho_hat) * 1.1, 500)
        q_fit = Q_fd1(rho_range)

        plt.figure(figsize=(8, 6))
        plt.scatter(
            flattened_rho_hat,
            flattened_q_hat,
            color="gray",
            alpha=0.7,
            label="Data",
            s=1,
        )
        plt.plot(rho_range, q_fit, linewidth=2.5, label="Fitted FD1", zorder=10)
        plt.axvline(
            rho_crit_opt,
            color="red",
            linestyle="--",
            label=f"ρ_crit = {rho_crit_opt:.1f}",
        )
        plt.axhline(C_opt, color="blue", linestyle=":", label=f"C = {C_opt:.1f}")
        plt.xlabel("Density ρ (veh/km)")
        plt.ylabel("Flow q (veh/h)")
        plt.title("Fundamental Diagram Fit (FD1)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "rho_crit": rho_crit_opt,
        "V_free": V_free_opt,
        "a": a_opt,
        "C": C_opt,
        "Q_fd1": Q_fd1,
    }

def metanet_param_fit(
    v_hat,
    rho_hat,
    q_hat,
    T,
    l,
    initial_traffic_state,
    downstream_density,
    num_calibrated_segments,
    include_ramping=True,
    varylanes=True,
    lane_mapping=None,
    constraint_tol=1e-12,
    warmstart=None,
    prev_param_path=None,
    ramp_mapping=None,
    time_varying_ramps=False,
    fixed_inflows=None,
    use_A_regularizer=False,
    lambda_reg=1.0,
    x0=None,
    tee=True
):
    initial_flow_or = initial_traffic_state
    
    if initial_flow_or.ndim == 1:
        initial_flow_or = initial_flow_or.reshape(-1, 1)
    
    if downstream_density.ndim == 1:
        downstream_density = downstream_density.reshape(-1, 1)
    
     # Ensure no divide-by-zero

    num_timesteps, num_segments = v_hat.shape
    # print(num_timesteps, num_segments)
    # print(initial_flow_or.shape)
    # print(downstream_density.shape)

    model = ConcreteModel()
    model.t = RangeSet(0, num_timesteps - 1)
    # model.t_loss = RangeSet(0, num_timesteps - 1, 10)
    model.i = RangeSet(0, num_segments - 1)
    # model.i = RangeSet(0, num_calibrated_segments - 1)
    model.num_segments = num_segments
    model.num_calibrated_segments = num_calibrated_segments
    model.constraints = ConstraintList()

    # Fixed params
    model.T = Param(initialize=T)
    model.l = Param(initialize=l)

    # Number of lanes (per calibrated segment)
    if lane_mapping is not None:
        # print(lane_mapping)
        model.n_lanes = Param(model.i, initialize={i: lane_mapping[i] for i in model.i})
    elif varylanes:
        model.n_lanes = Var(model.i, bounds=(3, 5), initialize=3)
    else:
        model.n_lanes = Param(model.i, initialize=4)

    # Parameters to estimate
    if warmstart is not None:
        params = METANET_Params(path=warmstart, num_segments=num_calibrated_segments).get_params()

        model.eta_high = Var(model.i, bounds=(10.0, 90.0))
        model.tau = Var(model.i, bounds=(10.0 / 3600, 60.0 / 3600))
        model.K = Var(model.i, bounds=(5.0, 60.0))
        model.rho_crit = Var(model.i, bounds=(15, 100))
        model.v_free = Var(model.i, bounds=(70, 150))
        model.a = Var(model.i, bounds=(0.5, 5))

        for i in range(num_segments):
            model.eta_high[i].value = params["eta_high"][i]
            model.tau[i].value = params["tau"][i]
            model.K[i].value = params["K"][i]
            model.rho_crit[i].value = params["p_crit"][i]
            model.v_free[i].value = params["v_free"][i]
            model.a[i].value = params["a"][i]

    else:
        # model.eta_high = Var(model.i, bounds=(15.0, 60.0), initialize=30.0)
        # model.tau = Var(model.i, bounds=(15.0 / 3600, 60.0 / 3600), initialize=18 / 3600)
        # model.K = Var(model.i, bounds=(5.0, 60.0), initialize=40.0)
        # model.rho_crit = Var(model.i, bounds=(15, 100), initialize=37.45)
        # model.v_free = Var(model.i, bounds=(110, 150), initialize=120.0)
        # model.a = Var(model.i, bounds=(0.5, 5), initialize=1.4)
        model.eta_high = Var(model.i, bounds=(10.0, 90.0), initialize=30.0)
        model.tau = Var(
            model.i, bounds=(10.0 / 3600, 60.0 / 3600), initialize=18 / 3600
        )
        model.K = Var(model.i, bounds=(5.0, 60.0), initialize=40.0)
        model.rho_crit = Var(
            model.i, bounds=(15, 75), initialize=37.45
        )
        model.v_free = Var(model.i, bounds=(70, 150), initialize=120.0)
        model.a = Var(model.i, bounds=(0.5, 5), initialize=1.4)

        ## I-24 Bounds
        # model.eta_high = Var(model.i, bounds=(10.0, 90.0), initialize=30.0)
        # model.tau = Var(
        #     model.i, bounds=(10.0 / 3600, 60.0 / 3600), initialize=18 / 3600
        # )
        # model.K = Var(model.i, bounds=(5, 60.0), initialize=40.0)
        # model.rho_crit = Var(
        #     model.i, bounds=(15, 100), initialize=37.45
        # )
        # model.v_free = Var(model.i, bounds=(70, 150), initialize=120.0)
        # model.a = Var(model.i, bounds=(0.5, 5), initialize=1.4)
    # model.eta_high = Var(model.i, bounds=(1.0, 90.0), initialize=30.0)
    # model.tau = Var(
    #     model.i, bounds=(1.0 / 3600, 60.0 / 3600), initialize=18 / 3600
    # )
    # model.K = Var(model.i, bounds=(1.0, 60.0), initialize=40.0)
    # model.rho_crit = Var(
    #     model.i, bounds=(10, np.max(rho_hat)), initialize=37.45 #np.max(rho_hat)
    # )
    # model.v_free = Var(model.i, bounds=(50, 150), initialize=120.0)
    # model.a = Var(model.i, bounds=(0.01, 10), initialize=1.4)

    # if include_ramping:
    #     model.beta = Var(model.i, bounds=(1e-3, 0.9), initialize=1e-3)
    #     model.r_inflow = Var(model.i, bounds=(1e-3, 2000), initialize=1e-3)

    if include_ramping:    
        if time_varying_ramps:
            model.beta = Var(model.i, bounds=(1e-3, 0.6), initialize=1e-3)
            model.r_inflow = Var(model.t, model.i, bounds=(1e-3, 2000), initialize=1e-3)
            
            if ramp_mapping is not None:
                assert("on_ramps" in ramp_mapping and "off_ramps" in ramp_mapping), "If include_ramping is True, ramp_mapping must contain 'on_ramps' and 'off_ramps' keys."

                on_ramps = ramp_mapping["on_ramps"]
                off_ramps = ramp_mapping["off_ramps"]
                for i in range(num_segments):
                    if not on_ramps[i]:
                        for t in range(num_timesteps):
                            model.r_inflow[t, i].fix(1e-3)

                    if not off_ramps[i]:
                        if tee:
                            print(f"seg {i} is not an off-ramp, fixing beta to 1e-3")
                        model.beta[i].fix(1e-3)

                    if fixed_inflows is not None and i in fixed_inflows:
                        for t in range(num_timesteps):
                            model.r_inflow[t, i].fix(fixed_inflows[i][t])

            if warmstart is not None:
                for i in range(num_segments):
                    for t in range(num_timesteps):
                        model.beta[i].value = params["beta"][i]
                        model.r_inflow[t, i].value = params["r"][t, i]
        else:
            model.beta = Var(model.i, bounds=(1e-3, 0.9), initialize=1e-3)
            model.r_inflow = Var(model.i, bounds=(1e-3, 2000), initialize=1e-3)
            
            if ramp_mapping is not None:
                assert("on_ramps" in ramp_mapping and "off_ramps" in ramp_mapping), "If include_ramping is True, ramp_mapping must contain 'on_ramps' and 'off_ramps' keys."

                on_ramps = ramp_mapping["on_ramps"]
                off_ramps = ramp_mapping["off_ramps"]
                for i in range(num_segments):
                    if not on_ramps[i]:
                        model.r_inflow[i].fix(1e-3)

                    if not off_ramps[i]:
                        if tee:
                            print(f"seg {i} is not an off-ramp, fixing beta to 1e-3")
                        model.beta[i].fix(1e-3)

            if warmstart is not None:
                for i in range(num_segments):
                    for t in range(num_timesteps):
                        model.beta[i].value = params["beta"][i]
                        model.r_inflow[i].value = params["r"][i]

    else:
        model.beta = Var(model.i, bounds=(0.0, 0.0), initialize=0.0)
        model.r_inflow = Var(model.i, bounds=(0.0, 0.0), initialize=0.0)
    # Variables to predict (per-lane values)

    model.v_pred = Var(
        model.t,
        model.i,
        bounds=(1e-3, 150),
        initialize={(t, i): float(v_hat[t, i]) for t in model.t for i in model.i},
    )
    model.rho_pred = Var(
        model.t,
        model.i,
        bounds=(1e-3, 400),
        initialize={(t, i): float(rho_hat[t, i]) for t in model.t for i in model.i},
    )
    # model.q_pred = Var(
    #     model.t,
    #     model.i,
    #     bounds=(1e-3, 10000),
    #     initialize={(t, i): float(q_hat[t, i]) for t in model.t for i in model.i},
    # )

    # Initial conditions
    for i in range(num_segments):
        model.constraints.add(model.v_pred[0, i] == v_hat[0, i].item())
        model.constraints.add(model.rho_pred[0, i] == rho_hat[0, i].item())

    # Observed data
    model.v_hat = Param(
        model.t,
        model.i,
        initialize={(t, i): float(v_hat[t, i]) for t in model.t for i in model.i},
    )
    model.rho_hat = Param(
        model.t,
        model.i,
        initialize={(t, i): float(rho_hat[t, i]) for t in model.t for i in model.i},
    )
    # model.q_hat = Param(
    #     model.t,
    #     model.i,
    #     initialize={(t, i): float(q_hat[t, i]) for t in model.t for i in model.i},
    # )

    # Dynamics functions
    def density_dynamics(current, inflow, outflow, T, l, lanes, beta, r_inflow):
        return current + T / l * (
            inflow - outflow / (1 - beta) + r_inflow
        ) # add r(t)- s(t)

    def calculate_V(m, rho, VSL, seg):
        return m.v_free[seg] * pyo.exp(
            -1 / m.a[seg] * (rho / m.rho_crit[seg]) ** m.a[seg]
        )

    def velocity_dynamics(
        m, current, prev_state, density, next_density, VSL, T, l, seg
    ):
        tau = m.tau[seg]
        eta = m.eta_high[seg]
        K = m.K[seg]
        v_eq = calculate_V(m, density, VSL, seg)
        term1 = T / tau * (v_eq - current)
        term2 = T / l * current * (prev_state - current)
        term3 = (eta * T) / (tau * l) * (next_density - density) / (density + K)
        return current + term1 + term2 - term3

    # Density dynamics
    def rho_update(m, t, i):
        if t == 0:
            return Constraint.Skip
        seg = i
        if i == 0:
            current = m.rho_pred[t - 1, 0]
            inflow = initial_flow_or[t - 1, 0]
            outflow = m.rho_pred[t - 1, i] * m.v_pred[t - 1, i]
        else:
            current = m.rho_pred[t - 1, i]
            inflow = m.rho_pred[t - 1, i - 1] * m.v_pred[t - 1, i - 1]
            outflow = m.rho_pred[t - 1, i] * m.v_pred[t - 1, i]
        if include_ramping:
            return m.rho_pred[t, i] == density_dynamics(
                current,
                inflow,
                outflow,
                model.T,
                model.l,
                model.n_lanes[i],
                model.beta[i],
                model.r_inflow[t, i] if time_varying_ramps else model.r_inflow[i],
            )
        else:
            return m.rho_pred[t, i] == density_dynamics(
                current, inflow, outflow, model.T, model.l, model.n_lanes[i], 0.0, 0.0
            )

    model.rho_dyn = Constraint(model.t, model.i, rule=rho_update)
    # Velocity dynamics
    VSL = 150

    def v_update(m, t, i):
        seg = i
        if t == 0:
            return Constraint.Skip

        current = m.v_pred[t - 1, i]
        prev_state = m.v_pred[t - 1, i]
        density = m.rho_pred[t - 1, i] / m.n_lanes[seg]

        if num_segments == 1:
            # single-segment case
            next_density = downstream_density[t - 1] / m.n_lanes[seg]
        elif i == 0:
            # first segment in a multi-segment block
            next_density = m.rho_pred[t - 1, i + 1] / m.n_lanes[seg + 1]
        elif i == num_segments - 1:
            # last segment in block
            prev_state = m.v_pred[t - 1, i - 1]
            next_density = downstream_density[t - 1] / m.n_lanes[seg]
        else:
            # interior segment
            prev_state = m.v_pred[t - 1, i - 1]
            next_density = m.rho_pred[t - 1, i + 1] / m.n_lanes[seg + 1]

        return m.v_pred[t, i] == velocity_dynamics(
            m, current, prev_state, density, next_density, VSL, m.T, m.l, seg
        )

    model.v_dyn = Constraint(model.t, model.i, rule=v_update)

    if x0 is not None:                                                # <<<
        _num_seg = num_segments                                        # <<<
        for i in range(_num_seg):                                      # <<<
            if "eta_high"  in x0: model.eta_high[i].value  = float(x0["eta_high"][i])   # <<<
            if "tau"       in x0: model.tau[i].value        = float(x0["tau"][i])        # <<<
            if "K"         in x0: model.K[i].value          = float(x0["K"][i])          # <<<
            if "rho_crit"  in x0: model.rho_crit[i].value   = float(x0["rho_crit"][i])  # <<<
            if "v_free"    in x0: model.v_free[i].value     = float(x0["v_free"][i])     # <<<
            if "a"         in x0: model.a[i].value          = float(x0["a"][i])          # <<<
            if include_ramping:                                         # <<<
                if "beta"     in x0: model.beta[i].value     = float(x0["beta"][i])      # <<<
                if "r_inflow" in x0 and not time_varying_ramps:        # <<<
                    model.r_inflow[i].value = float(x0["r_inflow"][i]) # <<<
 

    def compute_A_regularizer_pyomo(m, t):
        """
        Compute the Frobenius norm squared of the parameter-dependent
        blocks (3 and 4) of A_t using Pyomo variables, summed over
        all segments at time step t.

        Block 3: d(v_{i+1})/d(rho)  -- depends on tau, eta_high, K, v_free, rho_crit, a
        Block 4: d(v_{i+1})/d(v)    -- depends on tau
        """
        reg = 0.0
        N = m.num_segments

        for i in m.i:
            # --- segment-specific Pyomo variables ---
            tau_i    = m.tau[i]
            nu_i     = m.eta_high[i]
            kap_i    = m.K[i]
            v_f_i    = m.v_free[i]
            rho_cr_i = m.rho_crit[i]
            alpha_i  = m.a[i]

            # density is per-lane, matching your v_update convention
            rho_i    = m.rho_pred[t, i] / m.n_lanes[i]
            rho_next = (m.rho_pred[t, i + 1] / m.n_lanes[i + 1]
                        if i < N - 1
                        else downstream_density[t] / m.n_lanes[i])

            # V'(rho_i) using Pyomo exp
            dV_i = -(v_f_i / rho_cr_i) \
                    * (rho_i / rho_cr_i) ** (alpha_i - 1) \
                    * pyo.exp(-(1.0 / alpha_i) * (rho_i / rho_cr_i) ** alpha_i)

            # --- Block 3 entries ---
            # self: d(v_{i,t+1})/d(rho_{i,t})
            b3_self = (m.T / tau_i) * dV_i \
                    + (nu_i * m.T / (tau_i * m.l)) \
                    * (rho_next + kap_i) / (rho_i + kap_i) ** 2

            # downstream neighbour: d(v_{i,t+1})/d(rho_{i+1,t})
            b3_down = (-(nu_i * m.T) / (tau_i * m.l * (rho_i + kap_i))
                        if i < N - 1 else 0.0)

            # --- Block 4 entries ---
            v_i    = m.v_pred[t, i]
            v_prev = m.v_pred[t, i - 1] if i > 0 else m.v_pred[t, i]

            # self: d(v_{i,t+1})/d(v_{i,t})
            b4_self = 1.0 - m.T / tau_i + (m.T / m.l) * (v_prev - 2.0 * v_i)

            # upstream neighbour: d(v_{i,t+1})/d(v_{i-1,t})
            b4_up = ((m.T / m.l) * v_i if i > 0 else 0.0)

            reg += b3_self ** 2 + b3_down ** 2 + b4_self ** 2 + b4_up ** 2

        return reg

    # Objective: per-lane error
    def loss_fn(m):
        v_max = max(m.v_hat[t, i] for t in m.t for i in m.i)
        rho_max = max(m.rho_hat[t, i] for t in m.t for i in m.i)
        #q_max = max(m.q_hat[t, i] for t in m.t for i in m.i)
        loss_fn = sum(
            (20 * ((m.v_pred[t, i] - m.v_hat[t, i]) / v_max) ** 2)
            + ((m.rho_pred[t, i] - m.rho_hat[t, i]) / rho_max) ** 2
            #+ ((m.q_pred[t, i] - m.q_hat[t, i]) / q_max) ** 2
            for t in m.t
            for i in m.i
        )
        # Add smoothness constraint to params from prev_param_path
        if prev_param_path is not None:
            prev_params = METANET_Params(path=prev_param_path, num_segments=num_calibrated_segments).get_params()
            for i in m.i:
                loss_fn += 10.0 * ((m.eta_high[i] - prev_params["eta_high"][i]) / 90.0) ** 2
                loss_fn += 10.0 * ((m.tau[i] - prev_params["tau"][i]) / (60.0 / 3600)) ** 2
                loss_fn += 10.0 * ((m.K[i] - prev_params["K"][i]) / 60.0) ** 2
                loss_fn += 10.0 * ((m.rho_crit[i] - prev_params["p_crit"][i]) / 100.0) ** 2
                loss_fn += 10.0 * ((m.v_free[i] - prev_params["v_free"][i]) / 150.0) ** 2
                loss_fn += 10.0 * ((m.a[i] - prev_params["a"][i]) / 5.0) ** 2
                loss_fn += 10.0 * ((m.beta[i] - prev_params["beta"][i]) / 0.9) ** 2
                loss_fn += 10.0 * ((m.r_inflow[i] - prev_params["r"][i]) / 2000.0) ** 2
        
         # Jacobian regularizer: penalize large ||A_t||_F over trajectory
        if use_A_regularizer:
            reg_sum = sum(
                compute_A_regularizer_pyomo(m, t)
                for t in m.t
                if t > 0      # skip t=0 since e_0 = 0
            )
            loss_fn += lambda_reg * reg_sum

        return loss_fn
    
    model.loss = Objective(rule=loss_fn, sense=minimize)

    # Solve
    solver = SolverFactory("ipopt")
    # solver.options["tol"] = 1e-15
    # solver.options["constr_viol_tol"] = 1e-10    # constraint violation tolerance
    # solver.options["acceptable_tol"] = 1e-9      # early stopping criterion
    # solver.options["acceptable_constr_viol_tol"] = 1e-9
    # solver.options["dual_inf_tol"] = 1e-10       # dual infeasibility tolerance
    # solver.options["compl_inf_tol"] = 1e-10       
    solver.options["max_iter"] = 20000
    solver.options['acceptable_constr_viol_tol'] = constraint_tol
    solver.options['constr_viol_tol'] = constraint_tol
    # solver.options['acceptable_tol'] = constraint_tol
    # solver.options['tol'] = constraint_tol
    # solver.options['tol'] = constraint_tol
    # solver.options['acceptable_tol'] = constraint_tol
    # solver.options['nlp_scaling_max_gradient'] = 10

    # solver.options["nlp_scaling_method"] = "none"  # disable IPOPT's internal scaling
    # solver.options["dual_inf_tol"] = 1e-11
    # solver.options["acceptable_dual_inf_tol"] = 1e-11
    #solver.options['nlp_scaling_method'] = 'equilibration-based'  # disable IPOPT's internal scaling
    if warmstart is not None:
        solver.options['warm_start_init_point'] = 'yes'
        solver.options['mu_init'] = 1e-6
        solver.options['warm_start_bound_push'] = 0.001
        solver.options['warm_start_mult_bound_push'] = 0.001
    solver_results = solver.solve(model, tee=tee)

    return model, solver_results

def metanet_param_fit_robust(
    v_hat,
    rho_hat,
    q_hat,
    T,
    l,
    initial_traffic_state,
    downstream_density,
    num_calibrated_segments,
    include_ramping=True,
    varylanes=True,
    lane_mapping=None,
    constraint_tol=1e-12,
    # robust-specific
    S=25,                     # number of scenarios
    bc_noise_percent=10.0,     # passed into generate_perturbations (see note above)
    seed=0,
    objective_mode="minmax",   # "minmax" or "mean_plus_worst"
    lam_worst=0.2,              # only used for "mean_plus_worst"
    warmstart=None,
    ramp_mapping=None,
    prev_param_path=None
):
    """
    Scenario-based robust calibration:
      - shared parameters across scenarios
      - scenario-specific states (rho_pred, v_pred)
      - objective uses rho_pred*v_pred for flow fit (no q_pred anywhere)

    Uncertainty is applied to boundary conditions:
      - upstream inflow (initial_traffic_state)
      - downstream density (downstream_density)
    """
    print("Number of robust scenarios S =", S, "with bc noise percent =", bc_noise_percent)
    # --- reshape boundary conditions to (T, 1) like your current code ---
    initial_flow_or = np.asarray(initial_traffic_state)
    if initial_flow_or.ndim == 1:
        initial_flow_or = initial_flow_or.reshape(-1, 1)

    downstream_density = np.asarray(downstream_density)
    if downstream_density.ndim == 1:
        downstream_density = downstream_density.reshape(-1, 1)

    v_hat = np.asarray(v_hat, dtype=float)
    rho_hat = np.asarray(rho_hat, dtype=float)
    q_hat = np.asarray(q_hat, dtype=float)

    num_timesteps, num_segments = v_hat.shape

    # --- generate scenario perturbations for inflows ---
    inflow_s = generate_perturbations(
        initial_flow_or, percent_noise=bc_noise_percent, seed=seed, num=S
    )

    # down_s = generate_perturbations(
    #     downstream_density, percent_noise=bc_noise_percent, seed=seed + 1, num=S
    # )

    # Optional: clip to keep physical positivity
    # inflow_s = np.clip(inflow_s, 1e-3, None)


    m = ConcreteModel()
    m.s = RangeSet(0, S - 1)
    m.t = RangeSet(0, num_timesteps - 1)
    m.i = RangeSet(0, num_segments - 1)
    m.constraints = ConstraintList()

    # Fixed params
    m.T = Param(initialize=float(T))
    m.l = Param(initialize=float(l))

    # Number of lanes (per segment) — matches your pattern
    if lane_mapping is not None:
        m.n_lanes = Param(m.i, initialize={i: lane_mapping[i] for i in range(num_segments)})
    elif varylanes:
        m.n_lanes = Var(m.i, bounds=(3, 5), initialize=3)
    else:
        m.n_lanes = Param(m.i, initialize=4)

    # Shared parameters to estimate (same as your metanet_param_fit)
    if warmstart is not None:
        params = METANET_Params(path=warmstart, num_segments=num_calibrated_segments).get_params()

        m.eta_high = Var(m.i, bounds=(10.0, 90.0))
        m.tau = Var(m.i, bounds=(10.0 / 3600, 60.0 / 3600))
        m.K = Var(m.i, bounds=(5.0, 60.0))
        m.rho_crit = Var(m.i, bounds=(15, 100))
        m.v_free = Var(m.i, bounds=(70, 150))
        m.a = Var(m.i, bounds=(0.5, 5))

        for i in range(num_segments):
            m.eta_high[i].value = params["eta_high"][i]
            m.tau[i].value = params["tau"][i]
            m.K[i].value = params["K"][i]
            m.rho_crit[i].value = params["p_crit"][i]
            m.v_free[i].value = params["v_free"][i]
            m.a[i].value = params["a"][i]
    
    else:
        m.eta_high = Var(m.i, bounds=(10.0, 90.0), initialize=30.0)
        m.tau = Var(
            m.i, bounds=(10.0 / 3600, 60.0 / 3600), initialize=18 / 3600
        )
        m.K = Var(m.i, bounds=(5.0, 60.0), initialize=40.0)
        m.rho_crit = Var(
            m.i, bounds=(15, 100), initialize=37.45
        )
        m.v_free = Var(m.i, bounds=(70, 150), initialize=120.0)
        m.a = Var(m.i, bounds=(0.5, 5), initialize=1.4)

    # m.eta_high = Var(m.i, bounds=(10.0, 90.0), initialize=30.0)
    # m.tau = Var(m.i, bounds=(1.0 / 3600, 60.0 / 3600), initialize=18 / 3600)
    # m.K = Var(m.i, bounds=(5.0, 60.0), initialize=40.0)
    # m.rho_crit = Var(m.i, bounds=(15, float(np.max(rho_hat))), initialize=37.45)
    # m.v_free = Var(m.i, bounds=(70, 150), initialize=120.0)
    # m.a = Var(m.i, bounds=(0.5, 5), initialize=1.4)

    if include_ramping:
        assert("on_ramps" in ramp_mapping and "off_ramps" in ramp_mapping), "If include_ramping is True, ramp_mapping must contain 'on_ramps' and 'off_ramps' keys."

        if ramp_mapping is not None:
            on_ramps = ramp_mapping["on_ramps"]
            off_ramps = ramp_mapping["off_ramps"]
            for i in range(num_segments):
                if on_ramps[i]:
                    m.r_inflow[i] = Var(bounds=(0, 2000), initialize=1e-3)
                else:
                    m.r_inflow[i] = Param(bounds=(0.0, 0.0), initialize=0.0)

                if off_ramps[i]:
                    m.beta[i] = Var(bounds=(1e-3, 0.9), initialize=1e-3)
                else:
                    m.beta[i] = Param(bounds=(1e-3, 1e-3), initialize=1e-3)
        else:
            # Assume there is an on and off ramp at every segment
            m.beta = Var(m.i, bounds=(1e-3, 0.9), initialize=1e-3)
            m.r_inflow = Var(m.i, bounds=(0, 2000), initialize=1e-3)

        if warmstart is not None:
            for i in range(num_segments):
                m.beta[i].value = params["beta"][i]
                m.r_inflow[i].value = params["r"][i]
    else:
        m.beta = Var(m.i, bounds=(0.0, 0.0), initialize=0.0)
        m.r_inflow = Var(m.i, bounds=(0.0, 0.0), initialize=0.0)

    # Scenario-specific predicted states
    # if warmstart is not None:
    #     v_init = np.load(warmstart + "/v_pred.npy")
    #     p_init = np.load(warmstart + "/rho_pred.npy")

    #     print(v_init.shape, p_init.shape)

    #     # v_init = np.zeros((num_timesteps, num_segments, S))
    #     # p_init = np.zeros((num_timesteps, num_segments, S))

    #     # # Simulate v and rho for each scenario using warmstart params

    #     # num_lanes = np.load(f"{warmstart}/num_lanes.npy")

    #     # for s in range(S):
    #     #     init_traffic_state = (rho_hat[0, :] / num_lanes, v_hat[0, :], inflow_s[s, 0, 0], 0)

    #     #     rho_sim, v_sim, _, tts_sim = run_metanet_sim(
    #     #         T, 
    #     #         l, 
    #     #         init_traffic_state,
    #     #         inflow_s[s, :, 0],
    #     #         downstream_density,
    #     #         params,
    #     #         vsl_speeds=None,
    #     #         lanes={i: num_lanes[i] for i in range(num_calibrated_segments)},
    #     #         plotting=True,
    #     #         real_data=True
    #     #     )

    #     #     v_init[:, :, s] = v_sim[0:-1, :].copy()
    #     #     p_init[:, :, s] = rho_sim[0:-1, :].copy() * num_lanes[np.newaxis, :]

    # else:
    # v_init = v_hat.copy()
    # p_init = rho_hat.copy()
    
    # q_init = p_init * v_init

    m.v_pred = Var(m.s, m.t, m.i, bounds=(1e-3, 150),
                   initialize={(s, t, i): float(v_hat[t, i]) for s in m.s for t in m.t for i in m.i})
    m.rho_pred = Var(m.s, m.t, m.i, bounds=(1e-3, 400),
                     initialize={(s, t, i): float(rho_hat[t, i]) for s in m.s for t in m.t for i in m.i})
    # m.q_pred = Var(m.s, m.t, m.i, bounds=(1e-3, 10000),
    #               initialize={(s, t, i): float(q_hat[t, i]) for s in m.s for t in m.t for i in m.i})

    # Observed data params (shared across scenarios)
    m.v_hat = Param(m.t, m.i, initialize={(t, i): float(v_hat[t, i]) for t in range(num_timesteps) for i in range(num_segments)})
    m.rho_hat = Param(m.t, m.i, initialize={(t, i): float(rho_hat[t, i]) for t in range(num_timesteps) for i in range(num_segments)})
    #m.q_hat = Param(m.t, m.i, initialize={(t, i): float(q_hat[t, i]) for t in range(num_timesteps) for i in range(num_segments)})

    # Boundary conditions (scenario-specific)
    m.initial_flow = Param(m.s, m.t, initialize={(s, t): float(inflow_s[s, t, 0]) for s in range(S) for t in range(num_timesteps)})
    #m.downstream_density = Param(m.s, m.t, initialize={(s, t): float(down_s[s, t, 0]) for s in range(S) for t in range(num_timesteps)})

    # --- dynamics functions (same form as your code) ---
    def density_dynamics(current, inflow, outflow, T_, l_, lanes_, beta_, r_inflow_):
        # lanes_ is unused here, but kept to mirror your signature
        return current + T_ / l_ * (inflow - outflow / (1 - beta_) + r_inflow_)

    def calculate_V(mm, rho_per_lane, VSL, seg):
        return mm.v_free[seg] * pyo.exp(-1 / mm.a[seg] * (rho_per_lane / mm.rho_crit[seg]) ** mm.a[seg])

    def velocity_dynamics(mm, current, prev_state, density, next_density, VSL, T_, l_, seg):
        tau = mm.tau[seg]
        eta = mm.eta_high[seg]
        K = mm.K[seg]
        v_eq = calculate_V(mm, density, VSL, seg)
        term1 = T_ / tau * (v_eq - current)
        term2 = T_ / l_ * current * (prev_state - current)
        term3 = (eta * T_) / (tau * l_) * (next_density - density) / (density + K)
        return current + term1 + term2 - term3

    # --- initial conditions (for every scenario) ---
    def init_v_rule(mm, s, i):
        return mm.v_pred[s, 0, i] == mm.v_hat[0, i]
    m.init_v = Constraint(m.s, m.i, rule=init_v_rule)

    def init_rho_rule(mm, s, i):
        return mm.rho_pred[s, 0, i] == mm.rho_hat[0, i]
    m.init_rho = Constraint(m.s, m.i, rule=init_rho_rule)

    # --- density dynamics ---
    def rho_update(mm, s, t, i):
        if t == 0:
            return Constraint.Skip

        if i == 0:
            current = mm.rho_pred[s, t - 1, 0]
            inflow = mm.initial_flow[s, t - 1]
            outflow = mm.rho_pred[s, t - 1, i] * mm.v_pred[s, t - 1, i]
        else:
            current = mm.rho_pred[s, t - 1, i]
            inflow = mm.rho_pred[s, t - 1, i - 1] * mm.v_pred[s, t - 1, i - 1]
            outflow = mm.rho_pred[s, t - 1, i] * mm.v_pred[s, t - 1, i]

        beta_ = mm.beta[i] if include_ramping else 0.0
        r_in_ = mm.r_inflow[i] if include_ramping else 0.0

        return mm.rho_pred[s, t, i] == density_dynamics(
            current, inflow, outflow, mm.T, mm.l, mm.n_lanes[i], beta_, r_in_
        )

    m.rho_dyn = Constraint(m.s, m.t, m.i, rule=rho_update)

    # --- velocity dynamics ---
    VSL = 150

    def v_update(mm, s, t, i):
        if t == 0:
            return Constraint.Skip

        seg = i
        current = mm.v_pred[s, t - 1, i]
        prev_state = mm.v_pred[s, t - 1, i]
        density = mm.rho_pred[s, t - 1, i] / mm.n_lanes[seg]

        if num_segments == 1:
            next_density = downstream_density[t - 1] / mm.n_lanes[seg]
        elif i == 0:
            next_density = mm.rho_pred[s, t - 1, i + 1] / mm.n_lanes[seg + 1]
        elif i == num_segments - 1:
            prev_state = mm.v_pred[s, t - 1, i - 1]
            next_density = downstream_density[t - 1] / mm.n_lanes[seg]
        else:
            prev_state = mm.v_pred[s, t - 1, i - 1]
            next_density = mm.rho_pred[s, t - 1, i + 1] / mm.n_lanes[seg + 1]

        return mm.v_pred[s, t, i] == velocity_dynamics(
            mm, current, prev_state, density, next_density, VSL, mm.T, mm.l, seg
        )

    m.v_dyn = Constraint(m.s, m.t, m.i, rule=v_update)

    # --- robust objective ---
    v_max = max(m.v_hat[t, i] for t in m.t for i in m.i)
    rho_max = max(m.rho_hat[t, i] for t in m.t for i in m.i)
    # q_max = max(m.q_hat[t, i] for t in m.t for i in m.i)

    # scenario loss expression (uses rho*v directly; no q_pred)
    def scenario_loss(mm, s):
        loss_fn = sum(
            (20 * ((mm.v_pred[s, t, i] - mm.v_hat[t, i]) / v_max) ** 2)
            + ((mm.rho_pred[s, t, i] - mm.rho_hat[t, i]) / rho_max) ** 2
            #+ ((mm.q_pred[s, t, i] - mm.q_hat[t, i]) / q_max) ** 2
            for t in mm.t
            for i in mm.i
        )
        # Add smoothness constraint to params from prev_param_path
        if prev_param_path is not None:
            prev_params = METANET_Params(path=prev_param_path, num_segments=num_calibrated_segments).get_params()
            for i in mm.i:
                loss_fn += 10.0 * ((mm.eta_high[i] - prev_params["eta_high"][i]) / 90.0) ** 2
                loss_fn += 10.0 * ((mm.tau[i] - prev_params["tau"][i]) / (60.0 / 3600)) ** 2
                loss_fn += 10.0 * ((mm.K[i] - prev_params["K"][i]) / 60.0) ** 2
                loss_fn += 10.0 * ((mm.rho_crit[i] - prev_params["p_crit"][i]) / 100.0) ** 2
                loss_fn += 10.0 * ((mm.v_free[i] - prev_params["v_free"][i]) / 150.0) ** 2
                loss_fn += 10.0 * ((mm.a[i] - prev_params["a"][i]) / 5.0) ** 2
                loss_fn += 10.0 * ((mm.beta[i] - prev_params["beta"][i]) / 0.9) ** 2
                loss_fn += 10.0 * ((mm.r_inflow[i] - prev_params["r"][i]) / 2000.0) ** 2
                
        return loss_fn

    m.L = pyo.Expression(m.s, rule=scenario_loss)

    # epigraph for worst-case loss
    if objective_mode == "mean":
        m.obj = Objective(
            expr=(sum(m.L[s] for s in m.s) / S),
            sense=minimize
        )
    elif objective_mode in {"minmax", "mean_plus_worst"}:
        m.z = Var(bounds=(0, None), initialize=0.0)

        def z_ge_loss(mm, s):
            return mm.z >= mm.L[s]
        m.z_con = Constraint(m.s, rule=z_ge_loss)

        if objective_mode == "minmax":
            m.obj = Objective(expr=m.z, sense=minimize)
        elif objective_mode == "mean_plus_worst":
            m.obj = Objective(
                expr=(1.0 - lam_worst) * (sum(m.L[s] for s in m.s) / S) + lam_worst * m.z,
                sense=minimize
            )
    else:
        raise ValueError("objective_mode must be 'minmax' or 'mean_plus_worst' or 'mean'")

    # Solve
    solver = pyo.SolverFactory("ipopt")
    solver.options["max_iter"] = 20000
    solver.options["acceptable_constr_viol_tol"] = constraint_tol
    solver.options["constr_viol_tol"] = constraint_tol

    if warmstart is not None:
        solver.options['warm_start_init_point'] = 'yes'
        solver.options['mu_init'] = 1e-3
        # solver.options['warm_start_bound_push'] = 0.001
        # solver.options['warm_start_mult_bound_push'] = 0.001
    # solver.options['hessian_approximation'] = 'limited-memory'

    solver.solve(m, tee=True)

    return m


def run_calibration(
    rho_hat,
    q_hat,
    T,
    l,
    num_calibrated_segments=1,
    sep_boundary_conditions=None,
    include_ramping=True,
    varylanes=True,
    lane_mapping=None,
    ramp_mapping=None,
    smoothing=True,
    constraint_tol=1e-12,
    robust_opt=None,
    warmstart=None,
    prev_param_path=None,
    time_varying_ramps=False,
    fixed_inflows=None,
    use_A_regularizer=False,
    lambda_reg=1,
    x0=None,
    tee=True
):
    """
    Run METANET parameter calibration with configurable segment grouping.

    Parameters
    ----------
    rho_hat : np.ndarray
        Density measurements (time, segment).
    q_hat : np.ndarray
        Flow measurements (time, segment).
    T : float
        Time step (hours).
    l : float
        Segment length (km).
    num_calibrated_segments : int
        Number of consecutive segments to calibrate at a time.

    Returns
    -------
    results : dict
        Dictionary with concatenated predictions and parameter arrays.
    """

    def calculate_V(m, rho, VSL, seg):
        return m.v_free[seg] * pyo.exp(
            -1 / m.a[seg] * (rho / m.rho_crit[seg]) ** m.a[seg]
        )

    def velocity_dynamics(
        m, current, prev_state, density, next_density, VSL, T, l, seg
    ):
        tau = m.tau[seg]
        eta = m.eta_high[seg]
        K = m.K[seg]
        v_eq = calculate_V(m, density, VSL, seg)
        term1 = T / tau * (v_eq - current)
        term2 = T / l * current * (prev_state - current)
        term3 = (eta * T) / (tau * l) * (next_density - density) / (density + K)
        return current + term1 + term2 - term3

    # Ensure no divide-by-zero
    # rho_hat = np.where(rho_hat == 0.0, 1e-3, rho_hat)
    # q_hat = np.where(q_hat == 0.0, 1e-3, q_hat)
    v_hat = q_hat / rho_hat
    # v_hat = np.where(v_hat == 0.0, 1e-3, v_hat)

    # Initialize results storage
    results = {
        "v_pred": [],
        "rho_pred": [],
        "tau": [],
        "K": [],
        "eta_high": [],
        "rho_crit": [],
        "v_free": [],
        "a": [],
        "num_lanes": [],
    }
        # results["gamma"] = []
    results["beta"] = []
    results["r_inflow"] = []
        

    total_segments = rho_hat.shape[1] - 2 if sep_boundary_conditions is None else rho_hat.shape[1] # exclude first + last for boundary cond
    # Loop through groups of segments
    start_segment = 1 if sep_boundary_conditions is None else 0
    end_segment = start_segment + total_segments

    assert (total_segments == num_calibrated_segments)

    # Boundary conditions depend on group position
    if sep_boundary_conditions is not None:
        initial_flow = sep_boundary_conditions["initial_flow"]
        downstream_density = sep_boundary_conditions["downstream_density"]
    else:
        initial_flow = q_hat[:, 0]  # upstream inflow
        downstream_density = rho_hat[:, -1]

    if smoothing:
        initial_flow = smooth_inflow(initial_flow)  # upstream inflow
        downstream_density = smooth_inflow(downstream_density)  # downstream density

    # print("--------Calib boundary conditions--------")
    # print("Initial flow:", initial_flow[0:10])
    # print("Downstream density:", downstream_density[0:10])
    # print("Initial velocity:", segment_v_hat[0, :])
    # print("Initial density:", segment_rho_hat[0, :])
    # print("Initial flow:", segment_q_hat[0, :])
    # Run calibration for this block

    # print(initial_flow[0:10])
    # print(downstream_density[0:10]/4)
    # print(v_hat[0,:])
    # print(rho_hat[0,:])

    segment_v_hat = v_hat[:, start_segment:end_segment]
    segment_rho_hat = rho_hat[:, start_segment:end_segment]
    segment_q_hat = q_hat[:, start_segment:end_segment]
    
    if warmstart is not None:
        print("Using warmstart from:", warmstart)

    if robust_opt is None:
        res_model, _solver_res = metanet_param_fit(
            segment_v_hat,
            segment_rho_hat,
            segment_q_hat,
            T,
            l,
            initial_flow,
            downstream_density,
            num_calibrated_segments,
            include_ramping=include_ramping,
            varylanes=varylanes,
            lane_mapping=lane_mapping,
            constraint_tol=constraint_tol,
            prev_param_path=prev_param_path,
            ramp_mapping=ramp_mapping,
            time_varying_ramps=time_varying_ramps,
            fixed_inflows=fixed_inflows,
            use_A_regularizer=use_A_regularizer,
            lambda_reg=lambda_reg, x0=x0,
            tee=tee
        )
    else:
        assert isinstance(robust_opt, RobustOptConfig)
        assert robust_opt.objective_mode in ["minmax", "mean_plus_worst", "mean"]

        res_model = metanet_param_fit_robust(
            segment_v_hat,
            segment_rho_hat,
            segment_q_hat,
            T, 
            l,
            initial_flow,
            downstream_density,
            num_calibrated_segments,
            include_ramping=include_ramping,
            varylanes=varylanes,
            lane_mapping=lane_mapping,
            constraint_tol=constraint_tol,
            S=robust_opt.S,
            bc_noise_percent=robust_opt.bc_noise_percent,
            seed=robust_opt.seed,
            objective_mode=robust_opt.objective_mode,
            lam_worst=robust_opt.lam_worst,
            warmstart=warmstart,
            prev_param_path=prev_param_path,
            ramp_mapping=ramp_mapping
        )

        _solver_res = None  # not used downstream for now, but could be returned if desired

    from pyomo.opt import SolverStatus, TerminationCondition            # <<<
    if _solver_res is not None:                                         # <<<
        _tc = _solver_res.solver.termination_condition                  # <<<
        _ss = _solver_res.solver.status                                 # <<<
    else:                                                               # <<<
        _tc = TerminationCondition.unknown                              # <<<
        _ss = SolverStatus.unknown  

    num_timesteps, num_segments = segment_v_hat.shape

    v_pred_array = np.zeros((num_timesteps, num_segments))
    rho_pred_array = np.zeros((num_timesteps, num_segments))

    for t in range(num_timesteps):
        for i in range(num_segments):
            if robust_opt is not None:
                # For robust case, take average over scenarios
                v_pred_array[t, i] = value(res_model.v_pred[0, t, i])
                rho_pred_array[t, i] = value(res_model.rho_pred[0, t, i])
            else:
                v_pred_array[t, i] = value(res_model.v_pred[t, i])
                rho_pred_array[t, i] = value(res_model.rho_pred[t, i])
    
    # Append predictions
    if len(results["v_pred"]) == 0:
        results["v_pred"] = v_pred_array
        results["rho_pred"] = rho_pred_array
    else:
        results["v_pred"] = np.concatenate(
            [results["v_pred"], v_pred_array], axis=1
        )
        results["rho_pred"] = np.concatenate(
            [results["rho_pred"], rho_pred_array], axis=1
        )

    # Append parameter arrays
    results["tau"].extend([value(res_model.tau[i]) for i in range(num_segments)])
    results["K"].extend([value(res_model.K[i]) for i in range(num_segments)])
    results["eta_high"].extend(
        [value(res_model.eta_high[i]) for i in range(num_segments)]
    )
    results["rho_crit"].extend(
        [value(res_model.rho_crit[i]) for i in range(num_segments)]
    )
    results["v_free"].extend(
        [value(res_model.v_free[i]) for i in range(num_segments)]
    )
    results["a"].extend([value(res_model.a[i]) for i in range(num_segments)])
    results["num_lanes"].extend(
        [value(res_model.n_lanes[i]) for i in range(num_segments)]
    )
    # if include_ramping:
        # results["gamma"].extend([value(res_model.gamma[i]) for i in range(num_segments)])
    results["beta"].extend(
        [value(res_model.beta[i]) for i in range(num_segments)]
    )
    if time_varying_ramps:
        results["r_inflow"].extend(
            [[value(res_model.r_inflow[t, i]) for i in range(num_segments)] for t in range(num_timesteps)]
        )
    else:
        results["r_inflow"].extend(
            [value(res_model.r_inflow[i]) for i in range(num_segments)]
        )

    # Convert parameter lists to numpy arrays
    for key in ["tau", "K", "eta_high", "rho_crit", "v_free", "a", "num_lanes"]:
        results[key] = np.array(results[key])
    # if include_ramping:
        # results["gamma"] = np.array(results["gamma"])
    results["beta"] = np.array(results["beta"])
    results["r_inflow"] = np.array(results["r_inflow"])

    results["obj_val"] = float(value(res_model.loss))              # <<<
    results["solver_status"] = str(_ss)                                # <<<
    results["termination_condition"] = str(_tc)                        # <<<
    return results



