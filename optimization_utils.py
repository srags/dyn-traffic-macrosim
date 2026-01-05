import numpy as np
import matplotlib.pyplot as plt
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
import pyomo.environ as pyo


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
    enforce_ramps=False
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
    model.t_loss = RangeSet(0, num_timesteps - 1, 10)
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
        print(lane_mapping)
        model.n_lanes = Param(model.i, initialize={i: lane_mapping[i] for i in model.i})
    elif varylanes:
        model.n_lanes = Var(model.i, bounds=(3, 5), initialize=3)
    else:
        model.n_lanes = Param(model.i, initialize=4)

    # Parameters to estimate
    model.eta_high = Var(model.i, bounds=(10.0, 90.0), initialize=30.0)
    model.tau = Var(
        model.i, bounds=(1.0 / 3600, 60.0 / 3600), initialize=18 / 3600
    )
    model.K = Var(model.i, bounds=(5.0, 60.0), initialize=40.0)
    model.rho_crit = Var(
        model.i, bounds=(15, np.max(rho_hat)), initialize=37.45
    )
    model.v_free = Var(model.i, bounds=(70, 150), initialize=120.0)
    model.a = Var(model.i, bounds=(0.5, 5), initialize=1.4)

    if include_ramping:
        # model.gamma = Var(model.i, bounds=(0.5, 1.5), initialize=1)
        model.beta = Var(model.i, bounds=(1e-3, 0.9), initialize=1e-3)
        model.r_inflow = Var(model.i, bounds=(1e-3, 2000), initialize=1e-3)
        if enforce_ramps:
            for r_ind in [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]:
                model.constraints.add(model.r_inflow[r_ind] == 0.0)
            for b_ind in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13]:
                model.constraints.add(model.beta[b_ind] == 0.0)

        # model.beta = Var(model.i, bounds=(0.0, 0.0), initialize=0.0)
        # model.r_inflow = Var(model.i, bounds=(-2000, 2000), initialize=200)
        # model.beta = Var(model.i, bounds=(-1.0, 1.0), initialize=0.5)
        # model.r_inflow = Var(model.i, bounds=(0.0, 0.0), initialize=0.0)
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
    model.q_pred = Var(
        model.t,
        model.i,
        bounds=(1e-3, 10000),
        initialize={(t, i): float(q_hat[t, i]) for t in model.t for i in model.i},
    )

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
    model.q_hat = Param(
        model.t,
        model.i,
        initialize={(t, i): float(q_hat[t, i]) for t in model.t for i in model.i},
    )

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
                model.r_inflow[i],
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

    # Objective: per-lane error
    def loss_fn(m):
        v_max = max(m.v_hat[t, i] for t in m.t for i in m.i)
        rho_max = max(m.rho_hat[t, i] for t in m.t for i in m.i)
        q_max = max(m.q_hat[t, i] for t in m.t for i in m.i)

        return sum(
            (20 * ((m.v_pred[t, i] - m.v_hat[t, i]) / v_max) ** 2)
            + ((m.rho_pred[t, i] - m.rho_hat[t, i]) / rho_max) ** 2
            + ((m.q_pred[t, i] - m.q_hat[t, i]) / q_max) ** 2
            for t in m.t
            for i in m.i
        )

    # def loss_fn(m):
    #     # --- Precompute max values for normalization ---
    #     v_max = max(m.v_hat[t, i] for t in m.t for i in m.i)
    #     rho_max = max(m.rho_hat[t, i] for t in m.t for i in m.i)
    #     q_max = max(m.q_hat[t, i] for t in m.t for i in m.i)

    #     # --- Precompute inflow_hat (fixed, based on data) ---
    #     inflow_hat = np.zeros((num_timesteps, num_segments))
    #     inflow_hat[:, 0] = q_hat[:, 0] - initial_flow_or[:, 0]
    #     inflow_hat[:, 1:] = q_hat[:, 1:] - q_hat[:, :-1]

    #     # --- Construct inflow_pred expression using Pyomo vars ---
    #     inflow_pred_expr = {}
    #     for t in m.t:
    #         inflow_pred_expr[t, 0] = m.q_pred[t, 0] - initial_flow_or[t, 0]
    #         for i in m.i:
    #             if i > 0:
    #                 inflow_pred_expr[t, i] = m.q_pred[t, i] - m.q_pred[t, i - 1]

    #     # --- Loss expression ---
    #     return sum(
    #         20 * ((m.v_pred[t, i] - m.v_hat[t, i]) / v_max) ** 2
    #         + ((m.rho_pred[t, i] - m.rho_hat[t, i]) / rho_max) ** 2
    #         + ((inflow_pred_expr[t, i] - inflow_hat[t, i]) / q_max) ** 2
    #         for t in m.t
    #         for i in m.i
    #     )

    # def loss_fn(m):
    #     return sum(
    #         (m.v_pred[t, i] - m.v_hat[t, i])**2 #+ (m.rho_pred[t, i] - m.rho_hat[t, i])**2 + (m.q_pred[t, i] - m.q_hat[t, i])**2
    #         for t in m.t
    #         for i in m.i
    #     )
    # def loss_fn(m):
    #     selected_segments = {3, 6, 9}  # segments to include
    #     return sum(
    #         (m.v_pred[t, i] - m.v_hat[t, i])**2 #+ (m.rho_pred[t, i] - m.rho_hat[t, i])**2 + (m.q_pred[t, i] - m.q_hat[t, i])**2
    #         for t in m.t
    #         for i in m.i
    #         if i in selected_segments
    #     )

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
    # solver.options["dual_inf_tol"] = 1e-11
    # solver.options["acceptable_dual_inf_tol"] = 1e-11
    solver.solve(model, tee=False)

    return model


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
    smoothing=True,
    constraint_tol=1e-12,
    enforce_ramps=False
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

    res_model = metanet_param_fit(
        v_hat,
        rho_hat,
        q_hat,
        T,
        l,
        initial_flow,
        downstream_density,
        num_calibrated_segments,
        include_ramping=include_ramping,
        varylanes=varylanes,
        lane_mapping=lane_mapping,
        constraint_tol=constraint_tol,
        enforce_ramps=enforce_ramps
    )

    num_timesteps, num_segments = v_hat.shape

    v_pred_array = np.zeros((num_timesteps, num_segments))
    rho_pred_array = np.zeros((num_timesteps, num_segments))

    for t in range(num_timesteps):
        for i in range(num_segments):
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
    return results


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
