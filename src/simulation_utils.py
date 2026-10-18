
import numpy as np
from typing import Dict, Tuple, Optional

# ----------------------------
# Core METANET helper functions
# ----------------------------

def density_dynamics_raw(current: float, inflow: float, outflow: float, lanes: int,
                         T: float, l: float, beta: float = 0.0,
                         r: float = 0.0) -> float:
    """Return the unclipped METANET density update for one segment.

    Keeping the raw update separate lets stress tests detect that the numerical
    dynamics attempted to produce a negative or non-finite density. Normal
    simulation callers still receive the clipped value from
    :func:`density_dynamics`.
    """
    return current + T / (l * lanes) * (inflow - outflow / (1 - beta) + r)


def density_dynamics(current: float, inflow: float, outflow: float, lanes: int, T: float, l: float,
                    beta: float = 0.0, r: float = 0.0) -> float:
    """Update density with the existing non-negative numerical safeguard."""
    raw_density = density_dynamics_raw(current, inflow, outflow, lanes, T, l, beta, r)
    return max(0, raw_density)


def flow_dynamics(density: float, velocity: float, lanes: int) -> float:
    """Fundamental relation: q = rho * v * lanes."""
    return density * velocity * lanes


def queue_dynamics(current: float, demand: float, flow_origin: float, T: float) -> float:
    return current + T * (demand - flow_origin)


def calculate_V(rho: float, v_ctrl: float, a: float, p_crit: float, v_free: float = 150.0) -> float:
    """Desired speed function V(rho), capped by control speed v_ctrl."""
    # prevent overflow warnings
    # p_crit += 1e-4
    # a += 1e-4
    # assert - (rho / p_crit) ** a / a < 700, f"Overflow in desired speed calculation, pow too large: { - (rho / p_crit) ** a / a}, rho={rho}, p_crit={p_crit}, a={a}"
    # if rho <= 0:
    #     print("Warning: non-positive density in calculate_V:", rho)
    uncontrolled_speed = v_free * np.exp(- (rho / p_crit) ** a / a)
    return min(uncontrolled_speed, v_ctrl)


def calculate_V_arr(rho_arr: np.ndarray, v_ctrl_arr: np.ndarray, a: float, p_crit: float, v_free: float) -> np.ndarray:
    """Vectorized desired speed function. Not used in sim, but useful for plotting."""
    return np.minimum(v_free * np.exp(- (rho_arr / p_crit) ** a / a), v_ctrl_arr)


def velocity_dynamics_raw_MN(current: float,
                             prev_state: float,
                             density: float,
                             next_density: float,
                             v_ctrl: float,
                             T: float,
                             l: float,
                             eta_high: float = 30.0,
                             K: float = 40.0,
                             tau: float = 18 / 3600,
                             a: float = 1.4,
                             p_crit: float = 37.45,
                             v_free: float = 120.0) -> float:
    """Return the unclipped one-step METANET velocity update."""
    return (
        current
        + T / tau * (calculate_V(density, v_ctrl, a, p_crit, v_free) - current)
        + T / l * current * (prev_state - current)
        - (eta_high * T) / (tau * l) * (next_density - density) / (density + K)
    )


def velocity_dynamics_MN(current: float,
                         prev_state: float,
                         density: float,
                         next_density: float,
                         v_ctrl: float,
                         T: float,
                         l: float,
                         eta_high: float = 30.0,
                         K: float = 40.0,
                         tau: float = 18 / 3600,
                         a: float = 1.4,
                         p_crit: float = 37.45,
                         v_free: float = 120.0) -> float:
    """One-step METANET velocity update with its existing positive floor."""
    raw_velocity = velocity_dynamics_raw_MN(
        current,
        prev_state,
        density,
        next_density,
        v_ctrl,
        T,
        l,
        eta_high=eta_high,
        K=K,
        tau=tau,
        a=a,
        p_crit=p_crit,
        v_free=v_free,
    )
    return max(1e-4, raw_velocity)


def origin_flow_dynamics_MN(demand: float,
                            density_first: float,
                            queue: float,
                            lanes: int,
                            T: float,
                            p_max: float = 180.0,
                            p_crit: float = 37.45,
                            q_capacity: float = 2200.0) -> float:
    """Origin (on-ramp) sending/merging flow constraint."""
    return min(
        demand + queue / T,
        lanes * q_capacity * (p_max - density_first) / (p_max - p_crit),
        lanes * q_capacity,
    )

def _get_time_space_param(param, t: int, i: int):
    """Helper to index params that may be scalar, 1D (over i), or 2D (over t,i)."""
    if np.ndim(param) == 0:
        return float(param)
    if np.ndim(param) == 1:
        return float(param[i])
    # assume 2D
    return float(param[t, i])


def metanet_step(t: int,
                 density_t: np.ndarray,
                 velocity_t: np.ndarray,
                 queue_t: float,
                 flow_origin_t: float,
                 *,
                 T: float,
                 l: float,
                 vsl_speeds: np.ndarray,
                 demand: np.ndarray,
                 downstream_density: np.ndarray,
                 params: Dict[str, np.ndarray],
                 lanes: Dict[int, int],
                 real_data: bool = False,
                 return_diagnostics: bool = False,
                 ) -> tuple:
    """Compute a single simulation step (t -> t+1) for METANET.

    Args:
        t: current time index.
        density_t, velocity_t: arrays of shape (num_segments,).
        queue_t, flow_origin_t: scalars at time t.
        T, l: discretization time and segment length.
        vsl_speeds: (time_steps, num_segments) control speeds.
        demand: (time_steps,) exogenous demand at origin.
        downstream_density: (time_steps,) boundary density at downstream end.
        params: dict with keys 'beta','r','gamma','eta_high','K','tau','a','p_crit','v_free','q_capacity'.
        lanes: dict mapping segment index -> number of lanes.
        real_data: if True, use demand directly for first cell inflow; otherwise use origin flow.
        upstream_velocity: optional (time_steps,) boundary velocity for the first cell.
        return_diagnostics: if True, append raw pre-clipping values and clipping
            flags to the normal return tuple. The default preserves the original
            five-value API.

    Returns:
        density_tp1, velocity_tp1, queue_tp1, flow_origin_tp1, flow_tp1

    Notes:
        - flow_tp1 is per-segment flow at t+1 (shape (num_segments,)).
    """
    num_segments = density_t.shape[0]
    density_tp1 = np.empty_like(density_t, dtype=float)
    raw_density_tp1 = np.empty_like(density_t, dtype=float)

    # --- density update ---
    for i in range(num_segments):
        beta = _get_time_space_param(params["beta"], t if np.ndim(params["beta"]) == 2 else 0, i)
        r = _get_time_space_param(params["r"], t if np.ndim(params["r"]) == 2 else 0, i)
        # gamma = _get_time_space_param(params["gamma"], 0, i)

        if i == 0:
            inflow = demand[t] if real_data else flow_origin_t
            outflow = density_t[i] * velocity_t[i] * lanes[i]
            raw_density_tp1[i] = density_dynamics_raw(
                density_t[i], inflow, outflow, lanes[i], T, l, beta=beta, r=r
            )
        else:
            inflow = density_t[i - 1] * velocity_t[i - 1] * lanes[i - 1]
            outflow = density_t[i] * velocity_t[i] * lanes[i]
            raw_density_tp1[i] = density_dynamics_raw(
                density_t[i], inflow, outflow, lanes[i], T, l, beta=beta, r=r
            )
        density_tp1[i] = max(0, raw_density_tp1[i])

    # --- velocity update ---
    velocity_tp1 = np.empty_like(velocity_t, dtype=float)
    raw_velocity_tp1 = np.empty_like(velocity_t, dtype=float)
    for i in range(num_segments):
        kwargs = dict(
            eta_high=_get_time_space_param(params["eta_high"], t, i),
            K=_get_time_space_param(params["K"], t, i),
            tau=_get_time_space_param(params["tau"], t, i),
            a=_get_time_space_param(params["a"], t, i),
            p_crit=_get_time_space_param(params["p_crit"], t, i),
            v_free=_get_time_space_param(params["v_free"], t, i),
        )
        if i == 0:
            prev_vel = velocity_t[i]
            next_dens = density_t[i + 1] if num_segments > 1 else downstream_density[t]
            raw_velocity_tp1[i] = velocity_dynamics_raw_MN(
                velocity_t[i], prev_vel, density_t[i], next_dens, vsl_speeds[t, i], T, l, **kwargs
            )
        elif i == num_segments - 1:
            raw_velocity_tp1[i] = velocity_dynamics_raw_MN(
                velocity_t[i], velocity_t[i - 1], density_t[i], downstream_density[t], vsl_speeds[t, i], T, l, **kwargs
            )
        else:
            raw_velocity_tp1[i] = velocity_dynamics_raw_MN(
                velocity_t[i], velocity_t[i - 1], density_t[i], density_t[i + 1], vsl_speeds[t, i], T, l, **kwargs
            )
        velocity_tp1[i] = max(1e-4, raw_velocity_tp1[i])

    # --- queue update (only when not using real data) ---
    if not real_data:
        queue_tp1 = queue_dynamics(queue_t, demand[t], flow_origin_t, T)
    else:
        queue_tp1 = queue_t

    # --- per-segment flow at t+1 ---
    flow_tp1 = np.empty_like(density_tp1, dtype=float)
    for i in range(num_segments):
        flow_tp1[i] = flow_dynamics(density_tp1[i], velocity_tp1[i], lanes[i])

    # --- origin flow update (only when not using real data) ---
    if t+1 >= demand.shape[0]:
        flow_origin_tp1 = 0
    elif real_data:
        flow_origin_tp1 = flow_origin_t
    else:
        pcrit0 = _get_time_space_param(params["p_crit"], t+1, 0)
        qcap0 = _get_time_space_param(params["q_capacity"], t+1, 0)
        flow_origin_tp1 = origin_flow_dynamics_MN(
            demand[t + 1], density_tp1[0], queue_tp1, lanes[0], T, p_max=180.0, p_crit=pcrit0, q_capacity=qcap0
        )

    if return_diagnostics:
        diagnostics = {
            "raw_density": raw_density_tp1.copy(),
            "raw_velocity": raw_velocity_tp1.copy(),
            "density_clipped": raw_density_tp1 < 0,
            "velocity_clipped": raw_velocity_tp1 < 1e-4,
            "nonfinite_raw_density": ~np.isfinite(raw_density_tp1),
            "nonfinite_raw_velocity": ~np.isfinite(raw_velocity_tp1),
        }
        return density_tp1, velocity_tp1, queue_tp1, flow_origin_tp1, flow_tp1, diagnostics

    return density_tp1, velocity_tp1, queue_tp1, flow_origin_tp1, flow_tp1


def run_metanet_sim(T: float,
                    l: float,
                    init_traffic_state: Tuple[np.ndarray, np.ndarray, float, float],
                    demand: np.ndarray,
                    downstream_density: np.ndarray,
                    params: Dict[str, np.ndarray],
                    vsl_speeds: Optional[np.ndarray] = None,
                    lanes: Optional[Dict[int, int]] = None,
                    plotting: bool = False,
                    real_data: bool = False,
                    opt: bool = False):
    """Run a METANET simulation for the provided horizon by repeatedly calling `metanet_step`.

    This preserves the original function's outputs for compatibility:
      - If plotting=True: returns (density, velocity, queue, total_travel_time)
      - If opt=True: returns (density, velocity, queue, flow_origin, V_fd, total_travel_time)
      - Else: returns ((density_T, velocity_T, flow_origin_T, queue_T), total_travel_time)
    """
    time_steps = downstream_density.shape[0]
    num_segments = init_traffic_state[0].shape[0]
    # print(f"Running sim for time steps {time_steps}, segments {num_segments}")

    if vsl_speeds is None:
        vsl_speeds = np.full((time_steps, num_segments), 1000)  # effectively no speed limit
    if lanes is None or len(lanes) == 0:
        lanes = {i: 1 for i in range(num_segments)}

    initial_density, initial_velocity, initial_flow_or, initial_queue = init_traffic_state

    # Allocate histories
    density = np.zeros((time_steps + 1, num_segments), dtype=float)
    velocity = np.zeros((time_steps + 1, num_segments), dtype=float)
    flow = np.zeros((time_steps + 1, num_segments), dtype=float)
    queue = np.zeros((time_steps + 1, 1), dtype=float)
    flow_origin = np.zeros((time_steps + 1, 1), dtype=float)

    # Initial conditions
    density[0] = initial_density
    velocity[0] = initial_velocity
    flow[0] = np.array([initial_density[i] * initial_velocity[i] * lanes[i] for i in range(num_segments)], dtype=float)
    flow_origin[0, 0] = initial_flow_or
    queue[0, 0] = initial_queue

    # Main loop
    for t in range(time_steps):
        d_tp1, v_tp1, q_tp1, fo_tp1, f_tp1 = metanet_step(
            t,
            density[t],
            velocity[t],
            queue[t, 0],
            flow_origin[t, 0],
            T=T,
            l=l,
            vsl_speeds=vsl_speeds,
            demand=demand,
            downstream_density=downstream_density,
            params=params,
            lanes=lanes,
            real_data=real_data
        )
        density[t + 1] = d_tp1
        velocity[t + 1] = v_tp1
        queue[t + 1, 0] = q_tp1
        flow[t + 1] = f_tp1
        flow_origin[t + 1, 0] = fo_tp1

    # Compute total travel time
    total_travel_time = T * (
        sum([np.sum(density[:, i]) * lanes[i] * l for i in range(num_segments)])
        + np.sum(queue)
    )

    if plotting:
        return density, velocity, queue, total_travel_time
    elif opt:
        V_fd = calculate_V_arr(
            density[0:-1],
            vsl_speeds,
            params["a"][0],
            params["p_crit"][0],
            params["v_free"][0],
        )
        return density, velocity, queue, flow_origin, V_fd, total_travel_time
    else:
        final_tuple = (density[-1], velocity[-1], float(flow_origin[-1, 0]), float(queue[-1, 0]))
        return final_tuple, total_travel_time
