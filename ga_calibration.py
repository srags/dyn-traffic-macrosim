"""
METANET Parameter Calibration with Optuna (NSGA-II)
=====================================================
Calibrates per-segment METANET parameters by wrapping your existing
run_metanet_sim function inside an Optuna study using the NSGA-II
evolutionary sampler.

Install dependencies:
    pip install optuna

Usage:
    1. Fill in the YOUR DATA section at the bottom with your actual inputs.
    2. Adjust PARAM_BOUNDS if needed.
    3. Run:  python metanet_optuna_calibration.py
"""

import warnings

from optimization_utils import smooth_inflow
warnings.filterwarnings("ignore")

import numpy as np
import optuna
from optuna.samplers import NSGAIISampler
import matplotlib.pyplot as plt
from simulation_utils import run_metanet_sim  # existing METANET simulation function
from mpc_utils import save_results  # helper to save calibrated parameters

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PARAMETER METADATA
# ─────────────────────────────────────────────────────────────────────────────

# # Per-parameter search bounds — edit to match your network.
# PARAM_BOUNDS: dict[str, tuple[float, float]] = {
#     "tau":        (0.005,  0.030),   # relaxation time              [h]
#     "K":          (10.0,   80.0),    # density-smoothing constant   [veh/km/lane]
#     "eta_high":   (20.0,  150.0),    # anticipation constant        [km²/h]
#     "p_crit":     ( 0.1,    1.0),    # critical occupancy fraction  [-]
#     "v_free":     (80.0,  140.0),    # free-flow speed              [km/h]
#     "a":          ( 1.0,    4.0),    # speed-density exponent       [-]
#     "q_capacity": (1500., 2400.),    # capacity flow                [veh/h/lane]
#     "r":          ( 0.0,    1.0),    # [-]
#     "beta":       ( 0.0,    1.0),    # [-]
#     "gamma":      ( 0.0,    1.0),    # [-]
# }

PARAM_BOUNDS = {                                                       # <<<
    "eta_high": (10.0,        90.0),                                         # <<<
    "tau":      (10.0/3600,   60.0/3600),                                    # <<<
    "K":        (5.0,         60.0),                                         # <<<
    "p_crit": (15.0,        75.0),                                         # <<<
    "v_free":   (70.0,        150.0),                                        # <<<
    "a":        (0.5,         5.0),                                          # <<<
    "q_capacity": (2200.0,     2200.0),                                       # <<<
    "r":        (0.0,         2000),                                          # <<<
    "beta":     (0.0,         0.7),                                          # <<
}        

PARAM_NAMES = list(PARAM_BOUNDS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ENCODING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def trial_to_params(
    trial:           optuna.Trial,
    n_segments:      int,
    on_ramp_segs:    set,
    off_ramp_segs:   set,
) -> dict:
    """
    Ask Optuna to suggest one float per (parameter, segment) combination
    and pack the suggestions into the params dict expected by run_metanet_sim.
 
    Parameter names in the Optuna study follow the pattern  "<name>_seg<i>",
    e.g. "tau_seg0", "tau_seg1", "v_free_seg0", ...
 
    Ramp constraints
    ----------------
    r    : only suggested for on-ramp segments;  fixed to 0 elsewhere.
    beta : only suggested for off-ramp segments; fixed to 0 elsewhere.
    """
    params = {}
    for name in PARAM_NAMES:
        lo, hi = PARAM_BOUNDS[name]
        seg_vals = []
        for i in range(n_segments):
            if name == "r" and i not in on_ramp_segs:
                seg_vals.append(0.0)
            elif name == "beta" and i not in off_ramp_segs:
                seg_vals.append(0.0)
            else:
                seg_vals.append(trial.suggest_float(f"{name}_seg{i}", lo, hi))
        params[name] = seg_vals
    return params


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FITNESS / OBJECTIVE
# ─────────────────────────────────────────────────────────────────────────────

def nrmse(sim: np.ndarray, obs: np.ndarray) -> float:
    """Normalised RMSE: RMSE / peak-to-peak range of observations."""
    obs_range = np.ptp(obs)
    if obs_range < 1e-6:
        return 0.0
    return float(np.sqrt(np.mean((sim - obs) ** 2)) / obs_range)


def make_objective(
    n_segments:         int,
    T:                  float,
    l:                  float,
    init_traffic_state,
    data_inflow,
    downstream_density,
    num_lanes_array:    np.ndarray,
    ramp_mapping:       dict,
    obs_speed:          np.ndarray,   # shape (K, I)
    obs_density:        np.ndarray,   # shape (K, I)
    weight_speed:       float = 0.7,
    weight_density:        float = 0.3,
):
    """
    Returns an Optuna objective function that:
      1. Asks Optuna for one float per (parameter, segment) pair.
      2. Passes the resulting params dict to your run_metanet_sim.
      3. Returns the weighted normalised RMSE over speed and flow.

    All simulation inputs are captured in the closure.
    """
    lanes = {i: int(num_lanes_array[i]) for i in range(n_segments)}
    on_ramp_segs  = set(i for i, v in enumerate(ramp_mapping["on_ramps"]) if v == 1)
    off_ramp_segs = set(i for i, v in enumerate(ramp_mapping["off_ramps"]) if v == 1)

    v_max   = float(obs_speed.max())
    rho_max = float(obs_density.max())

    def objective(trial: optuna.Trial) -> float:
        params = trial_to_params(trial, n_segments, on_ramp_segs, off_ramp_segs)
        try:
            rho_sim, v_sim, _, _ = run_metanet_sim(
                T,
                l,
                init_traffic_state,
                data_inflow,
                downstream_density,
                params,
                vsl_speeds=None,
                lanes=lanes,
                plotting=True,   # never plot during optimisation
                real_data=True,
            )
        except Exception as e:
            # Penalise simulations that crash (e.g. numerical instability)
            raise optuna.exceptions.TrialPruned(f"Simulation failed: {e}")

        v_pred = np.asarray(v_sim)[0:-1]   # drop the initial state row → shape (K, I)
        rho_pred = np.asarray(rho_sim)[0:-1]

        speed_err = ((v_pred   - obs_speed)   / v_max)   ** 2
        density_err = ((rho_pred - obs_density) / rho_max) ** 2

        loss = float(np.sum(
            weight_speed * speed_err + weight_density * density_err
        ))
        # speed_err = nrmse(v_hat, obs_speed)
        # density_err  = nrmse(rho_hat, obs_density)
        return loss

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# 4.  STUDY RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(
    objective,
    n_trials:        int  = 1000,
    population_size: int  = 50,
    crossover_prob:  float = 0.7,
    mutation_prob:   float = 0.15,
    seed:            int  = 42,
    n_jobs:          int  = 1,     # set > 1 to parallelise across CPU cores
    study_name:      str  = "metanet_calibration",
    storage:         str = None,  # e.g. "sqlite:///metanet.db" for persistence
    show_progress:   bool = True,
) -> optuna.Study:
    """
    Create and run an Optuna study using the NSGA-II evolutionary sampler.

    NSGA-II is a genetic algorithm variant well-suited to this problem:
      - Real-valued chromosomes with bounded parameters
      - No gradient information required
      - Population-based, so diverse solutions are explored in parallel
      - Elitism via non-dominated sorting (even for a single objective)

    Parameters
    ----------
    n_trials        : total number of simulation evaluations
    population_size : NSGA-II population size per generation
                      (n_trials / population_size ≈ number of generations)
    seed            : random seed for reproducibility
    n_jobs          : parallel workers (-1 = all available cores)
    storage         : Optuna storage URL; None = in-memory only
    show_progress   : show tqdm progress bar
    """
    sampler = NSGAIISampler(
        population_size=population_size,
        seed=seed,
        crossover_prob=crossover_prob,       # probability of crossover vs reproduction
        mutation_prob=mutation_prob,       # None → Optuna sets 1/n_params automatically
        swapping_prob=0.5,        # SBX crossover gene-swap probability
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=(storage is not None),
    )

    # Suppress Optuna's per-trial console output; we print our own summary
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=show_progress,
        catch=(Exception,),   # log failures instead of crashing the study
    )

    return study


# ─────────────────────────────────────────────────────────────────────────────
# 5.  RESULTS & VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def extract_best_params(study: optuna.Study, n_segments: int, ramp_mapping: dict) -> dict:
    """Decode the best Optuna trial back into a run_metanet_sim params dict."""
    best          = study.best_trial
    on_ramp_segs  = set(i for i, v in enumerate(ramp_mapping["on_ramps"])  if v == 1)
    off_ramp_segs = set(i for i, v in enumerate(ramp_mapping["off_ramps"]) if v == 1)
    params        = {}
    for name in PARAM_NAMES:
        seg_vals = []
        for i in range(n_segments):
            if name == "r" and i not in on_ramp_segs:
                seg_vals.append(0.0)
            elif name == "beta" and i not in off_ramp_segs:
                seg_vals.append(0.0)
            else:
                seg_vals.append(best.params[f"{name}_seg{i}"])
        params[name] = seg_vals
    return params

def print_param_table(params: dict, n_segments: int):
    col_w = 10
    header = f"{'Parameter':<12}" + "".join(f"{'seg'+str(i):>{col_w}}" for i in range(n_segments))
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for name in PARAM_NAMES:
        row = f"{name:<12}" + "".join(f"{v:>{col_w}.4f}" for v in params[name])
        print(row)
    print("─" * len(header))


def plot_convergence(study: optuna.Study, save_path: str = "metanet_convergence.png"):
    """Plot best and median fitness vs. number of completed trials."""
    trials = [t for t in study.trials if t.value is not None]
    trials.sort(key=lambda t: t.number)

    numbers = [t.number for t in trials]
    values  = [t.value  for t in trials]

    best_so_far = np.minimum.accumulate(values)
    # Rolling median over a window ≈ population_size
    window = max(1, len(values) // 20)
    rolling_med = [
        np.median(values[max(0, i - window): i + 1])
        for i in range(len(values))
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(numbers, values,      ".", alpha=0.3, ms=3, color="steelblue", label="Trial fitness")
    ax.plot(numbers, rolling_med, "-", lw=1.5,   color="darkorange",  label=f"Rolling median (w={window})")
    ax.plot(numbers, best_so_far, "-", lw=2,     color="crimson",     label="Best so far")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Weighted normalised RMSE")
    ax.set_title("METANET Calibration — Optuna / NSGA-II Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📊  Convergence plot saved → {save_path}")
    plt.show()


def plot_fit(
    study:              optuna.Study,
    n_segments:         int,
    T:                  float,
    l:                  float,
    init_traffic_state,
    data_inflow,
    downstream_density,
    num_lanes_array,
    ramp_mapping:       dict,
    obs_speed:          np.ndarray,   # shape (K, I)
    save_path:          str = "metanet_ts_diagram.png",
):
    """
    Plot velocity time-space diagrams comparing observed vs calibrated speed.
 
    Each diagram is a 2D heatmap with:
      x-axis : time  [min]
      y-axis : space (segment positions along the freeway) [km]
      colour : mean speed [km/h]
 
    Observed and calibrated are shown side-by-side, with a difference panel
    (calibrated − observed) to highlight where the model under/over-estimates.
    """
    best_params = extract_best_params(study, n_segments, ramp_mapping)
    lanes = {i: int(num_lanes_array[i]) for i in range(n_segments)}
 
    _, v_sim, q_sim, _ = run_metanet_sim(
        T, l, init_traffic_state, data_inflow, downstream_density,
        best_params, vsl_speeds=None, lanes=lanes,
        plotting=True, real_data=True,
    )
 
    v_hat = np.asarray(v_sim)[1:]   # (K, I)
    K     = obs_speed.shape[0]
 
    # ── Spatial axis: cumulative segment midpoints along the freeway ──────────
    lengths     = np.full(n_segments, l, dtype=float)   # (I,)          # (I,)
    seg_starts  = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
    seg_mids    = seg_starts + lengths / 2             # midpoint of each segment [km]
    total_len   = lengths.sum()
 
    # ── Time axis ─────────────────────────────────────────────────────────────
    t_min = np.arange(K) * T * 60                     # [min]
 
    # ── Colour scale: shared across all three panels ──────────────────────────
    v_min = 0.0
    v_max = max(obs_speed.max(), v_hat.max())
 
    # Diverging scale for the difference panel
    diff      = v_hat - obs_speed                      # (K, I)
    abs_max   = np.abs(diff).max()
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    fig.suptitle(
        "METANET Calibration — Velocity Time-Space Diagrams",
        fontsize=13, fontweight="bold",
    )
 
    def _heatmap(ax, data, cmap, vmin, vmax, title, cbar_label):
        """
        data  : (K, I) array  — rows = time, cols = space
        We transpose to get space on y-axis, time on x-axis.
        pcolormesh treats each cell as centred on the segment midpoint / time step.
        """
        # Build cell edges from midpoints for pcolormesh
        dt      = T * 60                                          # step width [min]
        t_edges = np.append(t_min - dt / 2, t_min[-1] + dt / 2)
 
        half_l  = lengths / 2
        s_edges = np.concatenate((
            [seg_mids[0]  - half_l[0]],
            (seg_mids[:-1] + seg_mids[1:]) / 2,
            [seg_mids[-1] + half_l[-1]],
        ))
 
        mesh = ax.pcolormesh(
            t_edges, s_edges, data.T,
            cmap=cmap, vmin=vmin, vmax=vmax, shading="flat",
        )
        cb = fig.colorbar(mesh, ax=ax, pad=0.02)
        cb.set_label(cbar_label, fontsize=9)
 
        ax.set_xlabel("Time [min]", fontsize=10)
        ax.set_ylabel("Location [km]", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(t_edges[0],  t_edges[-1])
        ax.set_ylim(s_edges[0],  s_edges[-1])
        ax.set_yticks(seg_mids)
        ax.set_yticklabels([f"{x:.2f}" for x in seg_mids], fontsize=7)
        return mesh
 
    _heatmap(axes[0], obs_speed, "RdYlGn", v_min, v_max,
             "Observed speed",    "Speed [km/h]")
    _heatmap(axes[1], v_hat,     "RdYlGn", v_min, v_max,
             "Calibrated speed",  "Speed [km/h]")
    _heatmap(axes[2], diff,      "coolwarm", -abs_max, abs_max,
             "Difference (cal − obs)", "Δ Speed [km/h]")
 
    plt.savefig(save_path, dpi=150)
    print(f"📊  Time-space diagram saved → {save_path}")
    plt.show()
 


# ─────────────────────────────────────────────────────────────────────────────
# 6.  ENTRY POINT — fill in YOUR DATA below
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── YOUR DATA ─────────────────────────────────────────────────────────────
    # Replace each item with your actual variables.

    N_SEGMENTS = 14               # number of calibrated segments

    T = 10 / 3600                # simulation time step [h]  (10 seconds here)

    l = 0.4       # segment lengths [km]

    # --- Your measured traffic data (numpy arrays) ---
    q_hat = np.load("data/flow_10sec_400m_1hr.npy")#[240:270, :]
    rho_hat = np.load("data/density_10sec_400m_1hr.npy")#[240:270, :]
    rho_hat = np.where(rho_hat == 0.0, 1e-3, rho_hat)
    q_hat = np.where(q_hat == 0.0, 1e-3, q_hat)
    v_hat = q_hat / rho_hat


    num_lanes_array = np.load("data/lane_mapping.npy")[1:-1]
    ramp_mapping = {"on_ramps": np.load("data/on_ramp_mapping.npy")[1:-1], "off_ramps": np.load("data/off_ramp_mapping.npy")[1:-1]}

    downstream_density = smooth_inflow(rho_hat[:, -1]) / num_lanes_array[-1]
    data_inflow = smooth_inflow(q_hat[:, 0])

    scaled_rho_hat = rho_hat[:, 1:-1] / np.array(num_lanes_array)
    init_traffic_state = (scaled_rho_hat[0, :], v_hat[0, 1:-1], data_inflow[0], 0)
    # ── BUILD OBJECTIVE ───────────────────────────────────────────────────────
    objective = make_objective(
        n_segments=N_SEGMENTS,
        T=T,
        l=l,
        init_traffic_state=init_traffic_state,
        data_inflow=data_inflow,
        downstream_density=downstream_density,
        num_lanes_array=num_lanes_array,
        ramp_mapping=ramp_mapping,
        obs_speed=v_hat[:, 1:-1],
        obs_density=rho_hat[:, 1:-1]/num_lanes_array,
        weight_speed=20,
        weight_density=1,
    )

    # ── RUN CALIBRATION ───────────────────────────────────────────────────────
    # With population_size=50 and n_trials=500 you get ~10 generations.
    # Rule of thumb: aim for at least 20–30 generations, so n_trials ≥ 20 × population_size.
    study = run_calibration(
        objective,
        n_trials=2500,
        population_size=30,
        crossover_prob=0.7,
        mutation_prob=0.15,
        seed=42,
        n_jobs=1,            # increase to parallelise (run_metanet_sim must be thread-safe)
        study_name="metanet_calibration",
        storage="sqlite:///metanet2500.db",        # e.g. "sqlite:///metanet.db" to persist results across runs
    )

    # ── PRINT & SAVE RESULTS ──────────────────────────────────────────────────
    print(f"\n✓  Best fitness : {study.best_value:.6f}")
    best_params = extract_best_params(study, N_SEGMENTS, ramp_mapping)
    print_param_table(best_params, N_SEGMENTS)
    
    results_dir = "ga_calibration_results/trials2500"
    save_results(
        results={**best_params, "num_lanes": num_lanes_array},
        RESULTS_DIR=results_dir,
        from_param_loader=True,
    )
    print(f"💾  Best parameters saved → {results_dir}")

    # ── PLOTS ─────────────────────────────────────────────────────────────────
    plot_convergence(study, save_path=f"{results_dir}/convergence.png")
    plot_fit(
        study, N_SEGMENTS, T, l,
        init_traffic_state, data_inflow, downstream_density,
        num_lanes_array, ramp_mapping, v_hat[:, 1:-1], save_path=f"{results_dir}/ts_diagram.png"
    )