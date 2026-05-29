import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import re
from matplotlib.colors import to_rgb, to_hex
import colorsys
from pathlib import Path
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
 
from simulation_utils import *
from param_loader import METANET_Params
import optimization_utils as opt


#### Helper functions ####
def compute_perturbation_mapes(T, l, num_calibrated_segments,
                                run_metanet_sim, mape_fn,
                                rho_hat, q_hat, v_hat, params,
                                lanes, base_rho_init, base_downstream, base_inflow,
                                segment_idx=7, bc_timestep=180,
                                perturb_pct=np.arange(-0.1, 0.101, 0.001)):
    """
    Compute MAPE curves for ±perturbation of each parameter/BC.
    
    All preprocessing (smoothing, lane selection) is done by the caller,
    so this function just runs the sim with whatever it's given.
    """

    def run_sim(perturbed_params, downstream_density, data_inflow, rho_init):
        init_traffic_state = (rho_init[0, :], v_hat[0, 1:-1], data_inflow[0], 0)
        rho_sim, v_sim, queue_sim, tts_sim = run_metanet_sim(
            T=T, l=l,
            init_traffic_state=init_traffic_state,
            demand=data_inflow,
            downstream_density=downstream_density,
            params=perturbed_params,
            vsl_speeds=None,
            lanes=lanes,
            plotting=True,
            real_data=True,
        )
        return v_sim

    v_true_interior = v_hat[:, 1:-1]

    # Sanity check
    v_sim_baseline = run_sim(params, base_downstream, base_inflow, base_rho_init)
    baseline_mape = mape_fn(v_true_interior, v_sim_baseline[:-1, :])
    print(f"  Baseline MAPE: {baseline_mape:.4f}")

    param_keys = ['eta_high', 'K', 'tau', 'p_crit', 'v_free', 'a']
    bc_keys = [] #['downstream_density', 'initial_flow']
    all_keys = param_keys + bc_keys

    results = {}
    for key in all_keys:
        print(f"  Perturbing {key}...")
        mape_list = []
        for pct in perturb_pct:
            p_params = {k: np.array(v, dtype=float).copy() for k, v in params.items()}
            p_downstream = base_downstream.copy()
            p_inflow = base_inflow.copy()

            if key == 'downstream_density':
                p_downstream[bc_timestep] *= (1 + pct)
            elif key == 'initial_flow':
                p_inflow[bc_timestep] *= (1 + pct)
            else:
                param_arr = np.array(params[key], dtype=float)
                if param_arr.ndim == 1:
                    p_params[key][segment_idx] = param_arr[segment_idx] * (1 + pct)
                elif param_arr.ndim == 2:
                    p_params[key][:, segment_idx] = param_arr[:, segment_idx] * (1 + pct)
                else:
                    p_params[key] = float(param_arr) * (1 + pct)

            v_sim = run_sim(p_params, p_downstream, p_inflow, base_rho_init)
            mape_list.append(mape_fn(v_true_interior, v_sim[:-1, :]))

        results[key] = np.array(mape_list)

    return results

def _make_cache_name(cfg, perturb_pct):
    segment_idx = cfg.get("segment_idx", 7)
    if segment_idx == "all":
        raise ValueError(
            "segment_idx='all' is an aggregate view and should be loaded from "
            "existing per-segment cache files, not saved as a direct run."
        )

    bc_timestep = cfg.get("bc_timestep", 180)
    label = cfg["label"].replace(" ", "_").replace("/", "_")
    return (
        f"{label}"
        f"_seg{segment_idx}"
        f"_bc{bc_timestep}"
        f"_n{len(perturb_pct)}"
        f"_pmin{perturb_pct[0]:.4f}"
        f"_pmax{perturb_pct[-1]:.4f}.npz"
    )

def _save_perturbation_results(filepath, results, perturb_pct, cfg):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "perturb_pct": np.array(perturb_pct, dtype=float),
        "keys": np.array(list(results.keys()), dtype=object),
        "label": np.array(cfg["label"], dtype=object),
        "segment_idx": np.array(cfg.get("segment_idx", 7)),
        "bc_timestep": np.array(cfg.get("bc_timestep", 180)),
    }

    for k, v in results.items():
        save_dict[f"result__{k}"] = np.array(v)

    np.savez_compressed(filepath, **save_dict)
    print(f"  Saved perturbation data to: {filepath}")

def _load_perturbation_results(filepath):
    filepath = Path(filepath)
    data = np.load(filepath, allow_pickle=True)

    perturb_pct = data["perturb_pct"]
    keys = [str(k) for k in data["keys"]]

    results = {}
    for k in keys:
        results[k] = data[f"result__{k}"]

    metadata = {
        "label": str(data["label"].item()) if "label" in data else None,
        "segment_idx": int(data["segment_idx"]) if "segment_idx" in data else None,
        "bc_timestep": int(data["bc_timestep"]) if "bc_timestep" in data else None,
    }

    return results, perturb_pct, metadata


def _load_average_over_all_segments(cache_dir, cfg, perturb_pct):
    """
    Load all cached segment files for this config and return the average result
    across segments for each perturbation key.

    Assumes cache files were saved one segment at a time.
    """
    cache_dir = Path(cache_dir)
    label = cfg["label"].replace(" ", "_").replace("/", "_")
    bc_timestep = cfg.get("bc_timestep", 180)

    # Match files like:
    # label_seg7_bc180_n201_pmin-0.1000_pmax0.1000.npz
    pattern = re.compile(
        rf"^{re.escape(label)}_seg(\d+)_bc{bc_timestep}"
        rf"_n{len(perturb_pct)}"
        rf"_pmin{perturb_pct[0]:.4f}"
        rf"_pmax{perturb_pct[-1]:.4f}\.npz$"
    )

    matching_files = []
    for f in cache_dir.glob("*.npz"):
        if pattern.match(f.name):
            matching_files.append(f)

    if not matching_files:
        raise FileNotFoundError(
            f"No cached segment files found in {cache_dir} for label='{cfg['label']}' "
            f"with bc_timestep={bc_timestep} and the requested perturbation grid."
        )

    matching_files = sorted(matching_files)
    print(f"  Found {len(matching_files)} cached segment files. Averaging over all segments...")

    loaded_results = []
    loaded_pct_ref = None

    for f in matching_files:
        results, loaded_pct, meta = _load_perturbation_results(f)

        if loaded_pct_ref is None:
            loaded_pct_ref = loaded_pct
        elif not np.allclose(loaded_pct_ref, loaded_pct):
            raise ValueError(f"Perturbation grid mismatch in cache file: {f}")

        loaded_results.append(results)

    keys = loaded_results[0].keys()
    avg_results = {}

    for key in keys:
        stacked = np.stack([res[key] for res in loaded_results], axis=0)
        avg_results[key] = np.mean(stacked, axis=0)

    metadata = {
        "label": cfg["label"],
        "segment_idx": "all",
        "bc_timestep": bc_timestep,
        "num_segments_averaged": len(matching_files),
    }

    return avg_results, loaded_pct_ref, metadata

#############
def plot_percentile_curves(root_dirs, desired_legend_order, metric="mean", percentile=False, g_fontsize=24, save_path=None):
    """
    Plot percentile curves from multiple root directories only if they contain noise_robustness_results.csv files. 
    Run evaluate robustness function to generate these files for different methods and horizons. 
    The metric argument can be "mean" or "max" to plot either the average MAPE or worst-case MAPE across noise perturbations.
    """

    blue_purple_colors = {
        0 :  "#4ACEBC",
        1:  "#4F8FCF",
        2:  "#002DF4",
        5:  "#5644FA",
        10: "#8B59FF",
        15: "#BB4EFF",  # highlight this one
        20: "#E45DFF",
        30: "#FF5DBE",
    }
    
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
        "legend.fontsize": g_fontsize - 2,
    })
        
    fig, ax = plt.subplots(figsize=(13, 7))

    base_colors = {
        'ocp': "#FF6600",      # gray
        'robust': "#2c3da0",
        'rl': "#1b9e77",
    }

    counters = {k: 0 for k in base_colors.keys()}

    def shade_color(hex_color, idx):
        """Return a slightly different shade for successive indices."""
        r, g, b = to_rgb(hex_color)
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        v = max(0.35, v * (1.0 - 0.12 * idx))
        s = max(0.25, s * (1.0 - 0.05 * idx))
        nr, ng, nb = colorsys.hsv_to_rgb(h, s, v)
        return to_hex((nr, ng, nb))

    def extract_horizon(label, root_dir):
        """Try to extract MPC horizon from folder name/path."""
        text = f"{label} {root_dir}"
        patterns = [
            r'control_h_[_=\s-]*(\d+)',
            r'horizon[_=\s-]*(\d+)',
            r'hor[_=\s-]*(\d+)',
            r'\bmpc[_=\s-]*(\d+)\b',
            r'\bN[_=\s-]*(\d+)\b'
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None

    # First pass: load data and determine method for each root_dir
    entries = []
    for root_dir in root_dirs:
        try:
            data = np.genfromtxt(
                f"{root_dir}/noise_robustness_results.csv",
                delimiter=',',
                names=True,
                dtype=None,
                encoding=None
            )
        except Exception as e:
            print(f"Could not load {root_dir}: {e}")
            continue

        noise_levels = np.array(data['noise'], dtype=float)
        mean = np.array(data['mean'], dtype=float) if 'mean' in data.dtype.names else None
        percentile_5 = np.array(data['percentile_5'], dtype=float) if 'percentile_5' in data.dtype.names else None
        percentile_95 = np.array(data['percentile_95'], dtype=float) if 'percentile_95' in data.dtype.names else None
        max_vals = np.array(data['max'], dtype=float) if 'max' in data.dtype.names else None

        order = np.argsort(noise_levels)
        noise_levels = noise_levels[order]
        if mean is not None:
            mean = mean[order]
        if percentile_5 is not None:
            percentile_5 = percentile_5[order]
        if percentile_95 is not None:
            percentile_95 = percentile_95[order]
        if max_vals is not None:
            max_vals = max_vals[order]

        label = os.path.basename(root_dir.rstrip('/')) or root_dir
        rd_lower = root_dir.lower()

        if 'ocp' in rd_lower:
            method = 'OCP'
        elif 'mpc' in rd_lower and 'robust' not in rd_lower:
            method = 'MPC'
        elif 'robust' in rd_lower or 'robust' in label.lower():
            method = 'Robust'
        elif 'rl' in rd_lower:
            method = 'RL'
        else:
            method = 'Other'

        horizon = extract_horizon(label, root_dir)

        entries.append({
            'root_dir': root_dir,
            'label': label,
            'method': method,
            'horizon': horizon,
            'noise_levels': noise_levels,
            'mean': mean,
            'p5': percentile_5,
            'p95': percentile_95,
            'max': max_vals
        })

    if len(entries) == 0:
        print("No valid data found in provided root directories.")
        return

    def get_metric_array(e):
        if metric == "max":
            return e.get('max', None)
        else:
            return e.get('mean', None)

    ylabel = " Average MAPE (%)" if metric != "max" else "Worst-case MAPE (%)"

    # Sort MPC entries by horizon if available
    mpc_entries = [e for e in entries if e['method'] == 'MPC']
    mpc_entries = sorted(
        mpc_entries,
        key=lambda e: (e['horizon'] is None, e['horizon'] if e['horizon'] is not None else 1e9)
    )

    # Purple shades, not too light
    # mpc_cmap = plt.get_cmap("summer")

    def purple_to_teal(n):
        dark_purple = np.array(mcolors.to_rgb("#8430a8"))
        teal        = np.array(mcolors.to_rgb("#2fbda3"))
        return [mcolors.to_hex((1 - t) * dark_purple + t * teal) for t in np.linspace(0, 1, n)]


    mpc_colors = purple_to_teal(len(mpc_entries))
    mpc_color_dict = {e['horizon']: color for color, e in zip(mpc_colors, mpc_entries)}
    # evenly space the MPC colors from purple to blue
    #mpc_cmap = mpl.cm.get_cmap("cool", num_mpc + 2)  # +2 to avoid very light colors at the start
    # 
    #     # mpc_color_map = {}
    # for e in mpc_entries:
    #     mpc_color_map[e['label']] = blue_purple_colors[e['horizon'] // 6]

        

    for e in entries:
        label = e['label']
        method = e['method']
        rd_lower = e['root_dir'].lower()
        marker = None
        marker_size = None

        if method == 'MPC':
            color = mpc_color_dict.get(e['horizon'], None)
            linewidth = 3 if e['horizon'] == 90 else 3
            linestyle = '-' if e['horizon'] == 90 else '-'
            marker = None if e['horizon'] == 90 else None
            marker_size = 10 if e['horizon'] == 90 else None
            if e['horizon'] is not None:
                legend_label = f"Dynamic: {e['horizon'] * 10 / 60:.1f} min" #f"Dynamic: {e['horizon'] * 10 // 60} min"
            else:
                legend_label = "Dynamic"
        elif method == 'OCP':
            color = "orange"
            linewidth = 3
            linestyle = '-'
            legend_label = "Static"
        elif 'robust' in rd_lower or 'robust' in label.lower():
            key = 'robust'
            color = shade_color(base_colors[key], counters[key])
            counters[key] += 2
            linewidth = 2.0
            legend_label = label
        elif "rl" in rd_lower or "rl" in label.lower():
            key = 'rl'
            color = shade_color(base_colors[key], counters[key])
            counters[key] += 2
            linewidth = 3.0
            legend_label = "RL: 10 sec"
        else:
            color = None
            linewidth = 2.0
            legend_label = label

        vals = get_metric_array(e)
        if vals is None:
            print(f"Entry {label} missing requested metric '{metric}'; skipping.")
            continue

        ax.plot(
            e['noise_levels'],
            vals,
            label=legend_label,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
            markersize=marker_size
        )

        if percentile and metric != "max":
            ax.fill_between(
                e['noise_levels'],
                e['p5'],
                e['p95'],
                color=color,
                alpha=0.18,
                linewidth=0
            )

    ax.set_xlim([1e-10, 5])
    ax.set_ylim(bottom=0)
    ax.set_xscale('log')
    ax.set_xlabel('Noise added to inflows (%)', fontsize=g_fontsize)
    ax.set_ylabel(ylabel, fontsize=g_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=g_fontsize - 2)
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))

    ordered_labels = [lab for lab in desired_legend_order if lab in label_to_handle]
    ordered_handles = [label_to_handle[lab] for lab in ordered_labels]

    ax.legend(
        ordered_handles,
        ordered_labels,
        ncol=2,
    )
    #ax.grid(False, which='both', ls='--', linewidth=0.5)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

def plot_sim_vs_gt(v_gt, v_sim, T, l, noise, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Compute common color scale ---
    vmin = min(
        np.min(v_gt),
        np.min(v_sim),
    )
    vmax = max(
        np.max(v_gt),
        np.max(v_sim),
    )

    # --- Prediction ---
    im0 = axes[0].imshow(
        v_gt.T,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        interpolation="none",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Segment Index")
    axes[0].set_title("GT Velocity", fontsize=14)

    # --- Simulation ---
    im1 = axes[1].imshow(
        v_sim.T,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        interpolation="none",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_xlabel("Time Step")
    axes[1].set_title(f"Simulation Velocity at {noise}% noise", fontsize=14)

    # ---- One shared colorbar in its own axis ----
    # Make room on the right for the colorbar
    fig.subplots_adjust(right=0.88)

    # [left, bottom, width, height] in figure coordinates
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im0, cax=cbar_ax)
    cbar.set_label("Velocity (km/hr)", fontsize=14)

    # Convert x label to minutes
    for ax in axes:
        ax.set_xlabel("Time (minutes)")

    # Convert x ticks to minutes
    num_time_steps = v_gt.shape[0]
    time_step_indices = np.arange(0, num_time_steps, max(1, num_time_steps // 6))
    time_step_labels = [f"{(i * T * 60):.0f}" for i in time_step_indices]
    for ax in axes:
        ax.set_xticks(time_step_indices)
        ax.set_xticklabels(time_step_labels)

    # Convert y axis to kilometers
    for ax in axes:
        num_segments = v_gt.shape[1]
        segment_indices = np.arange(0, num_segments, max(1, num_segments // 6))
        segment_labels = [f"{(i * l):.1f}" for i in segment_indices]
        ax.set_yticks(segment_indices)
        ax.set_yticklabels(segment_labels)
        ax.set_ylabel("Distance (km)")

    # label font size
    for ax in axes:
        ax.title.set_fontsize(16)
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        ax.tick_params(axis='both', which='major', labelsize=12)    
    
    # Save the figure
    # plt.tight_layout()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # plt.tight_layout()
    fig.savefig(f"{save_dir}/sim_vs_gt_noise_{noise}.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

def run_and_plot_perturbations(param_configs, T, l, num_calibrated_segments,
                               run_metanet_sim, mape_fn,
                               perturb_pct=np.arange(-0.1, 0.101, 0.001),
                               save_path=None,
                               cache_dir="perturbation_cache",
                               use_cache=True,
                               load_only=False,
                               g_fontsize=20):
    
    tick_fs = g_fontsize
    label_fs = g_fontsize + 1
    legend_fs = g_fontsize + 1
    title_fs = g_fontsize + 8

    # ── Load shared ground truth data ─────────────────────────────────────────
    rho_hat = np.load("data/density_10sec_400m_1hr.npy")
    q_hat = np.load("data/flow_10sec_400m_1hr.npy")
    v_hat = q_hat / rho_hat

    lane_mapping = np.load("data/lane_mapping.npy")
    rho_hat = rho_hat / lane_mapping

    # ── Run or load perturbation for each config ──────────────────────────────
    all_results = {}
    loaded_perturb_pct = None
    cache_dir = Path(cache_dir)

    for cfg in param_configs:
        label = cfg["label"]
        true_labels = {"OCP": "Static", "MPC_90": "Dynamic \n (Ours)", "RL": "RL"}
        segment_idx = cfg.get("segment_idx", 7)

        print(f"\n{'='*50}")
        print(f"Config: {label} ({cfg['results_dir']})")
        print(f"Segment: {segment_idx}")
        print(f"{'='*50}")

        # Special case: average all cached segment results
        if segment_idx == "all":
            if not (use_cache or load_only):
                raise ValueError(
                    "segment_idx='all' only supports loading from cache. "
                    "Run and save individual segments first, then use segment_idx='all'."
                )

            results, loaded_pct, meta = _load_average_over_all_segments(
                cache_dir=cache_dir,
                cfg=cfg,
                perturb_pct=perturb_pct,
            )
            all_results[label] = results
            loaded_perturb_pct = loaded_pct
            continue

        cache_file = cache_dir / _make_cache_name(cfg, perturb_pct)
        print(f"Cache:  {cache_file}")

        if cache_file.exists() and (use_cache or load_only):
            print("  Loading existing perturbation data...")
            results, loaded_pct, meta = _load_perturbation_results(cache_file)
            all_results[label] = results
            loaded_perturb_pct = loaded_pct
            continue

        if load_only:
            raise FileNotFoundError(
                f"load_only=True, but cache file does not exist:\n{cache_file}"
            )

        print("  No cache found, running perturbations...")

        params = METANET_Params(
            path=cfg["results_dir"],
            control_h=cfg["control_hor"],
            num_segments=num_calibrated_segments,
            num_timesteps=rho_hat.shape[0]
        ).get_params()

        lanes = {i: cfg["lanes"][i] for i in range(num_calibrated_segments)}

        if cfg.get("smooth_rho", False):
            base_rho_init = opt.smooth_inflow(rho_hat[:, 1:-1])
        else:
            base_rho_init = rho_hat[:, 1:-1].copy()

        if cfg.get("smooth_bc", False):
            base_downstream = opt.smooth_inflow(rho_hat[:, -1])
            base_inflow = opt.smooth_inflow(q_hat[:, 0])
        else:
            base_downstream = rho_hat[:, -1].copy()
            base_inflow = q_hat[:, 0].copy()

        results = compute_perturbation_mapes(
            T=T, l=l,
            num_calibrated_segments=num_calibrated_segments,
            run_metanet_sim=run_metanet_sim,
            mape_fn=mape_fn,
            rho_hat=rho_hat, q_hat=q_hat, v_hat=v_hat,
            params=params,
            lanes=lanes,
            base_rho_init=base_rho_init,
            base_downstream=base_downstream,
            base_inflow=base_inflow,
            segment_idx=segment_idx,
            bc_timestep=cfg.get("bc_timestep", 180),
            perturb_pct=perturb_pct,
        )

        all_results[label] = results
        _save_perturbation_results(cache_file, results, perturb_pct, cfg)

    # If loaded from file, use the saved perturbation axis
    if loaded_perturb_pct is not None:
        perturb_pct = loaded_perturb_pct

    # ── Plotting ──────────────────────────────────────────────────────────────
    # Plotting
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": "Times New Roman", #["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],  # Use Computer Modern for text
        "mathtext.fontset": "cm",   # Computer Modern for math
        "pdf.fonttype": 42,  # Embed fonts in PDF output
        "ps.fonttype": 42,  # Embed fonts in PS output
        "axes.unicode_minus": False,
        "text.usetex": False
    })
    
    perturb_x = perturb_pct * 100
    zero_idx = np.argmin(np.abs(perturb_pct))
    all_keys = ['eta_high', 'K', 'tau', 'p_crit', 'v_free', 'a']

    # LaTeX/Greek subplot titles
    title_map = {
        'eta_high': r'$\eta$',
        'K': r'$\kappa$',
        'tau': r'$\tau$',
        'p_crit': r'$\rho_{\mathrm{cr}}$',
        'v_free': r'$v_{\mathrm{free}}$',
        'a': r'$a$',
        'downstream_density': r'$\rho_{\mathrm{down}}$',
        'initial_flow': r'$q_{\mathrm{in}}$',
    }

    n_keys = len(all_keys)
    nrows = 1 if n_keys < 3 else 2
    ncols = math.ceil(n_keys / nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    colors = ["orange", "#BB4EFF", "blue"]#plt.cm.tab10.colors

    legend_handles = []
    for idx, label in enumerate(all_results):
        color = colors[idx % len(colors)]
        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=1.5, label=true_labels[label])
        )

    fig.legend(
        handles=legend_handles,
        loc='right',
        bbox_to_anchor=(1.07, 0.5),
        frameon=True,
        fontsize=legend_fs,
    )

    # -------------------------------------------------------------------------
    # First pass: compute global y-limits shared by all subplots
    # -------------------------------------------------------------------------
    global_ymin = np.inf
    global_ymax = -np.inf

    for key in all_keys:
        for label, results in all_results.items():
            baseline = results[key][zero_idx]
            yvals = results[key] - baseline
            global_ymin = min(global_ymin, np.min(yvals))
            global_ymax = max(global_ymax, np.max(yvals))

    # Optional padding so lines do not sit on the border
    if np.isclose(global_ymin, global_ymax):
        pad = 1e-6 if global_ymin == 0 else 0.05 * abs(global_ymin)
    else:
        pad = 0.05 * (global_ymax - global_ymin)

    global_ymin -= pad
    global_ymax += pad

    # -------------------------------------------------------------------------
    # Second pass: plot
    # -------------------------------------------------------------------------
    for plot_idx, (ax, key) in enumerate(zip(axes[:n_keys], all_keys)):
        ax.set_facecolor('white')

        for idx, (label, results) in enumerate(all_results.items()):
            color = colors[idx % len(colors)]
            baseline = results[key][zero_idx]
            ax.plot(perturb_x, results[key] - baseline, color=color, linewidth=2)

        ax.axvline(0, color='black', linewidth=0.8, linestyle=':', alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.8, linestyle=':', alpha=0.7)
        ax.grid(True, color='lightgrey', linewidth=0.5, linestyle='-', alpha=0.8)
        ax.set_axisbelow(True)

        ax.set_title(title_map.get(key, key), fontsize=title_fs, fontweight='normal', pad=6)
        ax.set_xlim(perturb_x[0], perturb_x[-1])
        ax.set_ylim(global_ymin, global_ymax)
        ax.tick_params(labelsize=tick_fs)

        # Only show x-axis on bottom row
        row_idx = plot_idx // ncols
        is_bottom_row = (row_idx == nrows - 1)

        if is_bottom_row:
            ax.set_xlabel('Perturbation (%)', fontsize=label_fs)
        else:
            ax.set_xlabel('')
            ax.tick_params(axis='x', which='both', labelbottom=False, bottom=False)

        # Only show y-axis label on left column
        col_idx = plot_idx % ncols
        is_left_col = (col_idx == 0)

        if is_left_col:
            ax.set_ylabel('\u0394MAPE from baseline', fontsize=label_fs)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', which='both', labelleft=False)

        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(0.8)

    # hide any unused axes
    for ax in axes[n_keys:]:
        ax.set_visible(False)

    # Optional overall title
    # fig.suptitle(
    #     f'Sensitivity Analysis — Segment {param_configs[0].get("segment_idx", 7)} | '
    #     f'BC Timestep {param_configs[0].get("bc_timestep", 180)}',
    #     fontsize=suptitle_fs,
    #     y=1.02
    # )

    # plt.tight_layout()
    plt.show()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\nSaved plot to {save_path}")

    return all_results

def plot_fd_diagrams(params_list, scenario_info, labels, save_path=None):
    q_sims = dict()
    rho_sims = dict()

    for i in range(len(params_list)):
        params = params_list[i]
        rho_sim, v_sim, _, tts_sim = run_metanet_sim(
            scenario_info["T"], 
            scenario_info["l"], 
            scenario_info["init_traffic_state"],
            scenario_info["data_inflow"],
            scenario_info["downstream_density"],
            params,
            vsl_speeds=None,
            lanes={i: scenario_info["lanes"][i] for i in range(scenario_info["num_calibrated_segments"])},
            plotting=True,
            real_data=True
        )

        all_rho_pred = rho_sim * np.array(scenario_info["lanes"])  # scale back to total density using number of lanes
        all_q_pred = all_rho_pred * v_sim

        q_sims[labels[i]] = all_q_pred.reshape(-1, 1)
        rho_sims[labels[i]] = all_rho_pred.reshape(-1, 1)

    q_hat = scenario_info["q_hat"].reshape(-1, 1)
    rho_hat = scenario_info["rho_hat"].reshape(-1, 1) * np.array(scenario_info["lanes"])[-1]  # scale back to total density using number of lanes

    g_fontsize = 17
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": "Times New Roman", #["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],  # Use Computer Modern for text
        "mathtext.fontset": "cm",   # Computer Modern for math
        "pdf.fonttype": 42,  # Embed fonts in PDF output
        "ps.fonttype": 42,  # Embed fonts in PS output
        "axes.unicode_minus": False,
        "text.usetex": False,
        "font.size": g_fontsize,
        "axes.labelsize": g_fontsize,
        "xtick.labelsize": g_fontsize - 2,
        "ytick.labelsize": g_fontsize - 2,
        "legend.fontsize": g_fontsize - 1,
    })
    colors = ["orange", "forestgreen",  "#BB4EFF", "blue"]#plt.cm.tab10.colors

    fig, ax = plt.subplots(len(params_list), 1, figsize=(7, 4.5*len(params_list)), sharex=True)

    for i in range(len(params_list)):
        # Top subplot
        ax[i].scatter(rho_hat, q_hat, color="gray", alpha=0.7, s=1, label="Data (gt)")
        ax[i].scatter(rho_sims[labels[i]], q_sims[labels[i]], alpha=0.6, s=1, label="Predicted", color=colors[i])
        if i == len(params_list) - 1:
            ax[i].set_xlabel("Density (veh/km)")
        ax[i].set_ylabel("Flow (veh/hr)")
        # ax[i].set_title("Standard Fundamental Diagram")
        ax[i].grid(True)
        ax[i].set_ylim(0, np.max(all_q_pred) * 1.4)

    # Shared legend
    handles = [
        Line2D([0], [0], marker='o', linestyle='None', color='gray', markersize=8, alpha=0.7, label='Ground Truth Data'),
    ]

    for i in range(len(labels)):
        handles.append(
            Line2D([0], [0], marker='o', linestyle='None', color=colors[i], markersize=8, alpha=0.6, label=labels[i])
        )

    if len(labels) <= 2:
        fig.legend(handles=handles, loc='upper center', ncol=len(labels)+1, bbox_to_anchor=(0.555, 1.04))
    else:
        fig.legend(handles=handles, loc='upper center', ncol = 2, bbox_to_anchor=(0.555, 1.045))

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()