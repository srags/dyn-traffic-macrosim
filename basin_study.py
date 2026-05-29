"""
basin_study.py  —  Basin-of-attraction landscape mapping for METANET calibration
==================================================================================
Workflow
--------
  Phase 1 (Global):  Latin Hypercube sample N starting points in parameter
                     space, run IPOPT from each, record converged solution +
                     objective.  PCA-project starting points to 2D and plot a
                     colour-coded basin map + interpolated objective surface.

  Phase 2 (Local):   Continuation sweep: walk along a 1D path in PCA space,
                     using each converged solution as the warm-start for the
                     next neighbouring point.  Reveals how the landscape
                     changes continuously.

Usage
-----
  1.  Apply the changes in optimization_utils_changes.py to optimization_utils.py
  2.  Fill in the "USER CONFIG" section below with your actual data/settings
  3.  python basin_study.py

Dependencies (beyond your existing env):
    pip install scipy scikit-learn matplotlib joblib
"""

from __future__ import annotations

import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from scipy.interpolate import griddata
from scipy.stats import qmc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from pyomo.opt import TerminationCondition

# Your (modified) calibration entry-point
import optimization_utils as opt

warnings.filterwarnings("ignore", category=UserWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# USER CONFIG — fill these in before running
# ═══════════════════════════════════════════════════════════════════════════════

# --- Your measured traffic data (numpy arrays) ---
q_hat = np.load("data/flow_10sec_400m_1hr.npy")#[240:270, :]
rho_hat = np.load("data/density_10sec_400m_1hr.npy")#[240:270, :]
rho_hat = np.where(rho_hat == 0.0, 1e-3, rho_hat)
q_hat = np.where(q_hat == 0.0, 1e-3, q_hat)

lane_map = np.load("data/lane_mapping.npy")
lane_mapping = {i: float(lane_map[i+1]) for i in range(len(lane_map)-2)}
ramp_mapping = {"on_ramps": np.load("data/on_ramp_mapping.npy")[1:-1], "off_ramps": np.load("data/off_ramp_mapping.npy")[1:-1]}

# --- Fixed calibration settings (mirrors your normal run_calibration call) ---
CALIB_KWARGS = dict(
    rho_hat                = rho_hat,      # fill in
    q_hat                  = q_hat,        # fill in
    T                      = 10 / 3600,    # time step in hours
    l                      = 0.4,          # segment length in km
    num_calibrated_segments= rho_hat.shape[1] - 2,            # number of interior segments
    include_ramping        = True,
    varylanes              = False,
    lane_mapping           = lane_mapping,
    ramp_mapping           = ramp_mapping,
    smoothing              = True,
    constraint_tol         = 1e-12,         # looser tol is fine for landscape study
    tee                    = False,         # suppress IPOPT output for cleaner logs
)

# --- Basin study settings ---
N_TRIALS   = 100      # total IPOPT solves  (start with 50 to test, then 200+)
N_CLUSTERS = None        # k-means clusters on converged solutions
DISTANCE_THRESHOLD = 10  # for agglomerative clustering (set to None to disable)
N_WORKERS  = 1        # parallel processes (set to 1 to debug)
SEED       = 42
OUTPUT_DIR = "basin_results/500_trials"  # where to save results + plots
PERTURB_PCT = 0  # how far from the best solution to sample new starting points

# ═══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

def lhs_sample(N: int, num_segments: int, seed: int = 42) -> np.ndarray:
    """
    Latin Hypercube sample N points in (6 * num_segments)-D parameter space.
    Returns array of shape (N, 6 * num_segments), scaled to actual bounds.

    Parameter ordering (repeated per segment):
        eta_high | tau | K | rho_crit | v_free | a
    """
    dim      = len(opt.CALIB_PARAM_NAMES) * num_segments
    sampler  = qmc.LatinHypercube(d=dim, seed=seed)
    unit_smp = sampler.random(N)                          # (N, dim) in [0,1]

    lbs = np.array([opt.CALIB_PARAM_BOUNDS[p][0] for p in opt.CALIB_PARAM_NAMES for _ in range(num_segments)])
    ubs = np.array([opt.CALIB_PARAM_BOUNDS[p][1] for p in opt.CALIB_PARAM_NAMES for _ in range(num_segments)])
    return qmc.scale(unit_smp, lbs, ubs)                  # (N, dim)


def vec_to_x0(vec: np.ndarray, num_segments: int) -> dict:
    """
    Flat parameter vector (6*num_segments,) → x0 dict expected by metanet_param_fit.

    x0["eta_high"] = array of shape (num_segments,)  etc.
    """
    x0 = {}
    for j, pname in enumerate(opt.CALIB_PARAM_NAMES):
        x0[pname] = vec[j * num_segments:(j + 1) * num_segments].copy()
    return x0


def results_to_conv_vec(results: dict, num_segments: int) -> np.ndarray:
    """Extract converged parameter vector from run_calibration results dict."""
    return np.concatenate([np.atleast_1d(results[p])[:num_segments]
                           for p in opt.CALIB_PARAM_NAMES])


# ─────────────────────────────────────────────────────────────────────────────
# Single-trial runner  (called in parallel)
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_trial(
    trial_id: int,
    x0_vec: np.ndarray,
    num_segments: int,
    calib_kwargs: dict,
) -> dict:
    """
    Run one IPOPT solve from starting point x0_vec.
    Returns a result dict that is safe to pickle (no Pyomo objects).
    """
    x0 = vec_to_x0(x0_vec, num_segments)
    try:
        results = opt.run_calibration(x0=x0, **calib_kwargs)

        tc = results.get("termination_condition", "unknown")
        converged = tc in {
            str(TerminationCondition.optimal),
            str(TerminationCondition.locallyOptimal),
            "optimal", "locallyOptimal",
        }

        return {
            "trial_id"   : trial_id,
            "x0_vec"     : x0_vec,
            "conv_vec"   : results_to_conv_vec(results, num_segments) if converged else None,
            "obj_val"    : results["obj_val"] if converged else np.nan,
            "status"     : results.get("solver_status", "unknown"),
            "termination": tc,
            "converged"  : converged,
        }
    except Exception as exc:
        return {
            "trial_id"   : trial_id,
            "x0_vec"     : x0_vec,
            "conv_vec"   : None,
            "obj_val"    : np.nan,
            "status"     : "error",
            "termination": str(exc),
            "converged"  : False,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — global basin map
# ─────────────────────────────────────────────────────────────────────────────

def run_basin_study(
    calib_kwargs: dict,
    num_segments: int,
    N: int         = 200,
    n_clusters: int = 5,
    n_workers: int  = 4,
    seed: int       = 42,
    output_dir: str = "basin_results",
) -> list[dict]:
    """
    Phase 1: run N IPOPT solves from LHS starting points in parallel.

    Returns
    -------
    trials : list of result dicts, also saved to  output_dir/trials.pkl
    """
    os.makedirs(output_dir, exist_ok=True)

    # Test bounds

    x0_matrix = lhs_sample(N=5, num_segments=num_seg, seed=42)
    for i in range(5):
        x0 = vec_to_x0(x0_matrix[i], num_seg)
        for pname in opt.CALIB_PARAM_NAMES:
            lb, ub = opt.CALIB_PARAM_BOUNDS[pname]
            assert np.all(x0[pname] >= lb) and np.all(x0[pname] <= ub), \
                f"Trial {i}: {pname} = {x0[pname]} out of bounds [{lb}, {ub}]"
    print("All starting points within bounds ✓")

    x0_matrix = lhs_sample(N, num_segments, seed=seed)   # (N, 6*num_seg)
    print(f"[basin] {N} LHS starting points sampled in "
          f"{len(opt.CALIB_PARAM_NAMES) * num_segments}D space")

    # ── parallel IPOPT solves ─────────────────────────────────────────────────
    print(f"[basin] launching {N} solves on {n_workers} workers ...")
    trials = Parallel(n_jobs=n_workers, verbose=5)(
        delayed(_run_one_trial)(i, x0_matrix[i], num_segments, calib_kwargs)
        for i in range(N)
    )

    with open(os.path.join(output_dir, "trials.pkl"), "wb") as f:
        pickle.dump({"trials": trials, "x0_matrix": x0_matrix}, f)
    print(f"[basin] results saved to {output_dir}/trials.pkl")

    # ── plot ──────────────────────────────────────────────────────────────────
    analyse_and_plot(trials, x0_matrix, num_segments,
                     output_dir=output_dir, distance_threshold=DISTANCE_THRESHOLD)
    return trials


# ─────────────────────────────────────────────────────────────────────────────
# Analysis and plotting
# ─────────────────────────────────────────────────────────────────────────────

def analyse_and_plot(
    trials: list[dict],
    x0_matrix: np.ndarray,
    num_segments: int,
    distance_threshold: float = 0.05,
    output_dir: str = "basin_results",
    tsne_perplexity: float = None,   # None = auto-set to min(30, M//3)
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
 
    # ── split good / bad ──────────────────────────────────────────────────────
    good = [t for t in trials if t["converged"] and t["conv_vec"] is not None]
    bad  = [t for t in trials if not t["converged"] or t["conv_vec"] is None]
    print(f"\n[basin] converged={len(good)}  failed/infeasible={len(bad)}  "
          f"total={len(trials)}")
 
    if len(good) < 3:
        print("[basin] Too few converged trials to analyse.")
        return {}
 
    # ── arrays ────────────────────────────────────────────────────────────────
    ids      = np.array([t["trial_id"] for t in good])
    x0_good  = x0_matrix[ids]
    conv_mat = np.vstack([t["conv_vec"] for t in good])
    obj_vals = np.array([t["obj_val"]   for t in good])
    M        = len(good)

    # Ensure clean float64 — None values anywhere cause the whole array to go string dtype
    conv_mat = conv_mat.astype(np.float64)

    # Drop any rows with NaN/Inf (can happen if a param hit a bound and returned None)
    finite_mask = np.isfinite(conv_mat).all(axis=1)
    if not finite_mask.all():
        print(f"[basin] Dropping {(~finite_mask).sum()} trials with non-finite param values")
        conv_mat  = conv_mat[finite_mask]
        obj_vals  = obj_vals[finite_mask]
        ids       = ids[finite_mask]
        x0_good   = x0_good[finite_mask]
    
    # ── normalise converged solutions ─────────────────────────────────────────
    scaler    = StandardScaler()
    conv_norm = scaler.fit_transform(conv_mat)

    scaler_x0 = StandardScaler()
    x0_norm   = scaler_x0.fit_transform(x0_good)
 
    # ── t-SNE on converged solutions ──────────────────────────────────────────
    if tsne_perplexity is None:
        tsne_perplexity = float(min(30, max(5, M // 3)))
    print(f"[basin] Running t-SNE (M={M}, perplexity={tsne_perplexity:.0f}) ...")
 
    from sklearn.manifold import TSNE
    tsne = TSNE(
        n_components=2,
        perplexity=30, #tsne_perplexity,
        random_state=42,
        n_iter=1000,
        init="pca",
    )

    init_2d = tsne.fit_transform(x0_norm).astype(np.float64)
    conv_2d = tsne.fit_transform(conv_norm).astype(np.float64)
 
    # ── dendrogram + agglomerative clustering ─────────────────────────────────
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.cluster import AgglomerativeClustering
 
    Z = linkage(conv_norm, method="complete")
    fig0, ax0 = plt.subplots(figsize=(12, 4))
    dendrogram(Z, no_labels=True, ax=ax0, color_threshold=distance_threshold)
    ax0.axhline(distance_threshold, color="red", linestyle="--",
                label=f"threshold={distance_threshold}")
    ax0.set_title("Dendrogram — gaps indicate distinct basins")
    ax0.set_ylabel("Distance at merge")
    ax0.legend()
    plt.tight_layout()
    _save(fig0, output_dir, "dendrogram.png")
 
    agg = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage="complete",
    )
    basin_labels = agg.fit_predict(conv_norm)
    n_basins     = basin_labels.max() + 1
    print(f"[basin] Found {n_basins} distinct basins at threshold={distance_threshold}")
 
    # ── basin summary ─────────────────────────────────────────────────────────
    print("\n── Basin Summary " + "─" * 40)
    for k in range(n_basins):
        mask = basin_labels == k
        if mask.sum() == 0:
            continue
        print(f"  Basin {k}: {mask.sum():4d} starts | "
              f"obj  mean={obj_vals[mask].mean():.4f}  "
              f"std={obj_vals[mask].std():.4f}  "
              f"best={obj_vals[mask].min():.4f}")
    best = np.argmin(obj_vals)
    print(f"\n  ★ Global best: trial {ids[best]}  "
          f"obj={obj_vals[best]:.6f}  basin={basin_labels[best]}")
    
    tab20_colors = list(plt.cm.tab20.colors)
    extra_colors = ['black', 'gray']
    cmap = tab20_colors + extra_colors
    cmap = matplotlib.colors.ListedColormap(cmap)
    if n_basins > len(cmap.colors):
        cmap = cm.get_cmap("inferno", max(n_basins, 1))

    print(conv_mat[4, :])
    print(conv_mat[5, :])
 
    # ── Plot 1: basin map ─────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 7))
    scatter = ax1.scatter(
        conv_2d[:, 0], conv_2d[:, 1],
        c=basin_labels, cmap=cmap,
        vmin=-0.5, vmax=n_basins - 0.5,
        alpha=0.8, s=60, edgecolors="white", linewidths=0.4,
    )
    ax1.scatter(conv_2d[best, 0], conv_2d[best, 1],
                marker="*", s=300, color="gold", zorder=10, label="Global best")
    ax1.legend(fontsize=10)
    plt.colorbar(scatter, ax=ax1, label="Basin ID", ticks=range(n_basins))
    ax1.set_xlabel("t-SNE dim 1", fontsize=12)
    ax1.set_ylabel("t-SNE dim 2", fontsize=12)
    ax1.set_title(
        f"Basin of Attraction Map\nt-SNE of converged solutions  "
        f"(perplexity={tsne_perplexity:.0f})", fontsize=12)
    plt.tight_layout()
    _save(fig1, output_dir, "basin_map.png")

    # ── t-SNE on combined starting + converged points ─────────────────────────
    combined      = np.vstack([x0_norm, conv_norm])
    combined_2d   = tsne.fit_transform(combined).astype(np.float64)
    x0_2d         = combined_2d[:len(x0_norm)]
    conv_2d_combined = combined_2d[len(x0_norm):]

    fig, ax = plt.subplots(figsize=(10, 8))

    # converged solutions — circles
    # ax.scatter(
    #     conv_2d_combined[:, 0], conv_2d_combined[:, 1],
    #     c=basin_labels, cmap=cmap,
    #     vmin=-0.5, vmax=n_basins - 0.5,
    #     s=80, marker="o", edgecolors="white", linewidths=0.5,
    #     zorder=3, label="Converged",
    # )

    # starting points — triangles, same colour = same basin
    ax.scatter(
        x0_2d[:, 0], x0_2d[:, 1],
        c=basin_labels, cmap=cmap,
        vmin=-0.5, vmax=n_basins - 0.5,
        s=100, marker="^", alpha=1, edgecolors="white", linewidths=0.5,
        zorder=3, label="Start",
    )

    # arrows from start → converged
    # for i in range(len(conv_2d_combined)):
    #     ax.annotate(
    #         "", xy=conv_2d_combined[i], xytext=x0_2d[i],
    #         arrowprops=dict(arrowstyle="->", color="grey", alpha=0.3, lw=0.8),
    #         zorder=2,
    #     )

    # global best
    ax.scatter(
        conv_2d_combined[best, 0], conv_2d_combined[best, 1],
        marker="*", s=400, color="gold", zorder=5, label="Global best",
    )

    plt.colorbar(
        plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=-0.5, vmax=n_basins - 0.5)
        ),
        ax=ax, label="Basin ID", ticks=range(n_basins),
    )
    ax.set_xlabel("t-SNE dim 1", fontsize=12)
    ax.set_ylabel("t-SNE dim 2", fontsize=12)
    ax.set_title(
        f"Basin of Attraction Map\nTriangles=start  Circles=converged  Arrows=IPOPT path\n"
        f"(t-SNE fitted on both, perplexity={tsne_perplexity:.0f})",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save(fig, output_dir, "basin_map_combined.png")
 
    # ── Plot 2: objective surface ─────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    xi  = np.linspace(conv_2d[:, 0].min(), conv_2d[:, 0].max(), 200)
    yi  = np.linspace(conv_2d[:, 1].min(), conv_2d[:, 1].max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata(conv_2d, obj_vals, (Xi, Yi), method="cubic")
    cf = ax2.contourf(Xi, Yi, Zi, levels=40, cmap="viridis_r")
    ax2.scatter(conv_2d[:, 0], conv_2d[:, 1],
                c="white", s=8, alpha=0.35, edgecolors="none")
    ax2.scatter(conv_2d[best, 0], conv_2d[best, 1],
                marker="*", s=300, color="gold", zorder=10)
    plt.colorbar(cf, ax=ax2, label="Objective value")
    ax2.set_xlabel("t-SNE dim 1", fontsize=12)
    ax2.set_ylabel("t-SNE dim 2", fontsize=12)
    ax2.set_title("Objective Landscape\nt-SNE of converged solutions", fontsize=12)
    plt.tight_layout()
    _save(fig2, output_dir, "objective_surface.png")

    # ── Plot: v_free vs eta_high coloured by basin ────────────────────────────
    # Find which columns in conv_mat correspond to v_free and eta_high.
    # Layout is [eta_high_s0..sN, tau_s0..sN, K_s0..sN, rho_crit_s0..sN, v_free_s0..sN, a_s0..sN]
    eta_start   = opt.CALIB_PARAM_NAMES.index("eta_high") * num_segments
    v_free_start = opt.CALIB_PARAM_NAMES.index("v_free")  * num_segments

    # Average across segments so you get one scalar per trial
    eta_vals   = x0_good[:, eta_start  : eta_start   + num_segments].mean(axis=1)
    v_free_vals = x0_good[:, v_free_start : v_free_start + num_segments].mean(axis=1)

    fig_phys, ax_phys = plt.subplots(figsize=(8, 7))
    scatter = ax_phys.scatter(
        eta_vals, v_free_vals,
        c=basin_labels, cmap=cmap,
        vmin=-0.5, vmax=n_basins - 0.5,
        alpha=1, s=60,  marker="^", edgecolors="white", linewidths=0.4,
    )
    ax_phys.scatter(
        eta_vals[best], v_free_vals[best],
        marker="*", s=300, color="gold", zorder=10, label="Global best",
    )
    ax_phys.legend(fontsize=10)
    plt.colorbar(scatter, ax=ax_phys, label="Basin ID", ticks=range(n_basins))
    ax_phys.set_xlabel("η (eta_high) — averaged across segments", fontsize=12)
    ax_phys.set_ylabel("v_free — averaged across segments", fontsize=12)
    ax_phys.set_title("Basin Map\nPhysical parameter space (v_free vs η)", fontsize=12)
    plt.tight_layout()
    _save(fig_phys, output_dir, "basin_map_physical.png")
 
    # # ── Plot 3: per-basin objective boxplots ──────────────────────────────────
    # fig3, ax3 = plt.subplots(figsize=(max(6, n_basins * 1.5), 5))
    # basin_objs = [obj_vals[basin_labels == k] for k in range(n_basins)]
    # ax3.boxplot(basin_objs, labels=[f"Basin {k}" for k in range(n_basins)],
    #             patch_artist=True,
    #             boxprops=dict(facecolor="steelblue", alpha=0.6))
    # ax3.set_ylabel("Objective value", fontsize=12)
    # ax3.set_title("Objective distribution per basin", fontsize=13)
    # ax3.grid(True, axis="y", alpha=0.4)
    # plt.tight_layout()
    # _save(fig3, output_dir, "basin_objectives.png")
 
    # # ── Plot 4: parameter loadings ────────────────────────────────────────────
    # # t-SNE axes have no physical meaning, so we run a quick PCA on the
    # # converged solutions purely to get interpretable parameter loadings.
    # from sklearn.decomposition import PCA
    # pca_conv     = PCA(n_components=2).fit(conv_norm)
    # param_labels = [f"{p}[{i}]"
    #                 for p in opt.CALIB_PARAM_NAMES
    #                 for i in range(num_segments)]
    # fig4, axes4 = plt.subplots(1, 2, figsize=(14, 4))
    # for pc_idx, ax in enumerate(axes4):
    #     loadings = pca_conv.components_[pc_idx]
    #     colors   = ["#e05c5c" if v < 0 else "#5c9ee0" for v in loadings]
    #     ax.bar(range(len(param_labels)), loadings, color=colors, edgecolor="white")
    #     ax.axhline(0, color="black", linewidth=0.8)
    #     ax.set_xticks(range(len(param_labels)))
    #     ax.set_xticklabels(param_labels, rotation=45, ha="right", fontsize=8)
    #     ax.set_ylabel("Loading", fontsize=10)
    #     ax.set_title(
    #         f"PC{pc_idx+1} of converged solutions  "
    #         f"({pca_conv.explained_variance_ratio_[pc_idx]*100:.1f}% var)\n"
    #         f"(interpretation only — basin map uses t-SNE)",
    #         fontsize=10)
    # plt.tight_layout()
    # _save(fig4, output_dir, "pca_loadings.png")
 
    return dict(
        tsne=tsne, scaler=scaler,
        conv_2d=conv_2d, x0_good=x0_good,
        conv_mat=conv_mat, basin_labels=basin_labels,
        obj_vals=obj_vals, ids=ids, n_basins=n_basins,
    )

def get_best_params(output_dir: str = "basin_results") -> dict:
    """
    Load trials from disk and return the parameter dict of the best converged solution.

    Returns
    -------
    dict with keys: trial_id, obj_val, termination, and one key per param in
    CALIB_PARAM_NAMES each mapping to an array of length num_segments.
    """
    with open(os.path.join(output_dir, "trials.pkl"), "rb") as f:
        data = pickle.load(f)

    trials = data["trials"]

    # filter to converged only
    good = [t for t in trials if t["converged"] and t["conv_vec"] is not None]
    if len(good) == 0:
        raise ValueError("No converged trials found in results.")

    # find best
    best = min(good, key=lambda t: t["obj_val"])

    # unpack conv_vec back into named parameters
    conv_vec    = np.array(best["conv_vec"], dtype=np.float64)
    num_segments = len(conv_vec) // len(opt.CALIB_PARAM_NAMES)

    params = {
        pname: conv_vec[j * num_segments:(j + 1) * num_segments]
        for j, pname in enumerate(opt.CALIB_PARAM_NAMES)
    }

    params["obj_val"]     = best["obj_val"]
    params["trial_id"]    = best["trial_id"]
    params["termination"] = best["termination"]

    print(f"Best trial: {best['trial_id']}  obj={best['obj_val']:.6f}  [{best['termination']}]")
    for pname in opt.CALIB_PARAM_NAMES:
        print(f"  {pname}: {params[pname]}")

    return params

def jump_and_optimize(
    best_params: dict,
    calib_kwargs: dict,
    num_segments: int,
    k: int = 20,
    perturb_pct: float = 10.0,
    distance_threshold: float = 0.05,
    n_workers: int = 4,
    seed: int = 0,
    output_dir: str = "basin_results",
) -> list[dict]:
    """
    Perturb the best known solution k times, re-optimise from each perturbation,
    and check whether the solver returns to the best solution.
    Also runs one unperturbed baseline solve (k+1 total).

    Parameters
    ----------
    best_params       : dict returned by get_best_params()
    perturb_pct       : perturbation magnitude as % of each parameter's bound width
    distance_threshold: used to decide whether a re-run "recovered" the best solution

    Returns
    -------
    List of k+1 result dicts (baseline first, then k perturbed).
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ── best solution as a flat vector ────────────────────────────────────────
    best_vec = np.concatenate([
        np.atleast_1d(best_params[p])[:num_segments]
        for p in opt.CALIB_PARAM_NAMES
    ]).astype(np.float64)

    lbs = np.array([opt.CALIB_PARAM_BOUNDS[p][0]
                    for p in opt.CALIB_PARAM_NAMES
                    for _ in range(num_segments)])
    ubs = np.array([opt.CALIB_PARAM_BOUNDS[p][1]
                    for p in opt.CALIB_PARAM_NAMES
                    for _ in range(num_segments)])
    bound_widths = ubs - lbs

    # ── k perturbed starting points ───────────────────────────────────────────
    perturbed_starts = []
    for _ in range(k):
        if perturb_pct == 0.0:
            perturbed_starts.append(best_vec.copy())
        else:
            noise     = rng.uniform(-1, 1, size=len(best_vec))
            delta     = noise * (perturb_pct / 100.0) * bound_widths
            perturbed = np.clip(best_vec + delta, lbs, ubs)
            perturbed_starts.append(perturbed)

    # ── run baseline (unperturbed) first, then k perturbed in parallel ────────
    print(f"[jump] Running 1 unperturbed baseline + {k} perturbed solves "
          f"(perturb_pct={perturb_pct}%, n_workers={n_workers}) ...")
    
    print(np.isclose(best_vec, perturbed_starts[0]))

    baseline_trial = _run_one_trial(0, best_vec, num_segments, calib_kwargs)
    baseline_trial["is_baseline"] = True
    print(f"[jump] Baseline obj={baseline_trial['obj_val']:.6f}  "
          f"[{baseline_trial['termination']}]")

    perturbed_trials = Parallel(n_jobs=n_workers, verbose=5)(
        delayed(_run_one_trial)(i + 1, perturbed_starts[i], num_segments, calib_kwargs)
        for i in range(k)
    )
    for t in perturbed_trials:
        t["is_baseline"] = False

    jump_trials = [baseline_trial] + perturbed_trials

    for i in range(1, k + 1):
        t = jump_trials[i]
        print(f"[jump] Trial {t['trial_id']:2d}  obj={t['obj_val']:.6f}  "
              f"[{t['termination']}]  "
              f"perturbed={'yes' if not t['is_baseline'] else 'no'}")

    # ── check recovery ────────────────────────────────────────────────────────
    all_conv = np.vstack(
        [t["conv_vec"] for t in jump_trials if t["conv_vec"] is not None] + [best_vec]
    ).astype(np.float64)
    scaler = StandardScaler()
    scaler.fit(all_conv)

    best_vec_norm = scaler.transform(best_vec.reshape(1, -1))

    for t in jump_trials:
        if t["conv_vec"] is not None:
            conv_norm         = scaler.transform(
                np.array(t["conv_vec"], dtype=np.float64).reshape(1, -1)
            )
            dist              = np.linalg.norm(conv_norm - best_vec_norm)
            t["dist_to_best"] = dist
            t["recovered"]    = dist < distance_threshold
        else:
            t["dist_to_best"] = np.nan
            t["recovered"]    = False

    n_converged = sum(t["converged"] for t in jump_trials)
    n_recovered = sum(t["recovered"] for t in jump_trials)
    print(f"\n[jump] Converged: {n_converged}/{k+1}  |  "
          f"Recovered best: {n_recovered}/{k+1} "
          f"({100*n_recovered/(k+1):.1f}%)")

    if not baseline_trial["recovered"]:
        print("[jump] WARNING: unperturbed baseline did not recover the best solution.")

    # ── save ──────────────────────────────────────────────────────────────────
    with open(os.path.join(output_dir, "jump_trials.pkl"), "wb") as f:
        pickle.dump({"jump_trials": jump_trials, "best_vec": best_vec,
                     "perturb_pct": perturb_pct}, f)

    # ── separate baseline from perturbed for plotting ─────────────────────────
    good_jumps  = [t for t in perturbed_trials if t["conv_vec"] is not None]
    baseline_ok = baseline_trial["conv_vec"] is not None

    if len(good_jumps) < 1:
        print("[jump] Too few converged perturbed trials to plot.")
        return jump_trials

    perturb_vecs = np.vstack([perturbed_starts[t["trial_id"] - 1] for t in good_jumps])
    conv_vecs    = np.vstack([t["conv_vec"] for t in good_jumps]).astype(np.float64)
    recovered    = np.array([t["recovered"]    for t in good_jumps])
    dist_to_best = np.array([t["dist_to_best"] for t in good_jumps])
    obj_vals     = np.array([t["obj_val"]       for t in good_jumps])
    colors       = np.where(recovered, "tab:green", "tab:red")
    M            = len(good_jumps)

    # ── parameter indices ─────────────────────────────────────────────────────
    eta_start    = opt.CALIB_PARAM_NAMES.index("eta_high") * num_segments
    v_free_start = opt.CALIB_PARAM_NAMES.index("v_free")   * num_segments

    # perturbed starting point coords
    eta_start_vals   = perturb_vecs[:, eta_start   : eta_start   + num_segments].mean(axis=1)
    vfree_start_vals = perturb_vecs[:, v_free_start : v_free_start + num_segments].mean(axis=1)

    # converged endpoint coords
    eta_conv_vals    = conv_vecs[:, eta_start   : eta_start   + num_segments].mean(axis=1)
    vfree_conv_vals  = conv_vecs[:, v_free_start : v_free_start + num_segments].mean(axis=1)

    # best solution coords
    eta_best   = best_vec[eta_start   : eta_start   + num_segments].mean()
    vfree_best = best_vec[v_free_start : v_free_start + num_segments].mean()

    # baseline converged coords
    if baseline_ok:
        baseline_vec        = np.array(baseline_trial["conv_vec"], dtype=np.float64)
        eta_baseline_conv   = baseline_vec[eta_start   : eta_start   + num_segments].mean()
        vfree_baseline_conv = baseline_vec[v_free_start : v_free_start + num_segments].mean()

    # ── Plot 1: recovery map in physical parameter space ──────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 8))

    # perturbed starts — triangles
    ax1.scatter(eta_start_vals, vfree_start_vals,
                c=colors, s=80, marker="^", alpha=0.6,
                edgecolors="white", linewidths=0.5, zorder=3)

    # converged endpoints — circles
    ax1.scatter(eta_conv_vals, vfree_conv_vals,
                c=colors, s=80, marker="o",
                edgecolors="white", linewidths=0.5, zorder=3)

    # arrows from perturbed start → converged endpoint
    for i in range(M):
        ax1.annotate(
            "", xy=(eta_conv_vals[i], vfree_conv_vals[i]),
                xytext=(eta_start_vals[i], vfree_start_vals[i]),
            arrowprops=dict(arrowstyle="->",
                            color="green" if recovered[i] else "red",
                            alpha=0.4, lw=0.8),
            zorder=2,
        )

    # baseline — diamond
    # if baseline_ok:
    #     ax1.scatter(eta_baseline_conv, vfree_baseline_conv,
    #                 marker="D", s=200,
    #                 color="tab:green" if baseline_trial["recovered"] else "tab:red",
    #                 edgecolors="black", linewidths=1.0, zorder=6,
    #                 label=f"Baseline (unperturbed)  obj={baseline_trial['obj_val']:.4f}")

    # best solution — star
    ax1.scatter(eta_best, vfree_best,
                marker="*", s=500, color="gold",
                zorder=7, label="Best solution")

    # legend proxies
    ax1.scatter([], [], marker="o", color="tab:green",
                label=f"Recovered ({n_recovered}/{k+1})")
    ax1.scatter([], [], marker="o", color="tab:red",
                label=f"Did not recover ({k+1-n_recovered}/{k+1})")
    ax1.scatter([], [], marker="^", color="grey", alpha=0.6, label="Perturbed start")

    ax1.legend(fontsize=9)
    ax1.set_xlabel("η (eta_high) — averaged across segments", fontsize=12)
    ax1.set_ylabel("v_free — averaged across segments", fontsize=12)
    ax1.set_title(
        f"Jump & Re-optimise: Recovery Map\n"
        f"Perturbation={perturb_pct}%  |  "
        f"Recovered {n_recovered}/{k+1}  (◆ = unperturbed baseline)",
        fontsize=12,
    )
    plt.tight_layout()
    _save(fig1, output_dir, "jump_recovery_map.png")

    # ── Plot 2: distance to best vs objective ─────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.scatter(dist_to_best, obj_vals, c=colors, s=60,
                edgecolors="white", linewidths=0.5, zorder=3)
    if baseline_ok:
        ax2.scatter(baseline_trial["dist_to_best"], baseline_trial["obj_val"],
                    marker="D", s=150,
                    color="tab:green" if baseline_trial["recovered"] else "tab:red",
                    edgecolors="black", linewidths=1.0,
                    zorder=5, label="Baseline (unperturbed)")
    ax2.axvline(distance_threshold, color="black", linestyle="--",
                label=f"Recovery threshold ({distance_threshold})")
    ax2.axhline(best_params["obj_val"], color="gold", linestyle="--",
                label=f"Best obj ({best_params['obj_val']:.4f})")
    ax2.set_xlabel("Distance to best solution (normalised)", fontsize=12)
    ax2.set_ylabel("Converged objective value", fontsize=12)
    ax2.set_title("Distance to Best vs Objective\n"
                  "Green = recovered  Red = different local minimum", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig2, output_dir, "jump_distance_vs_obj.png")

    # ── Plot 3: per-parameter deviation from best ─────────────────────────────
    param_labels = [f"{p}[{i}]"
                    for p in opt.CALIB_PARAM_NAMES
                    for i in range(num_segments)]
    deviations   = np.abs(conv_vecs - best_vec) / (ubs - lbs)

    fig3, ax3 = plt.subplots(figsize=(14, 5))
    ax3.boxplot(deviations, labels=param_labels, patch_artist=True,
                boxprops=dict(facecolor="steelblue", alpha=0.6))
    ax3.set_ylabel("Absolute deviation from best\n(normalised by bound width)", fontsize=11)
    ax3.set_title("Per-parameter Deviation of Re-optimised Solutions from Best", fontsize=12)
    ax3.set_xticklabels(param_labels, rotation=45, ha="right", fontsize=8)
    ax3.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig3, output_dir, "jump_param_deviations.png")

    return jump_trials

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — continuation sweep along PC1
# ─────────────────────────────────────────────────────────────────────────────

def run_continuation_sweep(
    landscape: dict,
    calib_kwargs: dict,
    num_segments: int,
    n_steps: int    = 50,
    pc_axis: int    = 0,        # 0 = PC1, 1 = PC2
    output_dir: str = "basin_results",
) -> list[dict]:
    """
    Phase 2: walk along the chosen PC axis in steps, using each converged
    solution as the warm-start for the next step.

    This is the "continuation / homotopy" strategy — far cheaper per solve
    than Phase 1 because IPOPT starts very close to a known solution.

    Parameters
    ----------
    landscape : dict returned by analyse_and_plot
    n_steps   : number of equally-spaced steps along the PC axis
    pc_axis   : which principal component to walk along (0=PC1, 1=PC2)
    """
    pca      = landscape["pca"]
    scaler   = landscape["scaler"]
    x0_2d    = landscape["x0_2d"]
    x0_good  = landscape["x0_good"]
    obj_vals = landscape["obj_vals"]

    # ── find the starting-point with the smallest x0_2d on the chosen axis ───
    axis_vals = x0_2d[:, pc_axis]
    sorted_idx = np.argsort(axis_vals)        # left → right along axis

    # Evenly sub-sample n_steps points from the sorted order
    step_indices = np.round(
        np.linspace(0, len(sorted_idx) - 1, n_steps)
    ).astype(int)
    step_indices = sorted_idx[step_indices]

    print(f"\n[continuation] {n_steps} steps along PC{pc_axis+1}")
    path_trials = []
    prev_conv_vec = None

    for k, idx in enumerate(step_indices):
        x0_vec = x0_good[idx].copy()

        # If we have a previous converged solution, blend it with the new x0.
        # Using 100 % of the previous solution is the true continuation; using
        # the LHS point from Phase 1 is safer if the basin switches.
        if prev_conv_vec is not None:
            x0_vec = prev_conv_vec.copy()        # pure continuation

        x0 = vec_to_x0(x0_vec, num_segments)
        print(f"  step {k+1:3d}/{n_steps}  PC{pc_axis+1}={x0_2d[idx, pc_axis]:.3f}",
              end="  ", flush=True)
        try:
            results = opt.run_calibration(x0=x0, **calib_kwargs)
            conv_vec = results_to_conv_vec(results, num_segments)
            obj      = results["obj_val"]
            tc       = results.get("termination_condition", "?")
            print(f"obj={obj:.5f}  [{tc}]")
            prev_conv_vec = conv_vec
            path_trials.append({
                "step": k, "pc_val": x0_2d[idx, pc_axis],
                "x0_vec": x0_vec, "conv_vec": conv_vec,
                "obj_val": obj, "termination": tc,
            })
        except Exception as exc:
            print(f"ERROR: {exc}")
            prev_conv_vec = None    # restart from LHS on next step
            path_trials.append({
                "step": k, "pc_val": x0_2d[idx, pc_axis],
                "x0_vec": x0_vec, "conv_vec": None,
                "obj_val": np.nan, "termination": str(exc),
            })

    # ── plot continuation path ────────────────────────────────────────────────
    pc_vals  = [t["pc_val"]  for t in path_trials]
    obj_path = [t["obj_val"] for t in path_trials]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(pc_vals, obj_path, "o-", markersize=5, color="steelblue")
    ax.set_xlabel(f"PC{pc_axis+1} coordinate", fontsize=12)
    ax.set_ylabel("Converged objective", fontsize=12)
    ax.set_title(f"Continuation sweep along PC{pc_axis+1}", fontsize=13)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    _save(fig, output_dir, f"continuation_pc{pc_axis+1}.png")

    with open(os.path.join(output_dir, "continuation_trials.pkl"), "wb") as f:
        pickle.dump(path_trials, f)

    return path_trials


# ─────────────────────────────────────────────────────────────────────────────
# Small utilities
# ─────────────────────────────────────────────────────────────────────────────

def _label_axes(ax, pca, title=""):
    ax.set_xlabel(f"PC1  ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2  ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=11)
    ax.set_title(title, fontsize=12)


def _save(fig, directory, filename):
    path = os.path.join(directory, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[basin] saved → {path}")
    plt.show()


def load_results(output_dir: str = "basin_results") -> tuple[list, np.ndarray]:
    """Reload a previous run from disk."""
    with open(os.path.join(output_dir, "trials.pkl"), "rb") as f:
        data = pickle.load(f)
    return data["trials"], data["x0_matrix"]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    num_seg = CALIB_KWARGS["num_calibrated_segments"]

    # ── Phase 1: global basin map ─────────────────────────────────────────────
    # trials = run_basin_study(
    #     calib_kwargs=CALIB_KWARGS,
    #     num_segments=num_seg,
    #     N=N_TRIALS,
    #     n_clusters=N_CLUSTERS,
    #     n_workers=N_WORKERS,
    #     seed=SEED,
    #     output_dir=OUTPUT_DIR,
    # )

    best_params = get_best_params(OUTPUT_DIR)

    jump_and_optimize(
        best_params,
        calib_kwargs=CALIB_KWARGS,
        num_segments=num_seg,
        k=2,
        perturb_pct=PERTURB_PCT,
        distance_threshold=DISTANCE_THRESHOLD,
        n_workers=N_WORKERS,
        seed=SEED,
        output_dir=OUTPUT_DIR,
    )
    # Reload and re-analyse any time (without re-running IPOPT):
    # trials, x0_matrix = load_results(OUTPUT_DIR)
    # landscape = analyse_and_plot(trials, x0_matrix, num_seg, output_dir=OUTPUT_DIR, 
    #                              distance_threshold=DISTANCE_THRESHOLD)

    # ── Phase 2: continuation sweep along PC1 ────────────────────────────────
    # (comment out if you only want Phase 1)
    # trials, x0_matrix = load_results(OUTPUT_DIR)
    # landscape = analyse_and_plot(trials, x0_matrix, num_seg, OUTPUT_DIR, DISTANCE_THRESHOLD)
    # run_continuation_sweep(landscape, CALIB_KWARGS, num_seg,
    #                        n_steps=60, pc_axis=0, output_dir=OUTPUT_DIR)
