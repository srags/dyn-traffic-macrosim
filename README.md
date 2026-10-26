# Dynamic Calibration for Traffic Macrosimulation
This repo contains the code, data and results for paper [Dynamic Gradient-Based Calibration for Robust and Accurate Traffic Macrosimulation](https://arxiv.org/pdf/2605.19056v2). This paper is published at 2026 IEEE Intelligent Systems Transportation Conference. Please cite the work when using this repo.

```bibtex
@misc{raghavan2026dynamic,
  title         = {Dynamic Gradient-Based Calibration for Robust and Accurate Traffic Macrosimulation},
  author        = {Raghavan, Shreyaa and Hickert, Cameron and Chan, Monica and Wu, Cathy},
  year          = {2026},
  eprint        = {2605.19056},
  archivePrefix = {arXiv},
  primaryClass  = {eess.SY},
  url           = {https://arxiv.org/abs/2605.19056}
}
```

## Environment setup
Tested on macOS with Python 3.14 and IPOPT 3.14.16.

1. Install the IPOPT solver (used through Pyomo; not pip-installable):
   ```bash
   brew install ipopt
   ```
   or `conda install -c conda-forge ipopt=3.14.16`. Check that `ipopt --version` works.
2. Install Python 3.14 if `python3.14` isn't available.
3. Create the environment and install the requirements:
   ```bash
   python3.14 -m venv calib_env
   source calib_env/bin/activate
   pip install -r requirements.txt
   ```
4. Register the environment as a Jupyter kernel, then select `calib_env` as the kernel in the notebooks:
   ```bash
   python -m ipykernel install --user --name calib_env --display-name calib_env
   ```

## Data
Download the raw I-24 MOTION trajectory data from the official [I-24 MOTION website](https://i24motion.org/) and place the files in the top-level `raw_i24_data/` folder. `experiments/DataProcessing.ipynb` reads the raw trajectories from there.

## Running Dynamic Calibration
The code for dynamically calibrating to the I-24 Data can be found in `experiments/Rolling_calibration.ipynb`, with the core simulation functions in `src/simulation_utils.py` and the calibration functions in `src/optimization_utils.py` and `src/mpc_utils.py`. Code to replicate all the figures in the paper and more can be found in `experiments/plotting.ipynb` and `src/plotting.py`.

## Adaptive Stress Testing proof of concept

`experiments/ast_poc.py` runs three small searches against both the static
calibration saved by `experiments/Optimization.ipynb` and the saved rolling
calibration schedule (`control_h_6`, updated once per minute):

1. Standard AST over a synthetic probability model for VSL commands.
2. Minimum-intervention falsification over the same VSL commands.
3. Standard AST over discretized Gaussian upstream-inflow perturbations.

The initial failure event is physical or numerical invalidity: a non-finite
state, a negative raw density or velocity update, density above 190
veh/km/lane, or velocity above 250 km/h. Raw values are checked before the
simulator's existing non-negative clipping. The limits are adjustable with
`--max-density`, `--max-velocity`, `--min-density` and `--min-velocity` (in
both `ast_poc.py` and `ast_sweep.py`). The minimums default to 0; a positive
minimum (e.g. `--min-velocity 5`) makes the check stricter, and a negative one
(e.g. `--min-velocity -1`) tolerates small undershoots. Failures below a
non-zero minimum are reported as `low_raw_velocity` / `low_raw_density`
instead of `negative_raw_*`. Every calibration first passes an
undisturbed validity gate over the window. The dynamic case is deliberately
**frozen**: it replays the previously calibrated time-varying parameters but
does not recalibrate in response to an AST disturbance.

The VSL action grid defaults to speeds from 40 through 75 mph in five-mph
increments, plus an unrestricted action. Values are converted to km/h
before being passed to METANET. Smaller reductions from 75 mph receive higher
nominal probability, and minimum-intervention cost increases by one point per
mph of additional reduction. The bounds, step, and probability-decay scale are
available as command-line options.

Run all three modes from the repository root:

```bash
conda activate ast
python experiments/ast_poc.py --mode all --iterations 2000
```

For a quick smoke test, select one mode and a smaller search budget:

```bash
python experiments/ast_poc.py --mode inflow-ast --iterations 50
```

The default `--calibration-mode both` produces the paired comparison. Use
`--calibration-mode static` or `--calibration-mode dynamic` to run only one,
and use `--dynamic-control-horizon` to select another saved dynamic schedule.

`--calibration-dir` accepts several static calibration folders; each is run
separately and shown side by side in the plots. Pass
`--calibration-step-seconds` with the time step each folder was calibrated with
(one value for all, or one per folder; it must divide the 10 s data step, and
shorter steps repeat each observation as `Optimization.ipynb` does). Labels
default to folder names; override them with `--calibration-labels`. If a
calibration's undisturbed run already fails, it is recorded and its searches
are skipped.

```bash
python experiments/ast_poc.py --mode vsl-ast --calibration-mode static \
  --window-minutes 60 --action-minutes 5 --iterations 100 \
  --calibration-dir calibration/i24/i24_11_30/calibration_static/fixed_ramping \
                    calibration/i24/i24_11_30/calibration_static/fixed_ramping_T=5 \
  --calibration-step-seconds 10 5 \
  --json-output artifacts/ast_poc/multi.json --time-output artifacts/ast_poc/multi_time.png
```

The script only prints results by default. Pass `--json-output PATH` to save a
machine-readable summary. Three optional figure paths save standalone bar
charts for time to failure and raw boundary severity, plus the standalone
results table. The plots place static and frozen-dynamic outcomes alongside
each other; each output supports PNG, PDF, or SVG:

```bash
python experiments/ast_poc.py --mode all --iterations 2000 \
  --json-output artifacts/ast_poc/results.json \
  --time-output artifacts/ast_poc/time_to_first_invalid_state.png \
  --severity-output artifacts/ast_poc/boundary_severity.png \
  --table-output artifacts/ast_poc/discovered_sequences.png
```

The plotting implementation is in `src/ast_plotting.py`. Saved results can be
rendered again without rerunning MCTS:

```bash
python experiments/plot_ast_results.py artifacts/ast_poc/results.json \
  --time-output artifacts/ast_poc/time_to_first_invalid_state.png \
  --severity-output artifacts/ast_poc/boundary_severity.png \
  --table-output artifacts/ast_poc/discovered_sequences.png
```

Run `python experiments/ast_poc.py --help` for the configurable window,
thresholds, noise scale, MCTS settings, and input paths.

The proof-of-concept VSL values, probabilities, and intervention costs are
explicit modeling assumptions pending domain review. Online recalibration,
deferred failure modes, and other extensions are recorded in
`AST_POSSIBLE_EXTENSIONS.md`.

### Parameter sweep

`experiments/ast_sweep.py` repeats the static `vsl-ast` search (default: 15-minute
window, 1-minute actions, 100 MCTS iterations) over sampled values of `tau`,
`eta_high`, `K`, `p_crit`, `v_free` and `a`. `--sampling` picks how:

- `uniform` (default): one value per parameter, used on every segment, drawn
  uniformly from the calibration bounds pulled in by 5% of each range
  (`--inset`).
- `calibration`: around the calibration in `--calibration-dir`. Each sample
  draws one factor per parameter from U[1 − `--spread`, 1 + `--spread`]
  (default 0.1) and multiplies that parameter's calibrated value on every
  segment, so the differences between segments are kept. Values pushed past a
  calibration bound are clipped to it (no inset); `clipped_values` counts how
  many of the 84 segment values were clipped. Sample 0 is the unperturbed
  calibration.

Ramp terms keep their calibrated per-segment values in both modes.

Each sample first simulates the undisturbed trajectory over the whole window and
records whether (and when) it fails, its distance to failure, and its velocity
and density MAPE against the I-24 data. `--velocity-threshold` and
`--density-threshold` (in %) each discard samples whose MAPE is above them; set
either, both or neither (the default, which discards nothing). Discarded samples
still count towards `--n-samples`, and `discard_reason` records which MAPE was
too high. Kept samples with a valid baseline then get the VSL search. Each row's
`outcome` is `discarded`, `failed_nominal`, `failed_vsl`, `safe` or `error`.

Results are kept in memory and the outcome table is printed; nothing is
resumable. Figures are always saved, using `--fig-output` as a filename prefix
(default `artifacts/ast_sweep/sweep`; a prefix ending in `/` saves into that
folder): `<prefix>_outcomes.png` (discarded / failed / safe counts),
`<prefix>_ttf_vs_velocity_mape.png` and `<prefix>_ttf_vs_density_mape.png`
(time to failure vs. each MAPE for samples tested under VSL), and
`<prefix>_best_safe_time_space.png` (the safe sample with the lowest average
MAPE vs. the data; with `--sampling calibration`, sample 0, the unperturbed
centre, is left out).

`--json-output PATH` also saves a JSON laid out like `ast_poc.py`'s: `scenario`,
`action_models`, `sampling`, `outcomes`, a compact `samples` list (parameters or
factors, outcome, time to failure, both MAPEs) and `highlights` (the best safe
sample and the five fastest VSL failures, rerun to record their `nominal` and
`searches` in the same format as `ast_poc.py`). Without it, no JSON is written.

```bash
python experiments/ast_sweep.py --n-samples 200 --fig-output artifacts/ast_sweep/test_200
python experiments/ast_sweep.py --n-samples 10000 --velocity-threshold 70 --density-threshold 60 \
  --fig-output artifacts/ast_sweep/uniform --json-output artifacts/ast_sweep/uniform.json
python experiments/ast_sweep.py --sampling calibration --spread 0.1 \
  --calibration-dir calibration/i24/i24_11_30/calibration_static/fixed_ramping \
  --fig-output artifacts/ast_sweep/around_fixed_ramping
```

### Synthetic data

Both `ast_poc.py` and `ast_sweep.py` accept `--synthetic`. It reads the data
and boundary conditions from `--data-dir` (default `data/synthetic`) with
separate boundary conditions: every segment in `rho_hat`/`v_hat` is simulated,
the upstream inflow comes from `inflow.npy` and the downstream density from
`downstream_density.npy` (both as saved, unsmoothed). The parameters are the defaults from `param_loader.METANET_Params` (no ramps).

```bash
python experiments/ast_poc.py --synthetic --mode vsl-ast --window-minutes 60 --action-minutes 5
python experiments/ast_sweep.py --synthetic --sampling calibration --spread 0.1 \
  --fig-output artifacts/ast_sweep/synthetic/near_default \
  --json-output artifacts/ast_sweep/synthetic/near_default.json
```

Run the focused unit tests with:

```bash
python -m unittest discover -s tests -v
```
