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
simulator's existing non-negative clipping. Every calibration first passes an
undisturbed 15-minute validity gate. The dynamic case is deliberately
**frozen**: it replays the previously calibrated time-varying parameters but
does not recalibrate in response to an AST disturbance.

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

Run the focused unit tests with:

```bash
python -m unittest discover -s tests -v
```
