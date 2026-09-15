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
