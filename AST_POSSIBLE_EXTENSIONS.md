# Adaptive Stress Testing: Possible Extensions

This file records ideas that are intentionally outside the initial proof of
concept. The first implementation remains small and understandable: static
parameters and a frozen dynamic parameter schedule, a 15-minute simulation
window, actions updated once per minute, physical/numerical invalidity as the
failure event, and a simple tree-search solver.

## Additional failure definitions

### Loss of predictive accuracy

Treat the model as failed when its simulated trajectory diverges too far from
observed traffic data. Keep this interchangeable with the initial
physical/numerical-validity detector by giving every failure definition the
same small interface: a Boolean event flag, a continuous distance-to-failure,
and diagnostic details.

Candidate error measures include velocity MAE or RMSE, a joint velocity and
density error, and the worst rolling-window error. Clarify whether a disturbed
inflow represents measurement corruption or a real change in physical demand
before comparing the resulting trajectory with unchanged observations.

### Unphysical oscillations

Add oscillation as its own failure subtype rather than mixing it into the
initial validity checks. Possible definitions include excessive one-step
velocity change, excessive normalized second difference, or repeated large
sign reversals in velocity or density changes. Thresholds should be justified
in physical units and checked against undisturbed trajectories before use.

### Operational traffic failures

Possible events include severe congestion, excessive density, excessive
origin queue, spillback, excessive total travel time, or prolonged low speed.
These are appropriate when disturbances represent real traffic conditions
rather than corrupted model inputs.

## Dynamic calibration

### Frozen dynamic parameter schedule

Implemented in the proof of concept: load the parameter sequence produced by
rolling calibration and treat it as a fixed time-varying schedule during each
AST rollout. This provides a direct comparison with static calibration because
only the parameter source changes; the AST state, actions, transition wrapper,
failure detector, and solver remain the same.

### Online recalibration under stress

Rerun rolling calibration at every calibration boundary inside each AST
episode, using the disturbed inputs or measurements. This tests the dynamic
calibration procedure itself rather than a previously saved parameter
schedule. It is substantially more expensive because each AST rollout then
contains repeated IPOPT solves.

## Distance-to-failure and parameter landscapes

Record more than a binary failure result. Useful robustness summaries include:

- The log-likelihood or action cost of the least stressful failure found.
- The number of action intervals and simulated time until first failure.
- The state-space margin at the closest non-failing point.
- The segment and failure subtype responsible for the first violation.
- The minimum disturbance magnitude needed to reach failure.

These quantities could define an empirical safety score for a calibrated
parameter set. Repeating AST over many parameter sets could then produce a
parameter-safety landscape and help identify parameter regions that are both
accurate and far from failure. Comparisons must use the same action model,
failure thresholds, horizon, and search budget.

## VSL and controller extensions

- **Before treating the proof-of-concept VSL setup as final, check with
  Shreyaa:** review the proposed VSL actions (`unrestricted`, 120, 100, 80,
  and 60 km/h), their nominal probability distribution, and their
  minimum-intervention costs. The initial values are modeling assumptions for
  a simple experiment, not empirically validated operational frequencies.

  | VSL action | Nominal probability | Intervention cost |
  | --- | ---: | ---: |
  | Unrestricted | 0.50 | 0 |
  | 120 km/h | 0.25 | 1 |
  | 100 km/h | 0.15 | 2 |
  | 80 km/h | 0.07 | 3 |
  | 60 km/h | 0.03 | 4 |

- Stress test faulty, delayed, missing, or adversarial VSL commands.
- Add an actual VSL controller and let AST control environmental disturbances
  while the controller chooses the VSL actions.
- Allow spatially varying VSL values after first testing one segment or one
  fixed segment group.
- Add limits on VSL switching frequency and magnitude.

## Additional disturbances

Later action spaces could include downstream boundary-density error, ramp-flow
error, demand surges, lane-count or lane-closure changes, initial-state error,
sensor dropout, and correlated or temporally smooth noise. Test one disturbance
family at a time before combining them.

## Longer horizons and alternative solvers

- Extend from the initial 15-minute window to the full observation horizon.
- Compare action update intervals other than one minute.
- Compare Monte Carlo tree search with cross-entropy search or other
  trajectory-optimization methods.
- Explore reinforcement-learning solvers only after the simulator wrapper,
  event definitions, likelihoods, and diagnostics have been validated.
- Compare every adaptive solver against random sampling with an equal number
  of simulator calls.
