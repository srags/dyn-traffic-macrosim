"""Focused tests for the AST proof-of-concept and simulator instrumentation."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ast_utils import (  # noqa: E402
    ASTEnvironment,
    MCTSConfig,
    PhysicalFailureConfig,
    SearchObjective,
    TrafficState,
    VSLAction,
    VSLActionModel,
    assess_physical_failure,
    run_mcts,
)
from simulation_utils import calculate_V, metanet_step  # noqa: E402


def simple_params(*, tau: float = 18 / 3600) -> dict:
    """Return a one-segment parameter set suitable for deterministic tests."""

    return {
        "beta": np.array([0.0]),
        "r": np.array([0.0]),
        "eta_high": np.array([0.0]),
        "K": np.array([40.0]),
        "tau": np.array([tau]),
        "a": np.array([1.4]),
        "p_crit": np.array([40.0]),
        "v_free": np.array([110.0]),
        "q_capacity": np.array([2200.0]),
    }


class SimulatorDiagnosticsTests(unittest.TestCase):
    def test_v_ctrl_caps_desired_speed(self) -> None:
        unrestricted = calculate_V(
            rho=1.0, v_ctrl=1000.0, a=1.4, p_crit=40.0, v_free=110.0
        )
        controlled = calculate_V(
            rho=1.0, v_ctrl=60.0, a=1.4, p_crit=40.0, v_free=110.0
        )

        self.assertGreater(unrestricted, 60.0)
        self.assertEqual(controlled, 60.0)

    def test_metanet_step_reports_raw_density_before_clipping(self) -> None:
        result = metanet_step(
            0,
            density_t=np.array([1.0]),
            velocity_t=np.array([100.0]),
            queue_t=0.0,
            flow_origin_t=0.0,
            T=1.0,
            l=0.4,
            vsl_speeds=np.array([[1000.0]]),
            demand=np.array([0.0]),
            downstream_density=np.array([1.0]),
            params=simple_params(),
            lanes={0: 1},
            real_data=True,
            return_diagnostics=True,
        )
        density, _velocity, _queue, _origin, _flow, diagnostics = result

        self.assertLess(diagnostics["raw_density"][0], 0.0)
        self.assertEqual(density[0], 0.0)
        self.assertTrue(diagnostics["density_clipped"][0])

    def test_vsl_changes_the_actual_velocity_transition(self) -> None:
        common = dict(
            t=0,
            density_t=np.array([1.0]),
            velocity_t=np.array([100.0]),
            queue_t=0.0,
            flow_origin_t=100.0,
            T=10 / 3600,
            l=0.4,
            demand=np.array([100.0]),
            downstream_density=np.array([1.0]),
            params=simple_params(),
            lanes={0: 1},
            real_data=True,
        )
        unrestricted = metanet_step(
            vsl_speeds=np.array([[1000.0]]), **common
        )[1]
        controlled = metanet_step(vsl_speeds=np.array([[60.0]]), **common)[1]

        self.assertLess(controlled[0], unrestricted[0])

    def test_failure_detector_applies_absolute_upper_limits(self) -> None:
        state = TrafficState(
            density=np.array([191.0]),
            velocity=np.array([100.0]),
            flow_origin=0.0,
            queue=0.0,
            time_index=3,
        )
        assessment = assess_physical_failure(
            state,
            {"raw_density": state.density, "raw_velocity": state.velocity},
            PhysicalFailureConfig(max_density=190.0, max_velocity=250.0),
        )

        self.assertTrue(assessment.failed)
        self.assertEqual(assessment.subtype, "excessive_density")
        self.assertEqual(assessment.threshold, 190.0)

    def test_failure_detector_applies_configurable_lower_limits(self) -> None:
        def assess(raw_density: float, raw_velocity: float, **limits):
            state = TrafficState(
                density=np.array([max(raw_density, 0.0)]),
                velocity=np.array([max(raw_velocity, 1e-4)]),
                flow_origin=0.0,
                queue=0.0,
            )
            return assess_physical_failure(
                state,
                {
                    "raw_density": np.array([raw_density]),
                    "raw_velocity": np.array([raw_velocity]),
                },
                PhysicalFailureConfig(**limits),
            )

        # Default: anything below zero is unsafe, with the original name.
        default = assess(20.0, -0.01)
        self.assertTrue(default.failed)
        self.assertEqual(default.subtype, "negative_raw_velocity")

        # A higher minimum velocity flags slow but positive speeds.
        slow = assess(20.0, 3.0, min_velocity=5.0)
        self.assertTrue(slow.failed)
        self.assertEqual(slow.subtype, "low_raw_velocity")
        self.assertEqual(slow.threshold, 5.0)
        self.assertFalse(assess(20.0, 6.0, min_velocity=5.0).failed)

        # A negative minimum tolerates small undershoots.
        self.assertFalse(assess(20.0, -0.5, min_velocity=-1.0).failed)

        # Density minimum works the same way.
        sparse = assess(1.0, 50.0, min_density=2.0)
        self.assertTrue(sparse.failed)
        self.assertEqual(sparse.subtype, "low_raw_density")

        with self.assertRaises(ValueError):
            PhysicalFailureConfig(min_velocity=300.0)


class ASTSearchTests(unittest.TestCase):
    def test_proof_of_concept_vsl_model_uses_five_mph_steps(self) -> None:
        model = VSLActionModel.proof_of_concept()
        controlled = model.actions[1:]

        self.assertEqual(len(model.actions), 9)
        self.assertEqual(model.actions[0].label, "unrestricted")
        self.assertEqual(controlled[0].label, "75 mph")
        self.assertEqual(controlled[-1].label, "40 mph")
        controlled_mph = np.array(
            [action.speed / VSLActionModel.MPH_TO_KMH for action in controlled]
        )
        np.testing.assert_allclose(np.diff(controlled_mph), -5.0)
        self.assertAlmostEqual(
            sum(model.action_probability(action) for action in model.actions), 1.0
        )
        self.assertEqual(
            [model.intervention_cost(action) for action in model.actions[:3]],
            [0.0, 1.0, 6.0],
        )
        self.assertEqual(model.intervention_cost(controlled[-1]), 36.0)
        self.assertGreater(
            model.action_probability(controlled[0]),
            model.action_probability(controlled[-1]),
        )

    def test_nominal_check_and_mcts_find_low_vsl_failure(self) -> None:
        environment = ASTEnvironment(
            T=10 / 3600,
            l=0.4,
            initial_state=TrafficState(
                density=np.array([1.0]),
                velocity=np.array([100.0]),
                flow_origin=100.0,
                queue=0.0,
            ),
            nominal_demand=np.array([100.0]),
            downstream_density=np.array([1.0]),
            # A short tau makes a severe one-step VSL change overshoot below
            # zero while the unrestricted baseline remains finite and valid.
            params=simple_params(tau=1 / 3600),
            lanes={0: 1},
            failure_config=PhysicalFailureConfig(
                max_density=190.0, max_velocity=250.0
            ),
            action_interval_steps=1,
        )
        model = VSLActionModel(
            actions=(
                VSLAction("unrestricted", 1000.0),
                VSLAction("60", 60.0),
            ),
            probabilities=(0.9, 0.1),
            costs=(0.0, 1.0),
        )

        self.assertFalse(environment.run_nominal().failed)
        result = run_mcts(
            environment,
            model,
            SearchObjective.MIN_INTERVENTION,
            MCTSConfig(iterations=4, seed=2),
        )

        self.assertIsNotNone(result.best_failure)
        self.assertEqual(result.best_failure.action_labels, ("60",))
        self.assertEqual(
            result.best_failure.failure.subtype, "negative_raw_velocity"
        )

    def test_run_can_continue_past_first_failure(self) -> None:
        environment = ASTEnvironment(
            T=10 / 3600,
            l=0.4,
            initial_state=TrafficState(
                density=np.array([1.0]),
                velocity=np.array([100.0]),
                flow_origin=100.0,
                queue=0.0,
            ),
            nominal_demand=np.full(3, 100.0),
            downstream_density=np.full(3, 1.0),
            params=simple_params(tau=1 / 3600),
            lanes={0: 1},
            action_interval_steps=1,
        )
        model = VSLActionModel(
            actions=(VSLAction("unrestricted", 1000.0), VSLAction("60", 60.0)),
            probabilities=(0.9, 0.1),
            costs=(0.0, 1.0),
        )
        actions = [model.actions[1], model.actions[0], model.actions[0]]

        stopped = environment.run(actions, model)
        continued = environment.run(actions, model, stop_on_failure=False)

        # Both report the same first failure, one step (10 s) in.
        for episode in (stopped, continued):
            self.assertTrue(episode.failed)
            self.assertEqual(episode.failure.time_index, 1)
            self.assertAlmostEqual(episode.time_to_failure_minutes, 10 / 60)
        # Only the continued run simulates the whole horizon.
        self.assertEqual(stopped.velocity_history.shape[0], 2)
        self.assertEqual(continued.velocity_history.shape[0], 4)
        self.assertEqual(continued.simulated_steps, 3)


if __name__ == "__main__":
    unittest.main()
