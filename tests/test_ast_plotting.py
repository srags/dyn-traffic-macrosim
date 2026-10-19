"""Smoke tests for the reproducible AST result figures."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ast_plotting import (  # noqa: E402
    save_boundary_severity,
    save_discovered_sequences,
    save_time_to_first_invalid_state,
)


class ASTPlottingTests(unittest.TestCase):
    def test_focused_figures_are_saved(self) -> None:
        def episode(*, failed: bool, raw_value: float, time: float | None) -> dict:
            return {
                "failed": failed,
                "actions": ["80"] if failed else ["nominal"],
                "time_to_failure_minutes": time,
                "negative_log_likelihood": 2.0,
                "intervention_cost": 3.0,
                "closest_nonfailure_subtype": "negative_raw_velocity",
                "closest_nonfailure_raw_value": raw_value,
                "closest_nonfailure_signed_margin": -0.01,
                "failure_subtype": "negative_raw_velocity" if failed else None,
                "failure_raw_value": raw_value if failed else None,
                "failure_signed_margin": 0.01 if failed else None,
                "failure_segment": 2 if failed else None,
            }

        static_nominal = episode(failed=False, raw_value=0.001, time=None)
        static_failure = episode(failed=True, raw_value=-2.9, time=1.0)
        dynamic_nominal = episode(failed=False, raw_value=0.002, time=None)
        dynamic_failure = episode(failed=True, raw_value=-1.8, time=2.0)
        results = {
            "scenario": {
                "window_minutes": 15,
                "action_minutes": 1,
                "max_density": 190.0,
                "max_velocity": 250.0,
            },
            "calibrations": {
                "static": {
                    "label": "Static calibration",
                    "nominal": static_nominal,
                    "searches": {"vsl-ast": {"best_episode": static_failure}},
                },
                "dynamic": {
                    "label": "Frozen dynamic calibration (h=6, 1 min)",
                    "control_horizon_steps": 6,
                    "nominal": dynamic_nominal,
                    "searches": {"vsl-ast": {"best_episode": dynamic_failure}},
                },
            },
            "action_models": {
                "vsl": {
                    "actions": [
                        {
                            "label": "unrestricted",
                            "probability": 0.9,
                            "intervention_cost": 0.0,
                        },
                        {
                            "label": "80",
                            "probability": 0.1,
                            "intervention_cost": 3.0,
                        },
                    ]
                }
            },
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            time_path = Path(temporary_directory) / "time.png"
            severity_path = Path(temporary_directory) / "severity.png"
            table_path = Path(temporary_directory) / "table.png"
            saved_time = save_time_to_first_invalid_state(
                results, time_path, dpi=72
            )
            saved_severity = save_boundary_severity(
                results, severity_path, dpi=72
            )
            saved_table = save_discovered_sequences(results, table_path, dpi=72)

            self.assertEqual(saved_time, time_path.resolve())
            self.assertEqual(saved_severity, severity_path.resolve())
            self.assertEqual(saved_table, table_path.resolve())
            self.assertGreater(time_path.stat().st_size, 1_000)
            self.assertGreater(severity_path.stat().st_size, 1_000)
            self.assertGreater(table_path.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
