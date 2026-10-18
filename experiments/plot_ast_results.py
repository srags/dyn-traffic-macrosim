"""Regenerate focused AST figures from a saved JSON result.

Example
-------
python experiments/plot_ast_results.py artifacts/ast_poc/results.json \
    --time-output artifacts/ast_poc/time_to_first_invalid_state.png \
    --severity-output artifacts/ast_poc/boundary_severity.png \
    --table-output artifacts/ast_poc/discovered_sequences.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ast_plotting import (  # noqa: E402
    save_boundary_severity,
    save_discovered_sequences,
    save_time_to_first_invalid_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="JSON output previously saved by ast_poc.py.")
    parser.add_argument(
        "--time-output",
        help="Time-to-failure chart destination (.png, .pdf, or .svg).",
    )
    parser.add_argument(
        "--severity-output",
        help="Boundary-severity chart destination (.png, .pdf, or .svg).",
    )
    parser.add_argument(
        "--table-output",
        help="Discovered-sequences table destination (.png, .pdf, or .svg).",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.time_output and not args.severity_output and not args.table_output:
        raise SystemExit(
            "Specify --time-output, --severity-output, --table-output, or a combination."
        )
    with Path(args.results).open(encoding="utf-8") as input_file:
        results = json.load(input_file)
    if args.time_output:
        output_path = save_time_to_first_invalid_state(
            results, args.time_output, dpi=args.dpi
        )
        print(f"Saved time-to-failure chart to {output_path}")
    if args.severity_output:
        output_path = save_boundary_severity(
            results, args.severity_output, dpi=args.dpi
        )
        print(f"Saved boundary-severity chart to {output_path}")
    if args.table_output:
        output_path = save_discovered_sequences(
            results, args.table_output, dpi=args.dpi
        )
        print(f"Saved discovered-sequences table to {output_path}")


if __name__ == "__main__":
    main()
