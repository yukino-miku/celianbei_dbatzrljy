"""Run the complete question-two strategy and charging-parameter analysis."""

from __future__ import annotations

import argparse
import json

from celianbei_dbatzrljy.question2_pipeline import run_question2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutation-samples", type=int, default=19_999)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--grouped-bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_question2(
        permutation_samples=args.permutation_samples,
        bootstrap_samples=args.bootstrap_samples,
        grouped_bootstrap_samples=args.grouped_bootstrap_samples,
        make_plots=not args.skip_plots,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
