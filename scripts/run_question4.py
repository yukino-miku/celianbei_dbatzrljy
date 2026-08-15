"""Run question four charge-time/lifetime multi-objective optimization."""

from __future__ import annotations

import argparse
import json

from celianbei_dbatzrljy.question4_pipeline import run_question4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_question4(
        bootstrap_samples=args.bootstrap_samples,
        make_plots=not args.skip_plots,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
