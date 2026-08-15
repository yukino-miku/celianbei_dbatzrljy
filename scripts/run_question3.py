"""Run leakage-safe pseudo-testing, test forecasts, and conditional EOL estimates."""

from __future__ import annotations

import argparse
import json

from celianbei_dbatzrljy.question3_pipeline import run_question3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eol-bootstrap-samples", type=int, default=300)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run_question3(eol_bootstrap_samples=args.eol_bootstrap_samples,
                                   make_plots=not args.skip_plots), ensure_ascii=False, indent=2))
