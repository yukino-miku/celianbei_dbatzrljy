"""Run the non-destructive unified quadratic Q1--Q4 refactor."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from celianbei_dbatzrljy.unified_pipeline import run_unified_pipeline
from celianbei_dbatzrljy.unified_plotting import generate_unified_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use small bootstrap counts for development only")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    if args.quick:
        counts = dict(
            battery_bootstrap_samples=20,
            strategy_bootstrap_samples=40,
            optimization_bootstrap_samples=40,
            test_eol_bootstrap_samples=20,
        )
    else:
        counts = dict(
            battery_bootstrap_samples=300,
            strategy_bootstrap_samples=1_000,
            optimization_bootstrap_samples=1_000,
            test_eol_bootstrap_samples=300,
        )
    results = run_unified_pipeline(PROJECT_ROOT, **counts)
    if not args.skip_figures:
        generate_unified_figures(results)
    print(f"Unified outputs: {results['output_root']}")
    print(f"Selected parameterization: {results['manifest']['parameterization']}")
    print(f"Selected short-horizon model: {results['manifest']['short_horizon_model']}")


if __name__ == "__main__":
    main()
