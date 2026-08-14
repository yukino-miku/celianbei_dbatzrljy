"""Audit the untouched competition CSV files and print a JSON report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def build_report() -> dict[str, object]:
    summary_path = RAW_DIR / "battery_summary.csv"
    cycle_path = RAW_DIR / "cycle_train.csv"
    summary = pd.read_csv(summary_path)
    cycles = pd.read_csv(cycle_path)

    cycle_lengths = cycles.groupby("battery_id")["cycle"].max()
    joined = summary.join(cycle_lengths.rename("max_cycle"), on="battery_id")
    policies = summary.groupby("policy", dropna=False).agg(
        batteries=("battery_id", "size"),
        test_batteries=("prediction_test", "sum"),
    )

    return {
        "files": {
            summary_path.name: {"sha256": sha256(summary_path)},
            cycle_path.name: {"sha256": sha256(cycle_path)},
        },
        "summary": {
            "shape": list(summary.shape),
            "battery_ids_unique": bool(summary["battery_id"].is_unique),
            "policy_count": int(summary["policy"].nunique()),
            "test_battery_count": int(summary["prediction_test"].sum()),
            "missing_by_column": summary.isna().sum().astype(int).to_dict(),
        },
        "cycles": {
            "shape": list(cycles.shape),
            "duplicate_battery_cycle_rows": int(
                cycles.duplicated(subset=["battery_id", "cycle"]).sum()
            ),
            "missing_by_column": cycles.isna().sum().astype(int).to_dict(),
            "anomaly_counts": {
                "capacity_gt_1_15": int(cycles["capacity"].gt(1.15).sum()),
                "SOH_gt_1_05": int(cycles["SOH"].gt(1.05).sum()),
                "SOH_smooth_gt_1_05": int(cycles["SOH_smooth"].gt(1.05).sum()),
                "IR_le_0": int(cycles["IR"].le(0).sum()),
            },
        },
        "consistency": {
            "non_test_with_200_cycles": int(
                ((joined["prediction_test"] == 0) & (joined["max_cycle"] == 200)).sum()
            ),
            "test_with_150_cycles": int(
                ((joined["prediction_test"] == 1) & (joined["max_cycle"] == 150)).sum()
            ),
            "strategy_replicates": {
                policy: {
                    "batteries": int(row["batteries"]),
                    "test_batteries": int(row["test_batteries"]),
                }
                for policy, row in policies.iterrows()
            },
        },
    }


if __name__ == "__main__":
    print(json.dumps(build_report(), ensure_ascii=False, indent=2))
