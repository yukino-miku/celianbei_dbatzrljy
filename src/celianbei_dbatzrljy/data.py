"""Input validation and conservative, traceable cleaning for question one."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from .config import CYCLES_PATH, SUMMARY_PATH


SUMMARY_COLUMNS = {
    "battery_id",
    "global_id",
    "dataset_id",
    "local_id",
    "policy",
    "C1",
    "Q1",
    "C2",
    "initial_capacity",
    "mean_chargetime",
    "mean_IR",
    "mean_Tavg",
    "prediction_test",
}
CYCLE_COLUMNS = {
    "battery_id",
    "cycle",
    "capacity",
    "SOH",
    "SOH_smooth",
    "chargetime",
    "IR",
    "Tavg",
    "policy",
}


@dataclass(frozen=True)
class CleanedData:
    summary: pd.DataFrame
    cycles: pd.DataFrame
    anomalies: pd.DataFrame
    audit: pd.DataFrame


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two untouched competition CSV files."""
    return pd.read_csv(SUMMARY_PATH), pd.read_csv(CYCLES_PATH)


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def validate_raw_data(summary: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, keys, physical domains, censoring, and cross-file consistency."""
    _require_columns(summary, SUMMARY_COLUMNS, "battery_summary.csv")
    _require_columns(cycles, CYCLE_COLUMNS, "cycle_train.csv")

    checks: list[dict[str, object]] = []

    def add(check: str, value: object, expected: object, passed: bool) -> None:
        checks.append(
            {"check": check, "observed": value, "expected": expected, "passed": bool(passed)}
        )

    add("summary_rows", len(summary), 49, len(summary) == 49)
    add("cycle_rows", len(cycles), 9350, len(cycles) == 9350)
    add("unique_battery_id", summary["battery_id"].nunique(), 49, summary["battery_id"].is_unique)
    add(
        "duplicate_battery_cycle",
        int(cycles.duplicated(["battery_id", "cycle"]).sum()),
        0,
        not cycles.duplicated(["battery_id", "cycle"]).any(),
    )
    add("duplicate_summary_rows", int(summary.duplicated().sum()), 0, not summary.duplicated().any())
    add("test_batteries", int(summary["prediction_test"].sum()), 9, summary["prediction_test"].sum() == 9)
    add("strategies", summary["policy"].nunique(), 9, summary["policy"].nunique() == 9)
    add(
        "cycle_positive_integer",
        bool((cycles["cycle"] > 0).all() and np.allclose(cycles["cycle"], cycles["cycle"].round())),
        True,
        bool((cycles["cycle"] > 0).all() and np.allclose(cycles["cycle"], cycles["cycle"].round())),
    )

    joined = cycles.merge(
        summary[["battery_id", "policy", "prediction_test", "dataset_id"]],
        on="battery_id",
        suffixes=("_cycle", "_summary"),
        validate="many_to_one",
    )
    mismatch = int((joined["policy_cycle"] != joined["policy_summary"]).sum())
    add("strategy_label_mismatch", mismatch, 0, mismatch == 0)

    lengths = cycles.groupby("battery_id")["cycle"].agg(["min", "max", "count"])
    length_check = summary.join(lengths, on="battery_id")
    train_ok = (
        (length_check.loc[length_check["prediction_test"] == 0, "min"] == 1)
        & (length_check.loc[length_check["prediction_test"] == 0, "max"] == 200)
        & (length_check.loc[length_check["prediction_test"] == 0, "count"] == 200)
    ).all()
    test_ok = (
        (length_check.loc[length_check["prediction_test"] == 1, "min"] == 1)
        & (length_check.loc[length_check["prediction_test"] == 1, "max"] == 150)
        & (length_check.loc[length_check["prediction_test"] == 1, "count"] == 150)
    ).all()
    add("non_test_cycle_coverage", int(train_ok), True, bool(train_ok))
    add("test_cycle_coverage", int(test_ok), True, bool(test_ok))

    unexpected_summary_missing = int(summary.drop(columns=["C1"]).isna().sum().sum())
    add("unexpected_summary_missing", unexpected_summary_missing, 0, unexpected_summary_missing == 0)
    add("cycle_missing", int(cycles.isna().sum().sum()), 0, not cycles.isna().any().any())
    add("C1_missing", int(summary["C1"].isna().sum()), 3, summary["C1"].isna().sum() == 3)
    add("capacity_positive", int((cycles["capacity"] <= 0).sum()), 0, (cycles["capacity"] > 0).all())
    add("SOH_positive", int((cycles["SOH"] <= 0).sum()), 0, (cycles["SOH"] > 0).all())
    add("Tavg_physical", int((~cycles["Tavg"].between(-20, 80)).sum()), 0, cycles["Tavg"].between(-20, 80).all())
    add(
        "chargetime_positive",
        int((cycles["chargetime"] <= 0).sum()),
        0,
        (cycles["chargetime"] > 0).all(),
    )

    audit = pd.DataFrame(checks)
    failed = audit.loc[~audit["passed"]]
    if not failed.empty:
        raise ValueError(f"Raw-data validation failed:\n{failed.to_string(index=False)}")
    return audit


def _rolling_hampel_mask(
    values: pd.Series,
    *,
    window: int,
    sigma: float,
    absolute_floor: float,
) -> pd.Series:
    """Flag only isolated deviations that exceed both a robust and absolute threshold."""
    median = values.rolling(window=window, center=True, min_periods=max(5, window // 2)).median()
    residual = (values - median).abs()
    mad = residual.rolling(window=window, center=True, min_periods=max(5, window // 2)).median()
    threshold = np.maximum(sigma * 1.4826 * mad.fillna(0.0), absolute_floor)
    return residual > threshold


def _repair_local(series: pd.Series, mask: pd.Series, *, limit: int = 2) -> pd.Series:
    repaired = series.astype(float).mask(mask)
    repaired = repaired.interpolate(method="linear", limit=limit, limit_direction="both")
    if repaired.isna().any():
        raise ValueError("Conservative local interpolation left unresolved values")
    return repaired


def robust_lowess(cycles: pd.Series, values: pd.Series) -> np.ndarray:
    """Robust LOWESS using an approximately 25-cycle local span."""
    span_cycles = 25.0
    frac = float(np.clip(span_cycles / len(values), 0.08, 0.25))
    return lowess(
        endog=values.to_numpy(dtype=float),
        exog=cycles.to_numpy(dtype=float),
        frac=frac,
        it=3,
        delta=0.0,
        return_sorted=False,
    )


def clean_cycle_data(summary: pd.DataFrame, cycles: pd.DataFrame) -> CleanedData:
    """Conservatively repair explicit local anomalies and add a robust LOWESS curve."""
    audit = validate_raw_data(summary, cycles)
    meta = summary[
        [
            "battery_id",
            "global_id",
            "dataset_id",
            "local_id",
            "policy",
            "C1",
            "Q1",
            "C2",
            "initial_capacity",
            "prediction_test",
        ]
    ].rename(columns={"policy": "policy_summary"})
    data = cycles.rename(
        columns={
            "capacity": "capacity_raw",
            "SOH": "SOH_raw",
            "SOH_smooth": "SOH_smooth_official",
            "IR": "IR_raw",
        }
    ).merge(meta, on="battery_id", validate="many_to_one")
    data["strategy"] = data["policy"]

    cleaned_groups: list[pd.DataFrame] = []
    for _, group in data.groupby("battery_id", sort=True):
        g = group.sort_values("cycle").copy()
        soh_local = _rolling_hampel_mask(
            g["SOH_raw"], window=11, sigma=6.0, absolute_floor=0.02
        )
        capacity_local = _rolling_hampel_mask(
            g["capacity_raw"], window=11, sigma=6.0, absolute_floor=0.03
        )
        physical_soh = ~g["SOH_raw"].between(0.70, 1.10)
        physical_capacity = ~g["capacity_raw"].between(0.70, 1.30)
        g["flag_soh_capacity_outlier"] = (
            soh_local | capacity_local | physical_soh | physical_capacity
        )
        g["flag_ir_invalid"] = (~g["IR_raw"].between(0.001, 0.10)) | g["IR_raw"].isna()
        g["flag_temperature_invalid"] = (~g["Tavg"].between(-20, 80)) | g["Tavg"].isna()
        g["flag_chargetime_invalid"] = (g["chargetime"] <= 0) | g["chargetime"].isna()

        g["SOH_clean"] = _repair_local(g["SOH_raw"], g["flag_soh_capacity_outlier"])
        g["capacity_clean"] = g["capacity_raw"].astype(float)
        g.loc[g["flag_soh_capacity_outlier"], "capacity_clean"] = (
            g.loc[g["flag_soh_capacity_outlier"], "SOH_clean"]
            * g.loc[g["flag_soh_capacity_outlier"], "initial_capacity"]
        )
        g["IR_clean"] = _repair_local(g["IR_raw"], g["flag_ir_invalid"])
        g["SOH_smooth_robust"] = robust_lowess(g["cycle"], g["SOH_clean"])
        cleaned_groups.append(g)

    cleaned = pd.concat(cleaned_groups, ignore_index=True)
    flag_columns = [column for column in cleaned.columns if column.startswith("flag_")]
    anomaly_mask = cleaned[flag_columns].any(axis=1)
    anomalies = cleaned.loc[
        anomaly_mask,
        [
            "battery_id",
            "cycle",
            "strategy",
            "capacity_raw",
            "capacity_clean",
            "SOH_raw",
            "SOH_clean",
            "SOH_smooth_official",
            "SOH_smooth_robust",
            "IR_raw",
            "IR_clean",
            *flag_columns,
        ],
    ].copy()
    return CleanedData(summary=summary.copy(), cycles=cleaned, anomalies=anomalies, audit=audit)
