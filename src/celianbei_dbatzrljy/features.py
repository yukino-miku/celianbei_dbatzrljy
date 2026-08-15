"""Interpretable early-degradation features for individual batteries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from .config import SLOPE_WINDOWS, TARGET_CYCLES


def robust_slope(cycle: pd.Series | np.ndarray, values: pd.Series | np.ndarray) -> float:
    x = np.asarray(cycle, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    return float(theilslopes(y[mask], x[mask]).slope)


def local_level(group: pd.DataFrame, target: int, radius: int = 5) -> float:
    if target < group["cycle"].min() or target > group["cycle"].max():
        return float("nan")
    exact = group.loc[group["cycle"] == target, "SOH_smooth_robust"]
    if not exact.empty:
        return float(exact.iloc[0])
    local = group.loc[group["cycle"].between(target - radius, target + radius)]
    if len(local) < 3:
        return float("nan")
    slope, intercept, _, _ = theilslopes(local["SOH_smooth_robust"], local["cycle"])
    return float(intercept + slope * target)


def _quadratic_curvature(group: pd.DataFrame) -> float:
    x = group["cycle"].to_numpy(dtype=float) / 100.0
    y = group["SOH_smooth_robust"].to_numpy(dtype=float)
    if len(x) < 10:
        return float("nan")
    coefficient = np.polyfit(x, y, deg=2)[0]
    return float(2.0 * coefficient)


def extract_battery_features(cleaned_cycles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for battery_id, group in cleaned_cycles.groupby("battery_id", sort=True):
        g = group.sort_values("cycle")
        row: dict[str, object] = {
            "battery_id": int(battery_id),
            "strategy": g["strategy"].iloc[0],
            "dataset_id": int(g["dataset_id"].iloc[0]),
            "prediction_test": int(g["prediction_test"].iloc[0]),
            "n_observed_cycles": int(g["cycle"].max()),
        }
        for target in TARGET_CYCLES:
            row[f"SOH_{target}"] = local_level(g, target)
        for start, end in SLOPE_WINDOWS:
            window = g.loc[g["cycle"].between(start, end)]
            row[f"SOH_slope_{start}_{end}_per100"] = 100.0 * robust_slope(
                window["cycle"], window["SOH_smooth_robust"]
            )

        residual = g["SOH_clean"] - g["SOH_smooth_robust"]
        residual_median = float(np.median(residual))
        row["SOH_noise_sd"] = float(residual.std(ddof=1))
        row["SOH_noise_mad"] = float(1.4826 * np.median(np.abs(residual - residual_median)))
        row["SOH_curvature_per100sq"] = _quadratic_curvature(g)
        row["SOH_slope_change_late_minus_early"] = (
            row["SOH_slope_150_200_per100"] - row["SOH_slope_50_100_per100"]
            if pd.notna(row["SOH_slope_150_200_per100"])
            else float("nan")
        )
        row["SOH_capacity_outlier_count"] = int(g["flag_soh_capacity_outlier"].sum())
        row["IR_invalid_count"] = int(g["flag_ir_invalid"].sum())

        for source, label in (
            ("chargetime", "chargetime"),
            ("Tavg", "Tavg"),
            ("IR_clean", "IR"),
        ):
            row[f"mean_{label}_observed"] = float(g[source].mean())
            row[f"{label}_slope_per100"] = 100.0 * robust_slope(g["cycle"], g[source])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("battery_id").reset_index(drop=True)
