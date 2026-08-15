"""Strategy-level robust summaries for question one."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_SEED


def make_strategy_mapping(summary: pd.DataFrame) -> pd.DataFrame:
    strategies = summary["policy"].drop_duplicates().tolist()
    return pd.DataFrame(
        {
            "strategy": strategies,
            "strategy_code": [f"S{index}" for index in range(1, len(strategies) + 1)],
        }
    )


def _bootstrap_mean_ci(values: pd.Series, seed: int, samples: int = 5000) -> tuple[float, float]:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return np.nan, np.nan
    if len(clean) == 1:
        return float(clean[0]), float(clean[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(clean, size=(samples, len(clean)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _metric_statistics(values: pd.Series, seed: int) -> dict[str, float | int]:
    clean = values.dropna().astype(float)
    ci_low, ci_high = _bootstrap_mean_ci(clean, seed)
    return {
        "n_valid": int(len(clean)),
        "mean": float(clean.mean()) if len(clean) else np.nan,
        "median": float(clean.median()) if len(clean) else np.nan,
        "std": float(clean.std(ddof=1)) if len(clean) > 1 else np.nan,
        "q1": float(clean.quantile(0.25)) if len(clean) else np.nan,
        "q3": float(clean.quantile(0.75)) if len(clean) else np.nan,
        "mean_ci95_low": ci_low,
        "mean_ci95_high": ci_high,
    }


def build_strategy_summary(
    battery_features: pd.DataFrame,
    lifetimes: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    training = battery_features.loc[battery_features["prediction_test"] == 0].merge(
        lifetimes, on=["battery_id", "strategy"], validate="one_to_one"
    )
    training = training.merge(mapping, on="strategy", validate="many_to_one")
    metrics = {
        "SOH_150": "SOH_150",
        "SOH_200": "SOH_200",
        "slope_50_200_per100": "SOH_slope_50_200_per100",
        "estimated_life": "eol_point",
    }
    rows: list[dict[str, object]] = []
    for strategy_index, (strategy, group) in enumerate(training.groupby("strategy", sort=False)):
        row: dict[str, object] = {
            "strategy": strategy,
            "strategy_code": group["strategy_code"].iloc[0],
            "n_training_batteries": int(len(group)),
            "n_conditionally_stable_lifetimes": int(
                group["reliability"].eq("conditionally_stable").sum()
            ),
            "n_moderate_model_sensitivity": int(
                group["reliability"].eq("moderate_model_sensitivity").sum()
            ),
            "n_high_model_sensitivity": int(
                group["reliability"].eq("high_model_sensitivity").sum()
            ),
            "n_wide_intervals": int(group["reliability"].eq("wide_interval").sum()),
            "n_unresolved_lifetimes": int(group["eol_point"].isna().sum()),
        }
        for metric_index, (label, column) in enumerate(metrics.items()):
            stats = _metric_statistics(
                group[column], RANDOM_SEED + strategy_index * 100 + metric_index
            )
            row.update({f"{label}_{name}": value for name, value in stats.items()})
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values("strategy_code", key=lambda s: s.str[1:].astype(int)).reset_index(drop=True)


def make_battery_results(
    battery_features: pd.DataFrame,
    lifetimes: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    return (
        battery_features.loc[battery_features["prediction_test"] == 0]
        .merge(lifetimes, on=["battery_id", "strategy"], validate="one_to_one")
        .merge(mapping, on="strategy", validate="many_to_one")
        .sort_values("battery_id")
        .reset_index(drop=True)
    )
