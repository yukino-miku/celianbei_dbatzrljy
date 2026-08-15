"""Constrained degradation models, time-ordered validation, and EOL uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

from .config import (
    CANDIDATE_MODELS,
    EOL_THRESHOLD,
    MAX_EOL_CYCLE,
    RANDOM_SEED,
    TRUNCATION_CUTOFFS,
)
from .data import robust_lowess
from .features import robust_slope


def predict_model(model: str, cycle: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    x = np.asarray(cycle, dtype=float) / 100.0
    if model == "linear":
        a, d = parameters
        return a - d * x
    if model == "quadratic":
        a, d1, d2 = parameters
        return a - d1 * x - d2 * x**2
    if model == "power":
        a, d, power = parameters
        return a - d * np.power(np.maximum(x, 0.0), power)
    if model == "exponential":
        a, d, rate = parameters
        return a - d * np.expm1(np.clip(rate * x, None, 50.0))
    raise KeyError(f"Unknown model: {model}")


def _initial_and_bounds(model: str, cycle: np.ndarray, values: np.ndarray):
    x_range = max((cycle.max() - cycle.min()) / 100.0, 0.5)
    observed_drop = max(float(values[0] - values[-1]), 1e-4)
    a0 = float(np.clip(np.median(values[: min(10, len(values))]), 0.90, 1.10))
    d0 = float(np.clip(observed_drop / x_range, 1e-4, 0.2))
    if model == "linear":
        return np.array([a0, d0]), (np.array([0.85, 0.0]), np.array([1.15, 0.50]))
    if model == "quadratic":
        return np.array([a0, d0 * 0.7, d0 * 0.15]), (
            np.array([0.85, 0.0, 0.0]),
            np.array([1.15, 0.50, 0.25]),
        )
    if model == "power":
        return np.array([a0, d0, 1.0]), (
            np.array([0.85, 1e-8, 0.25]),
            np.array([1.15, 0.50, 3.00]),
        )
    if model == "exponential":
        return np.array([a0, max(d0, 1e-4), 0.50]), (
            np.array([0.85, 1e-8, 0.01]),
            np.array([1.15, 0.50, 3.00]),
        )
    raise KeyError(model)


@dataclass(frozen=True)
class ModelFit:
    model: str
    parameters: np.ndarray | None
    success: bool
    status: str
    rmse_fit: float
    eol_cycle: float

    def predict(self, cycle: np.ndarray) -> np.ndarray:
        if self.parameters is None:
            return np.full_like(np.asarray(cycle, dtype=float), np.nan)
        return predict_model(self.model, np.asarray(cycle, dtype=float), self.parameters)


def estimate_eol(model: str, parameters: np.ndarray, max_cycle: float = MAX_EOL_CYCLE) -> float:
    def crossing(cycle: float) -> float:
        return float(predict_model(model, np.array([cycle]), parameters)[0] - EOL_THRESHOLD)

    if crossing(0.0) <= 0:
        return 0.0
    if crossing(max_cycle) > 0:
        return float("nan")
    try:
        return float(brentq(crossing, 0.0, max_cycle, maxiter=500))
    except ValueError:
        return float("nan")


def fit_degradation_model(model: str, cycle: np.ndarray, values: np.ndarray) -> ModelFit:
    cycle = np.asarray(cycle, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(cycle) & np.isfinite(values)
    cycle, values = cycle[mask], values[mask]
    if len(cycle) < 20:
        return ModelFit(model, None, False, "insufficient_data", np.nan, np.nan)
    initial, bounds = _initial_and_bounds(model, cycle, values)
    try:
        result = least_squares(
            lambda parameters: predict_model(model, cycle, parameters) - values,
            x0=initial,
            bounds=bounds,
            loss="soft_l1",
            f_scale=5e-4,
            max_nfev=5000,
        )
    except (ValueError, RuntimeError, FloatingPointError):
        return ModelFit(model, None, False, "fit_failed", np.nan, np.nan)
    if not result.success or not np.all(np.isfinite(result.x)):
        return ModelFit(model, None, False, "fit_failed", np.nan, np.nan)

    prediction = predict_model(model, cycle, result.x)
    rmse = float(np.sqrt(np.mean((prediction - values) ** 2)))
    eol = estimate_eol(model, result.x)
    grid_end = min(float(MAX_EOL_CYCLE), float(eol) if np.isfinite(eol) else MAX_EOL_CYCLE)
    grid = np.linspace(0.0, grid_end, 1000)
    monotone = bool(np.all(np.diff(predict_model(model, grid, result.x)) <= 1e-9))
    if not monotone:
        return ModelFit(model, result.x, False, "nonmonotone", rmse, np.nan)
    if np.isfinite(eol) and eol <= cycle.max():
        return ModelFit(model, result.x, False, "eol_before_observed_end", rmse, eol)
    status = "ok" if np.isfinite(eol) else "no_crossing_within_5000"
    return ModelFit(model, result.x, True, status, rmse, eol)


def _fit_row(battery_id: int, model: str, cutoff: int, fit: ModelFit) -> dict[str, object]:
    parameters = None if fit.parameters is None else json.dumps(fit.parameters.tolist())
    return {
        "battery_id": battery_id,
        "model": model,
        "cutoff": cutoff,
        "fit_success": fit.success,
        "fit_status": fit.status,
        "fit_rmse": fit.rmse_fit,
        "eol_cycle": fit.eol_cycle,
        "parameters_json": parameters,
    }


def run_truncation_validation(
    cleaned_cycles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = cleaned_cycles.loc[cleaned_cycles["prediction_test"] == 0]
    validation_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []

    for battery_id, group in train.groupby("battery_id", sort=True):
        g = group.sort_values("cycle")
        for model in CANDIDATE_MODELS:
            for cutoff in (*TRUNCATION_CUTOFFS, 200):
                fitting = g.loc[g["cycle"] <= cutoff]
                fitting_values = (
                    fitting["SOH_smooth_robust"].to_numpy()
                    if cutoff == 200
                    else robust_lowess(fitting["cycle"], fitting["SOH_clean"])
                )
                fit = fit_degradation_model(
                    model,
                    fitting["cycle"].to_numpy(),
                    fitting_values,
                )
                fit_rows.append(_fit_row(int(battery_id), model, cutoff, fit))
                if cutoff == 200:
                    continue
                future = g.loc[g["cycle"] > cutoff]
                prediction = fit.predict(future["cycle"].to_numpy())
                error = prediction - future["SOH_clean"].to_numpy()
                actual_slope = robust_slope(future["cycle"], future["SOH_clean"])
                predicted_slope = robust_slope(future["cycle"], pd.Series(prediction))
                valid_prediction = bool(fit.success and np.all(np.isfinite(prediction)))
                validation_rows.append(
                    {
                        "battery_id": int(battery_id),
                        "strategy": g["strategy"].iloc[0],
                        "model": model,
                        "cutoff": cutoff,
                        "n_future": len(future),
                        "fit_status": fit.status,
                        "MAE": float(np.mean(np.abs(error))) if valid_prediction else np.nan,
                        "RMSE": float(np.sqrt(np.mean(error**2))) if valid_prediction else np.nan,
                        "MaxAE": float(np.max(np.abs(error))) if valid_prediction else np.nan,
                        "actual_future_slope_per100": 100.0 * actual_slope,
                        "predicted_future_slope_per100": 100.0 * predicted_slope,
                        "trend_sign_match": bool(
                            valid_prediction
                            and np.sign(actual_slope) == np.sign(predicted_slope)
                        ),
                        "eol_cycle": fit.eol_cycle,
                    }
                )

    validation = pd.DataFrame(validation_rows)
    fits = pd.DataFrame(fit_rows)
    pivot = fits.pivot(index=["battery_id", "model"], columns="cutoff", values="eol_cycle")
    for cutoff in (100, 150, 200):
        if cutoff not in pivot:
            pivot[cutoff] = np.nan
    pivot = pivot.rename(columns={100: "eol_100", 150: "eol_150", 200: "eol_200"}).reset_index()
    pivot["relative_change_100_to_200"] = (
        (pivot["eol_100"] - pivot["eol_200"]).abs() / pivot["eol_200"]
    )
    pivot["relative_change_150_to_200"] = (
        (pivot["eol_150"] - pivot["eol_200"]).abs() / pivot["eol_200"]
    )
    pivot["valid_eol_count"] = pivot[["eol_100", "eol_150", "eol_200"]].notna().sum(axis=1)
    pivot["eol_cv"] = pivot[["eol_100", "eol_150", "eol_200"]].std(axis=1) / pivot[
        ["eol_100", "eol_150", "eol_200"]
    ].mean(axis=1)
    return validation, fits, pivot


def summarize_and_select_model(
    validation: pd.DataFrame, fits: pd.DataFrame, stability: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, object]] = []
    for model in CANDIDATE_MODELS:
        model_validation = validation.loc[validation["model"] == model]
        model_fits = fits.loc[(fits["model"] == model) & (fits["cutoff"] == 200)]
        model_stability = stability.loc[stability["model"] == model]
        row: dict[str, object] = {
            "model": model,
            "MAE_100_median": model_validation.loc[model_validation["cutoff"] == 100, "MAE"].median(),
            "MAE_150_median": model_validation.loc[model_validation["cutoff"] == 150, "MAE"].median(),
            "RMSE_100_median": model_validation.loc[model_validation["cutoff"] == 100, "RMSE"].median(),
            "RMSE_150_median": model_validation.loc[model_validation["cutoff"] == 150, "RMSE"].median(),
            "MaxAE_median": model_validation["MaxAE"].median(),
            "trend_match_rate": model_validation["trend_sign_match"].mean(),
            "eol_150_to_200_change_median": model_stability["relative_change_150_to_200"].median(),
            "eol_100_to_200_change_median": model_stability["relative_change_100_to_200"].median(),
            "eol_200_coverage": model_fits["eol_cycle"].notna().mean(),
            "fit_success_rate": model_fits["fit_success"].mean(),
        }
        rows.append(row)
    summary = pd.DataFrame(rows)

    ascending_metrics = [
        "MAE_100_median",
        "MAE_150_median",
        "RMSE_100_median",
        "RMSE_150_median",
        "MaxAE_median",
        "eol_150_to_200_change_median",
        "eol_100_to_200_change_median",
    ]
    descending_metrics = ["trend_match_rate", "eol_200_coverage", "fit_success_rate"]
    for metric in ascending_metrics:
        summary[f"rank_{metric}"] = summary[metric].rank(method="average", ascending=True, na_option="bottom")
    for metric in descending_metrics:
        summary[f"rank_{metric}"] = summary[metric].rank(method="average", ascending=False, na_option="bottom")

    summary["selection_score"] = (
        0.14 * summary["rank_MAE_100_median"]
        + 0.14 * summary["rank_MAE_150_median"]
        + 0.10 * summary["rank_RMSE_100_median"]
        + 0.10 * summary["rank_RMSE_150_median"]
        + 0.08 * summary["rank_MaxAE_median"]
        + 0.17 * summary["rank_eol_150_to_200_change_median"]
        + 0.07 * summary["rank_eol_100_to_200_change_median"]
        + 0.08 * summary["rank_trend_match_rate"]
        + 0.08 * summary["rank_eol_200_coverage"]
        + 0.04 * summary["rank_fit_success_rate"]
    )
    summary = summary.sort_values("selection_score").reset_index(drop=True)
    summary["selected"] = False
    summary.loc[0, "selected"] = True
    return summary, str(summary.loc[0, "model"])


def _moving_block_residuals(residuals: np.ndarray, rng: np.random.Generator, block: int) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=float)
    n = len(residuals)
    block = min(block, n)
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([residuals[start : start + block] for start in starts])[:n]


def estimate_training_lifetimes(
    cleaned_cycles: pd.DataFrame,
    fits: pd.DataFrame,
    selected_model: str,
    *,
    bootstrap_samples: int = 200,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    train = cleaned_cycles.loc[cleaned_cycles["prediction_test"] == 0]
    all_200 = fits.loc[fits["cutoff"] == 200]
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(random_seed)

    for battery_id, group in train.groupby("battery_id", sort=True):
        g = group.sort_values("cycle")
        cycle = g["cycle"].to_numpy(dtype=float)
        y = g["SOH_smooth_robust"].to_numpy(dtype=float)
        fit = fit_degradation_model(selected_model, cycle, y)
        bootstrap_eol: list[float] = []
        if fit.parameters is not None:
            fitted = fit.predict(cycle)
            residual = g["SOH_clean"].to_numpy(dtype=float) - fitted
            residual = residual - np.median(residual)
            for _ in range(bootstrap_samples):
                pseudo = fitted + _moving_block_residuals(residual, rng, block=10)
                boot_fit = fit_degradation_model(selected_model, cycle, pseudo)
                if boot_fit.success and np.isfinite(boot_fit.eol_cycle):
                    bootstrap_eol.append(float(boot_fit.eol_cycle))
        bootstrap_array = np.asarray(bootstrap_eol, dtype=float)
        valid_fraction = len(bootstrap_array) / bootstrap_samples
        if len(bootstrap_array) >= max(20, int(0.20 * bootstrap_samples)):
            ci_low, median, ci_high = np.quantile(bootstrap_array, [0.025, 0.5, 0.975])
        else:
            ci_low = median = ci_high = np.nan

        model_eols = all_200.loc[all_200["battery_id"] == battery_id, "eol_cycle"].dropna()
        disagreement = (
            float((model_eols.max() - model_eols.min()) / model_eols.median())
            if len(model_eols) >= 2 and model_eols.median() > 0
            else np.nan
        )
        if not np.isfinite(fit.eol_cycle):
            reliability = "unresolved_no_crossing"
            reason = "selected model does not reach SOH=0.8 within 5000 cycles"
        elif valid_fraction < 0.80:
            reliability = "unstable_bootstrap"
            reason = "fewer than 80% of block-bootstrap fits yield a bounded EOL"
        elif np.isfinite(ci_low) and (ci_high - ci_low) / median > 0.50:
            reliability = "wide_interval"
            reason = "conditional 95% interval width exceeds 50% of the median EOL"
        elif np.isfinite(disagreement) and disagreement > 0.75:
            reliability = "high_model_sensitivity"
            reason = "candidate-model EOL range exceeds 75% of its median"
        elif np.isfinite(disagreement) and disagreement > 0.35:
            reliability = "moderate_model_sensitivity"
            reason = "candidate-model EOL range is 35%-75% of its median"
        else:
            reliability = "conditionally_stable"
            reason = "stable conditional on the selected model and 5000-cycle horizon"

        rows.append(
            {
                "battery_id": int(battery_id),
                "strategy": g["strategy"].iloc[0],
                "selected_model": selected_model,
                "eol_point": fit.eol_cycle,
                "eol_bootstrap_median": median,
                "eol_ci95_low": ci_low,
                "eol_ci95_high": ci_high,
                "eol_ci95_width_ratio": (
                    float((ci_high - ci_low) / median)
                    if np.isfinite(ci_low) and np.isfinite(median) and median > 0
                    else np.nan
                ),
                "bootstrap_valid_fraction": valid_fraction,
                "bootstrap_valid_n": len(bootstrap_array),
                "candidate_model_eol_min": model_eols.min() if not model_eols.empty else np.nan,
                "candidate_model_eol_max": model_eols.max() if not model_eols.empty else np.nan,
                "candidate_model_disagreement_ratio": disagreement,
                "reliability": reliability,
                "reliability_note": reason,
                "parameters_json": None if fit.parameters is None else json.dumps(fit.parameters.tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values("battery_id").reset_index(drop=True)
