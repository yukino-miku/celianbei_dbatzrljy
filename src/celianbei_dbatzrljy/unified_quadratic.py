"""Unified constrained-quadratic ageing model used by Questions 1--4.

The module deliberately keeps the previous question-specific pipelines intact.
All artefacts generated here are written to the ``unified_quadratic_v2`` result
tree by :mod:`unified_pipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import sqrt
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from .config import EOL_THRESHOLD, MAX_EOL_CYCLE, RANDOM_SEED
from .features import robust_slope
from .models import fit_degradation_model, predict_model


PARAMETER_RESPONSES = ("d1", "d2")
STABLE_RESPONSES = ("R100", "A")
PHASE_SLOPES = (
    "SOH_slope_1_50_per100",
    "SOH_slope_50_100_per100",
    "SOH_slope_100_150_per100",
    "SOH_slope_150_200_per100",
    "SOH_slope_50_200_per100",
)


def quadratic_soh(cycle: np.ndarray | float, a: float, d1: float, d2: float) -> np.ndarray:
    """Evaluate ``a - d1*n/100 - d2*(n/100)^2``."""
    x = np.asarray(cycle, dtype=float) / 100.0
    return a - d1 * x - d2 * x**2


def quadratic_derived(d1: np.ndarray | float, d2: np.ndarray | float) -> tuple[np.ndarray, ...]:
    """Return instantaneous 100/200-cycle loss rates and acceleration."""
    d1a = np.asarray(d1, dtype=float)
    d2a = np.asarray(d2, dtype=float)
    return d1a + 2.0 * d2a, d1a + 4.0 * d2a, 2.0 * d2a


def stable_to_native(r100: np.ndarray | float, acceleration: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Map the stable ``(R100, A)`` parameterization to physical ``(d1,d2)``.

    Predictions are projected onto ``d1>=0,d2>=0``.  The unprojected values
    should still be retained by callers for model diagnostics.
    """
    r = np.asarray(r100, dtype=float)
    a = np.asarray(acceleration, dtype=float)
    a_physical = np.maximum(a, 0.0)
    r_physical = np.maximum(r, a_physical)
    return r_physical - a_physical, a_physical / 2.0


def analytic_eol(
    a: np.ndarray | float,
    d1: np.ndarray | float,
    d2: np.ndarray | float,
    *,
    threshold: float = EOL_THRESHOLD,
    max_cycle: float = MAX_EOL_CYCLE,
) -> np.ndarray:
    """Analytic first crossing of the unified quadratic SOH curve.

    Invalid/non-degrading curves and crossings beyond ``max_cycle`` are NaN.
    """
    aa, b, c = np.broadcast_arrays(
        np.asarray(a, dtype=float), np.asarray(d1, dtype=float), np.asarray(d2, dtype=float)
    )
    result = np.full(aa.shape, np.nan, dtype=float)
    immediate = aa <= threshold
    result[immediate] = 0.0
    linear = (~immediate) & (c <= 1e-14) & (b > 1e-14)
    result[linear] = 100.0 * (aa[linear] - threshold) / b[linear]
    curved = (~immediate) & (c > 1e-14) & (b >= 0.0)
    discriminant = b[curved] ** 2 + 4.0 * c[curved] * (aa[curved] - threshold)
    result[curved] = 100.0 * (-b[curved] + np.sqrt(np.maximum(discriminant, 0.0))) / (
        2.0 * c[curved]
    )
    result[(result < 0.0) | (result > max_cycle)] = np.nan
    return result


def _moving_block(values: np.ndarray, rng: np.random.Generator, block: int = 10) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = len(values)
    block = min(block, n)
    starts = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
    return np.concatenate([values[s : s + block] for s in starts])[:n]


def parse_quadratic_parameters(value: object) -> tuple[float, float, float]:
    """Parse the baseline JSON parameter field with strict length checks."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan, np.nan, np.nan
    parsed = json.loads(str(value))
    if len(parsed) != 3:
        raise ValueError(f"Expected three quadratic parameters, received {parsed!r}")
    return tuple(float(v) for v in parsed)


def build_battery_parameter_table(battery_results: pd.DataFrame) -> pd.DataFrame:
    """Expand baseline quadratic results into the unified parameter table."""
    frame = battery_results.copy()
    parameters = pd.DataFrame(
        [parse_quadratic_parameters(v) for v in frame["parameters_json"]],
        columns=["a", "d1", "d2"],
        index=frame.index,
    )
    frame = pd.concat([frame, parameters], axis=1)
    frame["quadratic_fit_rmse"] = np.nan
    frame["quadratic_eol"] = analytic_eol(frame["a"], frame["d1"], frame["d2"])
    frame["R100"], frame["R200"], frame["A"] = quadratic_derived(frame["d1"], frame["d2"])
    tolerance = 1e-8
    frame["d1_lower_boundary"] = frame["d1"] <= tolerance
    frame["d2_lower_boundary"] = frame["d2"] <= tolerance
    frame["a_lower_boundary"] = frame["a"] <= 0.850001
    frame["a_upper_boundary"] = frame["a"] >= 1.149999
    frame["physical_parameter_status"] = np.select(
        [
            frame[["a", "d1", "d2"]].isna().any(axis=1),
            (frame["d1"] < -tolerance) | (frame["d2"] < -tolerance),
            frame["quadratic_eol"].isna(),
        ],
        ["missing_fit", "violates_nonnegative_loss", "no_crossing_within_horizon"],
        default="physical",
    )
    keep = [
        "battery_id", "strategy", "strategy_code", "dataset_id", "prediction_test", "a", "d1", "d2",
        "R100", "R200", "A", "quadratic_fit_rmse", "quadratic_eol",
        "d1_lower_boundary", "d2_lower_boundary", "a_lower_boundary", "a_upper_boundary",
        "physical_parameter_status", *PHASE_SLOPES, "SOH_150", "SOH_200",
        "initial_capacity", "mean_chargetime_observed", "mean_Tavg_observed",
        "mean_IR_observed", "reliability",
    ]
    return frame[[c for c in keep if c in frame.columns]].sort_values("battery_id").reset_index(drop=True)


def attach_quadratic_fit_rmse(
    parameters: pd.DataFrame, candidate_fits: pd.DataFrame
) -> pd.DataFrame:
    fit_rmse = candidate_fits.loc[
        (candidate_fits["model"] == "quadratic") & (candidate_fits["cutoff"] == 200),
        ["battery_id", "fit_rmse", "fit_status"],
    ].rename(columns={"fit_rmse": "quadratic_fit_rmse", "fit_status": "quadratic_fit_status"})
    result = parameters.drop(columns=["quadratic_fit_rmse"], errors="ignore").merge(
        fit_rmse, on="battery_id", how="left", validate="one_to_one"
    )
    return result


def bootstrap_battery_parameters(
    cleaned_cycles: pd.DataFrame,
    *,
    samples: int = 300,
    seed: int = RANDOM_SEED + 501,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Moving-block residual bootstrap for each training battery's parameters."""
    rng = np.random.default_rng(seed)
    draws: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    train = cleaned_cycles.loc[cleaned_cycles["prediction_test"] == 0]
    for battery_id, group in train.groupby("battery_id", sort=True):
        g = group.sort_values("cycle")
        cycle = g["cycle"].to_numpy(float)
        y = g["SOH_smooth_robust"].to_numpy(float)
        fit = fit_degradation_model("quadratic", cycle, y)
        if fit.parameters is None:
            continue
        fitted = predict_model("quadratic", cycle, fit.parameters)
        residual = g["SOH_clean"].to_numpy(float) - fitted
        residual -= np.nanmedian(residual)
        battery_draws: list[tuple[float, ...]] = []
        for iteration in range(samples):
            pseudo = fitted + _moving_block(residual, rng)
            boot = fit_degradation_model("quadratic", cycle, pseudo)
            if boot.parameters is None:
                continue
            a, d1, d2 = boot.parameters
            r100, r200, acceleration = quadratic_derived(d1, d2)
            eol = analytic_eol(a, d1, d2).item()
            row = (
                float(a), float(d1), float(d2), float(r100), float(r200),
                float(acceleration), float(eol) if np.isfinite(eol) else np.nan,
            )
            battery_draws.append(row)
            draws.append(
                {
                    "battery_id": int(battery_id), "iteration": iteration,
                    "a": row[0], "d1": row[1], "d2": row[2], "R100": row[3],
                    "R200": row[4], "A": row[5], "quadratic_eol": row[6],
                }
            )
        array = np.asarray(battery_draws, dtype=float)
        summary: dict[str, object] = {
            "battery_id": int(battery_id),
            "strategy": g["strategy"].iloc[0],
            "dataset_id": int(g["dataset_id"].iloc[0]),
            "requested_samples": samples,
            "valid_samples": len(array),
            "valid_fraction": len(array) / samples,
        }
        if len(array):
            for j, name in enumerate(("a", "d1", "d2", "R100", "R200", "A", "quadratic_eol")):
                finite = array[:, j][np.isfinite(array[:, j])]
                summary[f"{name}_median"] = float(np.median(finite)) if len(finite) else np.nan
                summary[f"{name}_ci95_low"] = float(np.quantile(finite, 0.025)) if len(finite) else np.nan
                summary[f"{name}_ci95_high"] = float(np.quantile(finite, 0.975)) if len(finite) else np.nan
            summary["d1_boundary_fraction"] = float(np.mean(array[:, 1] <= 1e-8))
            summary["d2_boundary_fraction"] = float(np.mean(array[:, 2] <= 1e-8))
            summary["d1_d2_correlation"] = (
                float(np.corrcoef(array[:, 1], array[:, 2])[0, 1])
                if np.std(array[:, 1]) > 0 and np.std(array[:, 2]) > 0 else np.nan
            )
        summaries.append(summary)
    return pd.DataFrame(draws), pd.DataFrame(summaries)


def build_cutoff_parameter_stability(candidate_fits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare native and stable quadratic parameters at cutoffs 100/150/200."""
    source = candidate_fits.loc[candidate_fits["model"] == "quadratic"].copy()
    parsed = pd.DataFrame(
        [parse_quadratic_parameters(v) for v in source["parameters_json"]],
        columns=["a", "d1", "d2"], index=source.index,
    )
    source = pd.concat([source, parsed], axis=1)
    source["R100"], source["R200"], source["A"] = quadratic_derived(source["d1"], source["d2"])
    wide = source.pivot(index="battery_id", columns="cutoff", values=["a", "d1", "d2", "R100", "R200", "A", "eol_cycle"])
    wide.columns = [f"{name}_{int(cutoff)}" for name, cutoff in wide.columns]
    wide = wide.reset_index()
    for name in ("a", "d1", "d2", "R100", "R200", "A", "eol_cycle"):
        for earlier in (100, 150):
            denominator = wide[f"{name}_200"].abs().clip(lower=1e-6)
            wide[f"{name}_relative_change_{earlier}_200"] = (
                (wide[f"{name}_{earlier}"] - wide[f"{name}_200"]).abs() / denominator
            )
    rows = []
    for name in ("d1", "d2", "R100", "A", "eol_cycle"):
        for earlier in (100, 150):
            values = wide[f"{name}_relative_change_{earlier}_200"].replace([np.inf, -np.inf], np.nan)
            rows.append(
                {
                    "parameter": name,
                    "comparison": f"{earlier}_vs_200",
                    "median_relative_change": float(values.median()),
                    "q25_relative_change": float(values.quantile(0.25)),
                    "q75_relative_change": float(values.quantile(0.75)),
                    "finite_n": int(values.notna().sum()),
                }
            )
    return wide, pd.DataFrame(rows)


def identifiability_summary(
    battery_parameters: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    cutoff_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """Diagnose compensation and choose the strategy-response parameterization."""
    correlation = float(battery_parameters[["d1", "d2"]].corr().iloc[0, 1])
    boundary = float(
        (battery_parameters["d1_lower_boundary"] | battery_parameters["d2_lower_boundary"]).mean()
    )
    boot_corr = float(bootstrap_summary["d1_d2_correlation"].median())
    native_stability = cutoff_summary.loc[
        cutoff_summary["parameter"].isin(["d1", "d2"]), "median_relative_change"
    ].median()
    stable_stability = cutoff_summary.loc[
        cutoff_summary["parameter"].isin(["R100", "A"]), "median_relative_change"
    ].median()
    compensation = abs(correlation) >= 0.65 or abs(boot_corr) >= 0.65 or boundary >= 0.25
    selected = "R100_A" if compensation or stable_stability < native_stability else "d1_d2"
    table = pd.DataFrame(
        [
            {"diagnostic": "cross_battery_d1_d2_correlation", "value": correlation},
            {"diagnostic": "median_within_battery_bootstrap_d1_d2_correlation", "value": boot_corr},
            {"diagnostic": "either_loss_parameter_boundary_fraction", "value": boundary},
            {"diagnostic": "native_cutoff_instability_median", "value": native_stability},
            {"diagnostic": "stable_cutoff_instability_median", "value": stable_stability},
            {"diagnostic": "compensation_flag", "value": float(compensation)},
            {"diagnostic": "selected_parameterization_R100_A", "value": float(selected == "R100_A")},
        ]
    )
    return table, selected


def extract_early_health_features(cleaned_cycles: pd.DataFrame) -> pd.DataFrame:
    """Extract strictly cycle-1--50 electro-thermal association features."""
    rows: list[dict[str, object]] = []
    for battery_id, group in cleaned_cycles.groupby("battery_id", sort=True):
        g = group.sort_values("cycle")
        early = g.loc[g["cycle"] <= 50]
        first = early.loc[early["cycle"].between(1, 10)]
        late = early.loc[early["cycle"].between(41, 50)]
        row: dict[str, object] = {
            "battery_id": int(battery_id), "strategy": g["strategy"].iloc[0],
            "dataset_id": int(g["dataset_id"].iloc[0]),
            "prediction_test": int(g["prediction_test"].iloc[0]),
            "C1": float(g["C1"].iloc[0]) if pd.notna(g["C1"].iloc[0]) else np.nan,
            "Q1": float(g["Q1"].iloc[0]), "C2": float(g["C2"].iloc[0]),
            "initial_capacity": float(g["initial_capacity"].iloc[0]),
        }
        specifications = {
            "IR": "IR_clean", "Tavg": "Tavg", "chargetime": "chargetime",
        }
        for label, column in specifications.items():
            row[f"{label}0"] = float(first[column].median())
            row[f"{label}_mean_1_50"] = float(early[column].mean())
            row[f"{label}_delta_early"] = float(late[column].median() - first[column].median())
            row[f"{label}_slope_1_50_per100"] = float(100.0 * robust_slope(early["cycle"], early[column]))
        rows.append(row)
    return pd.DataFrame(rows)


def build_strategy_table(
    battery_parameters: pd.DataFrame,
    battery_results: pd.DataFrame,
    summary: pd.DataFrame,
    early_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create battery- and strategy-level design tables without pseudo-replication."""
    meta = summary[["battery_id", "C1", "Q1", "C2"]]
    battery = battery_parameters.merge(meta, on="battery_id", how="left", validate="one_to_one")
    battery["mean_chargetime"] = battery["mean_chargetime_observed"]
    supplement = battery_results[["battery_id", "SOH_150", "SOH_200", *PHASE_SLOPES]]
    battery = battery.merge(supplement, on="battery_id", how="left", suffixes=("", "_feature"), validate="one_to_one")
    battery = battery.merge(
        early_features.drop(columns=["strategy", "dataset_id", "prediction_test", "C1", "Q1", "C2"]),
        on="battery_id", how="left", validate="one_to_one",
    )
    response_columns = ["a", "d1", "d2", "R100", "R200", "A", "quadratic_eol", "SOH_150", "SOH_200", *PHASE_SLOPES]
    early_columns = [
        c for c in early_features.columns
        if c not in {"battery_id", "strategy", "dataset_id", "prediction_test", "C1", "Q1", "C2"}
    ]
    aggregations = {c: (c, "median") for c in [*response_columns, *early_columns] if c in battery.columns}
    if "mean_chargetime" in battery.columns:
        aggregations["mean_chargetime"] = ("mean_chargetime", "median")
    aggregations.update(n_batteries=("battery_id", "size"), a_sd=("a", "std"), eol_sd=("quadratic_eol", "std"))
    strategy = (
        battery.groupby(["strategy", "strategy_code", "dataset_id", "C1", "Q1", "C2"], dropna=False)
        .agg(**aggregations).reset_index()
    )
    strategy["parameterized"] = strategy["C1"].notna()
    strategy["main_dataset3"] = (strategy["dataset_id"] == 3) & strategy["parameterized"]
    return battery, strategy


def add_phase_exposures(frame: pd.DataFrame, p: float) -> pd.DataFrame:
    result = frame.copy()
    q = result["Q1"] / 80.0
    result["exposure_p"] = float(p)
    result["E1"] = q * result["C1"] ** p
    result["E2"] = (1.0 - q) * result["C2"] ** p
    result["stress_sum_benchmark"] = result["E1"] + result["E2"]
    return result


@dataclass
class StandardizedRidge:
    predictors: tuple[str, ...]
    responses: tuple[str, ...]
    alpha: float
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    model: Ridge

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = self.x_scaler.transform(frame[list(self.predictors)].to_numpy(float))
        z = self.model.predict(x)
        z = np.asarray(z)
        if z.ndim == 1:
            z = z[:, None]
        values = self.y_scaler.inverse_transform(z)
        return pd.DataFrame(values, columns=self.responses, index=frame.index)

    def coefficients(self) -> pd.DataFrame:
        coef = np.asarray(self.model.coef_)
        if coef.ndim == 1:
            coef = coef[None, :]
        rows = []
        for j, response in enumerate(self.responses):
            for k, predictor in enumerate(self.predictors):
                rows.append(
                    {
                        "response": response, "predictor": predictor,
                        "standardized_beta": float(coef[j, k]), "alpha": self.alpha,
                    }
                )
        return pd.DataFrame(rows)


def fit_standardized_ridge(
    frame: pd.DataFrame,
    predictors: Sequence[str],
    responses: Sequence[str],
    alpha: float,
) -> StandardizedRidge:
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x = x_scaler.fit_transform(frame[list(predictors)].to_numpy(float))
    y = y_scaler.fit_transform(frame[list(responses)].to_numpy(float))
    model = Ridge(alpha=float(alpha)).fit(x, y)
    return StandardizedRidge(tuple(predictors), tuple(responses), float(alpha), x_scaler, y_scaler, model)


def loso_predictions(
    frame: pd.DataFrame,
    predictors: Sequence[str],
    responses: Sequence[str],
    alpha: float,
) -> pd.DataFrame:
    """Leave-one-strategy-out predictions with fold-local standardization."""
    rows: list[pd.DataFrame] = []
    for index in frame.index:
        train = frame.drop(index=index)
        test = frame.loc[[index]]
        model = fit_standardized_ridge(train, predictors, responses, alpha)
        prediction = model.predict(test)
        prediction.insert(0, "strategy", test["strategy"].iloc[0])
        prediction.insert(1, "strategy_code", test["strategy_code"].iloc[0])
        rows.append(prediction)
    return pd.concat(rows).sort_index()


def _condition_and_vif(frame: pd.DataFrame, predictors: Sequence[str]) -> pd.DataFrame:
    x = StandardScaler().fit_transform(frame[list(predictors)].to_numpy(float))
    design = np.column_stack([np.ones(len(x)), x])
    condition = float(np.linalg.cond(design))
    rows = []
    for j, predictor in enumerate(predictors):
        target = x[:, j]
        others = np.delete(x, j, axis=1)
        if others.shape[1] == 0:
            vif = 1.0
        else:
            fit = np.linalg.lstsq(np.column_stack([np.ones(len(x)), others]), target, rcond=None)[0]
            residual = target - np.column_stack([np.ones(len(x)), others]) @ fit
            r2 = 1.0 - np.sum(residual**2) / np.sum((target - target.mean()) ** 2)
            vif = float(1.0 / max(1.0 - r2, 1e-12))
        rows.append({"predictor": predictor, "VIF": vif, "condition_number": condition})
    return pd.DataFrame(rows)


def compare_strategy_models(
    strategy_table: pd.DataFrame,
    responses: Sequence[str],
    *,
    p_grid: Iterable[float] = (1, 1.5, 2, 2.5, 3),
    alphas: Iterable[float] = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """LOSO comparison of raw parameters and separate phase exposures."""
    frame = strategy_table.loc[strategy_table["main_dataset3"]].copy()
    models: list[tuple[str, float | None, list[str]]] = [("raw_parameters", None, ["C1", "Q1", "C2"])]
    for p in p_grid:
        models.append(("separate_phase_exposure", float(p), ["E1", "E2"]))
        models.append(("summed_stress_benchmark", float(p), ["stress_sum_benchmark"]))
    rows, predictions, collinearity = [], [], []
    for model_name, p, predictors in models:
        model_frame = add_phase_exposures(frame, p) if p is not None else frame.copy()
        for alpha in alphas:
            predicted = loso_predictions(model_frame, predictors, responses, float(alpha))
            normalized = []
            for response in responses:
                actual = model_frame.loc[predicted.index, response].to_numpy(float)
                estimate = predicted[response].to_numpy(float)
                scale = max(float(np.std(actual, ddof=0)), 1e-12)
                rmse = sqrt(mean_squared_error(actual, estimate))
                normalized.append(rmse / scale)
                rows.append(
                    {
                        "model": model_name, "p": p, "alpha": float(alpha),
                        "response": response, "loso_rmse": rmse,
                        "loso_mae": mean_absolute_error(actual, estimate),
                        "normalized_loso_rmse": rmse / scale,
                    }
                )
                for idx, actual_value, predicted_value in zip(predicted.index, actual, estimate):
                    predictions.append(
                        {
                            "model": model_name, "p": p, "alpha": float(alpha),
                            "strategy": model_frame.loc[idx, "strategy"],
                            "strategy_code": model_frame.loc[idx, "strategy_code"],
                            "response": response, "actual": actual_value,
                            "predicted": predicted_value,
                        }
                    )
            rows.append(
                {
                    "model": model_name, "p": p, "alpha": float(alpha),
                    "response": "combined", "loso_rmse": np.nan, "loso_mae": np.nan,
                    "normalized_loso_rmse": float(np.mean(normalized)),
                }
            )
        diagnostics = _condition_and_vif(model_frame, predictors)
        diagnostics.insert(0, "model", model_name)
        diagnostics.insert(1, "p", p)
        collinearity.append(diagnostics)
    diagnostic_rows = [record for table in collinearity for record in table.to_dict("records")]
    return pd.DataFrame(rows), pd.DataFrame(predictions), pd.DataFrame(diagnostic_rows)


def select_strategy_model(comparison: pd.DataFrame) -> pd.Series:
    combined = comparison.loc[comparison["response"] == "combined"].copy()
    direct = combined.loc[combined["model"] != "summed_stress_benchmark"]
    best = direct.sort_values(["normalized_loso_rmse", "model", "alpha"]).iloc[0]
    return best


def fit_selected_strategy_model(
    strategy_table: pd.DataFrame,
    responses: Sequence[str],
    selection: pd.Series,
) -> tuple[StandardizedRidge, pd.DataFrame]:
    frame = strategy_table.loc[strategy_table["main_dataset3"]].copy()
    if selection["model"] == "raw_parameters":
        predictors = ["C1", "Q1", "C2"]
    else:
        frame = add_phase_exposures(frame, float(selection["p"]))
        predictors = ["E1", "E2"]
    model = fit_standardized_ridge(frame, predictors, responses, float(selection["alpha"]))
    return model, frame


def grouped_bootstrap_strategy_model(
    battery_table: pd.DataFrame,
    selection: pd.Series,
    responses: Sequence[str],
    *,
    samples: int = 1_000,
    seed: int = RANDOM_SEED + 601,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strategy-first then battery-within-strategy coefficient bootstrap."""
    rng = np.random.default_rng(seed)
    source = battery_table.loc[(battery_table["dataset_id"] == 3) & battery_table["C1"].notna()].copy()
    groups = {name: group for name, group in source.groupby("strategy", sort=True)}
    strategies = np.asarray(list(groups))
    p = float(selection["p"]) if pd.notna(selection["p"]) else None
    predictors = ["C1", "Q1", "C2"] if selection["model"] == "raw_parameters" else ["E1", "E2"]
    draws, summaries = [], []
    for iteration in range(samples):
        sampled_names = rng.choice(strategies, len(strategies), replace=True)
        rows = []
        for draw_id, name in enumerate(sampled_names):
            group = groups[str(name)]
            sampled = group.iloc[rng.integers(0, len(group), len(group))]
            row = {
                "strategy": f"{name}__{draw_id}", "strategy_code": f"B{draw_id}",
                "C1": float(group["C1"].iloc[0]), "Q1": float(group["Q1"].iloc[0]),
                "C2": float(group["C2"].iloc[0]),
            }
            row.update({response: float(sampled[response].median()) for response in responses})
            rows.append(row)
        sample = pd.DataFrame(rows)
        if p is not None:
            sample = add_phase_exposures(sample, p)
        if np.linalg.matrix_rank(StandardScaler().fit_transform(sample[predictors])) < len(predictors):
            continue
        fitted = fit_standardized_ridge(sample, predictors, responses, float(selection["alpha"]))
        coefficients = fitted.coefficients()
        coefficients["iteration"] = iteration
        draws.append(coefficients)
    distribution = pd.concat(draws, ignore_index=True) if draws else pd.DataFrame()
    if not distribution.empty:
        for (response, predictor), group in distribution.groupby(["response", "predictor"]):
            values = group["standardized_beta"].to_numpy(float)
            summaries.append(
                {
                    "response": response, "predictor": predictor,
                    "requested_samples": samples, "valid_samples": group["iteration"].nunique(),
                    "beta_median": float(np.median(values)),
                    "beta_ci95_low": float(np.quantile(values, 0.025)),
                    "beta_ci95_high": float(np.quantile(values, 0.975)),
                    "same_sign_fraction": float(max(np.mean(values > 0), np.mean(values < 0))),
                }
            )
    return distribution, pd.DataFrame(summaries)


def predict_native_parameters(
    model: StandardizedRidge,
    frame: pd.DataFrame,
    parameterization: str,
) -> pd.DataFrame:
    """Predict strategy responses and convert them to physical d1/d2/EOL."""
    result = frame.copy()
    predicted = model.predict(result)
    for column in predicted:
        result[f"predicted_{column}_raw"] = predicted[column]
    if parameterization == "R100_A":
        d1, d2 = stable_to_native(predicted["R100"], predicted["A"])
    else:
        d1, d2 = np.maximum(predicted["d1"], 0.0), np.maximum(predicted["d2"], 0.0)
    result["predicted_d1"] = d1
    result["predicted_d2"] = d2
    result["predicted_R100"], result["predicted_R200"], result["predicted_A"] = quadratic_derived(d1, d2)
    result["predicted_a_ref"] = 1.0
    result["predicted_SOH200"] = quadratic_soh(200.0, 1.0, d1, d2)
    result["predicted_eol_a1"] = analytic_eol(1.0, d1, d2)
    result["projection_applied"] = (
        (result.get("predicted_d1_raw", result["predicted_d1"]) < 0)
        | (result.get("predicted_d2_raw", result["predicted_d2"]) < 0)
        | (result.get("predicted_A_raw", result["predicted_A"]) < 0)
    )
    return result


def strategy_eol_recreation(predicted: pd.DataFrame, strategy_table: pd.DataFrame) -> pd.DataFrame:
    observed = strategy_table[["strategy", "strategy_code", "quadratic_eol", "a", "d1", "d2"]].rename(
        columns={"quadratic_eol": "observed_quadratic_eol", "a": "observed_a",
                 "d1": "observed_d1", "d2": "observed_d2"}
    )
    result = predicted.merge(observed, on=["strategy", "strategy_code"], how="left", validate="one_to_one")
    result["observed_eol_a1"] = analytic_eol(1.0, result["observed_d1"], result["observed_d2"])
    result["predicted_minus_observed_eol_a1"] = result["predicted_eol_a1"] - result["observed_eol_a1"]
    result["a_offset_effect_on_observed_eol"] = result["observed_quadratic_eol"] - result["observed_eol_a1"]
    return result


def local_parameter_matches(strategy_table: pd.DataFrame) -> pd.DataFrame:
    """Nearest observed matches in standardized parameter space (dataset 3)."""
    frame = strategy_table.loc[strategy_table["main_dataset3"]].copy().reset_index(drop=True)
    x = StandardScaler().fit_transform(frame[["C1", "Q1", "C2"]].to_numpy(float))
    rows = []
    for i in range(len(frame)):
        distances = np.linalg.norm(x - x[i], axis=1)
        distances[i] = np.inf
        j = int(np.argmin(distances))
        rows.append(
            {
                "strategy": frame.loc[i, "strategy"], "strategy_code": frame.loc[i, "strategy_code"],
                "matched_strategy": frame.loc[j, "strategy"],
                "matched_strategy_code": frame.loc[j, "strategy_code"],
                "standardized_distance": float(distances[j]),
                "delta_C1": float(frame.loc[i, "C1"] - frame.loc[j, "C1"]),
                "delta_Q1": float(frame.loc[i, "Q1"] - frame.loc[j, "Q1"]),
                "delta_C2": float(frame.loc[i, "C2"] - frame.loc[j, "C2"]),
                "delta_R100": float(frame.loc[i, "R100"] - frame.loc[j, "R100"]),
                "delta_A": float(frame.loc[i, "A"] - frame.loc[j, "A"]),
                "delta_eol": float(frame.loc[i, "quadratic_eol"] - frame.loc[j, "quadratic_eol"]),
            }
        )
    return pd.DataFrame(rows)


def parameter_specific_matches(strategy_table: pd.DataFrame) -> pd.DataFrame:
    """Quasi-controls that maximize one contrast while minimizing nuisance distance."""
    frame = strategy_table.loc[strategy_table["main_dataset3"]].copy().reset_index(drop=True)
    parameters = ["C1", "Q1", "C2"]
    scales = frame[parameters].std(ddof=0).replace(0, 1.0)
    response_columns = [
        "d1", "d2", "R100", "A", "SOH_200", "quadratic_eol", "IR0", "Tavg_mean_1_50",
    ]
    rows = []
    for target in parameters:
        nuisance = [p for p in parameters if p != target]
        candidates = []
        for i in range(len(frame)):
            for j in range(i + 1, len(frame)):
                target_contrast = abs(float(frame.loc[i, target] - frame.loc[j, target])) / scales[target]
                nuisance_distance = float(np.linalg.norm(
                    (frame.loc[i, nuisance].to_numpy(float) - frame.loc[j, nuisance].to_numpy(float))
                    / scales[nuisance].to_numpy(float)
                ))
                score = nuisance_distance / max(target_contrast, 1e-9)
                candidates.append((score, nuisance_distance, target_contrast, i, j))
        for rank, (score, nuisance_distance, target_contrast, i, j) in enumerate(sorted(candidates)[:3], start=1):
            row = {
                "target_parameter": target, "match_rank": rank,
                "strategy_low": frame.loc[i, "strategy_code"], "strategy_high": frame.loc[j, "strategy_code"],
                "target_standardized_contrast": target_contrast,
                "nuisance_standardized_distance": nuisance_distance,
                "match_score": score,
                "match_quality": "good" if nuisance_distance <= 0.75 else "moderate" if nuisance_distance <= 1.25 else "weak",
            }
            for response in response_columns:
                if response in frame:
                    row[f"delta_{response}"] = float(frame.loc[j, response] - frame.loc[i, response])
            rows.append(row)
    return pd.DataFrame(rows)


def standardized_ols_coefficients(
    frame: pd.DataFrame,
    predictors: Sequence[str],
    responses: Sequence[str],
    *,
    model_name: str,
) -> pd.DataFrame:
    """Descriptive standardized OLS coefficients and conventional SEs."""
    x = StandardScaler().fit_transform(frame[list(predictors)].to_numpy(float))
    design = np.column_stack([np.ones(len(x)), x])
    rows = []
    for response in responses:
        y = StandardScaler().fit_transform(frame[[response]].to_numpy(float)).ravel()
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - design @ beta
        dof = max(len(y) - design.shape[1], 1)
        sigma2 = float(np.sum(residual**2) / dof)
        se = np.sqrt(np.maximum(np.diag(sigma2 * np.linalg.pinv(design.T @ design)), 0.0))
        for index, predictor in enumerate(predictors, start=1):
            rows.append(
                {
                    "model": model_name, "response": response, "predictor": predictor,
                    "standardized_beta": float(beta[index]), "conventional_standard_error": float(se[index]),
                    "n_strategies": len(frame),
                    "interpretation": "descriptive direction only; small-sample assumptions are not trusted",
                }
            )
    return pd.DataFrame(rows)


def early_feature_diagnostics(battery_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = [
        "initial_capacity", "IR0", "IR_delta_early", "IR_slope_1_50_per100",
        "Tavg0", "Tavg_mean_1_50", "Tavg_delta_early", "Tavg_slope_1_50_per100",
        "chargetime0", "chargetime_mean_1_50", "chargetime_delta_early",
        "chargetime_slope_1_50_per100",
    ]
    features = [feature for feature in features if feature in battery_table]
    correlation = battery_table[features].corr(method="spearman").stack().rename("spearman_rho").reset_index()
    correlation.columns = ["feature_1", "feature_2", "spearman_rho"]
    vif = _condition_and_vif(battery_table.dropna(subset=features), features)
    return correlation, vif


def structure_matched_comparison(strategy_table: pd.DataFrame) -> pd.DataFrame:
    """Exact-parameter S3 versus S9 comparison isolating the structure/batch contrast."""
    selected = strategy_table.loc[strategy_table["strategy_code"].isin(["S3", "S9"])].copy()
    if len(selected) != 2:
        return pd.DataFrame()
    s3 = selected.loc[selected["strategy_code"] == "S3"].iloc[0]
    s9 = selected.loc[selected["strategy_code"] == "S9"].iloc[0]
    rows = []
    for response in ("d1", "d2", "R100", "A", "SOH_200", "SOH_slope_50_200_per100", "quadratic_eol"):
        rows.append(
            {
                "comparison": "S9_NEWSTRUCTURE_minus_S3_old_structure",
                "response": response,
                "same_C1_Q1_C2": bool(np.allclose(
                    s3[["C1", "Q1", "C2"]].to_numpy(dtype=float),
                    s9[["C1", "Q1", "C2"]].to_numpy(dtype=float),
                )),
                "S3_value": float(s3[response]), "S9_value": float(s9[response]),
                "difference": float(s9[response] - s3[response]),
                "interpretation": "structure/batch-associated contrast; not a charging-parameter effect",
            }
        )
    return pd.DataFrame(rows)


def mechanism_model_comparison(battery_table: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-strategy-out comparison of M0/M1/M2 association models."""
    source = battery_table.loc[(battery_table["prediction_test"].fillna(0) == 0) if "prediction_test" in battery_table else np.ones(len(battery_table), dtype=bool)].copy()
    source = source.loc[source["C1"].notna()]
    models = {
        "M0_strategy": ["C1", "Q1", "C2"],
        "M1_plus_initial_capacity": ["C1", "Q1", "C2", "initial_capacity"],
        "M2_plus_early_health": [
            "C1", "Q1", "C2", "initial_capacity", "IR0", "IR_delta_early",
            "Tavg_mean_1_50", "Tavg_delta_early", "chargetime_mean_1_50",
            "chargetime_delta_early",
        ],
    }
    responses = ["SOH_slope_50_200_per100", "SOH_slope_100_150_per100", "SOH_slope_150_200_per100", "SOH_200"]
    rows = []
    for model_name, predictors in models.items():
        for response in responses:
            actual, estimate = [], []
            for held_out_strategy in source["strategy"].unique():
                train = source.loc[source["strategy"] != held_out_strategy].dropna(subset=[*predictors, response])
                test = source.loc[source["strategy"] == held_out_strategy].dropna(subset=[*predictors, response])
                if test.empty:
                    continue
                fit = fit_standardized_ridge(train, predictors, [response], alpha=3.0)
                predicted = fit.predict(test)[response].to_numpy(float)
                actual.extend(test[response].to_numpy(float))
                estimate.extend(predicted)
            scale = max(float(np.std(actual, ddof=0)), 1e-12)
            rows.append(
                {
                    "model": model_name, "response": response, "n_batteries": len(actual),
                    "validation": "leave_one_strategy_out",
                    "lobo_rmse": sqrt(mean_squared_error(actual, estimate)),
                    "normalized_lobo_rmse": sqrt(mean_squared_error(actual, estimate)) / scale,
                    "lobo_mae": mean_absolute_error(actual, estimate),
                }
            )
    return pd.DataFrame(rows)


def evidence_matrix(
    parameter_bootstrap_summary: pd.DataFrame,
    strategy_comparison: pd.DataFrame,
    mechanism_comparison: pd.DataFrame,
    structure_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Seven-channel evidence audit with deliberately conservative labels."""
    boot_stable = (
        parameter_bootstrap_summary["same_sign_fraction"].min() >= 0.8
        if not parameter_bootstrap_summary.empty else False
    )
    best = select_strategy_model(strategy_comparison)
    benchmark = strategy_comparison.loc[
        (strategy_comparison["response"] == "combined")
        & (strategy_comparison["model"] == "summed_stress_benchmark")
    ]["normalized_loso_rmse"].min()
    m = mechanism_comparison.pivot(index="response", columns="model", values="normalized_lobo_rmse")
    mechanism_gain = float((m["M1_plus_initial_capacity"] - m["M2_plus_early_health"]).median())
    rows = [
        ("observed_parameter_space", True, "six independent dataset-3 strategy points"),
        ("LOSO_generalization", float(best["normalized_loso_rmse"]) < 1.0, f"best normalized RMSE={best['normalized_loso_rmse']:.3f}"),
        ("bootstrap_coefficient_sign", boot_stable, "strategy-first grouped bootstrap"),
        ("separate_phase_vs_summed_stress", float(best["normalized_loso_rmse"]) <= float(benchmark), "separate phase model is required to outperform the retained benchmark"),
        ("early_health_increment", mechanism_gain > 0.05, f"median normalized LOBO improvement={mechanism_gain:.3f}"),
        ("exact_parameter_structure_contrast", not structure_comparison.empty, "S3/S9 expose unresolved structure/batch variation"),
        ("causal_identification", False, "observational sparse design with collinearity and batch confounding"),
    ]
    output = []
    for channel, supported, detail in rows:
        if channel == "causal_identification":
            label = "unsupported"
        elif supported and boot_stable and channel in {"LOSO_generalization", "bootstrap_coefficient_sign"}:
            label = "robust"
        elif supported:
            label = "directional_but_limited"
        else:
            label = "unsupported"
        output.append({"evidence_channel": channel, "classification": label, "detail": detail})
    return pd.DataFrame(output)


def predictor_evidence_matrix(
    raw_bootstrap: pd.DataFrame,
    exposure_bootstrap: pd.DataFrame,
    strategy_comparison: pd.DataFrame,
    mechanism_comparison: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """Evidence classification for each raw/exposure predictor across seven channels."""
    bootstrap = pd.concat(
        [raw_bootstrap.assign(model_family="raw"), exposure_bootstrap.assign(model_family="exposure")],
        ignore_index=True,
    )
    mechanism_wide = mechanism_comparison.pivot(index="response", columns="model", values="normalized_lobo_rmse")
    mechanism_gain = float(
        (mechanism_wide["M1_plus_initial_capacity"] - mechanism_wide["M2_plus_early_health"]).median()
    )
    best_scores = (
        strategy_comparison.loc[strategy_comparison["response"] == "combined"]
        .groupby("model")["normalized_loso_rmse"].min()
    )
    rows = []
    for predictor in ("C1", "Q1", "C2", "E1", "E2"):
        family = "raw" if predictor in {"C1", "Q1", "C2"} else "exposure"
        model_name = "raw_parameters" if family == "raw" else "separate_phase_exposure"
        subset = bootstrap.loc[
            (bootstrap["model_family"] == family) & (bootstrap["predictor"] == predictor)
        ]
        sign_stability = float(subset["same_sign_fraction"].min()) if len(subset) else np.nan
        loso = float(best_scores.get(model_name, np.nan))
        match_support = (
            bool((matches.loc[matches["target_parameter"] == predictor, "match_quality"] != "weak").any())
            if predictor in {"C1", "Q1", "C2"} else False
        )
        channels = {
            "direct_SOH_slope_statistics": "directional_but_limited",
            "quadratic_parameter_bridge": "directional_but_limited" if len(subset) else "unsupported",
            "LOSO": "robust" if loso < 0.8 else "directional_but_limited" if loso < 1.0 else "unsupported",
            "grouped_bootstrap": "robust" if sign_stability >= 0.9 else "directional_but_limited" if sign_stability >= 0.75 else "unsupported",
            "local_matching": "directional_but_limited" if match_support else "unsupported",
            "early_IR_T_charge_association": "directional_but_limited" if mechanism_gain > 0.05 else "unsupported",
            "batch_sensitivity": "unsupported",
        }
        robust_count = sum(value == "robust" for value in channels.values())
        directional_count = sum(value == "directional_but_limited" for value in channels.values())
        overall = "robust" if robust_count >= 3 and channels["LOSO"] == "robust" else (
            "directional_but_limited" if robust_count + directional_count >= 3 else "unsupported"
        )
        for channel, classification in channels.items():
            rows.append(
                {
                    "predictor": predictor, "evidence_channel": channel,
                    "classification": classification, "overall_classification": overall,
                    "minimum_bootstrap_sign_stability": sign_stability,
                    "best_combined_normalized_LOSO_RMSE": loso,
                }
            )
    return pd.DataFrame(rows)
