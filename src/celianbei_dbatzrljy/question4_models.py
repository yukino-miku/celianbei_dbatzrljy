"""Low-complexity surrogate models and trust-domain tools for question four."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import PROJECT_ROOT, RANDOM_SEED


PARAMETERS = ["C1", "Q1", "C2"]
P_VALUES = (2.0, 2.5, 3.0)
RESPONSE_COLUMNS = {
    "SOH200": "SOH_200_median",
    "slope50_200": "SOH_slope_50_200_per100_median",
    "EOL": "eol_point_median",
}
CHARGE_MODELS = (
    "physical_only",
    "physical_offset",
    "physical_q1_linear",
    "physical_q1_quadratic",
    "physical_affine",
)


def load_question4_data(project_root: Path = PROJECT_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse question-two strategy/battery tables without rerunning cleaning."""
    table_root = project_root / "outputs" / "question2" / "tables"
    strategies = pd.read_csv(table_root / "strategy_analysis_data.csv")
    batteries = pd.read_csv(table_root / "battery_analysis_data.csv")
    expected = {f"S{i}" for i in range(1, 10)}
    if set(strategies["strategy_code"]) != expected:
        raise ValueError("Question-two strategy table is incomplete.")
    if batteries["battery_id"].nunique() != 40:
        raise ValueError("Question-four inputs must contain exactly 40 non-test batteries.")
    strategies = add_physical_quantities(strategies)
    strategies["degradation_rate"] = -strategies["SOH_slope_50_200_per100_median"]
    strategies["SOH200_harm"] = 1.0 - strategies["SOH_200_median"]
    strategies["EOL_harm"] = -strategies["eol_point_median"]
    return strategies, batteries


def add_physical_quantities(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the physical time and an audit-only rendering of the supplied minus sign."""
    result = frame.copy()
    q1 = result["Q1"] / 100.0
    q2 = (80.0 - result["Q1"]) / 100.0
    result["phase1_ideal_minutes"] = 60.0 * q1 / result["C1"]
    result["phase2_ideal_minutes"] = 60.0 * q2 / result["C2"]
    result["T_ideal_physical"] = result["phase1_ideal_minutes"] + result["phase2_ideal_minutes"]
    result["T_ideal_as_written_minus"] = result["phase1_ideal_minutes"] - result["phase2_ideal_minutes"]
    if "mean_chargetime" in result:
        result["actual_minus_physical_minutes"] = result["mean_chargetime"] - result["T_ideal_physical"]
    return result


def parameterized_scope(strategies: pd.DataFrame, scope: str) -> pd.DataFrame:
    valid = strategies.dropna(subset=PARAMETERS).copy()
    if scope == "dataset3_only":
        return valid.query("dataset_id == 3").reset_index(drop=True)
    if scope == "all_parameterized":
        return valid.reset_index(drop=True)
    raise ValueError(f"Unknown scope: {scope}")


def soc_weighted_stress(frame: pd.DataFrame, p: float) -> np.ndarray:
    q1_fraction = frame["Q1"].to_numpy(float) / 80.0
    return q1_fraction * frame["C1"].to_numpy(float) ** p + (1.0 - q1_fraction) * frame[
        "C2"
    ].to_numpy(float) ** p


def _charge_base_design(frame: pd.DataFrame, model: str) -> tuple[np.ndarray, np.ndarray]:
    physical = frame["T_ideal_physical"].to_numpy(float)
    q = frame["Q1"].to_numpy(float) / 80.0
    if model == "physical_only":
        return physical, np.empty((len(frame), 0))
    if model == "physical_offset":
        return physical, np.ones((len(frame), 1))
    if model == "physical_q1_linear":
        return physical, np.column_stack([np.ones(len(frame)), q])
    if model == "physical_q1_quadratic":
        return physical, np.column_stack([np.ones(len(frame)), q, q**2])
    if model == "physical_affine":
        return np.zeros(len(frame)), np.column_stack([np.ones(len(frame)), physical])
    raise ValueError(f"Unknown charge model: {model}")


@dataclass
class ChargeTimeModel:
    name: str
    coefficients: np.ndarray
    covariance: np.ndarray
    residual_sigma: float
    cv_rmse: float = np.nan

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        base, design = _charge_base_design(frame, self.name)
        if design.shape[1] == 0:
            return base
        return base + design @ self.coefficients

    def prediction_se(self, frame: pd.DataFrame) -> np.ndarray:
        _, design = _charge_base_design(frame, self.name)
        if design.shape[1] == 0 or self.covariance.size == 0:
            fallback = self.cv_rmse if np.isfinite(self.cv_rmse) else self.residual_sigma
            return np.full(len(frame), fallback)
        variance = np.einsum("ij,jk,ik->i", design, self.covariance, design)
        return np.sqrt(np.maximum(variance, 0.0) + self.residual_sigma**2)


def fit_charge_time_model(frame: pd.DataFrame, model: str) -> ChargeTimeModel:
    base, design = _charge_base_design(frame, model)
    residual_target = frame["mean_chargetime"].to_numpy(float) - base
    if design.shape[1] == 0:
        residuals = residual_target
        sigma = float(np.sqrt(np.mean(residuals**2)))
        return ChargeTimeModel(model, np.array([]), np.empty((0, 0)), sigma)
    coefficients, *_ = np.linalg.lstsq(design, residual_target, rcond=None)
    residuals = residual_target - design @ coefficients
    dof = max(len(frame) - design.shape[1], 1)
    sigma2 = float(np.sum(residuals**2) / dof)
    covariance = sigma2 * np.linalg.pinv(design.T @ design)
    return ChargeTimeModel(model, coefficients, covariance, float(np.sqrt(sigma2)))


def charge_time_cross_validation(strategies: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in ("dataset3_only", "all_parameterized"):
        data = parameterized_scope(strategies, scope)
        for model in CHARGE_MODELS:
            predictions = []
            for index in range(len(data)):
                train = data.drop(index=index)
                fitted = fit_charge_time_model(train, model)
                predictions.append(float(fitted.predict(data.iloc[[index]])[0]))
            actual = data["mean_chargetime"].to_numpy(float)
            predictions_array = np.asarray(predictions)
            errors = predictions_array - actual
            rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "n_strategies": len(data),
                    "n_parameters": _charge_base_design(data, model)[1].shape[1],
                    "LOSO_MAE_minutes": float(np.mean(np.abs(errors))),
                    "LOSO_RMSE_minutes": float(np.sqrt(np.mean(errors**2))),
                    "LOSO_MaxAE_minutes": float(np.max(np.abs(errors))),
                    "bias_minutes": float(np.mean(errors)),
                }
            )
    return pd.DataFrame(rows)


def select_charge_time_model(validation: pd.DataFrame) -> str:
    """Prefer an empirical offset when it costs at most 5% LOSO RMSE versus the best model."""
    local = validation.query("scope == 'dataset3_only'").copy()
    best_rmse = local["LOSO_RMSE_minutes"].min()
    offset = local.query("model == 'physical_offset'").iloc[0]
    if offset["LOSO_RMSE_minutes"] <= best_rmse * 1.05:
        return "physical_offset"
    threshold = best_rmse * 1.10
    eligible = local.query("LOSO_RMSE_minutes <= @threshold")
    return str(eligible.sort_values(["n_parameters", "LOSO_RMSE_minutes"]).iloc[0]["model"])


def _proxy_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    response: str,
    model: str,
    p: float,
) -> np.ndarray:
    y = train[response].to_numpy(float)
    if model == "stress_linear":
        x_train = soc_weighted_stress(train, p).reshape(-1, 1)
        x_test = soc_weighted_stress(test, p).reshape(-1, 1)
        fitted = LinearRegression().fit(x_train, y)
    elif model == "stress_loglinear":
        x_train = soc_weighted_stress(train, p).reshape(-1, 1)
        x_test = soc_weighted_stress(test, p).reshape(-1, 1)
        if response == RESPONSE_COLUMNS["SOH200"]:
            transformed = np.log(np.maximum(1.0 - y, 1e-6))
            return 1.0 - np.exp(LinearRegression().fit(x_train, transformed).predict(x_test))
        if response == RESPONSE_COLUMNS["slope50_200"]:
            transformed = np.log(np.maximum(-y, 1e-6))
            return -np.exp(LinearRegression().fit(x_train, transformed).predict(x_test))
        transformed = np.log(np.maximum(y, 1e-6))
        return np.exp(LinearRegression().fit(x_train, transformed).predict(x_test))
    elif model == "stress_quadratic":
        stress_train = soc_weighted_stress(train, p)
        stress_test = soc_weighted_stress(test, p)
        x_train = np.column_stack([stress_train, stress_train**2])
        x_test = np.column_stack([stress_test, stress_test**2])
        fitted = make_pipeline(StandardScaler(), Ridge(alpha=0.5)).fit(x_train, y)
    elif model == "phase_exposure_linear":
        q_train = train["Q1"].to_numpy(float) / 80.0
        q_test = test["Q1"].to_numpy(float) / 80.0
        x_train = np.column_stack(
            [q_train * train["C1"].to_numpy(float) ** p, (1 - q_train) * train["C2"].to_numpy(float) ** p]
        )
        x_test = np.column_stack(
            [q_test * test["C1"].to_numpy(float) ** p, (1 - q_test) * test["C2"].to_numpy(float) ** p]
        )
        fitted = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x_train, y)
    elif model == "direct_parameter_ridge":
        x_train = train[PARAMETERS].to_numpy(float)
        x_test = test[PARAMETERS].to_numpy(float)
        fitted = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x_train, y)
    else:
        raise ValueError(f"Unknown proxy model: {model}")
    return fitted.predict(x_test)


def proxy_cross_validation(strategies: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    models = (
        "stress_linear",
        "stress_loglinear",
        "stress_quadratic",
        "phase_exposure_linear",
        "direct_parameter_ridge",
    )
    for scope in ("dataset3_only", "all_parameterized"):
        data = parameterized_scope(strategies, scope)
        for label, response in RESPONSE_COLUMNS.items():
            actual = data[response].to_numpy(float)
            scale = float(np.ptp(actual)) or 1.0
            for model in models:
                p_options = P_VALUES if model != "direct_parameter_ridge" else (np.nan,)
                for p in p_options:
                    predictions = []
                    for index in range(len(data)):
                        train = data.drop(index=index)
                        predictions.append(
                            float(_proxy_predictions(train, data.iloc[[index]], response, model, float(p or 2.0))[0])
                        )
                    errors = np.asarray(predictions) - actual
                    rows.append(
                        {
                            "scope": scope,
                            "response": label,
                            "response_column": response,
                            "model": model,
                            "p": p,
                            "n_strategies": len(data),
                            "LOSO_MAE": float(np.mean(np.abs(errors))),
                            "LOSO_RMSE": float(np.sqrt(np.mean(errors**2))),
                            "LOSO_MaxAE": float(np.max(np.abs(errors))),
                            "normalized_LOSO_RMSE": float(np.sqrt(np.mean(errors**2)) / scale),
                            "prediction_actual_correlation": float(np.corrcoef(predictions, actual)[0, 1]),
                        }
                    )
    return pd.DataFrame(rows)


def select_main_stress_p(validation: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    local = validation.query("scope == 'dataset3_only' and model == 'stress_loglinear'")
    summary = (
        local.groupby("p")["normalized_LOSO_RMSE"]
        .agg(["mean", "max"])
        .reset_index()
        .rename(columns={"mean": "mean_normalized_LOSO_RMSE", "max": "worst_normalized_LOSO_RMSE"})
    )
    summary["selected"] = summary["mean_normalized_LOSO_RMSE"].eq(summary["mean_normalized_LOSO_RMSE"].min())
    selected = float(summary.loc[summary["selected"], "p"].iloc[0])
    return selected, summary


@dataclass
class StressResponseModel:
    response: str
    p: float
    intercept: float
    slope: float
    covariance: np.ndarray
    residual_sigma: float
    n_observations: int
    transform: str = "identity"

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        linear = self.intercept + self.slope * soc_weighted_stress(frame, self.p)
        if self.transform == "soh_harm_log":
            return 1.0 - np.exp(linear)
        if self.transform == "degradation_log":
            return -np.exp(linear)
        if self.transform == "positive_log":
            return np.exp(linear)
        return linear

    def mean_prediction_se(self, frame: pd.DataFrame) -> np.ndarray:
        stress = soc_weighted_stress(frame, self.p)
        design = np.column_stack([np.ones(len(frame)), stress])
        variance = np.einsum("ij,jk,ik->i", design, self.covariance, design)
        linear_se = np.sqrt(np.maximum(variance, 0.0))
        if self.transform == "soh_harm_log":
            return np.exp(self.intercept + self.slope * stress) * linear_se
        if self.transform in {"degradation_log", "positive_log"}:
            return np.exp(self.intercept + self.slope * stress) * linear_se
        return linear_se


def fit_stress_response_model(
    frame: pd.DataFrame,
    response: str,
    p: float,
    *,
    enforce_physical_direction: bool = True,
) -> StressResponseModel:
    stress = soc_weighted_stress(frame, p)
    y = frame[response].to_numpy(float)
    if response == RESPONSE_COLUMNS["SOH200"]:
        transformed_y = np.log(np.maximum(1.0 - y, 1e-6))
        transform = "soh_harm_log"
        expected_sign = 1.0
    elif response == RESPONSE_COLUMNS["slope50_200"]:
        transformed_y = np.log(np.maximum(-y, 1e-6))
        transform = "degradation_log"
        expected_sign = 1.0
    elif response == RESPONSE_COLUMNS["EOL"]:
        transformed_y = np.log(np.maximum(y, 1e-6))
        transform = "positive_log"
        expected_sign = -1.0
    else:
        transformed_y = y
        transform = "identity"
        expected_sign = -1.0
    design = np.column_stack([np.ones(len(frame)), stress])
    coefficients, *_ = np.linalg.lstsq(design, transformed_y, rcond=None)
    if enforce_physical_direction and coefficients[1] * expected_sign < 0:
        coefficients = np.array([float(np.mean(transformed_y)), 0.0])
    residuals = transformed_y - design @ coefficients
    dof = max(len(frame) - 2, 1)
    sigma2 = float(np.sum(residuals**2) / dof)
    covariance = sigma2 * np.linalg.pinv(design.T @ design)
    return StressResponseModel(
        response=response,
        p=p,
        intercept=float(coefficients[0]),
        slope=float(coefficients[1]),
        covariance=covariance,
        residual_sigma=float(np.sqrt(sigma2)),
        n_observations=len(frame),
        transform=transform,
    )


def generate_parameter_grid(strategies: pd.DataFrame) -> pd.DataFrame:
    """Transparent 0.05 C / 1% SOC rectangular grid over observed dataset-three bounds."""
    local = parameterized_scope(strategies, "dataset3_only")
    c1_values = np.round(np.arange(local.C1.min(), local.C1.max() + 0.025, 0.05), 2)
    q1_values = np.arange(int(local.Q1.min()), int(local.Q1.max()) + 1, 1, dtype=float)
    c2_values = np.round(np.arange(local.C2.min(), local.C2.max() + 0.025, 0.05), 2)
    c1, q1, c2 = np.meshgrid(c1_values, q1_values, c2_values, indexing="ij")
    grid = pd.DataFrame({"C1": c1.ravel(), "Q1": q1.ravel(), "C2": c2.ravel()})
    return add_physical_quantities(grid)


def add_trust_domain_flags(grid: pd.DataFrame, observed: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    observed_x = observed[PARAMETERS].to_numpy(float)
    lower = observed_x.min(axis=0)
    span = np.maximum(observed_x.max(axis=0) - lower, 1e-12)
    observed_scaled = (observed_x - lower) / span
    grid_scaled = (grid[PARAMETERS].to_numpy(float) - lower) / span
    distances, indices = cKDTree(observed_scaled).query(grid_scaled, k=1)
    observed_distances, _ = cKDTree(observed_scaled).query(observed_scaled, k=2)
    trust_threshold = float(np.quantile(observed_distances[:, 1], 0.75))
    hull = ConvexHull(observed_x)
    triangulation = Delaunay(observed_x[hull.vertices])
    inside_hull = triangulation.find_simplex(grid[PARAMETERS].to_numpy(float)) >= 0
    result = grid.copy()
    result["nearest_observed_distance_normalized"] = distances
    result["nearest_observed_index"] = indices
    result["inside_rectangle"] = True
    result["inside_convex_hull"] = inside_hull
    result["inside_trust_region"] = distances <= trust_threshold + 1e-12
    result["inside_main_domain"] = result["inside_convex_hull"] & result["inside_trust_region"]
    metadata = {
        "C1_min": float(lower[0]), "C1_max": float(lower[0] + span[0]),
        "Q1_min": float(lower[1]), "Q1_max": float(lower[1] + span[1]),
        "C2_min": float(lower[2]), "C2_max": float(lower[2] + span[2]),
        "convex_hull_volume": float(hull.volume),
        "trust_distance_threshold_normalized": trust_threshold,
        "n_rectangle_grid_points": int(len(grid)),
        "n_convex_hull_grid_points": int(np.sum(inside_hull)),
        "n_trust_region_grid_points": int(np.sum(result["inside_trust_region"])),
        "n_main_domain_grid_points": int(np.sum(result["inside_main_domain"])),
    }
    return result, metadata


def add_surrogate_predictions(
    frame: pd.DataFrame,
    charge_model: ChargeTimeModel,
    response_models: dict[str, StressResponseModel],
) -> pd.DataFrame:
    result = frame.copy()
    result["predicted_charge_time"] = charge_model.predict(result)
    result["predicted_charge_time_se"] = charge_model.prediction_se(result)
    result["predicted_SOH200"] = response_models["SOH200"].predict(result)
    result["predicted_SOH200_se"] = response_models["SOH200"].mean_prediction_se(result)
    result["predicted_slope50_200"] = response_models["slope50_200"].predict(result)
    result["predicted_slope50_200_se"] = response_models["slope50_200"].mean_prediction_se(result)
    result["predicted_degradation_rate"] = -result["predicted_slope50_200"]
    result["predicted_degradation_rate_se"] = result["predicted_slope50_200_se"]
    result["predicted_EOL"] = response_models["EOL"].predict(result)
    result["predicted_EOL_se"] = response_models["EOL"].mean_prediction_se(result)
    result["stress_p"] = response_models["SOH200"].p
    result["soc_weighted_stress"] = soc_weighted_stress(result, response_models["SOH200"].p)
    return result


def pareto_front(frame: pd.DataFrame, degradation_column: str = "predicted_degradation_rate") -> pd.DataFrame:
    """Exact two-objective non-dominated filter for minimization of time and degradation."""
    ordered = frame.sort_values(["predicted_charge_time", degradation_column, "C1", "Q1", "C2"])
    keep: list[int] = []
    best_degradation = np.inf
    for index, value in zip(ordered.index, ordered[degradation_column].to_numpy(float)):
        if value < best_degradation - 1e-12:
            keep.append(index)
            best_degradation = value
    result = frame.loc[keep].sort_values("predicted_charge_time").copy()
    result["pareto_rank"] = np.arange(1, len(result) + 1)
    return result


def select_representative_points(
    feasible: pd.DataFrame,
    pareto: pd.DataFrame,
    degradation_column: str = "predicted_degradation_rate",
) -> pd.DataFrame:
    front = pareto.copy()
    time_span = max(front.predicted_charge_time.max() - front.predicted_charge_time.min(), 1e-12)
    degradation_span = max(front[degradation_column].max() - front[degradation_column].min(), 1e-12)
    front["normalized_time"] = (front.predicted_charge_time - front.predicted_charge_time.min()) / time_span
    front["normalized_degradation"] = (front[degradation_column] - front[degradation_column].min()) / degradation_span
    front["distance_to_ideal"] = np.sqrt(front.normalized_time**2 + front.normalized_degradation**2)
    knee_index = front["distance_to_ideal"].idxmin()
    non_extreme_limit = feasible[degradation_column].quantile(0.80)
    fast_pool = front.query(f"{degradation_column} <= @non_extreme_limit")
    if fast_pool.empty:
        fast_pool = front
    fast_index = fast_pool["predicted_charge_time"].idxmin()
    reasonable_time_limit = feasible["predicted_charge_time"].quantile(0.75)
    life_pool = front.query("predicted_charge_time <= @reasonable_time_limit")
    if life_pool.empty:
        life_pool = front
    life_index = life_pool[degradation_column].idxmin()
    selections = []
    for label, index in (("fast_charge", fast_index), ("knee", knee_index), ("longevity", life_index)):
        row = front.loc[index].copy()
        row["recommendation_type"] = label
        selections.append(row)
    return pd.DataFrame(selections).drop_duplicates(subset=["recommendation_type"]).reset_index(drop=True)


def fit_main_models(
    strategies: pd.DataFrame,
    charge_validation: pd.DataFrame,
    p: float,
    *,
    scope: str = "dataset3_only",
) -> tuple[ChargeTimeModel, dict[str, StressResponseModel]]:
    data = parameterized_scope(strategies, scope)
    charge_name = select_charge_time_model(charge_validation)
    charge = fit_charge_time_model(data, charge_name)
    cv_row = charge_validation.query("scope == 'dataset3_only' and model == @charge_name").iloc[0]
    charge.cv_rmse = float(cv_row["LOSO_RMSE_minutes"])
    responses = {
        label: fit_stress_response_model(data, response, p)
        for label, response in RESPONSE_COLUMNS.items()
    }
    return charge, responses


def bootstrap_recommendations(
    observed: pd.DataFrame,
    feasible_grid: pd.DataFrame,
    charge_model_name: str,
    p: float,
    samples: int = 300,
    seed: int = RANDOM_SEED + 800,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    for iteration in range(samples):
        indices = rng.integers(0, len(observed), len(observed))
        sample = observed.iloc[indices].reset_index(drop=True)
        charge = fit_charge_time_model(sample, charge_model_name)
        responses = {
            label: fit_stress_response_model(sample, response, p)
            for label, response in RESPONSE_COLUMNS.items()
        }
        evaluated = add_surrogate_predictions(feasible_grid, charge, responses)
        front = pareto_front(evaluated)
        selected = select_representative_points(evaluated, front)
        selected["bootstrap_iteration"] = iteration
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)
