"""Policy-informed forecasting and unified-quadratic multi-objective optimization."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_SEED
from .question3_models import FUTURE_CYCLES, extract_early_features
from .question4_models import (
    add_physical_quantities,
    add_trust_domain_flags,
    charge_time_cross_validation,
    fit_charge_time_model,
    generate_parameter_grid,
    select_charge_time_model,
)
from .unified_quadratic import (
    StandardizedRidge,
    add_phase_exposures,
    analytic_eol,
    fit_selected_strategy_model,
    fit_standardized_ridge,
    predict_native_parameters,
    quadratic_derived,
    quadratic_soh,
    stable_to_native,
)


def _monotone_future(values: np.ndarray, observed_150: float) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 0.75, 1.05)
    return np.minimum.accumulate(np.r_[observed_150, values])[1:]


@dataclass(frozen=True)
class PolicyPrior:
    mean_d1: float
    mean_d2: float
    sd_d1: float
    sd_d2: float
    source: str
    n_peer_batteries: int


def _strategy_model_frame(
    battery_table: pd.DataFrame,
    excluded_battery: int | None,
) -> pd.DataFrame:
    source = battery_table.copy()
    if excluded_battery is not None:
        source = source.loc[source["battery_id"] != excluded_battery]
    aggregation = (
        source.groupby(["strategy", "strategy_code", "dataset_id", "C1", "Q1", "C2"], dropna=False)
        .agg(d1=("d1", "median"), d2=("d2", "median"), R100=("R100", "median"),
             A=("A", "median"), n_batteries=("battery_id", "size"))
        .reset_index()
    )
    aggregation["parameterized"] = aggregation["C1"].notna()
    aggregation["main_dataset3"] = (aggregation["dataset_id"] == 3) & aggregation["parameterized"]
    return aggregation


def policy_prior_for_battery(
    battery_table: pd.DataFrame,
    target: pd.Series,
    selection: pd.Series,
    parameterization: str,
    *,
    excluded_battery: int | None,
) -> PolicyPrior:
    """Construct a policy prior without using the target battery's full trajectory."""
    development = battery_table.copy()
    if excluded_battery is not None:
        development = development.loc[development["battery_id"] != excluded_battery]
    peers = development.loc[development["strategy"] == target["strategy"]]
    strategy_frame = _strategy_model_frame(battery_table, excluded_battery)
    prediction: tuple[float, float] | None = None
    source = "same_strategy_peer_distribution"
    if int(target["dataset_id"]) == 3 and pd.notna(target["C1"]):
        try:
            responses = ["R100", "A"] if parameterization == "R100_A" else ["d1", "d2"]
            model, _ = fit_selected_strategy_model(strategy_frame, responses, selection)
            target_frame = pd.DataFrame([target])
            if selection["model"] != "raw_parameters":
                target_frame = add_phase_exposures(target_frame, float(selection["p"]))
            predicted = predict_native_parameters(model, target_frame, parameterization).iloc[0]
            prediction = float(predicted["predicted_d1"]), float(predicted["predicted_d2"])
            source = "leave_target_out_policy_parameter_model"
        except (ValueError, np.linalg.LinAlgError):
            prediction = None
    if prediction is None:
        reference = peers if len(peers) else development
        prediction = float(reference["d1"].median()), float(reference["d2"].median())
    residual_d1 = development["d1"] - development.groupby("strategy")["d1"].transform("median")
    residual_d2 = development["d2"] - development.groupby("strategy")["d2"].transform("median")
    sd_d1 = max(float(residual_d1.std(ddof=1)), float(development["d1"].std(ddof=1)) * 0.15, 5e-4)
    sd_d2 = max(float(residual_d2.std(ddof=1)), float(development["d2"].std(ddof=1)) * 0.15, 2e-4)
    return PolicyPrior(prediction[0], prediction[1], sd_d1, sd_d2, source, int(len(peers)))


def fit_policy_informed_quadratic(
    early: pd.DataFrame,
    prior: PolicyPrior,
    penalty: float,
) -> tuple[np.ndarray, float]:
    """Robust early-data fit with standardized shrinkage on d1/d2."""
    g = early.sort_values("cycle")
    cycle = g["cycle"].to_numpy(float)
    y = g["SOH_smooth_robust"].to_numpy(float)
    noise = max(float(1.4826 * np.median(np.abs(y - np.median(y)))), 8e-4)
    a0 = float(np.median(y[:10]))
    initial = np.array([np.clip(a0, 0.85, 1.15), prior.mean_d1, prior.mean_d2])

    def residual(parameters: np.ndarray) -> np.ndarray:
        data = (quadratic_soh(cycle, *parameters) - y) / (noise * sqrt(len(y)))
        if penalty <= 0:
            return data
        prior_residual = sqrt(penalty) * np.array(
            [(parameters[1] - prior.mean_d1) / prior.sd_d1,
             (parameters[2] - prior.mean_d2) / prior.sd_d2]
        )
        return np.r_[data, prior_residual]

    fit = least_squares(
        residual, initial, bounds=([0.85, 0.0, 0.0], [1.15, 0.5, 0.25]),
        loss="soft_l1", f_scale=1.0, max_nfev=4000,
    )
    prediction = quadratic_soh(cycle, *fit.x)
    rmse = float(np.sqrt(np.mean((prediction - y) ** 2)))
    return fit.x, rmse


def policy_pseudotest_predictions(
    cleaned_cycles: pd.DataFrame,
    battery_table: pd.DataFrame,
    selection: pd.Series,
    parameterization: str,
    penalties: Sequence[float] = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """LOBO simulation: cycles 1--150 in, cycles 151--200 out."""
    train = cleaned_cycles.loc[cleaned_cycles["prediction_test"] == 0]
    rows, diagnostics = [], []
    lookup = battery_table.set_index("battery_id")
    for battery_id, group in train.groupby("battery_id", sort=True):
        target = lookup.loc[int(battery_id)]
        early = group.loc[group["cycle"] <= 150]
        actual = group.set_index("cycle").loc[FUTURE_CYCLES.astype(int), "SOH_clean"].to_numpy(float)
        prior = policy_prior_for_battery(
            battery_table, target, selection, parameterization, excluded_battery=int(battery_id)
        )
        observed_150 = float(early.loc[early["cycle"] == 150, "SOH_smooth_robust"].iloc[0])
        for penalty in penalties:
            parameters, fit_rmse = fit_policy_informed_quadratic(early, prior, float(penalty))
            prediction = _monotone_future(quadratic_soh(FUTURE_CYCLES, *parameters), observed_150)
            error = prediction - actual
            diagnostics.append(
                {
                    "battery_id": int(battery_id), "strategy": target["strategy"],
                    "strategy_code": target["strategy_code"], "penalty": float(penalty),
                    "prior_source": prior.source, "prior_peer_n": prior.n_peer_batteries,
                    "prior_d1": prior.mean_d1, "prior_d2": prior.mean_d2,
                    "fitted_a": parameters[0], "fitted_d1": parameters[1], "fitted_d2": parameters[2],
                    "early_fit_rmse": fit_rmse, "future_MAE": float(np.mean(np.abs(error))),
                    "future_RMSE": float(np.sqrt(np.mean(error**2))),
                    "future_MaxAE": float(np.max(np.abs(error))),
                    "cycle200_error": float(error[-1]),
                }
            )
            for cycle, actual_soh, predicted_soh in zip(FUTURE_CYCLES.astype(int), actual, prediction):
                rows.append(
                    {
                        "battery_id": int(battery_id), "strategy": target["strategy"],
                        "strategy_code": target["strategy_code"], "penalty": float(penalty),
                        "cycle": int(cycle), "actual_soh": actual_soh,
                        "predicted_soh": predicted_soh,
                    }
                )
    diagnostic_frame = pd.DataFrame(diagnostics)
    score = diagnostic_frame.groupby("penalty").agg(
        mean_RMSE=("future_RMSE", "mean"), median_RMSE=("future_RMSE", "median"),
        p90_RMSE=("future_RMSE", lambda x: x.quantile(0.9)),
        worst_RMSE=("future_RMSE", "max"), cycle200_MAE=("cycle200_error", lambda x: np.mean(np.abs(x))),
    ).reset_index()
    score["selection_score"] = score["mean_RMSE"] + 0.25 * score["p90_RMSE"] + 0.1 * score["cycle200_MAE"]
    selected = float(score.sort_values(["selection_score", "penalty"]).iloc[0]["penalty"])
    score["selected"] = np.isclose(score["penalty"], selected)
    predictions = pd.DataFrame(rows)
    predictions["selected"] = np.isclose(predictions["penalty"], selected)
    return predictions, diagnostic_frame.merge(score[["penalty", "selected"]], on="penalty"), selected


def compare_policy_to_baseline(
    policy_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    selected_penalty: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy = policy_predictions.loc[np.isclose(policy_predictions["penalty"], selected_penalty)].copy()
    policy["model"] = "policy_informed_quadratic"
    policy = policy[["battery_id", "strategy", "strategy_code", "model", "cycle", "actual_soh", "predicted_soh"]]
    baseline = baseline_predictions.loc[
        baseline_predictions["model"].isin(["individual_quadratic", "adaptive_ensemble"])
    ].copy()
    combined = pd.concat([baseline, policy], ignore_index=True)
    combined["horizon"] = combined["cycle"] - 150
    coverage_rows = []
    for model, model_group in combined.groupby("model"):
        for battery_id, target in model_group.groupby("battery_id"):
            calibration = model_group.loc[model_group["battery_id"] != battery_id].copy()
            calibration["absolute_error"] = (
                calibration["predicted_soh"] - calibration["actual_soh"]
            ).abs()
            quantiles = calibration.groupby("horizon")["absolute_error"].quantile(0.95)
            target_error = (target["predicted_soh"] - target["actual_soh"]).abs()
            bounds = target["horizon"].map(quantiles).to_numpy(float)
            coverage_rows.append(
                {
                    "model": model, "battery_id": int(battery_id),
                    "interval_coverage_95": float(np.mean(target_error.to_numpy(float) <= bounds)),
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    rows = []
    for (battery_id, model), group in combined.groupby(["battery_id", "model"]):
        error = group["predicted_soh"].to_numpy(float) - group["actual_soh"].to_numpy(float)
        rows.append(
            {
                "battery_id": int(battery_id), "strategy": group["strategy"].iloc[0],
                "strategy_code": group["strategy_code"].iloc[0], "model": model,
                "MAE": float(np.mean(np.abs(error))), "RMSE": float(np.sqrt(np.mean(error**2))),
                "MaxAE": float(np.max(np.abs(error))), "cycle200_abs_error": float(abs(error[-1])),
                "trend_sign_match": bool(
                    np.sign(group["actual_soh"].iloc[-1] - group["actual_soh"].iloc[0])
                    == np.sign(group["predicted_soh"].iloc[-1] - group["predicted_soh"].iloc[0])
                ),
            }
        )
    battery = pd.DataFrame(rows)
    overall = battery.groupby("model").agg(
        n_batteries=("battery_id", "size"), MAE_mean=("MAE", "mean"),
        RMSE_mean=("RMSE", "mean"), RMSE_median=("RMSE", "median"),
        RMSE_p90=("RMSE", lambda x: x.quantile(0.9)), worst_battery_RMSE=("RMSE", "max"),
        cycle200_MAE=("cycle200_abs_error", "mean"), trend_match_rate=("trend_sign_match", "mean"),
    ).reset_index()
    interval_summary = coverage.groupby("model")["interval_coverage_95"].mean().reset_index()
    strategy_stability = battery.groupby(["model", "strategy_code"])["RMSE"].median().groupby("model").std(ddof=1).rename("strategy_median_RMSE_sd").reset_index()
    overall = overall.merge(interval_summary, on="model", how="left").merge(strategy_stability, on="model", how="left")
    return combined, battery.merge(overall[["model", "RMSE_mean"]].rename(columns={"RMSE_mean": "model_RMSE_mean"}), on="model"), overall


def feature_ablation_validation(
    cleaned_cycles: pd.DataFrame,
    battery_table: pd.DataFrame,
    selection: pd.Series,
    parameterization: str,
) -> pd.DataFrame:
    """Battery-level LOBO ablation predicting full-curve d1/d2 from cycle-150 features."""
    cycle_frame = cleaned_cycles.copy()
    if "strategy_code" not in cycle_frame:
        code_map = battery_table[["strategy", "strategy_code"]].drop_duplicates()
        cycle_frame = cycle_frame.merge(code_map, on="strategy", how="left", validate="many_to_one")
    features = extract_early_features(cycle_frame.loc[cycle_frame["cycle"] <= 150])
    data = features.loc[features["prediction_test"] == 0].merge(
        battery_table[["battery_id", "d1", "d2", "C1", "Q1", "C2"]],
        on="battery_id", suffixes=("", "_full"), validate="one_to_one",
    )
    soh = ["SOH_50", "SOH_100", "SOH_150", "slope_1_50", "slope_50_100", "slope_100_150", "slope_50_150", "curvature", "noise_mad"]
    groups = {
        "SOH_only": soh,
        "SOH_plus_IR": [*soh, "IR_mean", "IR_slope"],
        "SOH_plus_T": [*soh, "Tavg_mean", "Tavg_slope"],
        "SOH_plus_charge": [*soh, "chargetime_mean", "chargetime_slope"],
        "all_health": [*soh, "IR_mean", "IR_slope", "Tavg_mean", "Tavg_slope", "chargetime_mean", "chargetime_slope"],
    }
    policy_d1, policy_d2 = [], []
    lookup = battery_table.set_index("battery_id")
    for _, row in data.iterrows():
        prior = policy_prior_for_battery(
            battery_table, lookup.loc[int(row["battery_id"])], selection, parameterization,
            excluded_battery=int(row["battery_id"]),
        )
        policy_d1.append(prior.mean_d1)
        policy_d2.append(prior.mean_d2)
    data["policy_prior_d1"], data["policy_prior_d2"] = policy_d1, policy_d2
    groups["all_plus_policy_prior"] = [*groups["all_health"], "policy_prior_d1", "policy_prior_d2"]
    curve_cycles = FUTURE_CYCLES
    cycle_source = cleaned_cycles.loc[cleaned_cycles["prediction_test"] == 0]
    rows = []
    for name, predictors in groups.items():
        errors = []
        per_battery = []
        for index, target in data.iterrows():
            train = data.drop(index=index)
            x_scaler, y_scaler = StandardScaler(), StandardScaler()
            x_train = x_scaler.fit_transform(train[predictors].to_numpy(float))
            y_train = y_scaler.fit_transform(train[["d1", "d2"]].to_numpy(float))
            model = Ridge(alpha=3.0).fit(x_train, y_train)
            estimated = y_scaler.inverse_transform(model.predict(x_scaler.transform(target[predictors].to_numpy(float).reshape(1, -1))))[0]
            d1, d2 = np.maximum(estimated, 0.0)
            group = cycle_source.loc[cycle_source["battery_id"] == target["battery_id"]]
            y150 = float(target["SOH_150"])
            a = y150 + 1.5 * d1 + 2.25 * d2
            prediction = _monotone_future(quadratic_soh(curve_cycles, a, d1, d2), y150)
            actual = group.set_index("cycle").loc[curve_cycles.astype(int), "SOH_clean"].to_numpy(float)
            rmse = float(np.sqrt(np.mean((prediction - actual) ** 2)))
            errors.extend(prediction - actual)
            per_battery.append(rmse)
        rows.append(
            {
                "ablation": name, "n_predictors": len(predictors),
                "pointwise_RMSE": float(np.sqrt(np.mean(np.asarray(errors) ** 2))),
                "battery_RMSE_mean": float(np.mean(per_battery)),
                "battery_RMSE_p90": float(np.quantile(per_battery, 0.9)),
                "worst_battery_RMSE": float(np.max(per_battery)),
            }
        )
    result = pd.DataFrame(rows)
    baseline = float(result.loc[result["ablation"] == "SOH_only", "pointwise_RMSE"].iloc[0])
    result["relative_improvement_vs_SOH_only"] = (baseline - result["pointwise_RMSE"]) / baseline
    result["stable_improvement"] = (result["relative_improvement_vs_SOH_only"] >= 0.03) & (result["battery_RMSE_p90"] <= result.loc[result["ablation"] == "SOH_only", "battery_RMSE_p90"].iloc[0])
    return result


def forecast_test_with_policy(
    cleaned_cycles: pd.DataFrame,
    battery_table: pd.DataFrame,
    selection: pd.Series,
    parameterization: str,
    penalty: float,
    baseline_test_predictions: pd.DataFrame,
    final_short_model: str,
    *,
    bootstrap_samples: int = 300,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Forecast nine test batteries and refit the single quadratic EOL model."""
    test = cleaned_cycles.loc[cleaned_cycles["prediction_test"] == 1]
    mapping = test[["battery_id", "strategy", "dataset_id", "C1", "Q1", "C2"]].drop_duplicates()
    code_map = battery_table[["strategy", "strategy_code"]].drop_duplicates()
    mapping = mapping.merge(code_map, on="strategy", how="left", validate="many_to_one")
    rows, parameter_rows, curve_rows = [], [], []
    rng = np.random.default_rng(RANDOM_SEED + 777)
    baseline = baseline_test_predictions.set_index(["battery_id", "cycle"])
    for battery_id, group in test.groupby("battery_id", sort=True):
        target = mapping.loc[mapping["battery_id"] == battery_id].iloc[0]
        prior = policy_prior_for_battery(
            battery_table, target, selection, parameterization, excluded_battery=None
        )
        parameters150, early_rmse = fit_policy_informed_quadratic(group, prior, penalty)
        observed_150 = float(group.loc[group["cycle"] == 150, "SOH_smooth_robust"].iloc[0])
        policy_prediction = _monotone_future(quadratic_soh(FUTURE_CYCLES, *parameters150), observed_150)
        baseline_battery = baseline.loc[(int(battery_id),)].sort_index()
        if final_short_model == "adaptive_ensemble":
            selected_prediction = baseline_battery["predicted_soh"].to_numpy(float)
        else:
            selected_prediction = policy_prediction
        residual_scale = max(float(early_rmse), 3e-4)
        for offset, (cycle, value, policy_value) in enumerate(zip(FUTURE_CYCLES.astype(int), selected_prediction, policy_prediction)):
            if final_short_model == "adaptive_ensemble" and {
                "prediction_interval_95_low", "prediction_interval_95_high"
            }.issubset(baseline_battery.columns):
                interval_low = float(baseline_battery["prediction_interval_95_low"].iloc[offset])
                interval_high = float(baseline_battery["prediction_interval_95_high"].iloc[offset])
            else:
                interval_low = float(value - 1.96 * residual_scale)
                interval_high = float(value + 1.96 * residual_scale)
            rows.append(
                {
                    "battery_id": int(battery_id), "strategy": target["strategy"],
                    "strategy_code": target["strategy_code"], "cycle": int(cycle),
                    "predicted_soh": float(value), "policy_quadratic_prediction": float(policy_value),
                    "selected_short_model": final_short_model,
                    "prediction_interval_95_low": interval_low,
                    "prediction_interval_95_high": interval_high,
                }
            )
        augmented_cycle = np.r_[group["cycle"].to_numpy(float), FUTURE_CYCLES]
        augmented_soh = np.r_[group["SOH_smooth_robust"].to_numpy(float), selected_prediction]
        pseudo = pd.DataFrame({"cycle": augmented_cycle, "SOH_smooth_robust": augmented_soh})
        final_parameters, fit_rmse = fit_policy_informed_quadratic(pseudo, prior, penalty)
        a, d1, d2 = final_parameters
        eol = analytic_eol(a, d1, d2).item()
        eol150 = analytic_eol(*parameters150).item()
        fitted_augmented = quadratic_soh(augmented_cycle, *final_parameters)
        residual = augmented_soh - fitted_augmented
        residual -= np.median(residual)
        eol_draws = []
        for _ in range(bootstrap_samples):
            starts = rng.integers(0, max(len(residual) - 10 + 1, 1), int(np.ceil(len(residual) / 10)))
            resampled = np.concatenate([residual[start : start + 10] for start in starts])[: len(residual)]
            bootstrap_frame = pd.DataFrame(
                {"cycle": augmented_cycle, "SOH_smooth_robust": fitted_augmented + resampled}
            )
            bootstrap_parameters, _ = fit_policy_informed_quadratic(bootstrap_frame, prior, penalty)
            draw = analytic_eol(*bootstrap_parameters).item()
            if np.isfinite(draw):
                eol_draws.append(float(draw))
        eol_array = np.asarray(eol_draws, dtype=float)
        if len(eol_array):
            eol_low, eol_median, eol_high = np.quantile(eol_array, [0.025, 0.5, 0.975])
        else:
            eol_low = eol_median = eol_high = np.nan
        width_ratio = (eol_high - eol_low) / eol_median if np.isfinite(eol_median) and eol_median > 0 else np.nan
        reliability = (
            "lower" if len(eol_array) / bootstrap_samples < 0.8 or (np.isfinite(width_ratio) and width_ratio > 0.5)
            else "moderate" if np.isfinite(width_ratio) and width_ratio > 0.25
            else "higher_conditional"
        )
        r100, r200, acceleration = quadratic_derived(d1, d2)
        parameter_rows.append(
            {
                "battery_id": int(battery_id), "strategy": target["strategy"],
                "strategy_code": target["strategy_code"], "selected_short_model": final_short_model,
                "prior_source": prior.source, "penalty": penalty, "a": a, "d1": d1, "d2": d2,
                "R100": float(r100), "R200": float(r200), "A": float(acceleration),
                "eol_from_cycle150_only": float(eol150), "final_quadratic_eol": float(eol),
                "eol_change_after_short_forecast": float(eol - eol150),
                "eol_bootstrap_median": float(eol_median), "eol_ci95_low": float(eol_low),
                "eol_ci95_high": float(eol_high), "bootstrap_valid_fraction": len(eol_array) / bootstrap_samples,
                "eol_interval_width_ratio": float(width_ratio), "long_horizon_reliability": reliability,
                "augmented_fit_rmse": fit_rmse,
            }
        )
        end = min(float(eol) if np.isfinite(eol) else 1500.0, 2000.0)
        curve_cycles = np.arange(1, int(max(end, 200)) + 1)
        for cycle, value in zip(curve_cycles, quadratic_soh(curve_cycles, a, d1, d2)):
            curve_rows.append({"battery_id": int(battery_id), "cycle": int(cycle), "predicted_soh": float(value), "eol": float(eol)})
    return pd.DataFrame(rows), pd.DataFrame(parameter_rows), pd.DataFrame(curve_rows)


def choose_short_horizon_model(comparison: pd.DataFrame) -> str:
    indexed = comparison.set_index("model")
    policy = indexed.loc["policy_informed_quadratic"]
    ensemble = indexed.loc["adaptive_ensemble"]
    if policy["RMSE_mean"] <= 1.02 * ensemble["RMSE_mean"] and policy["worst_battery_RMSE"] <= 1.05 * ensemble["worst_battery_RMSE"]:
        return "policy_informed_quadratic"
    return "adaptive_ensemble"


def pareto_time_life(frame: pd.DataFrame, life_column: str) -> pd.DataFrame:
    """Exact non-dominated front for minimum time and maximum life."""
    ordered = frame.sort_values(["predicted_charge_time", life_column], ascending=[True, False])
    keep, best = [], -np.inf
    for index, value in zip(ordered.index, ordered[life_column].to_numpy(float)):
        if np.isfinite(value) and value > best + 1e-12:
            keep.append(index)
            best = value
    result = frame.loc[keep].sort_values("predicted_charge_time").copy()
    result["pareto_rank"] = np.arange(1, len(result) + 1)
    return result


def representative_time_life(frame: pd.DataFrame, front: pd.DataFrame, life_column: str) -> pd.DataFrame:
    """Choose fast, ideal-point compromise, and longevity representatives."""
    candidates = front.copy()
    t_span = max(float(candidates["predicted_charge_time"].max() - candidates["predicted_charge_time"].min()), 1e-12)
    l_span = max(float(candidates[life_column].max() - candidates[life_column].min()), 1e-12)
    candidates["normalized_time"] = (candidates["predicted_charge_time"] - candidates["predicted_charge_time"].min()) / t_span
    candidates["normalized_life_harm"] = (candidates[life_column].max() - candidates[life_column]) / l_span
    candidates["distance_to_ideal"] = np.sqrt(candidates["normalized_time"] ** 2 + candidates["normalized_life_harm"] ** 2)
    compromise = candidates["distance_to_ideal"].idxmin()
    life_floor = frame[life_column].quantile(0.20)
    fast_pool = candidates.loc[candidates[life_column] >= life_floor]
    fast = fast_pool["predicted_charge_time"].idxmin() if len(fast_pool) else candidates["predicted_charge_time"].idxmin()
    time_ceiling = frame["predicted_charge_time"].quantile(0.75)
    life_pool = candidates.loc[candidates["predicted_charge_time"] <= time_ceiling]
    longevity = life_pool[life_column].idxmax() if len(life_pool) else candidates[life_column].idxmax()
    rows = []
    for label, index in (("fast_charge", fast), ("ideal_point_compromise", compromise), ("longevity", longevity)):
        row = candidates.loc[index].copy()
        row["recommendation_type"] = label
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def prepare_optimization_grid(
    strategy_table: pd.DataFrame,
    strategy_model: StandardizedRidge,
    parameterization: str,
    selection: pd.Series,
) -> tuple[pd.DataFrame, dict[str, float]]:
    observed = strategy_table.loc[strategy_table["main_dataset3"]].copy()
    observed_time = observed.rename(columns={"mean_chargetime": "mean_chargetime"})
    observed_time = add_physical_quantities(observed_time)
    all_time = add_physical_quantities(strategy_table.copy())
    time_validation = charge_time_cross_validation(all_time)
    charge_name = select_charge_time_model(time_validation)
    charge_model = fit_charge_time_model(observed_time.reset_index(drop=True), charge_name)
    grid = generate_parameter_grid(observed_time)
    grid, metadata = add_trust_domain_flags(grid, observed_time)
    if selection["model"] != "raw_parameters":
        grid = add_phase_exposures(grid, float(selection["p"]))
    evaluated = predict_native_parameters(strategy_model, grid, parameterization)
    evaluated["predicted_charge_time"] = charge_model.predict(evaluated)
    evaluated["charge_time_model"] = charge_name
    metadata.update({"charge_time_model": charge_name, "charge_time_loso_rmse": float(time_validation.query("scope == 'dataset3_only' and model == @charge_name")["LOSO_RMSE_minutes"].iloc[0])})
    return evaluated, metadata | {"charge_validation": time_validation}


def bootstrap_optimization_life(
    feasible_grid: pd.DataFrame,
    battery_table: pd.DataFrame,
    selection: pd.Series,
    parameterization: str,
    responses: Sequence[str],
    *,
    samples: int = 1_000,
    seed: int = RANDOM_SEED + 901,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strategy-first/battery-within-strategy bootstrap of every candidate's EOL."""
    rng = np.random.default_rng(seed)
    source = battery_table.loc[(battery_table["dataset_id"] == 3) & battery_table["C1"].notna()].copy()
    groups = {name: group for name, group in source.groupby("strategy", sort=True)}
    names = np.asarray(list(groups))
    n_grid = len(feasible_grid)
    life = np.full((samples, n_grid), np.nan, dtype=np.float32)
    life_a_sensitivity = np.full((samples, n_grid), np.nan, dtype=np.float32)
    recommendation_rows = []
    p = float(selection["p"]) if pd.notna(selection["p"]) else None
    predictors = ["C1", "Q1", "C2"] if selection["model"] == "raw_parameters" else ["E1", "E2"]
    a_values = source["a"].dropna().to_numpy(float)
    valid = 0
    for iteration in range(samples):
        sampled_names = rng.choice(names, len(names), replace=True)
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
        if sample[predictors].drop_duplicates().shape[0] < 2:
            continue
        try:
            model = fit_standardized_ridge(sample, predictors, responses, float(selection["alpha"]))
            predicted = predict_native_parameters(model, feasible_grid, parameterization)
        except (ValueError, np.linalg.LinAlgError):
            continue
        life[iteration] = predicted["predicted_eol_a1"].to_numpy(np.float32)
        random_a = float(rng.choice(a_values))
        life_a_sensitivity[iteration] = analytic_eol(
            random_a, predicted["predicted_d1"], predicted["predicted_d2"]
        ).astype(np.float32)
        front = pareto_time_life(
            feasible_grid.assign(_life=life[iteration].astype(float)), "_life"
        )
        if len(front):
            selected = representative_time_life(
                feasible_grid.assign(_life=life[iteration].astype(float)), front, "_life"
            )
            selected["bootstrap_iteration"] = iteration
            recommendation_rows.append(selected[["bootstrap_iteration", "recommendation_type", "C1", "Q1", "C2", "predicted_charge_time", "_life"]])
        valid += 1
    result = feasible_grid.copy()
    result["bootstrap_valid_n"] = np.sum(np.isfinite(life), axis=0)
    result["eol_median"] = np.nanmedian(life, axis=0)
    result["eol_p10"] = np.nanquantile(life, 0.10, axis=0)
    result["eol_p90"] = np.nanquantile(life, 0.90, axis=0)
    result["eol_ci95_low"] = np.nanquantile(life, 0.025, axis=0)
    result["eol_ci95_high"] = np.nanquantile(life, 0.975, axis=0)
    result["eol_a_distribution_median"] = np.nanmedian(life_a_sensitivity, axis=0)
    result["eol_a_distribution_p10"] = np.nanquantile(life_a_sensitivity, 0.10, axis=0)
    result["bootstrap_valid_iterations"] = valid
    recommendations = pd.concat(recommendation_rows, ignore_index=True) if recommendation_rows else pd.DataFrame()
    return result, recommendations


def add_candidate_risk_and_curves(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = candidates.copy()
    result["life_interval_width_ratio"] = (
        (result["eol_p90"] - result["eol_p10"]) / result["eol_median"]
    )
    result["risk_grade"] = np.select(
        [
            ~result["inside_main_domain"],
            result["life_interval_width_ratio"] > 0.75,
            result["life_interval_width_ratio"] > 0.40,
        ],
        ["high_extrapolation_risk", "high_model_uncertainty", "moderate_model_uncertainty"],
        default="supported_model_region",
    )
    curve_rows = []
    for row in result.itertuples():
        eol = float(row.eol_median)
        cycles = np.arange(0, int(min(max(eol, 200), 2000)) + 1, 5)
        for cycle, soh in zip(cycles, quadratic_soh(cycles, 1.0, row.predicted_d1, row.predicted_d2)):
            curve_rows.append(
                {
                    "C1": row.C1, "Q1": row.Q1, "C2": row.C2,
                    "cycle": int(cycle), "predicted_soh": float(soh), "eol_median": eol,
                }
            )
    return result, pd.DataFrame(curve_rows)


def robustness_from_bootstrap(recommendation_draws: pd.DataFrame, recommendations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, target in recommendations.iterrows():
        group = recommendation_draws.loc[recommendation_draws["recommendation_type"] == target["recommendation_type"]]
        if group.empty:
            continue
        distance = np.sqrt(
            ((group["C1"] - target["C1"]) / 0.5) ** 2
            + ((group["Q1"] - target["Q1"]) / 10.0) ** 2
            + ((group["C2"] - target["C2"]) / 0.5) ** 2
        )
        rows.append(
            {
                "recommendation_type": target["recommendation_type"],
                "bootstrap_n": len(group), "median_parameter_distance": float(distance.median()),
                "within_one_scaled_unit_fraction": float((distance <= 1.0).mean()),
                "C1_p10": float(group["C1"].quantile(0.1)), "C1_p90": float(group["C1"].quantile(0.9)),
                "Q1_p10": float(group["Q1"].quantile(0.1)), "Q1_p90": float(group["Q1"].quantile(0.9)),
                "C2_p10": float(group["C2"].quantile(0.1)), "C2_p90": float(group["C2"].quantile(0.9)),
            }
        )
    return pd.DataFrame(rows)
