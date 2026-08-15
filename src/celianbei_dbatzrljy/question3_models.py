"""Leakage-safe short-horizon forecasting and EOL estimation for question three."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import theilslopes
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from .config import PROJECT_ROOT, RANDOM_SEED
from .features import robust_slope
from .models import fit_degradation_model


FUTURE_CYCLES = np.arange(151, 201, dtype=float)
BASE_MODELS = (
    "individual_quadratic",
    "local_linear",
    "strategy_template",
    "template_individual",
    "supervised_ridge",
)
NUMERIC_FEATURES = (
    "SOH_50", "SOH_100", "SOH_150",
    "slope_1_50", "slope_50_100", "slope_100_150", "slope_50_150",
    "curvature", "noise_mad", "IR_mean", "IR_slope",
    "Tavg_mean", "Tavg_slope", "chargetime_mean", "chargetime_slope",
)


@dataclass(frozen=True)
class Question3Data:
    cycles: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    features: pd.DataFrame
    strategy_codes: dict[str, str]


def load_question3_data(project_root) -> Question3Data:
    cycles = pd.read_csv(project_root / "data/processed/question1/cleaned_cycle_data.csv")
    mapping = pd.read_csv(project_root / "outputs/question1/tables/strategy_mapping.csv")
    strategy_codes = dict(zip(mapping["strategy"], mapping["strategy_code"]))
    cycles["strategy_code"] = cycles["strategy"].map(strategy_codes)
    train = cycles.loc[cycles["prediction_test"] == 0].copy()
    test = cycles.loc[cycles["prediction_test"] == 1].copy()
    if train["battery_id"].nunique() != 40 or test["battery_id"].nunique() != 9:
        raise ValueError("Question three requires exactly 40 training and 9 test batteries")
    if train.groupby("battery_id")["cycle"].max().ne(200).any():
        raise ValueError("Every training battery must contain cycles 1-200")
    if test.groupby("battery_id")["cycle"].max().ne(150).any() or (test["cycle"] > 150).any():
        raise ValueError("Test batteries may contain cycles 1-150 only")
    features = extract_early_features(cycles.loc[cycles["cycle"] <= 150])
    return Question3Data(cycles, train, test, features, strategy_codes)


def _level(group: pd.DataFrame, cycle: int) -> float:
    exact = group.loc[group["cycle"] == cycle, "SOH_smooth_robust"]
    if not exact.empty:
        return float(exact.iloc[0])
    local = group.loc[group["cycle"].between(cycle - 4, cycle + 4)]
    slope, intercept, _, _ = theilslopes(local["SOH_smooth_robust"], local["cycle"])
    return float(intercept + slope * cycle)


def _curvature(group: pd.DataFrame) -> float:
    x = group["cycle"].to_numpy(float) / 100.0
    return float(2 * np.polyfit(x, group["SOH_smooth_robust"].to_numpy(float), 2)[0])


def extract_early_features(early_cycles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for battery_id, group in early_cycles.groupby("battery_id", sort=True):
        g = group.sort_values("cycle").loc[lambda x: x["cycle"] <= 150]
        if g["cycle"].max() != 150:
            raise ValueError(f"Battery {battery_id} lacks cycle 150")
        row: dict[str, object] = {
            "battery_id": int(battery_id), "strategy": g["strategy"].iloc[0],
            "strategy_code": g["strategy_code"].iloc[0],
            "dataset_id": int(g["dataset_id"].iloc[0]),
            "prediction_test": int(g["prediction_test"].iloc[0]),
            "SOH_50": _level(g, 50), "SOH_100": _level(g, 100), "SOH_150": _level(g, 150),
        }
        for start, end, name in (
            (1, 50, "slope_1_50"), (50, 100, "slope_50_100"),
            (100, 150, "slope_100_150"), (50, 150, "slope_50_150"),
        ):
            w = g.loc[g["cycle"].between(start, end)]
            row[name] = 100.0 * robust_slope(w["cycle"], w["SOH_smooth_robust"])
        residual = g["SOH_clean"] - g["SOH_smooth_robust"]
        row["curvature"] = _curvature(g)
        row["noise_mad"] = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
        for source, name in (("IR_clean", "IR"), ("Tavg", "Tavg"), ("chargetime", "chargetime")):
            row[f"{name}_mean"] = float(g[source].mean())
            row[f"{name}_slope"] = 100.0 * robust_slope(g["cycle"], g[source])
        rows.append(row)
    return pd.DataFrame(rows)


def strategy_template(train_cycles: pd.DataFrame, strategy: str, exclude_battery: int | None) -> pd.DataFrame:
    frame = train_cycles.loc[train_cycles["strategy"] == strategy]
    if exclude_battery is not None:
        frame = frame.loc[frame["battery_id"] != exclude_battery]
    if frame["battery_id"].nunique() < 1:
        raise ValueError(f"No peer battery remains for strategy {strategy}")
    template = frame.groupby("cycle", as_index=False)["SOH_smooth_robust"].mean()
    return template.rename(columns={"SOH_smooth_robust": "template_soh"})


def _monotone(prediction: np.ndarray, observed_150: float) -> np.ndarray:
    bounded = np.clip(np.asarray(prediction, float), 0.75, 1.05)
    return np.minimum.accumulate(np.r_[observed_150, bounded])[1:]


def _template_individual_prediction(early: pd.DataFrame, template: pd.DataFrame) -> np.ndarray:
    merged = early.merge(template, on="cycle", validate="one_to_one")
    recent = merged.loc[merged["cycle"] >= 51].copy()
    z = (recent["cycle"].to_numpy(float) - 150.0) / 100.0
    x = np.column_stack([np.ones(len(z)), z, z**2])
    residual = recent["SOH_smooth_robust"].to_numpy(float) - recent["template_soh"].to_numpy(float)
    weights = 0.25 + 0.75 * (recent["cycle"].to_numpy(float) / 150.0) ** 2
    penalty = np.diag([1e-8, 1.5, 20.0])
    beta = np.linalg.solve(x.T @ (weights[:, None] * x) + penalty, x.T @ (weights * residual))
    future_template = template.set_index("cycle").loc[FUTURE_CYCLES.astype(int), "template_soh"].to_numpy(float)
    future_z = (FUTURE_CYCLES - 150.0) / 100.0
    correction = np.column_stack([np.ones(50), future_z, future_z**2]) @ beta
    return _monotone(future_template + correction, float(early.loc[early["cycle"] == 150, "SOH_smooth_robust"].iloc[0]))


def base_curve_predictions(early: pd.DataFrame, template: pd.DataFrame) -> dict[str, np.ndarray]:
    early = early.sort_values("cycle")
    observed_150 = float(early.loc[early["cycle"] == 150, "SOH_smooth_robust"].iloc[0])
    quadratic = fit_degradation_model(
        "quadratic", early["cycle"].to_numpy(float), early["SOH_smooth_robust"].to_numpy(float)
    ).predict(FUTURE_CYCLES)
    recent = early.loc[early["cycle"].between(100, 150)]
    slope = robust_slope(recent["cycle"], recent["SOH_smooth_robust"])
    local_linear = observed_150 + slope * (FUTURE_CYCLES - 150.0)
    raw_template = template.set_index("cycle").loc[FUTURE_CYCLES.astype(int), "template_soh"].to_numpy(float)
    return {
        "individual_quadratic": _monotone(quadratic, observed_150),
        "local_linear": _monotone(local_linear, observed_150),
        "strategy_template": _monotone(raw_template, observed_150),
        "template_individual": _template_individual_prediction(early, template),
    }


def _future_targets(train_cycles: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feature_index = features.set_index("battery_id")
    x = (FUTURE_CYCLES - 150.0) / 50.0
    design = np.column_stack([x, x**2])
    for battery_id, group in train_cycles.groupby("battery_id"):
        future = group.set_index("cycle").loc[FUTURE_CYCLES.astype(int), "SOH_smooth_robust"].to_numpy(float)
        y150 = float(feature_index.loc[battery_id, "SOH_150"])
        coefficients = np.linalg.lstsq(design, future - y150, rcond=None)[0]
        rows.append({"battery_id": int(battery_id), "future_linear": coefficients[0],
                     "future_quadratic": coefficients[1]})
    return pd.DataFrame(rows)


def _design_matrix(frame: pd.DataFrame, strategy_levels: list[str], mean=None, scale=None):
    numeric = frame[list(NUMERIC_FEATURES)].to_numpy(float)
    if mean is None:
        mean = numeric.mean(axis=0)
        scale = numeric.std(axis=0)
        scale[scale == 0] = 1.0
    z = (numeric - mean) / scale
    onehot = np.column_stack([(frame["strategy_code"].to_numpy() == level).astype(float)
                              for level in strategy_levels])
    return np.column_stack([z, onehot]), mean, scale


def _choose_alpha(frame: pd.DataFrame, targets: np.ndarray, strategy_levels: list[str]) -> float:
    alphas = np.logspace(-3, 2, 12)
    splitter = KFold(n_splits=5, shuffle=False)
    scores = np.zeros(len(alphas))
    for train_idx, val_idx in splitter.split(frame):
        x_train, mean, scale = _design_matrix(frame.iloc[train_idx], strategy_levels)
        x_val, _, _ = _design_matrix(frame.iloc[val_idx], strategy_levels, mean, scale)
        y_train = targets[train_idx]
        y_mean, y_scale = y_train.mean(axis=0), y_train.std(axis=0)
        y_scale[y_scale == 0] = 1.0
        for index, alpha in enumerate(alphas):
            model = Ridge(alpha=float(alpha)).fit(x_train, (y_train - y_mean) / y_scale)
            error = model.predict(x_val) - (targets[val_idx] - y_mean) / y_scale
            scores[index] += float(np.mean(error**2))
    return float(alphas[int(np.argmin(scores))])


def supervised_ridge_predictions(
    train_cycles: pd.DataFrame, all_features: pd.DataFrame
) -> tuple[dict[int, np.ndarray], dict[int, float], callable]:
    train_features = all_features.loc[all_features["prediction_test"] == 0].sort_values("battery_id").reset_index(drop=True)
    target_frame = _future_targets(train_cycles, train_features)
    train_features = train_features.merge(target_frame, on="battery_id", validate="one_to_one")
    target_columns = ["future_linear", "future_quadratic"]
    strategy_levels = sorted(all_features["strategy_code"].unique())
    predictions: dict[int, np.ndarray] = {}
    selected_alphas: dict[int, float] = {}
    curve_x = (FUTURE_CYCLES - 150.0) / 50.0
    curve_design = np.column_stack([curve_x, curve_x**2])
    for target_index, target_row in train_features.iterrows():
        development = train_features.drop(index=target_index).reset_index(drop=True)
        y = development[target_columns].to_numpy(float)
        alpha = _choose_alpha(development, y, strategy_levels)
        x, mean, scale = _design_matrix(development, strategy_levels)
        y_mean, y_scale = y.mean(axis=0), y.std(axis=0)
        y_scale[y_scale == 0] = 1.0
        model = Ridge(alpha=alpha).fit(x, (y - y_mean) / y_scale)
        x_target, _, _ = _design_matrix(target_row.to_frame().T, strategy_levels, mean, scale)
        coefficients = model.predict(x_target)[0] * y_scale + y_mean
        y150 = float(target_row["SOH_150"])
        predictions[int(target_row["battery_id"])] = _monotone(y150 + curve_design @ coefficients, y150)
        selected_alphas[int(target_row["battery_id"])] = alpha

    full_y = train_features[target_columns].to_numpy(float)
    full_alpha = _choose_alpha(train_features, full_y, strategy_levels)
    full_x, full_mean, full_scale = _design_matrix(train_features, strategy_levels)
    full_y_mean, full_y_scale = full_y.mean(axis=0), full_y.std(axis=0)
    full_y_scale[full_y_scale == 0] = 1.0
    full_model = Ridge(alpha=full_alpha).fit(full_x, (full_y - full_y_mean) / full_y_scale)

    def predict_new(feature_row: pd.DataFrame) -> np.ndarray:
        x_new, _, _ = _design_matrix(feature_row, strategy_levels, full_mean, full_scale)
        coefficients = full_model.predict(x_new)[0] * full_y_scale + full_y_mean
        y150 = float(feature_row["SOH_150"].iloc[0])
        return _monotone(y150 + curve_design @ coefficients, y150)

    predict_new.alpha = full_alpha
    return predictions, selected_alphas, predict_new


def deviation_score(
    target_feature: pd.Series, peer_features: pd.DataFrame, reference_features: pd.DataFrame,
    early: pd.DataFrame, template: pd.DataFrame,
) -> tuple[float, str]:
    compare = ["SOH_150", "slope_100_150", "IR_slope"]
    global_scale = reference_features[compare].std(ddof=0).replace(0, np.nan)
    peer_center = peer_features[compare].median()
    peer_scale = peer_features[compare].std(ddof=0)
    scale = np.maximum(peer_scale.fillna(0).to_numpy(float), 0.5 * global_scale.to_numpy(float))
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    z = (target_feature[compare].to_numpy(float) - peer_center.to_numpy(float)) / scale
    merged = early.merge(template, on="cycle")
    residual_rmse = float(np.sqrt(np.mean((merged["SOH_smooth_robust"] - merged["template_soh"]) ** 2)))
    residual_scale = max(float(reference_features["noise_mad"].median()) * 5.0, 0.001)
    score = float(np.sqrt(np.mean(np.r_[z, residual_rmse / residual_scale] ** 2)))
    flag = "marked_deviation" if score >= 2.0 else "moderate_deviation" if score >= 1.25 else "within_group"
    return score, flag


def forecast_training_pseudotests(data: Question3Data):
    supervised, alphas, predict_new = supervised_ridge_predictions(data.train, data.features)
    train_features = data.features.loc[data.features["prediction_test"] == 0].copy()
    prediction_rows: list[dict[str, object]] = []
    deviation_rows: list[dict[str, object]] = []
    for battery_id, group in data.train.groupby("battery_id", sort=True):
        strategy = group["strategy"].iloc[0]
        early = group.loc[group["cycle"] <= 150].copy()
        template = strategy_template(data.train, strategy, int(battery_id))
        predictions = base_curve_predictions(early, template)
        predictions["supervised_ridge"] = supervised[int(battery_id)]
        target_feature = train_features.loc[train_features["battery_id"] == battery_id].iloc[0]
        peers = train_features.loc[(train_features["strategy"] == strategy) & (train_features["battery_id"] != battery_id)]
        score, flag = deviation_score(target_feature, peers, train_features.drop(index=target_feature.name), early, template)
        deviation_rows.append({"battery_id": int(battery_id), "strategy": strategy,
                               "strategy_code": group["strategy_code"].iloc[0],
                               "deviation_score": score, "deviation_flag": flag,
                               "supervised_inner_alpha": alphas[int(battery_id)]})
        actual = group.set_index("cycle").loc[FUTURE_CYCLES.astype(int), "SOH_clean"].to_numpy(float)
        for model, prediction in predictions.items():
            for cycle, predicted, observed in zip(FUTURE_CYCLES.astype(int), prediction, actual):
                prediction_rows.append({"battery_id": int(battery_id), "strategy": strategy,
                                        "strategy_code": group["strategy_code"].iloc[0], "model": model,
                                        "cycle": cycle, "actual_soh": observed, "predicted_soh": predicted})
    base_predictions = pd.DataFrame(prediction_rows)
    deviations = pd.DataFrame(deviation_rows)
    ensemble, weights = build_adaptive_ensemble(base_predictions, deviations, exclude_target=True)
    predictions = pd.concat([base_predictions, ensemble], ignore_index=True)
    return predictions, deviations, weights, predict_new


def _battery_rmse(frame: pd.DataFrame) -> pd.Series:
    return frame.assign(sq=lambda x: (x["predicted_soh"] - x["actual_soh"]) ** 2).groupby("model")["sq"].mean().pow(.5)


def build_adaptive_ensemble(
    base_predictions: pd.DataFrame, deviations: pd.DataFrame, *, exclude_target: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    components = ["template_individual", "supervised_ridge", "individual_quadratic", "local_linear"]
    rows, weight_rows = [], []
    for battery_id, target in base_predictions.groupby("battery_id"):
        calibration = base_predictions.loc[base_predictions["battery_id"] != battery_id] if exclude_target else base_predictions
        rmse = _battery_rmse(calibration.loc[calibration["model"].isin(components)]).reindex(components)
        base = 1.0 / np.maximum(rmse.to_numpy(float), 1e-6) ** 2
        base /= base.sum()
        score = float(deviations.loc[deviations["battery_id"] == battery_id, "deviation_score"].iloc[0])
        similarity = float(1.0 / (1.0 + max(0.0, score - 1.0) ** 2))
        adaptive = base.copy()
        adaptive[0] *= similarity
        adaptive /= adaptive.sum()
        wide = target.loc[target["model"].isin(components)].pivot(index="cycle", columns="model",
                                                                   values="predicted_soh").reindex(columns=components)
        actual = target.drop_duplicates("cycle").set_index("cycle")["actual_soh"]
        prediction = wide.to_numpy(float) @ adaptive
        for cycle, predicted in zip(wide.index, prediction):
            rows.append({"battery_id": int(battery_id), "strategy": target["strategy"].iloc[0],
                         "strategy_code": target["strategy_code"].iloc[0], "model": "adaptive_ensemble",
                         "cycle": int(cycle), "actual_soh": float(actual.loc[cycle]),
                         "predicted_soh": float(predicted)})
        for component, base_weight, adaptive_weight in zip(components, base, adaptive):
            weight_rows.append({"battery_id": int(battery_id), "scope": "training_pseudotest",
                                "component": component, "base_weight": base_weight,
                                "similarity_factor": similarity, "adaptive_weight": adaptive_weight})
    return pd.DataFrame(rows), pd.DataFrame(weight_rows)


def prediction_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for (battery_id, model), group in predictions.groupby(["battery_id", "model"]):
        error = group["predicted_soh"].to_numpy(float) - group["actual_soh"].to_numpy(float)
        actual_slope = robust_slope(group["cycle"], group["actual_soh"])
        predicted_slope = robust_slope(group["cycle"], group["predicted_soh"])
        rows.append({"battery_id": int(battery_id), "strategy": group["strategy"].iloc[0],
                     "strategy_code": group["strategy_code"].iloc[0], "model": model,
                     "MAE": np.mean(np.abs(error)), "RMSE": np.sqrt(np.mean(error**2)),
                     "MaxAE": np.max(np.abs(error)),
                     "cycle200_error": float(error[group["cycle"].to_numpy() == 200][0]),
                     "cycle200_abs_error": abs(float(error[group["cycle"].to_numpy() == 200][0])),
                     "actual_future_slope_per100": 100 * actual_slope,
                     "predicted_future_slope_per100": 100 * predicted_slope,
                     "trend_sign_match": bool(np.sign(actual_slope) == np.sign(predicted_slope))})
    battery = pd.DataFrame(rows)
    overall_rows = []
    for model, group in battery.groupby("model"):
        point = predictions.loc[predictions["model"] == model]
        overall_rows.append({"model": model, "n_batteries": len(group),
                             "MAE_mean": group["MAE"].mean(), "MAE_median": group["MAE"].median(),
                             "RMSE_mean": group["RMSE"].mean(), "RMSE_median": group["RMSE"].median(),
                             "RMSE_p90": group["RMSE"].quantile(.9), "worst_battery_RMSE": group["RMSE"].max(),
                             "pointwise_RMSE": np.sqrt(np.mean((point["predicted_soh"]-point["actual_soh"])**2)),
                             "cycle200_MAE": group["cycle200_abs_error"].mean(),
                             "trend_match_rate": group["trend_sign_match"].mean(),
                             "strategy_median_RMSE_sd": group.groupby("strategy_code")["RMSE"].median().std(ddof=1)})
    overall = pd.DataFrame(overall_rows)
    strategy = battery.groupby(["model", "strategy", "strategy_code"], as_index=False).agg(
        n_batteries=("battery_id", "size"), MAE_mean=("MAE", "mean"), RMSE_mean=("RMSE", "mean"),
        RMSE_median=("RMSE", "median"), worst_RMSE=("RMSE", "max"),
        cycle200_MAE=("cycle200_abs_error", "mean"), trend_match_rate=("trend_sign_match", "mean"))
    return battery, overall.sort_values("RMSE_mean"), strategy


def forecast_test_batteries(
    data: Question3Data, pseudotest_predictions: pd.DataFrame, deviations_train: pd.DataFrame,
    predict_supervised,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_features = data.features.loc[data.features["prediction_test"] == 0]
    test_features = data.features.loc[data.features["prediction_test"] == 1]
    component_rmse = _battery_rmse(pseudotest_predictions.loc[pseudotest_predictions["model"].isin(BASE_MODELS)])
    components = ["template_individual", "supervised_ridge", "individual_quadratic", "local_linear"]
    base = 1.0 / np.maximum(component_rmse.reindex(components).to_numpy(float), 1e-6) ** 2
    base /= base.sum()
    selected_cv = pseudotest_predictions.loc[pseudotest_predictions["model"] == "adaptive_ensemble"].copy()
    selected_cv["sq_error"] = (selected_cv["predicted_soh"] - selected_cv["actual_soh"]) ** 2
    cv_strategy_rmse = selected_cv.groupby("strategy_code")["sq_error"].mean().pow(.5)
    prediction_rows, deviation_rows, weight_rows, component_rows = [], [], [], []
    for battery_id, group in data.test.groupby("battery_id", sort=True):
        strategy = group["strategy"].iloc[0]
        template = strategy_template(data.train, strategy, None)
        predictions = base_curve_predictions(group, template)
        feature = test_features.loc[test_features["battery_id"] == battery_id]
        predictions["supervised_ridge"] = predict_supervised(feature)
        peers = train_features.loc[train_features["strategy"] == strategy]
        score, flag = deviation_score(feature.iloc[0], peers, train_features, group, template)
        similarity = float(1.0 / (1.0 + max(0.0, score - 1.0) ** 2))
        strategy_rmse = float(cv_strategy_rmse.loc[group["strategy_code"].iloc[0]])
        if flag == "marked_deviation" or strategy_rmse > .001:
            short_reliability = "lower"
            reliability_reason = "marked peer deviation or difficult strategy in LOBO validation"
        elif flag == "moderate_deviation" or strategy_rmse > .0005:
            short_reliability = "moderate"
            reliability_reason = "moderate peer deviation or strategy-level LOBO error"
        else:
            short_reliability = "relatively_high"
            reliability_reason = "within peer group and low strategy-level LOBO error"
        adaptive = base.copy(); adaptive[0] *= similarity; adaptive /= adaptive.sum()
        ensemble = np.column_stack([predictions[name] for name in components]) @ adaptive
        spread = np.std(np.column_stack([predictions[name] for name in components]), axis=1, ddof=1)
        deviation_rows.append({"battery_id": int(battery_id), "strategy": strategy,
                               "strategy_code": group["strategy_code"].iloc[0],
                               "deviation_score": score, "deviation_flag": flag,
                               "template_similarity_factor": similarity,
                               "strategy_lobo_pointwise_rmse": strategy_rmse,
                               "short_horizon_reliability": short_reliability,
                               "short_horizon_reliability_reason": reliability_reason})
        for name, bw, aw in zip(components, base, adaptive):
            weight_rows.append({"battery_id": int(battery_id), "scope": "test_forecast", "component": name,
                                "base_weight": bw, "similarity_factor": similarity, "adaptive_weight": aw})
        for name, curve in predictions.items():
            for cycle, value in zip(FUTURE_CYCLES.astype(int), curve):
                component_rows.append({"battery_id": int(battery_id), "strategy": strategy,
                                       "strategy_code": group["strategy_code"].iloc[0], "model": name,
                                       "cycle": cycle, "predicted_soh": value})
        for cycle, value, model_spread in zip(FUTURE_CYCLES.astype(int), ensemble, spread):
            prediction_rows.append({"battery_id": int(battery_id), "strategy": strategy,
                                    "strategy_code": group["strategy_code"].iloc[0], "cycle": cycle,
                                    "predicted_soh": value, "model_disagreement_sd": model_spread})
    predictions = pd.DataFrame(prediction_rows)
    deviations = pd.DataFrame(deviation_rows)
    weights = pd.DataFrame(weight_rows)
    components_frame = pd.DataFrame(component_rows)
    predictions = add_prediction_intervals(predictions, pseudotest_predictions, deviations)
    return predictions, deviations, weights, components_frame


def add_prediction_intervals(
    test_predictions: pd.DataFrame, pseudotest_predictions: pd.DataFrame, deviations: pd.DataFrame,
) -> pd.DataFrame:
    calibration = pseudotest_predictions.loc[pseudotest_predictions["model"] == "adaptive_ensemble"].copy()
    calibration["absolute_error"] = (calibration["actual_soh"] - calibration["predicted_soh"]).abs()
    quantile = calibration.groupby("cycle")["absolute_error"].quantile(.95)
    result = test_predictions.copy()
    result["calibration_abs_error_q95"] = result["cycle"].map(quantile)
    result = result.merge(
        deviations[["battery_id", "deviation_score", "deviation_flag",
                    "strategy_lobo_pointwise_rmse", "short_horizon_reliability"]],
        on="battery_id",
    )
    inflation = 1.0 + 0.15 * np.maximum(0.0, result["deviation_score"] - 1.0)
    half_width = inflation * result["calibration_abs_error_q95"] + result["model_disagreement_sd"]
    result["prediction_interval_95_low"] = np.clip(result["predicted_soh"] - half_width, 0.75, 1.05)
    result["prediction_interval_95_high"] = np.clip(result["predicted_soh"] + half_width, 0.75, 1.05)
    result["interval_method"] = "LOBO horizon-wise 95% absolute-error quantile + ensemble disagreement"
    return result


def _fit_eol_from_hybrid(early: pd.DataFrame, future: np.ndarray) -> float:
    cycle = np.r_[early["cycle"].to_numpy(float), FUTURE_CYCLES]
    values = np.r_[early["SOH_smooth_robust"].to_numpy(float), future]
    return float(fit_degradation_model("quadratic", cycle, values).eol_cycle)


def _group_constrained_future(future: np.ndarray, template_future: np.ndarray, similarity: float) -> np.ndarray:
    group_weight = min(0.30, 0.30 * similarity)
    return (1.0 - group_weight) * future + group_weight * template_future


def eol_pseudotest_comparison(
    data: Question3Data, predictions: pd.DataFrame, deviations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    q1 = pd.read_csv(
        PROJECT_ROOT / "outputs/question1/tables/battery_question1_results.csv"
    ).set_index("battery_id")
    rows = []
    ensemble = predictions.loc[predictions["model"] == "adaptive_ensemble"]
    for battery_id, group in data.train.groupby("battery_id", sort=True):
        early = group.loc[group["cycle"] <= 150]
        future = ensemble.loc[ensemble["battery_id"] == battery_id].sort_values("cycle")["predicted_soh"].to_numpy(float)
        template = strategy_template(data.train, group["strategy"].iloc[0], int(battery_id))
        template_future = _template_individual_prediction(early, template)
        similarity = float(1.0 / (1.0 + max(0.0, float(deviations.loc[deviations["battery_id"] == battery_id,
                                                                           "deviation_score"].iloc[0]) - 1.0) ** 2))
        eol150 = float(fit_degradation_model("quadratic", early["cycle"], early["SOH_smooth_robust"]).eol_cycle)
        eol_pred = _fit_eol_from_hybrid(early, future)
        constrained_future = _group_constrained_future(future, template_future, similarity)
        eol_group = _fit_eol_from_hybrid(early, constrained_future)
        reference = float(q1.loc[battery_id, "eol_point"])
        for scheme, value in (("quadratic_150", eol150), ("quadratic_predicted_200", eol_pred),
                              ("group_constrained_quadratic", eol_group)):
            rows.append({"battery_id": int(battery_id), "strategy": group["strategy"].iloc[0],
                         "strategy_code": group["strategy_code"].iloc[0], "scheme": scheme,
                         "estimated_eol": value, "actual200_quadratic_reference": reference,
                         "absolute_difference_from_reference": abs(value-reference) if np.isfinite(value) else np.nan,
                         "relative_difference_from_reference": abs(value-reference)/reference if np.isfinite(value) else np.nan})
    battery = pd.DataFrame(rows)
    summary = battery.groupby("scheme", as_index=False).agg(
        valid_fraction=("estimated_eol", lambda x: np.isfinite(x).mean()),
        median_absolute_difference=("absolute_difference_from_reference", "median"),
        median_relative_difference=("relative_difference_from_reference", "median"),
        p90_relative_difference=("relative_difference_from_reference", lambda x: x.quantile(.9)),
        worst_relative_difference=("relative_difference_from_reference", "max"))
    summary["selection_score"] = summary["median_relative_difference"] + 0.5 * summary["p90_relative_difference"]
    summary.loc[summary["valid_fraction"] < .9, "selection_score"] += 10
    selected = str(summary.sort_values("selection_score").iloc[0]["scheme"])
    summary["selected"] = summary["scheme"].eq(selected)
    return battery, summary.sort_values("selection_score"), selected


def estimate_test_eol(
    data: Question3Data, test_predictions: pd.DataFrame, components: pd.DataFrame,
    deviations: pd.DataFrame, selected_scheme: str, pseudotest_predictions: pd.DataFrame,
    *, bootstrap_samples: int = 300,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED + 900)
    residual_vectors = pseudotest_predictions.loc[pseudotest_predictions["model"] == "adaptive_ensemble"].copy()
    residual_vectors["residual"] = residual_vectors["actual_soh"] - residual_vectors["predicted_soh"]
    residual_wide = residual_vectors.pivot(index="battery_id", columns="cycle", values="residual").to_numpy(float)
    rows, curves = [], []
    for battery_id, group in data.test.groupby("battery_id", sort=True):
        early = group.sort_values("cycle")
        prediction = test_predictions.loc[test_predictions["battery_id"] == battery_id].sort_values("cycle")["predicted_soh"].to_numpy(float)
        template = strategy_template(data.train, group["strategy"].iloc[0], None)
        template_future = _template_individual_prediction(early, template)
        similarity = float(deviations.loc[deviations["battery_id"] == battery_id, "template_similarity_factor"].iloc[0])
        candidate_future = {
            "quadratic_150": fit_degradation_model("quadratic", early["cycle"], early["SOH_smooth_robust"]).predict(FUTURE_CYCLES),
            "quadratic_predicted_200": prediction,
            "group_constrained_quadratic": _group_constrained_future(prediction, template_future, similarity),
        }
        eol_values = {
            "eol_quadratic_150": float(fit_degradation_model("quadratic", early["cycle"], early["SOH_smooth_robust"]).eol_cycle),
            "eol_quadratic_predicted_200": _fit_eol_from_hybrid(early, prediction),
            "eol_group_constrained_quadratic": _fit_eol_from_hybrid(early, candidate_future["group_constrained_quadratic"]),
        }
        selected_future = candidate_future[selected_scheme]
        point = (eol_values["eol_quadratic_150"] if selected_scheme == "quadratic_150"
                 else _fit_eol_from_hybrid(early, selected_future))
        bootstrap = []
        for _ in range(bootstrap_samples):
            residual = residual_wide[rng.integers(0, len(residual_wide))]
            pseudo_future = _monotone(selected_future + residual, float(early["SOH_smooth_robust"].iloc[-1]))
            value = _fit_eol_from_hybrid(early, pseudo_future)
            if np.isfinite(value): bootstrap.append(value)
        boot = np.asarray(bootstrap)
        low, median, high = np.quantile(boot, [.025, .5, .975]) if len(boot) >= 50 else (np.nan,)*3
        hybrid_cycle = np.r_[early["cycle"].to_numpy(float), FUTURE_CYCLES]
        hybrid_values = np.r_[early["SOH_smooth_robust"].to_numpy(float), selected_future]
        structural = []
        for model in ("linear", "quadratic", "power", "exponential"):
            value = fit_degradation_model(model, hybrid_cycle, hybrid_values).eol_cycle
            if np.isfinite(value): structural.append(value)
        disagreement = ((max(structural)-min(structural))/np.median(structural)) if len(structural) >= 2 else np.nan
        width_ratio = (high-low)/median if np.isfinite(median) else np.nan
        deviation_flag = deviations.loc[deviations["battery_id"] == battery_id, "deviation_flag"].iloc[0]
        if len(boot)/bootstrap_samples < .8 or (np.isfinite(width_ratio) and width_ratio > .5): reliability = "low"
        elif deviation_flag == "marked_deviation" or (np.isfinite(disagreement) and disagreement > .5): reliability = "moderate_low"
        elif deviation_flag == "moderate_deviation" or (np.isfinite(disagreement) and disagreement > .25): reliability = "moderate"
        else: reliability = "relatively_high_conditional"
        rows.append({"battery_id": int(battery_id), "strategy": group["strategy"].iloc[0],
                     "strategy_code": group["strategy_code"].iloc[0], **eol_values,
                     "selected_scheme": selected_scheme, "selected_eol_point": point,
                     "bootstrap_eol_median": median, "eol_ci95_low": low, "eol_ci95_high": high,
                     "bootstrap_valid_fraction": len(boot)/bootstrap_samples,
                     "eol_interval_width_ratio": width_ratio,
                     "candidate_model_disagreement_ratio": disagreement,
                     "deviation_flag": deviation_flag, "reliability": reliability,
                     "interpretation": "model-based extrapolation; not observed true lifetime"})
        fit = fit_degradation_model("quadratic", hybrid_cycle, hybrid_values)
        end = min(5000, int(np.ceil(point*1.08/10)*10)) if np.isfinite(point) else 5000
        grid = np.arange(1, end+1, 5)
        for cycle, value in zip(grid, fit.predict(grid)):
            curves.append({"battery_id": int(battery_id), "cycle": int(cycle), "predicted_soh": value,
                           "selected_scheme": selected_scheme})
    return pd.DataFrame(rows), pd.DataFrame(curves)
