"""Reproducible, non-destructive pipeline for the unified quadratic refactor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, RANDOM_SEED
from .question4_models import add_physical_quantities, fit_charge_time_model
from .unified_q3q4 import (
    add_candidate_risk_and_curves,
    bootstrap_optimization_life,
    choose_short_horizon_model,
    compare_policy_to_baseline,
    feature_ablation_validation,
    forecast_test_with_policy,
    pareto_time_life,
    policy_pseudotest_predictions,
    prepare_optimization_grid,
    representative_time_life,
    robustness_from_bootstrap,
)
from .unified_quadratic import (
    add_phase_exposures,
    attach_quadratic_fit_rmse,
    bootstrap_battery_parameters,
    build_battery_parameter_table,
    build_cutoff_parameter_stability,
    build_strategy_table,
    compare_strategy_models,
    evidence_matrix,
    early_feature_diagnostics,
    extract_early_health_features,
    fit_selected_strategy_model,
    grouped_bootstrap_strategy_model,
    identifiability_summary,
    local_parameter_matches,
    mechanism_model_comparison,
    parameter_specific_matches,
    predict_native_parameters,
    select_strategy_model,
    standardized_ols_coefficients,
    strategy_eol_recreation,
    structure_matched_comparison,
    predictor_evidence_matrix,
)


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "unified_quadratic_v2"
FIGURE_ROOT = PROJECT_ROOT / "figures" / "unified_quadratic_v2"


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _json_ready(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        native = float(value)
        return native if np.isfinite(native) else None
    if isinstance(value, np.integer):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _phase_slope_definition_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"feature": "slope_1_50", "cycle_window": "1-50", "role": "initial observed phase"},
            {"feature": "slope_50_100", "cycle_window": "50-100", "role": "early observed phase"},
            {"feature": "slope_100_150", "cycle_window": "100-150", "role": "middle observed phase"},
            {"feature": "slope_150_200", "cycle_window": "150-200", "role": "late observed phase"},
            {"feature": "slope_50_200", "cycle_window": "50-200", "role": "summary only; not a uniquely privileged ageing rate"},
        ]
    )


def _domain_sensitivity(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    definitions = {
        "rectangle": candidates.index == candidates.index,
        "convex_hull": candidates["inside_convex_hull"],
        "trust_region": candidates["inside_trust_region"],
        "convex_hull_intersection_trust": candidates["inside_main_domain"],
    }
    fronts, recommendations = [], []
    for name, mask in definitions.items():
        feasible = candidates.loc[mask].copy()
        front = pareto_time_life(feasible, "eol_median")
        front["domain"] = name
        selected = representative_time_life(feasible, front, "eol_median")
        selected["domain"] = name
        fronts.append(front)
        recommendations.append(selected)
    return pd.concat(fronts, ignore_index=True), pd.concat(recommendations, ignore_index=True)


def _p_sensitivity(
    strategy_table: pd.DataFrame,
    comparison: pd.DataFrame,
    responses: list[str],
    parameterization: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fronts, recommendations = [], []
    for p in (1.0, 1.5, 2.0, 2.5, 3.0):
        options = comparison.loc[
            (comparison["response"] == "combined")
            & (comparison["model"] == "separate_phase_exposure")
            & np.isclose(comparison["p"], p)
        ]
        selection = options.sort_values("normalized_loso_rmse").iloc[0]
        model, _ = fit_selected_strategy_model(strategy_table, responses, selection)
        evaluated, metadata = prepare_optimization_grid(
            strategy_table, model, parameterization, selection
        )
        feasible = evaluated.loc[evaluated["inside_main_domain"]].copy()
        feasible["eol_median"] = feasible["predicted_eol_a1"]
        front = pareto_time_life(feasible, "eol_median")
        front["p"] = p
        selected = representative_time_life(feasible, front, "eol_median")
        selected["p"] = p
        selected["normalized_loso_rmse"] = float(selection["normalized_loso_rmse"])
        fronts.append(front)
        recommendations.append(selected)
    return pd.concat(fronts, ignore_index=True), pd.concat(recommendations, ignore_index=True)


def _leave_one_strategy_sensitivity(
    strategy_table: pd.DataFrame,
    selection: pd.Series,
    responses: list[str],
    parameterization: str,
    evaluated_grid: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for omitted in strategy_table.loc[strategy_table["main_dataset3"], "strategy_code"]:
        reduced = strategy_table.loc[strategy_table["strategy_code"] != omitted].copy()
        model, _ = fit_selected_strategy_model(reduced, responses, selection)
        grid = evaluated_grid.copy()
        if selection["model"] != "raw_parameters" and "E1" not in grid:
            grid = add_phase_exposures(grid, float(selection["p"]))
        predicted = predict_native_parameters(model, grid, parameterization)
        predicted["predicted_charge_time"] = evaluated_grid["predicted_charge_time"].to_numpy(float)
        predicted["inside_main_domain"] = evaluated_grid["inside_main_domain"].to_numpy(bool)
        feasible = predicted.loc[predicted["inside_main_domain"]].copy()
        feasible["eol_median"] = feasible["predicted_eol_a1"]
        front = pareto_time_life(feasible, "eol_median")
        selected = representative_time_life(feasible, front, "eol_median")
        selected["omitted_strategy_code"] = omitted
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def _nearest_existing_comparison(
    recommendations: pd.DataFrame,
    evaluated_observed: pd.DataFrame,
) -> pd.DataFrame:
    observed = evaluated_observed.reset_index(drop=True)
    lower = observed[["C1", "Q1", "C2"]].min().to_numpy(float)
    span = np.maximum(observed[["C1", "Q1", "C2"]].max().to_numpy(float) - lower, 1e-12)
    rows = []
    for _, recommendation in recommendations.iterrows():
        distance = np.linalg.norm(
            (observed[["C1", "Q1", "C2"]].to_numpy(float) - recommendation[["C1", "Q1", "C2"]].to_numpy(float)) / span,
            axis=1,
        )
        nearest = observed.iloc[int(np.argmin(distance))]
        rows.append(
            {
                "recommendation_type": recommendation["recommendation_type"],
                "C1": recommendation["C1"], "Q1": recommendation["Q1"], "C2": recommendation["C2"],
                "nearest_strategy": nearest["strategy"], "nearest_strategy_code": nearest["strategy_code"],
                "nearest_distance_normalized": float(distance.min()),
                "delta_charge_time_minutes": float(recommendation["predicted_charge_time"] - nearest["predicted_charge_time"]),
                "delta_d1": float(recommendation["predicted_d1"] - nearest["predicted_d1"]),
                "delta_d2": float(recommendation["predicted_d2"] - nearest["predicted_d2"]),
                "delta_eol_median": float(recommendation["eol_median"] - nearest["predicted_eol_a1"]),
                "comparison_status": "model-predicted comparison; not experimental validation",
            }
        )
    return pd.DataFrame(rows)


def _old_new_model_comparison(
    new_comparison: pd.DataFrame,
    recreation: pd.DataFrame,
    project_root: Path,
) -> pd.DataFrame:
    rows = []
    combined = new_comparison.loc[new_comparison["response"] == "combined"]
    for name in ("raw_parameters", "separate_phase_exposure", "summed_stress_benchmark"):
        best = combined.loc[combined["model"] == name].sort_values("normalized_loso_rmse").iloc[0]
        rows.append(
            {
                "criterion": "Q2_LOSO_combined_normalized_RMSE", "model_family": name,
                "value": float(best["normalized_loso_rmse"]), "detail": f"p={best['p']}; alpha={best['alpha']}",
            }
        )
    rows.append(
        {
            "criterion": "observed_strategy_EOL_recreation_MAE_a_equals_1",
            "model_family": "unified_quadratic_parameter_bridge",
            "value": float(recreation["predicted_minus_observed_eol_a1"].abs().mean()),
            "detail": "dataset-3 observed strategies",
        }
    )
    old_validation_path = project_root / "outputs/question4/tables/degradation_proxy_validation.csv"
    if old_validation_path.exists():
        old = pd.read_csv(old_validation_path)
        subset = old.loc[old["scope"] == "dataset3_only"]
        for response, group in subset.groupby("response"):
            best = group.sort_values("normalized_LOSO_RMSE").iloc[0]
            rows.append(
                {
                    "criterion": f"old_Q4_LOSO_{response}",
                    "model_family": f"old_{best['model']}", "value": float(best["normalized_LOSO_RMSE"]),
                    "detail": f"p={best['p']}",
                }
            )
    old_recreation_path = project_root / "outputs/question4/tables/observed_strategy_surrogate_predictions.csv"
    if old_recreation_path.exists():
        old_recreation = pd.read_csv(old_recreation_path)
        actual_candidates = [c for c in old_recreation.columns if "eol" in c.lower() and ("actual" in c.lower() or "median" in c.lower())]
        predicted_candidates = [c for c in old_recreation.columns if "eol" in c.lower() and "predicted" in c.lower()]
        if actual_candidates and predicted_candidates:
            actual_column, predicted_column = actual_candidates[0], predicted_candidates[0]
            rows.append(
                {
                    "criterion": "observed_strategy_EOL_recreation_MAE_baseline_definition",
                    "model_family": "old_SOC_stress_surrogate",
                    "value": float((old_recreation[predicted_column] - old_recreation[actual_column]).abs().mean()),
                    "detail": f"{predicted_column} vs {actual_column}; not directly comparable to a=1 EOL",
                }
            )
    return pd.DataFrame(rows)


def run_unified_pipeline(
    project_root: Path = PROJECT_ROOT,
    *,
    battery_bootstrap_samples: int = 300,
    strategy_bootstrap_samples: int = 1_000,
    optimization_bootstrap_samples: int = 1_000,
    test_eol_bootstrap_samples: int = 300,
) -> dict[str, Any]:
    """Run the full Q1--Q4 refactor without modifying baseline artefacts."""
    output_root = project_root / "outputs" / "unified_quadratic_v2"
    figure_root = project_root / "figures" / "unified_quadratic_v2"
    output_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    q1_root, q2_root, q3_root, q4_root = [output_root / f"question{i}" / "tables" for i in range(1, 5)]

    q1_baseline = pd.read_csv(project_root / "outputs/question1/tables/battery_question1_results.csv")
    candidate_fits = pd.read_csv(project_root / "outputs/question1/tables/candidate_model_fits.csv")
    cycles = pd.read_csv(project_root / "data/processed/question1/cleaned_cycle_data.csv")
    summary = pd.read_csv(project_root / "data/raw/battery_summary.csv")

    battery_parameters = attach_quadratic_fit_rmse(
        build_battery_parameter_table(q1_baseline), candidate_fits
    )
    parameter_draws, parameter_bootstrap = bootstrap_battery_parameters(
        cycles, samples=battery_bootstrap_samples
    )
    cutoff_stability, cutoff_summary = build_cutoff_parameter_stability(candidate_fits)
    identifiability, parameterization = identifiability_summary(
        battery_parameters, parameter_bootstrap, cutoff_summary
    )
    early_health = extract_early_health_features(cycles)
    battery_table, strategy_table = build_strategy_table(
        battery_parameters, q1_baseline, summary, early_health
    )
    _write(battery_parameters, q1_root / "battery_quadratic_parameters.csv")
    _write(parameter_draws, q1_root / "battery_parameter_bootstrap_draws.csv")
    _write(parameter_bootstrap, q1_root / "battery_parameter_bootstrap_summary.csv")
    _write(cutoff_stability, q1_root / "cutoff_parameter_stability_by_battery.csv")
    _write(cutoff_summary, q1_root / "cutoff_parameter_stability_summary.csv")
    _write(identifiability, q1_root / "parameter_identifiability_diagnostics.csv")
    _write(_phase_slope_definition_table(), q1_root / "phase_slope_definitions.csv")

    responses = ["R100", "A"] if parameterization == "R100_A" else ["d1", "d2"]
    strategy_comparison, strategy_loso, collinearity = compare_strategy_models(
        strategy_table, responses
    )
    selection = select_strategy_model(strategy_comparison)
    strategy_model, model_frame = fit_selected_strategy_model(strategy_table, responses, selection)
    coefficient_distribution, coefficient_summary = grouped_bootstrap_strategy_model(
        battery_table, selection, responses, samples=strategy_bootstrap_samples
    )
    exposure_selection = (
        strategy_comparison.loc[
            (strategy_comparison["response"] == "combined")
            & (strategy_comparison["model"] == "separate_phase_exposure")
        ].sort_values("normalized_loso_rmse").iloc[0]
    )
    exposure_coefficient_distribution, exposure_coefficient_summary = grouped_bootstrap_strategy_model(
        battery_table, exposure_selection, responses, samples=strategy_bootstrap_samples
    )
    observed_prediction = predict_native_parameters(strategy_model, model_frame, parameterization)
    recreation = strategy_eol_recreation(observed_prediction, strategy_table)
    matches = local_parameter_matches(strategy_table)
    parameter_matches = parameter_specific_matches(strategy_table)
    structure_comparison = structure_matched_comparison(strategy_table)
    mechanism = mechanism_model_comparison(battery_table)
    early_correlation, early_vif = early_feature_diagnostics(battery_table)
    evidence = evidence_matrix(
        coefficient_summary, strategy_comparison, mechanism, structure_comparison
    )
    predictor_evidence = predictor_evidence_matrix(
        coefficient_summary, exposure_coefficient_summary, strategy_comparison,
        mechanism, parameter_matches,
    )
    raw_ols = standardized_ols_coefficients(
        strategy_table.loc[strategy_table["main_dataset3"]], ["C1", "Q1", "C2"],
        responses, model_name="raw_parameters_OLS",
    )
    exposure_ols_frame = add_phase_exposures(
        strategy_table.loc[strategy_table["main_dataset3"]], float(exposure_selection["p"])
    )
    exposure_ols = standardized_ols_coefficients(
        exposure_ols_frame, ["E1", "E2"], responses,
        model_name=f"separate_phase_exposure_OLS_p{float(exposure_selection['p']):g}",
    )
    ols_coefficients = pd.concat([raw_ols, exposure_ols], ignore_index=True)
    all_scope_table = strategy_table.copy()
    all_scope_table["main_dataset3"] = all_scope_table["parameterized"]
    all_scope_comparison, _, _ = compare_strategy_models(all_scope_table, responses)
    all_scope_comparison["scope"] = "all_parameterized"
    dataset3_comparison = strategy_comparison.copy()
    dataset3_comparison["scope"] = "dataset3_only"
    scope_sensitivity = pd.concat([dataset3_comparison, all_scope_comparison], ignore_index=True)
    _write(battery_table, q2_root / "battery_parameter_analysis_data.csv")
    _write(strategy_table, q2_root / "strategy_parameter_analysis_data.csv")
    _write(early_health, q2_root / "early_cycle_health_features.csv")
    _write(strategy_comparison, q2_root / "strategy_model_loso_comparison.csv")
    _write(strategy_loso, q2_root / "strategy_model_loso_predictions.csv")
    _write(collinearity, q2_root / "strategy_model_collinearity.csv")
    _write(pd.DataFrame([selection.to_dict() | {"parameterization": parameterization}]), q2_root / "selected_strategy_parameter_model.csv")
    _write(strategy_model.coefficients(), q2_root / "selected_strategy_model_coefficients.csv")
    _write(coefficient_distribution, q2_root / "grouped_bootstrap_coefficient_draws.csv")
    _write(coefficient_summary, q2_root / "grouped_bootstrap_coefficient_summary.csv")
    _write(exposure_coefficient_distribution, q2_root / "exposure_grouped_bootstrap_coefficient_draws.csv")
    _write(exposure_coefficient_summary, q2_root / "exposure_grouped_bootstrap_coefficient_summary.csv")
    _write(ols_coefficients, q2_root / "descriptive_ols_standardized_coefficients.csv")
    _write(recreation, q2_root / "observed_strategy_parameter_and_eol_recreation.csv")
    _write(matches, q2_root / "local_matched_strategy_comparisons.csv")
    _write(parameter_matches, q2_root / "parameter_specific_quasi_control_matches.csv")
    _write(structure_comparison, q2_root / "S3_S9_structure_batch_contrast.csv")
    _write(mechanism, q2_root / "early_health_mechanism_lobo_comparison.csv")
    _write(early_correlation, q2_root / "early_health_feature_correlations.csv")
    _write(early_vif, q2_root / "early_health_feature_vif.csv")
    _write(evidence, q2_root / "seven_channel_evidence_matrix.csv")
    _write(predictor_evidence, q2_root / "predictor_seven_channel_evidence_matrix.csv")
    _write(scope_sensitivity, q2_root / "dataset_scope_sensitivity.csv")

    policy_predictions, policy_diagnostics, penalty = policy_pseudotest_predictions(
        cycles, battery_table, selection, parameterization
    )
    baseline_pseudotest = pd.read_csv(
        project_root / "outputs/question3/tables/pseudotest_cycle_predictions.csv"
    )
    combined_predictions, model_battery_metrics, model_comparison = compare_policy_to_baseline(
        policy_predictions, baseline_pseudotest, penalty
    )
    final_short_model = choose_short_horizon_model(model_comparison)
    ablation = feature_ablation_validation(
        cycles, battery_table, selection, parameterization
    )
    baseline_test = pd.read_csv(
        project_root / "outputs/question3/tables/test_cycle151_200_predictions.csv"
    )
    test_predictions, test_eol, test_eol_curves = forecast_test_with_policy(
        cycles, battery_table, selection, parameterization, penalty, baseline_test,
        final_short_model, bootstrap_samples=test_eol_bootstrap_samples,
    )
    _write(policy_predictions, q3_root / "policy_quadratic_pseudotest_predictions_all_penalties.csv")
    _write(policy_diagnostics, q3_root / "policy_penalty_lobo_diagnostics.csv")
    _write(combined_predictions, q3_root / "selected_models_pseudotest_cycle_predictions.csv")
    _write(model_battery_metrics, q3_root / "selected_models_pseudotest_battery_metrics.csv")
    _write(model_comparison, q3_root / "selected_models_pseudotest_comparison.csv")
    _write(ablation, q3_root / "feature_ablation_lobo_results.csv")
    _write(pd.DataFrame([{"selected_penalty": penalty, "selected_short_horizon_model": final_short_model,
                          "final_eol_model": "same constrained quadratic refit to observed1-150 plus selected forecast151-200"}]),
           q3_root / "final_model_selection.csv")
    _write(test_predictions, q3_root / "test_cycle151_200_predictions.csv")
    _write(test_eol, q3_root / "test_quadratic_eol_estimates.csv")
    _write(test_eol_curves, q3_root / "test_quadratic_eol_curves.csv")

    evaluated_grid, metadata = prepare_optimization_grid(
        strategy_table, strategy_model, parameterization, selection
    )
    charge_validation = metadata.pop("charge_validation")
    feasible_grid = evaluated_grid.loc[evaluated_grid["inside_main_domain"]].copy().reset_index(drop=True)
    observed_main_for_nearest = strategy_table.loc[strategy_table["main_dataset3"]].reset_index(drop=True)
    nearest_position = feasible_grid["nearest_observed_index"].astype(int).to_numpy()
    feasible_grid["nearest_strategy"] = observed_main_for_nearest.iloc[nearest_position]["strategy"].to_numpy()
    feasible_grid["nearest_strategy_code"] = observed_main_for_nearest.iloc[nearest_position]["strategy_code"].to_numpy()
    candidates, bootstrap_recommendations = bootstrap_optimization_life(
        feasible_grid, battery_table, selection, parameterization, responses,
        samples=optimization_bootstrap_samples,
    )
    candidates, _ = add_candidate_risk_and_curves(candidates)
    pareto_median = pareto_time_life(candidates, "eol_median")
    pareto_median["front_type"] = "median_life"
    pareto_robust = pareto_time_life(candidates, "eol_p10")
    pareto_robust["front_type"] = "p10_robust_life"
    recommendations = representative_time_life(candidates, pareto_median, "eol_median")
    recommendations["robust_p10_pareto_member"] = recommendations.apply(
        lambda row: bool(((pareto_robust[["C1", "Q1", "C2"]] == row[["C1", "Q1", "C2"]].to_numpy()).all(axis=1)).any()), axis=1
    )
    recommendations, recommendation_curves = add_candidate_risk_and_curves(recommendations)
    robustness = robustness_from_bootstrap(bootstrap_recommendations, recommendations)
    domain_fronts, domain_recommendations = _domain_sensitivity(candidates)
    p_fronts, p_recommendations = _p_sensitivity(
        strategy_table, strategy_comparison, responses, parameterization
    )
    loso_recommendations = _leave_one_strategy_sensitivity(
        strategy_table, selection, responses, parameterization, evaluated_grid
    )
    observed_main = strategy_table.loc[strategy_table["main_dataset3"]].copy()
    observed_for_prediction = observed_main if selection["model"] == "raw_parameters" else add_phase_exposures(observed_main, float(selection["p"]))
    evaluated_observed = predict_native_parameters(strategy_model, observed_for_prediction, parameterization)
    charge_model_name = str(metadata["charge_time_model"])
    observed_time = add_physical_quantities(observed_main.reset_index(drop=True))
    charge_model = fit_charge_time_model(observed_time, charge_model_name)
    evaluated_observed["predicted_charge_time"] = charge_model.predict(observed_time)
    nearest_comparison = _nearest_existing_comparison(recommendations, evaluated_observed)
    old_new = _old_new_model_comparison(strategy_comparison, recreation, project_root)
    old_pareto_path = project_root / "outputs/question4/tables/pareto_candidates_main.csv"
    old_pareto = pd.read_csv(old_pareto_path) if old_pareto_path.exists() else pd.DataFrame()
    old_grid_path = project_root / "outputs/question4/tables/optimization_grid_trusted.csv"
    old_grid = pd.read_csv(old_grid_path) if old_grid_path.exists() else pd.DataFrame()
    _write(charge_validation, q4_root / "charge_time_model_loso_validation.csv")
    _write(pd.DataFrame([{k: v for k, v in metadata.items()}]), q4_root / "parameter_trust_domain.csv")
    _write(candidates, q4_root / "all_trusted_optimization_candidates.csv")
    _write(pareto_median, q4_root / "pareto_front_median_life.csv")
    _write(pareto_robust, q4_root / "pareto_front_robust_p10_life.csv")
    _write(recommendations, q4_root / "recommended_strategies.csv")
    _write(recommendation_curves, q4_root / "recommended_strategy_soh_curves.csv")
    _write(bootstrap_recommendations, q4_root / "bootstrap_recommendation_draws.csv")
    _write(robustness, q4_root / "bootstrap_recommendation_robustness.csv")
    _write(domain_fronts, q4_root / "trust_domain_sensitivity_pareto.csv")
    _write(domain_recommendations, q4_root / "trust_domain_sensitivity_recommendations.csv")
    _write(p_fronts, q4_root / "phase_exponent_sensitivity_pareto.csv")
    _write(p_recommendations, q4_root / "phase_exponent_sensitivity_recommendations.csv")
    _write(loso_recommendations, q4_root / "leave_one_strategy_out_recommendations.csv")
    _write(nearest_comparison, q4_root / "recommendation_nearest_existing_comparison.csv")
    _write(old_new, q4_root / "old_stress_vs_unified_model_comparison.csv")

    manifest = {
        "pipeline": "unified_quadratic_v2", "random_seed": RANDOM_SEED,
        "baseline_preserved": True, "parameterization": parameterization,
        "strategy_model": selection.to_dict(), "policy_penalty": penalty,
        "short_horizon_model": final_short_model,
        "battery_bootstrap_samples": battery_bootstrap_samples,
        "strategy_bootstrap_samples": strategy_bootstrap_samples,
        "optimization_bootstrap_samples": optimization_bootstrap_samples,
        "test_eol_bootstrap_samples": test_eol_bootstrap_samples,
        "main_eol_definition": "first n with 1-d1*n/100-d2*(n/100)^2 <= 0.8",
        "a_treatment": "a=1 main optimization; observed a distribution is sensitivity only",
    }
    (output_root / "manifest.json").write_text(
        json.dumps(_json_ready(manifest), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "manifest": manifest, "output_root": output_root, "figure_root": figure_root,
        "cycles": cycles, "battery_parameters": battery_parameters,
        "parameter_draws": parameter_draws, "parameter_bootstrap": parameter_bootstrap,
        "cutoff_stability": cutoff_stability, "identifiability": identifiability,
        "battery_table": battery_table, "strategy_table": strategy_table,
        "strategy_comparison": strategy_comparison, "strategy_loso": strategy_loso,
        "coefficient_summary": coefficient_summary, "recreation": recreation,
        "exposure_coefficient_summary": exposure_coefficient_summary,
        "mechanism": mechanism, "evidence": evidence, "predictor_evidence": predictor_evidence,
        "parameter_matches": parameter_matches,
        "combined_predictions": combined_predictions, "model_comparison": model_comparison,
        "ablation": ablation, "test_predictions": test_predictions, "test_eol": test_eol,
        "test_eol_curves": test_eol_curves, "candidates": candidates,
        "pareto_median": pareto_median, "pareto_robust": pareto_robust,
        "recommendations": recommendations, "recommendation_curves": recommendation_curves,
        "bootstrap_recommendations": bootstrap_recommendations,
        "domain_fronts": domain_fronts, "domain_recommendations": domain_recommendations,
        "p_fronts": p_fronts, "p_recommendations": p_recommendations,
        "loso_recommendations": loso_recommendations, "nearest_comparison": nearest_comparison,
        "charge_validation": charge_validation, "old_new": old_new,
        "old_pareto": old_pareto, "old_grid": old_grid,
    }
