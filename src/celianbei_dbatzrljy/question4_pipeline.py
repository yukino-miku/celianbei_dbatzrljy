"""Reproducible multi-objective optimization pipeline for question four."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import PROJECT_ROOT
from .question4_models import (
    CHARGE_MODELS,
    PARAMETERS,
    P_VALUES,
    RESPONSE_COLUMNS,
    add_surrogate_predictions,
    add_trust_domain_flags,
    bootstrap_recommendations,
    charge_time_cross_validation,
    fit_charge_time_model,
    fit_main_models,
    fit_stress_response_model,
    generate_parameter_grid,
    load_question4_data,
    parameterized_scope,
    pareto_front,
    proxy_cross_validation,
    select_charge_time_model,
    select_main_stress_p,
    select_representative_points,
)
from .question4_plotting import make_question4_plots


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _charge_loso_predictions(strategies: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("dataset3_only", "all_parameterized"):
        data = parameterized_scope(strategies, scope)
        for model in CHARGE_MODELS:
            for index in range(len(data)):
                train = data.drop(index=index)
                test = data.iloc[[index]]
                prediction = float(fit_charge_time_model(train, model).predict(test)[0])
                rows.append(
                    {
                        "scope": scope,
                        "model": model,
                        "strategy_code": test["strategy_code"].iloc[0],
                        "actual_charge_time": test["mean_chargetime"].iloc[0],
                        "predicted_charge_time": prediction,
                        "error_minutes": prediction - test["mean_chargetime"].iloc[0],
                    }
                )
    return pd.DataFrame(rows)


def _model_coefficients(charge_model, response_models: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, value in enumerate(charge_model.coefficients):
        rows.append(
            {
                "model_role": "charge_time",
                "response": "mean_chargetime",
                "model": charge_model.name,
                "transform": "physical_base_plus_correction",
                "term": f"correction_beta_{index}",
                "coefficient": value,
                "residual_sigma": charge_model.residual_sigma,
                "p": np.nan,
            }
        )
    if len(charge_model.coefficients) == 0:
        rows.append(
            {
                "model_role": "charge_time",
                "response": "mean_chargetime",
                "model": charge_model.name,
                "transform": "physical_formula",
                "term": "no_fitted_coefficient",
                "coefficient": 0.0,
                "residual_sigma": charge_model.residual_sigma,
                "p": np.nan,
            }
        )
    for label, model in response_models.items():
        for term, value in (("intercept", model.intercept), ("stress", model.slope)):
            rows.append(
                {
                    "model_role": "degradation_proxy",
                    "response": label,
                    "model": "stress_loglinear",
                    "transform": model.transform,
                    "term": term,
                    "coefficient": value,
                    "residual_sigma": model.residual_sigma,
                    "p": model.p,
                }
            )
    return pd.DataFrame(rows)


def _domain_sensitivity(evaluated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    masks = {
        "rectangle": evaluated["inside_rectangle"],
        "convex_hull": evaluated["inside_convex_hull"],
        "trust_region": evaluated["inside_trust_region"],
        "convex_hull_and_trust": evaluated["inside_main_domain"],
    }
    recommendations = []
    fronts = []
    for domain, mask in masks.items():
        feasible = evaluated.loc[mask].copy()
        front = pareto_front(feasible)
        front["domain"] = domain
        fronts.append(front)
        selected = select_representative_points(feasible, front)
        selected["domain"] = domain
        selected["n_feasible_points"] = len(feasible)
        selected["n_pareto_points"] = len(front)
        recommendations.append(selected)
    return pd.concat(recommendations, ignore_index=True), pd.concat(fronts, ignore_index=True)


def _p_sensitivity(
    observed: pd.DataFrame,
    feasible_grid: pd.DataFrame,
    charge_model,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    recommendations = []
    fronts = []
    for p in P_VALUES:
        responses = {
            label: fit_stress_response_model(observed, response, p)
            for label, response in RESPONSE_COLUMNS.items()
        }
        evaluated = add_surrogate_predictions(feasible_grid, charge_model, responses)
        front = pareto_front(evaluated)
        front["sensitivity_p"] = p
        fronts.append(front)
        selected = select_representative_points(evaluated, front)
        selected["sensitivity_p"] = p
        recommendations.append(selected)
    return pd.concat(recommendations, ignore_index=True), pd.concat(fronts, ignore_index=True)


def _response_sensitivity(evaluated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    definitions = {
        "slope50_200": "predicted_degradation_rate",
        "SOH200": "SOH200_harm_predicted",
        "EOL": "EOL_harm_predicted",
    }
    work = evaluated.copy()
    work["SOH200_harm_predicted"] = 1.0 - work["predicted_SOH200"]
    work["EOL_harm_predicted"] = -work["predicted_EOL"]
    recommendations = []
    fronts = []
    for response, column in definitions.items():
        front = pareto_front(work, column)
        front["optimization_response"] = response
        front["optimization_harm"] = front[column]
        fronts.append(front)
        selected = select_representative_points(work, front, column)
        selected["optimization_response"] = response
        selected["optimization_harm"] = selected[column]
        recommendations.append(selected)
    return pd.concat(recommendations, ignore_index=True), pd.concat(fronts, ignore_index=True)


def _scope_sensitivity(
    strategies: pd.DataFrame,
    charge_validation: pd.DataFrame,
    feasible_grid: pd.DataFrame,
    p: float,
) -> pd.DataFrame:
    rows = []
    selected_charge = select_charge_time_model(charge_validation)
    for scope in ("dataset3_only", "all_parameterized"):
        data = parameterized_scope(strategies, scope)
        charge = fit_charge_time_model(data, selected_charge)
        responses = {
            label: fit_stress_response_model(data, response, p)
            for label, response in RESPONSE_COLUMNS.items()
        }
        evaluated = add_surrogate_predictions(feasible_grid, charge, responses)
        selected = select_representative_points(evaluated, pareto_front(evaluated))
        selected["model_scope"] = scope
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def _loso_recommendations(
    observed: pd.DataFrame,
    feasible_grid: pd.DataFrame,
    charge_model_name: str,
    p: float,
) -> pd.DataFrame:
    rows = []
    for omitted in observed["strategy_code"]:
        sample = observed.query("strategy_code != @omitted")
        charge = fit_charge_time_model(sample, charge_model_name)
        responses = {
            label: fit_stress_response_model(sample, response, p)
            for label, response in RESPONSE_COLUMNS.items()
        }
        evaluated = add_surrogate_predictions(feasible_grid, charge, responses)
        selected = select_representative_points(evaluated, pareto_front(evaluated))
        selected["omitted_strategy"] = omitted
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def _weighted_recommendations(evaluated: pd.DataFrame) -> pd.DataFrame:
    result = []
    t = evaluated["predicted_charge_time"]
    d = evaluated["predicted_degradation_rate"]
    t_norm = (t - t.min()) / max(t.max() - t.min(), 1e-12)
    d_norm = (d - d.min()) / max(d.max() - d.min(), 1e-12)
    for label, time_weight, degradation_weight in (
        ("fast_preference", 0.7, 0.3),
        ("balanced", 0.5, 0.5),
        ("longevity_preference", 0.3, 0.7),
    ):
        score = time_weight * t_norm + degradation_weight * d_norm
        row = evaluated.loc[score.idxmin()].copy()
        row["weight_scenario"] = label
        row["time_weight"] = time_weight
        row["degradation_weight"] = degradation_weight
        row["normalized_objective_J"] = score.loc[score.idxmin()]
        result.append(row)
    return pd.DataFrame(result)


def _nearest_strategy_comparison(
    recommendations: pd.DataFrame,
    observed: pd.DataFrame,
    observed_predictions: pd.DataFrame,
) -> pd.DataFrame:
    lower = observed[PARAMETERS].min().to_numpy(float)
    span = (observed[PARAMETERS].max() - observed[PARAMETERS].min()).to_numpy(float)
    observed_scaled = (observed[PARAMETERS].to_numpy(float) - lower) / span
    tree = cKDTree(observed_scaled)
    rows = []
    for recommendation in recommendations.itertuples(index=False):
        x = np.array([recommendation.C1, recommendation.Q1, recommendation.C2])
        distance, index = tree.query((x - lower) / span)
        baseline = observed_predictions.iloc[int(index)]
        combined_eol_se = np.sqrt(recommendation.predicted_EOL_se**2 + baseline.predicted_EOL_se**2)
        delta_eol = recommendation.predicted_EOL - baseline.predicted_EOL
        rows.append(
            {
                "recommendation_type": recommendation.recommendation_type,
                "C1": recommendation.C1,
                "Q1": recommendation.Q1,
                "C2": recommendation.C2,
                "nearest_strategy_code": baseline.strategy_code,
                "nearest_distance_normalized": distance,
                "delta_charge_time_vs_nearest_actual_minutes": recommendation.predicted_charge_time
                - baseline.mean_chargetime,
                "delta_charge_time_proxy_minutes": recommendation.predicted_charge_time
                - baseline.predicted_charge_time,
                "delta_SOH200_proxy": recommendation.predicted_SOH200 - baseline.predicted_SOH200,
                "delta_degradation_rate_proxy": recommendation.predicted_degradation_rate
                - baseline.predicted_degradation_rate,
                "delta_EOL_proxy_cycles": delta_eol,
                "combined_EOL_model_se": combined_eol_se,
                "EOL_difference_interpretation": "predicted trend within model uncertainty"
                if abs(delta_eol) <= 1.96 * combined_eol_se
                else "model-predicted difference exceeds local coefficient uncertainty",
                "experimental_status": "surrogate recommendation; not experimentally validated",
            }
        )
    return pd.DataFrame(rows)


def _all_existing_strategy_comparison(
    recommendations: pd.DataFrame,
    observed_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for recommendation in recommendations.itertuples(index=False):
        for baseline in observed_predictions.itertuples(index=False):
            rows.append(
                {
                    "recommendation_type": recommendation.recommendation_type,
                    "existing_strategy_code": baseline.strategy_code,
                    "existing_strategy": baseline.strategy,
                    "delta_time_vs_observed_mean_minutes": recommendation.predicted_charge_time
                    - baseline.mean_chargetime,
                    "delta_SOH200_vs_observed_median": recommendation.predicted_SOH200
                    - baseline.SOH_200_median,
                    "delta_degradation_rate_vs_observed_median": recommendation.predicted_degradation_rate
                    + baseline.SOH_slope_50_200_per100_median,
                    "delta_EOL_vs_existing_quadratic_median_cycles": recommendation.predicted_EOL
                    - baseline.eol_point_median,
                    "delta_time_proxy_to_proxy_minutes": recommendation.predicted_charge_time
                    - baseline.predicted_charge_time,
                    "delta_SOH200_proxy_to_proxy": recommendation.predicted_SOH200 - baseline.predicted_SOH200,
                    "delta_degradation_rate_proxy_to_proxy": recommendation.predicted_degradation_rate
                    - baseline.predicted_degradation_rate,
                    "delta_EOL_proxy_to_proxy_cycles": recommendation.predicted_EOL - baseline.predicted_EOL,
                    "comparison_status": "candidate is surrogate-predicted; existing strategy metrics are observed/model summaries",
                }
            )
    return pd.DataFrame(rows)


def _marginal_benefit(front: pd.DataFrame, recommendations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = front.sort_values("predicted_charge_time").copy()
    ordered["delta_time_from_previous"] = ordered["predicted_charge_time"].diff()
    ordered["degradation_reduction_from_previous"] = -ordered["predicted_degradation_rate"].diff()
    ordered["EOL_gain_from_previous"] = ordered["predicted_EOL"].diff()
    ordered["degradation_reduction_per_added_minute"] = (
        ordered["degradation_reduction_from_previous"] / ordered["delta_time_from_previous"]
    )
    ordered["EOL_gain_per_added_minute"] = ordered["EOL_gain_from_previous"] / ordered["delta_time_from_previous"]
    selected = recommendations.set_index("recommendation_type")
    intervals = []
    for start, end in (("fast_charge", "knee"), ("knee", "longevity"), ("fast_charge", "longevity")):
        a, b = selected.loc[start], selected.loc[end]
        delta_t = b.predicted_charge_time - a.predicted_charge_time
        intervals.append(
            {
                "move": f"{start}_to_{end}",
                "added_charge_time_minutes": delta_t,
                "degradation_rate_reduction": a.predicted_degradation_rate - b.predicted_degradation_rate,
                "predicted_EOL_gain_cycles": b.predicted_EOL - a.predicted_EOL,
                "degradation_reduction_per_added_minute": (a.predicted_degradation_rate - b.predicted_degradation_rate)
                / max(delta_t, 1e-12),
                "predicted_EOL_gain_per_added_minute": (b.predicted_EOL - a.predicted_EOL) / max(delta_t, 1e-12),
            }
        )
    return ordered, pd.DataFrame(intervals)


def _bootstrap_summary(bootstrap: pd.DataFrame, reference: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    lower = observed[PARAMETERS].min().to_numpy(float)
    span = (observed[PARAMETERS].max() - observed[PARAMETERS].min()).to_numpy(float)
    rows = []
    for label, group in bootstrap.groupby("recommendation_type"):
        reference_row = reference.query("recommendation_type == @label").iloc[0]
        distances = np.sqrt(
            np.sum(((group[PARAMETERS].to_numpy(float) - reference_row[PARAMETERS].to_numpy(float)) / span) ** 2, axis=1)
        )
        median_distance = float(np.median(distances))
        grade = "stable" if median_distance <= 0.15 else "variable" if median_distance <= 0.30 else "unstable"
        row: dict[str, object] = {
            "recommendation_type": label,
            "bootstrap_samples": group["bootstrap_iteration"].nunique(),
            "median_normalized_parameter_distance_from_main": median_distance,
            "within_radius_0_20_fraction": float(np.mean(distances <= 0.20)),
            "stability_grade": grade,
        }
        for column in PARAMETERS + ["predicted_charge_time", "predicted_degradation_rate", "predicted_EOL"]:
            row[f"{column}_median"] = group[column].median()
            row[f"{column}_ci95_low"] = group[column].quantile(0.025)
            row[f"{column}_ci95_high"] = group[column].quantile(0.975)
        rows.append(row)
    return pd.DataFrame(rows)


def _robustness_assessment(
    reference: pd.DataFrame,
    observed: pd.DataFrame,
    p_recommendations: pd.DataFrame,
    response_recommendations: pd.DataFrame,
    scope_recommendations: pd.DataFrame,
    loso_recommendations: pd.DataFrame,
    domain_recommendations: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize coordinate drift; grades are capped because no candidate is experimentally tested."""
    lower = observed[PARAMETERS].min().to_numpy(float)
    span = (observed[PARAMETERS].max() - observed[PARAMETERS].min()).to_numpy(float)
    relevant_domains = domain_recommendations.query("domain in ['convex_hull','convex_hull_and_trust']")
    pools = pd.concat(
        [p_recommendations, response_recommendations, scope_recommendations, loso_recommendations, relevant_domains],
        ignore_index=True,
        sort=False,
    )
    rows = []
    for label, group in pools.groupby("recommendation_type"):
        target = reference.query("recommendation_type == @label").iloc[0]
        distances = np.sqrt(
            np.sum(((group[PARAMETERS].to_numpy(float) - target[PARAMETERS].to_numpy(float)) / span) ** 2, axis=1)
        )
        maximum = float(np.max(distances))
        coordinate_stability = "low" if maximum > 0.35 else "moderate" if maximum > 0.15 else "high"
        overall = "low" if coordinate_stability == "low" else "moderate"
        rows.append(
            {
                "recommendation_type": label,
                "n_sensitivity_recommendations": len(group),
                "median_normalized_parameter_drift": float(np.median(distances)),
                "p90_normalized_parameter_drift": float(np.quantile(distances, 0.90)),
                "maximum_normalized_parameter_drift": maximum,
                "coordinate_stability": coordinate_stability,
                "overall_recommendation_confidence": overall,
                "confidence_cap_reason": "model-derived only; six NEWSTRUCTURE strategy points; no experimental validation",
            }
        )
    return pd.DataFrame(rows)


def run_question4(
    *,
    bootstrap_samples: int = 300,
    make_plots: bool = True,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    strategies, batteries = load_question4_data(project_root)
    observed = parameterized_scope(strategies, "dataset3_only")
    output_root = project_root / "outputs" / "question4"
    table_root = output_root / "tables"
    figure_root = project_root / "figures" / "question4"

    charge_validation = charge_time_cross_validation(strategies)
    charge_loso = _charge_loso_predictions(strategies)
    proxy_validation = proxy_cross_validation(strategies)
    selected_p, p_selection = select_main_stress_p(proxy_validation)
    charge_model, response_models = fit_main_models(strategies, charge_validation, selected_p)

    grid, domain_metadata = add_trust_domain_flags(generate_parameter_grid(strategies), observed)
    evaluated_grid = add_surrogate_predictions(grid, charge_model, response_models)
    feasible = evaluated_grid.query("inside_main_domain").copy()
    main_front = pareto_front(feasible)
    main_recommendations = select_representative_points(feasible, main_front)
    main_recommendations["predicted_charge_time_ci95_low"] = (
        main_recommendations["predicted_charge_time"] - 1.96 * main_recommendations["predicted_charge_time_se"]
    )
    main_recommendations["predicted_charge_time_ci95_high"] = (
        main_recommendations["predicted_charge_time"] + 1.96 * main_recommendations["predicted_charge_time_se"]
    )
    main_recommendations["predicted_SOH200_ci95_low"] = (
        main_recommendations["predicted_SOH200"] - 1.96 * main_recommendations["predicted_SOH200_se"]
    ).clip(lower=0.0)
    main_recommendations["predicted_SOH200_ci95_high"] = (
        main_recommendations["predicted_SOH200"] + 1.96 * main_recommendations["predicted_SOH200_se"]
    ).clip(upper=1.0)
    main_recommendations["predicted_degradation_rate_ci95_low"] = (
        main_recommendations["predicted_degradation_rate"]
        - 1.96 * main_recommendations["predicted_degradation_rate_se"]
    ).clip(lower=0.0)
    main_recommendations["predicted_degradation_rate_ci95_high"] = (
        main_recommendations["predicted_degradation_rate"]
        + 1.96 * main_recommendations["predicted_degradation_rate_se"]
    )
    main_recommendations["predicted_EOL_ci95_low"] = (
        main_recommendations["predicted_EOL"] - 1.96 * main_recommendations["predicted_EOL_se"]
    ).clip(lower=0.0)
    main_recommendations["predicted_EOL_ci95_high"] = (
        main_recommendations["predicted_EOL"] + 1.96 * main_recommendations["predicted_EOL_se"]
    )
    main_recommendations["recommendation_scope"] = "dataset3 convex-hull and trust-region"
    main_recommendations["experimental_status"] = "surrogate recommendation; not experimentally validated"

    domain_recommendations, domain_fronts = _domain_sensitivity(evaluated_grid)
    p_recommendations, p_fronts = _p_sensitivity(observed, feasible[grid.columns], charge_model)
    response_recommendations, response_fronts = _response_sensitivity(feasible)
    scope_recommendations = _scope_sensitivity(strategies, charge_validation, feasible[grid.columns], selected_p)
    loso_recommendations = _loso_recommendations(
        observed, feasible[grid.columns], charge_model.name, selected_p
    )
    robustness = _robustness_assessment(
        main_recommendations,
        observed,
        p_recommendations,
        response_recommendations,
        scope_recommendations,
        loso_recommendations,
        domain_recommendations,
    )
    main_recommendations = main_recommendations.merge(robustness, on="recommendation_type", how="left")
    bootstrap = bootstrap_recommendations(
        observed,
        feasible[grid.columns],
        charge_model.name,
        selected_p,
        samples=bootstrap_samples,
    )
    bootstrap_summary = _bootstrap_summary(bootstrap, main_recommendations, observed)
    weighted = _weighted_recommendations(feasible)

    observed_predictions = add_surrogate_predictions(observed, charge_model, response_models)
    comparison = _nearest_strategy_comparison(main_recommendations, observed, observed_predictions)
    all_existing_comparison = _all_existing_strategy_comparison(main_recommendations, observed_predictions)
    marginal_curve, marginal_summary = _marginal_benefit(main_front, main_recommendations)

    physical_audit = strategies[
        [
            "strategy_code", "strategy", "dataset_id", "C1", "Q1", "C2", "mean_chargetime",
            "phase1_ideal_minutes", "phase2_ideal_minutes", "T_ideal_physical",
            "T_ideal_as_written_minus", "actual_minus_physical_minutes",
        ]
    ]
    model_selection = pd.DataFrame(
        [
            {
                "model_role": "charge_time",
                "selected_model": charge_model.name,
                "selected_p": np.nan,
                "selection_basis": "physical plus empirical offset: within 5% of best dataset-3 LOSO RMSE and removes mean residual bias",
                "primary_use": "Pareto time objective",
            },
            {
                "model_role": "degradation_rate",
                "selected_model": "monotone stress log-linear",
                "selected_p": selected_p,
                "selection_basis": "dataset-3 LOSO across SOH200, slope and EOL; log scale enforces positive degradation",
                "primary_use": "main Pareto degradation objective",
            },
            {
                "model_role": "SOH200",
                "selected_model": "monotone stress log-linear",
                "selected_p": selected_p,
                "selection_basis": "observed-response cross-check, not the sole objective",
                "primary_use": "auxiliary Pareto annotation",
            },
            {
                "model_role": "EOL",
                "selected_model": "monotone stress log-linear",
                "selected_p": selected_p,
                "selection_basis": "conditional quadratic EOL is retained only as uncertainty-sensitive annotation",
                "primary_use": "auxiliary Pareto annotation",
            },
        ]
    )
    parameter_domain = pd.DataFrame([domain_metadata]).assign(
        main_scope="dataset3 / NEWSTRUCTURE S4-S9",
        excluded_strategy="S2 (C1 missing)",
        main_domain="convex hull AND normalized nearest-neighbour trust region",
        grid_resolution="C1=0.05 C; Q1=1%SOC; C2=0.05 C",
    )

    tables = {
        "strategy_optimization_inputs": strategies,
        "charge_time_physical_audit": physical_audit,
        "charge_time_model_validation": charge_validation,
        "charge_time_loso_predictions": charge_loso,
        "degradation_proxy_validation": proxy_validation,
        "stress_exponent_selection": p_selection,
        "selected_surrogate_models": model_selection,
        "surrogate_model_coefficients": _model_coefficients(charge_model, response_models),
        "parameter_trust_domain": parameter_domain,
        "optimization_grid_trusted": feasible,
        "pareto_candidates_main": main_front,
        "recommended_strategies": main_recommendations,
        "recommendation_nearest_strategy_comparison": comparison,
        "recommendation_existing_strategy_comparison_all": all_existing_comparison,
        "domain_sensitivity_recommendations": domain_recommendations,
        "domain_sensitivity_pareto": domain_fronts,
        "stress_p_sensitivity_recommendations": p_recommendations,
        "stress_p_sensitivity_pareto": p_fronts,
        "response_sensitivity_recommendations": response_recommendations,
        "response_sensitivity_pareto": response_fronts,
        "dataset_scope_sensitivity": scope_recommendations,
        "leave_one_strategy_out_recommendations": loso_recommendations,
        "bootstrap_recommendations": bootstrap,
        "bootstrap_recommendation_summary": bootstrap_summary,
        "recommendation_robustness_assessment": robustness,
        "weighted_objective_sensitivity": weighted,
        "observed_strategy_surrogate_predictions": observed_predictions,
        "pareto_marginal_benefit_curve": marginal_curve,
        "representative_marginal_benefit": marginal_summary,
    }
    for name, frame in tables.items():
        _write(frame, table_root / f"{name}.csv")

    plot_inputs = {
        **tables,
        "all_strategy_inputs": strategies,
        "evaluated_grid": evaluated_grid,
    }
    if make_plots:
        figure_stems = make_question4_plots(plot_inputs, figure_root)
    else:
        figure_stems = sorted(path.stem for path in (figure_root / "png").glob("q4_*.png"))

    manifest = {
        "question": 4,
        "source": "question-one to question-three validated outputs; cleaning logic not rerun or modified",
        "optimization_scope": "dataset3 / NEWSTRUCTURE S4-S9",
        "continuous_parameter_exclusion": ["S2: C1 missing"],
        "selected_charge_time_model": charge_model.name,
        "selected_stress_exponent_p": selected_p,
        "main_degradation_objective": "positive 50-200 degradation rate from monotone stress log-linear proxy",
        "main_domain": "dataset3 observed convex hull intersected with normalized nearest-neighbour trust region",
        "grid_resolution": {"C1": 0.05, "Q1": 1.0, "C2": 0.05},
        "n_main_feasible_points": len(feasible),
        "n_main_pareto_points": len(main_front),
        "bootstrap_samples": bootstrap_samples,
        "recommendations_are_experimentally_validated": False,
        "table_files": [f"tables/{name}.csv" for name in tables],
        "figure_stems_png_and_svg": figure_stems,
        "limitations": [
            "only six parameterized NEWSTRUCTURE strategies support the main continuous response surface",
            "S9 has approximately one minute of charge-time residual unexplained by C1/Q1/C2",
            "S8 is isolated and influential for the stress-degradation relationship",
            "EOL is a constrained-quadratic extrapolation without observed full-lifetime labels",
            "recommended points are model candidates and require experimental validation",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
