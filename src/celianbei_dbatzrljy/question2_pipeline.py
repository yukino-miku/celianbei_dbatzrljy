"""Reproducible question-two analysis pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT
from .question2_plotting import make_question2_plots
from .question2_stats import (
    PRIMARY_RESPONSES,
    add_stress_index,
    collinearity_diagnostics,
    conclusion_matrix,
    grouped_bootstrap_coefficients,
    load_question2_data,
    pairwise_strategy_tests,
    parameter_model_comparison,
    parameter_ols_results,
    ridge_parameter_results,
    s8_diagnostic,
    strategy_difference_tests,
    stress_models,
    stress_p_search,
    univariate_associations,
)


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _strategy_summary(batteries: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260815 + 400)
    rows = []
    for code, group in batteries.groupby("strategy_code", sort=False):
        row: dict[str, object] = {
            "strategy_code": code,
            "strategy": group["strategy"].iloc[0],
            "dataset_id": int(group["dataset_id"].iloc[0]),
            "n_training_batteries": len(group),
            "C1": group["C1"].iloc[0], "Q1": group["Q1"].iloc[0], "C2": group["C2"].iloc[0],
        }
        for response in PRIMARY_RESPONSES:
            values = group[response].dropna().to_numpy(float)
            draws = rng.choice(values, size=(5_000, len(values)), replace=True)
            bootstrap_means = draws.mean(axis=1)
            bootstrap_medians = np.median(draws, axis=1)
            row.update({
                f"{response}_mean": np.mean(values), f"{response}_median": np.median(values),
                f"{response}_std": np.std(values, ddof=1) if len(values) > 1 else np.nan,
                f"{response}_q1": np.quantile(values, .25), f"{response}_q3": np.quantile(values, .75),
                f"{response}_min": np.min(values), f"{response}_max": np.max(values),
                f"{response}_mean_bootstrap_ci95_low": np.quantile(bootstrap_means, .025),
                f"{response}_mean_bootstrap_ci95_high": np.quantile(bootstrap_means, .975),
                f"{response}_median_bootstrap_ci95_low": np.quantile(bootstrap_medians, .025),
                f"{response}_median_bootstrap_ci95_high": np.quantile(bootstrap_medians, .975),
            })
        rows.append(row)
    return pd.DataFrame(rows).sort_values("strategy_code", key=lambda s: s.str[1:].astype(int))


def _batch_sensitivity(parameter_ols: pd.DataFrame, stress: pd.DataFrame) -> pd.DataFrame:
    parameter = parameter_ols.query("coefficient in ['C1','Q1','C2']").rename(
        columns={"coefficient": "predictor", "standardized_beta": "beta"}
    )[["scope", "response", "predictor", "n_strategies", "beta", "condition_number"]]
    parameter["model_family"] = "joint_parameter_OLS"
    stress_part = stress.rename(columns={"stress_standardized_beta": "beta"})[
        ["scope", "response", "n_strategies", "beta", "condition_number"]
    ]
    stress_part["predictor"] = "soc_weighted_stress"
    stress_part["model_family"] = "stress_index"
    return pd.concat([parameter, stress_part], ignore_index=True, sort=False)[
        ["model_family", "scope", "response", "predictor", "n_strategies", "beta", "condition_number"]
    ]


def _hypothesis_checks(associations: pd.DataFrame, s8: pd.DataFrame) -> pd.DataFrame:
    def metrics(predictor: str) -> str:
        part = associations.query("predictor == @predictor and response in @PRIMARY_RESPONSES")
        return "; ".join(
            f"{r.response}:rho={r.spearman_rho:.3f},p={r.exact_permutation_p:.3f}"
            for r in part.itertuples()
        )
    s8_row = s8.loc[s8["strategy_code"] == "S8"].iloc[0]
    return pd.DataFrame([
        {"hypothesis": "Higher C-rate necessarily means faster degradation", "evidence": metrics("C1") + " | " + metrics("C2"),
         "assessment_rule": "requires consistent observed SOH200/slope and EOL directions", "interpretation": "observational; inspect cross-metric consistency"},
        {"hypothesis": "High C2 exposure in middle/high SOC is more adverse than low-SOC exposure", "evidence": metrics("phase1_exposure") + " | " + metrics("phase2_exposure"),
         "assessment_rule": "compare phase-exposure association directions and stability", "interpretation": "trend only because phase exposures and batch are confounded"},
        {"hypothesis": "Q1 reflects duration allocation between charging phases", "evidence": metrics("Q1"),
         "assessment_rule": "interpret Q1 jointly with C1/C2, never as an isolated causal effect", "interpretation": "design variable is collinear with rates"},
        {"hypothesis": "S8 rapid degradation is explained by SOC-weighted stress", "evidence": f"stress_rank={s8_row['rank_soc_weighted_stress']:.0f}; SOH200_worst_rank={s8_row['rank_SOH_200_median']:.0f}; slope_worst_rank={s8_row['rank_SOH_slope_50_200_per100_median']:.0f}; EOL_worst_rank={s8_row['rank_eol_point_median']:.0f}",
         "assessment_rule": "stress rank and all outcome ranks should align", "interpretation": "descriptive explanation, not causal identification"},
    ])


def run_question2(
    *, permutation_samples: int = 19_999, bootstrap_samples: int = 2_000,
    grouped_bootstrap_samples: int = 1_000, make_plots: bool = True,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    output_root = project_root / "outputs" / "question2"
    table_root = output_root / "tables"
    figure_root = project_root / "figures" / "question2"
    data = load_question2_data(project_root)

    global_tests = strategy_difference_tests(data.batteries, permutation_samples=permutation_samples)
    pairwise = pairwise_strategy_tests(data.batteries, global_tests, bootstrap_samples=bootstrap_samples)
    collinearity, correlations = collinearity_diagnostics(data.strategies)
    parameter_ols = parameter_ols_results(data.strategies)
    ridge = ridge_parameter_results(data.strategies)
    stress_search, selected_p = stress_p_search(data.strategies)
    strategies = add_stress_index(data.strategies, selected_p)
    associations = univariate_associations(strategies, bootstrap_samples=bootstrap_samples)
    stress = stress_models(strategies)
    model_comparison = parameter_model_comparison(strategies)
    boot_distribution, boot_summary = grouped_bootstrap_coefficients(
        data.batteries, samples=grouped_bootstrap_samples
    )
    conclusions = conclusion_matrix(associations)
    s8 = s8_diagnostic(strategies)
    batch = _batch_sensitivity(parameter_ols, stress)
    hypotheses = _hypothesis_checks(associations, s8)

    tables = {
        "battery_analysis_data": data.batteries,
        "strategy_analysis_data": strategies,
        "strategy_robust_summary": _strategy_summary(data.batteries),
        "strategy_difference_tests": global_tests,
        "pairwise_strategy_tests": pairwise,
        "collinearity_diagnostics": collinearity,
        "parameter_correlations": correlations,
        "parameter_ols_results": parameter_ols,
        "ridge_parameter_results": ridge,
        "stress_p_search": stress_search,
        "stress_model_results": stress,
        "parameter_model_comparison": model_comparison,
        "grouped_bootstrap_coefficients": boot_distribution,
        "grouped_bootstrap_summary": boot_summary,
        "univariate_associations": associations,
        "cross_metric_conclusion_matrix": conclusions,
        "batch_sensitivity_coefficients": batch,
        "s8_diagnostic": s8,
        "hypothesis_checks": hypotheses,
    }
    for name, frame in tables.items():
        _write(frame, table_root / f"{name}.csv")

    plot_inputs = {
        "pairwise": pairwise, "collinearity": collinearity,
        "parameter_correlations": correlations, "stress_p_search": stress_search,
        "stress_models": stress, "bootstrap_summary": boot_summary,
        "model_comparison": model_comparison, "s8_diagnostic": s8,
        "parameter_ols": parameter_ols,
    }
    if make_plots:
        figure_stems = make_question2_plots(data.batteries, strategies, plot_inputs, figure_root)
    else:
        figure_stems = sorted(path.stem for path in (figure_root / "png").glob("q2_*.png"))
    manifest = {
        "question": 2,
        "source": "question-one validated outputs; no cleaning logic rerun or altered",
        "n_training_batteries": len(data.batteries),
        "n_strategies": len(strategies),
        "n_parameterized_strategies": int(strategies["C1"].notna().sum()),
        "excluded_from_parameter_regression": ["S2"],
        "selected_shared_stress_exponent_p": selected_p,
        "permutation_samples": permutation_samples,
        "association_bootstrap_samples": bootstrap_samples,
        "grouped_bootstrap_samples": grouped_bootstrap_samples,
        "table_files": [f"tables/{name}.csv" for name in tables],
        "figure_stems_png_and_svg": figure_stems,
        "limitations": [
            "strategy is partly confounded with dataset_id/NEWSTRUCTURE",
            "only 8 parameterized strategy combinations and 6 within dataset 3",
            "quadratic EOL is a model-based extrapolation and is triangulated with SOH200 and slope",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
