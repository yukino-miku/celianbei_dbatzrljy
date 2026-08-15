"""Reproducible leakage-safe pipeline for question three."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT
from .question3_models import (
    eol_pseudotest_comparison,
    estimate_test_eol,
    forecast_test_batteries,
    forecast_training_pseudotests,
    load_question3_data,
    prediction_metrics,
)
from .question3_plotting import make_question3_plots


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _interval_validation(pseudotest: pd.DataFrame, deviations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = pseudotest.query("model == 'adaptive_ensemble'").copy()
    selected["absolute_error"] = (selected["actual_soh"] - selected["predicted_soh"]).abs()
    rows = []
    for battery_id, group in selected.groupby("battery_id"):
        calibration = selected.loc[selected["battery_id"] != battery_id]
        q95 = calibration.groupby("cycle")["absolute_error"].quantile(.95)
        score = float(deviations.loc[deviations["battery_id"] == battery_id, "deviation_score"].iloc[0])
        inflation = 1.0 + .15 * max(0.0, score - 1.0)
        half_width = group["cycle"].map(q95).to_numpy(float) * inflation
        covered = group["absolute_error"].to_numpy(float) <= half_width
        rows.append({"battery_id": int(battery_id), "strategy": group["strategy"].iloc[0],
                     "strategy_code": group["strategy_code"].iloc[0],
                     "empirical_coverage": covered.mean(), "mean_half_width": half_width.mean(),
                     "max_half_width": half_width.max(), "deviation_score": score})
    battery = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "method": "leave-one-battery-out horizon-wise absolute-error quantile",
        "nominal_coverage": .95, "mean_battery_coverage": battery["empirical_coverage"].mean(),
        "median_battery_coverage": battery["empirical_coverage"].median(),
        "worst_battery_coverage": battery["empirical_coverage"].min(),
        "mean_half_width": battery["mean_half_width"].mean(),
        "note": "test intervals additionally include ensemble-model disagreement",
    }])
    return battery, summary


def _selection_table(model_summary: pd.DataFrame, eol_summary: pd.DataFrame) -> pd.DataFrame:
    adaptive = model_summary.loc[model_summary["model"] == "adaptive_ensemble"].iloc[0]
    best_baseline = model_summary.loc[model_summary["model"] != "adaptive_ensemble"].sort_values("RMSE_mean").iloc[0]
    return pd.DataFrame([{
        "short_horizon_selected_model": "adaptive_ensemble",
        "selected_RMSE_mean": adaptive["RMSE_mean"],
        "best_single_model": best_baseline["model"],
        "best_single_RMSE_mean": best_baseline["RMSE_mean"],
        "relative_RMSE_improvement_vs_best_single": (best_baseline["RMSE_mean"]-adaptive["RMSE_mean"])/best_baseline["RMSE_mean"],
        "selected_eol_scheme": eol_summary.loc[eol_summary["selected"], "scheme"].iloc[0],
        "selection_basis": "nested LOBO errors, worst-battery error, strategy stability, monotone physics; EOL selected by stability against actual-cycle-200 quadratic reference",
        "leakage_guard": "each outer training battery future is excluded from template, supervised fitting, preprocessing, alpha tuning, ensemble weights, and interval calibration",
    }])


def run_question3(
    *, eol_bootstrap_samples: int = 300, make_plots: bool = True,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    output_root = project_root / "outputs/question3"; table_root = output_root / "tables"
    figure_root = project_root / "figures/question3"
    data = load_question3_data(project_root)
    pseudotest, deviations_train, train_weights, predict_supervised = forecast_training_pseudotests(data)
    battery_metrics, model_summary, strategy_metrics = prediction_metrics(pseudotest)
    interval_battery, interval_summary = _interval_validation(pseudotest, deviations_train)
    test_predictions, deviations_test, test_weights, test_components = forecast_test_batteries(
        data, pseudotest, deviations_train, predict_supervised
    )
    eol_validation, eol_summary, selected_eol_scheme = eol_pseudotest_comparison(
        data, pseudotest, deviations_train
    )
    test_eol, eol_curves = estimate_test_eol(
        data, test_predictions, test_components, deviations_test, selected_eol_scheme,
        pseudotest, bootstrap_samples=eol_bootstrap_samples,
    )
    ensemble_weights = pd.concat([train_weights, test_weights], ignore_index=True)
    cycle200 = test_predictions.query("cycle == 200").merge(
        test_eol[["battery_id", "selected_eol_point", "eol_ci95_low", "eol_ci95_high", "reliability"]],
        on="battery_id", validate="one_to_one")
    selection = _selection_table(model_summary, eol_summary)

    tables = {
        "early150_features_all_batteries": data.features,
        "pseudotest_cycle_predictions": pseudotest,
        "pseudotest_battery_errors": battery_metrics,
        "model_comparison": model_summary,
        "strategy_prediction_errors": strategy_metrics,
        "training_deviation_diagnostics": deviations_train,
        "prediction_interval_validation_by_battery": interval_battery,
        "prediction_interval_validation_summary": interval_summary,
        "ensemble_weights": ensemble_weights,
        "test_cycle151_200_predictions": test_predictions,
        "test_component_predictions": test_components,
        "test_cycle200_and_eol_summary": cycle200,
        "test_deviation_diagnostics": deviations_test,
        "eol_pseudotest_by_battery": eol_validation,
        "eol_scheme_comparison": eol_summary,
        "test_eol_estimates": test_eol,
        "test_eol_curves": eol_curves,
        "final_model_selection": selection,
    }
    for name, frame in tables.items(): _write(frame, table_root / f"{name}.csv")

    figure_stems = []
    if make_plots:
        figure_stems = make_question3_plots(
            data.train, data.test, pseudotest, battery_metrics, model_summary,
            strategy_metrics, deviations_train, test_predictions, deviations_test,
            ensemble_weights, eol_validation, test_eol, eol_curves, figure_root,
        )
    elif (figure_root / "png").exists():
        figure_stems = sorted(path.stem for path in (figure_root / "png").glob("q3_*.png"))
    manifest = {
        "question": 3, "training_batteries": 40, "test_batteries": 9,
        "training_input_cycles_in_pseudotest": "1-150", "validated_future_cycles": "151-200",
        "test_input_cycles": "1-150 only", "test_prediction_cycles": "151-200",
        "short_horizon_selected_model": "adaptive_ensemble",
        "selected_eol_scheme": selected_eol_scheme,
        "eol_bootstrap_samples_per_test_battery": eol_bootstrap_samples,
        "supervised_full_training_alpha": float(getattr(predict_supervised, "alpha")),
        "table_files": [f"tables/{name}.csv" for name in tables],
        "figure_stems_png_and_svg": figure_stems,
        "leakage_guards": [
            "no test row beyond cycle 150 exists or is read",
            "each training pseudo-test target is excluded from its strategy template",
            "supervised preprocessing and alpha selection occur inside the outer leave-one-battery-out fold",
            "ensemble pseudo-test weights and interval calibration exclude the target battery",
            "global_id is retained only as existing metadata and never used as a feature or lookup key",
        ],
        "eol_caveat": "80% SOH lifetime is a constrained-model extrapolation, not an observed label",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
