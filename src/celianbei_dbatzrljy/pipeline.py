"""End-to-end reproducible pipeline for question one."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .config import (
    FIGURE_PNG_DIR,
    FIGURE_SVG_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    TABLE_DIR,
)
from .data import clean_cycle_data, load_raw_data
from .features import extract_battery_features
from .models import (
    estimate_training_lifetimes,
    run_truncation_validation,
    summarize_and_select_model,
)
from .plotting import generate_all_figures
from .summaries import build_strategy_summary, make_battery_results, make_strategy_mapping


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")


def run_question1(*, bootstrap_samples: int = 200, make_plots: bool = True) -> dict[str, object]:
    for directory in (PROCESSED_DIR, TABLE_DIR, FIGURE_PNG_DIR, FIGURE_SVG_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    summary, raw_cycles = load_raw_data()
    cleaned = clean_cycle_data(summary, raw_cycles)
    features = extract_battery_features(cleaned.cycles)
    validation, fits, stability = run_truncation_validation(cleaned.cycles)
    model_summary, selected_model = summarize_and_select_model(validation, fits, stability)
    lifetimes = estimate_training_lifetimes(
        cleaned.cycles,
        fits,
        selected_model,
        bootstrap_samples=bootstrap_samples,
    )
    mapping = make_strategy_mapping(summary)
    battery_results = make_battery_results(features, lifetimes, mapping)
    strategy_summary = build_strategy_summary(features, lifetimes, mapping)

    _write_csv(cleaned.cycles, PROCESSED_DIR / "cleaned_cycle_data.csv")
    _write_csv(cleaned.anomalies, TABLE_DIR / "anomaly_records.csv")
    _write_csv(cleaned.audit, TABLE_DIR / "data_validation_checks.csv")
    _write_csv(mapping, TABLE_DIR / "strategy_mapping.csv")
    _write_csv(features, TABLE_DIR / "battery_early_degradation_features.csv")
    _write_csv(validation, TABLE_DIR / "model_truncation_validation.csv")
    _write_csv(fits, TABLE_DIR / "candidate_model_fits.csv")
    _write_csv(stability, TABLE_DIR / "model_lifetime_stability.csv")
    _write_csv(model_summary, TABLE_DIR / "model_selection_summary.csv")
    _write_csv(lifetimes, TABLE_DIR / "battery_lifetime_estimates.csv")
    _write_csv(battery_results, TABLE_DIR / "battery_question1_results.csv")
    _write_csv(strategy_summary, TABLE_DIR / "strategy_question1_summary.csv")

    if make_plots:
        generate_all_figures(
            cleaned.cycles,
            features,
            validation,
            fits,
            stability,
            model_summary,
            battery_results,
            mapping,
            selected_model,
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": 1,
        "selected_model": selected_model,
        "bootstrap_samples_per_battery": bootstrap_samples,
        "training_batteries": int((summary["prediction_test"] == 0).sum()),
        "test_batteries_excluded_from_modeling": int((summary["prediction_test"] == 1).sum()),
        "cleaned_rows": int(len(cleaned.cycles)),
        "anomaly_rows": int(len(cleaned.anomalies)),
        "tables": sorted(path.name for path in TABLE_DIR.glob("*.csv")),
        "png_figures": sorted(path.name for path in FIGURE_PNG_DIR.glob("*.png")),
        "svg_figures": sorted(path.name for path in FIGURE_SVG_DIR.glob("*.svg")),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
