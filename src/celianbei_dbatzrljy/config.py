"""Shared configuration for the question-one analysis."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "question1"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "question1"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = PROJECT_ROOT / "figures" / "question1"
FIGURE_PNG_DIR = FIGURE_DIR / "png"
FIGURE_SVG_DIR = FIGURE_DIR / "svg"

SUMMARY_PATH = RAW_DIR / "battery_summary.csv"
CYCLES_PATH = RAW_DIR / "cycle_train.csv"

EOL_THRESHOLD = 0.80
MAX_EOL_CYCLE = 5000.0
RANDOM_SEED = 20260815
TARGET_CYCLES = (50, 100, 150, 200)
SLOPE_WINDOWS = ((1, 50), (50, 100), (100, 150), (150, 200), (50, 200))
CANDIDATE_MODELS = ("linear", "quadratic", "power", "exponential")
TRUNCATION_CUTOFFS = (100, 150)
