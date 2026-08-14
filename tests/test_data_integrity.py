from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def test_summary_integrity() -> None:
    summary = pd.read_csv(RAW_DIR / "battery_summary.csv")

    assert summary.shape == (49, 13)
    assert summary["battery_id"].is_unique
    assert summary["policy"].nunique() == 9
    assert summary["prediction_test"].sum() == 9
    assert summary.drop(columns="C1").isna().sum().sum() == 0
    assert summary["C1"].isna().sum() == 3


def test_cycle_integrity_and_censoring() -> None:
    summary = pd.read_csv(RAW_DIR / "battery_summary.csv")
    cycles = pd.read_csv(RAW_DIR / "cycle_train.csv")

    assert cycles.shape == (9350, 9)
    assert not cycles.duplicated(subset=["battery_id", "cycle"]).any()
    assert cycles.isna().sum().sum() == 0

    lengths = cycles.groupby("battery_id")["cycle"].max()
    joined = summary.join(lengths.rename("max_cycle"), on="battery_id")
    assert (joined.loc[joined["prediction_test"] == 0, "max_cycle"] == 200).all()
    assert (joined.loc[joined["prediction_test"] == 1, "max_cycle"] == 150).all()


def test_policy_labels_match_between_files() -> None:
    summary = pd.read_csv(RAW_DIR / "battery_summary.csv")
    cycles = pd.read_csv(RAW_DIR / "cycle_train.csv")

    joined = cycles.merge(
        summary[["battery_id", "policy"]],
        on="battery_id",
        suffixes=("_cycle", "_summary"),
        validate="many_to_one",
    )
    assert (joined["policy_cycle"] == joined["policy_summary"]).all()
