import numpy as np

from celianbei_dbatzrljy.data import clean_cycle_data, load_raw_data
from celianbei_dbatzrljy.features import extract_battery_features


def test_conservative_cleaning_flags_known_anomalies() -> None:
    summary, raw = load_raw_data()
    raw_copy = raw.copy(deep=True)
    cleaned = clean_cycle_data(summary, raw)

    spike = cleaned.cycles.loc[
        (cleaned.cycles["battery_id"] == 1) & (cleaned.cycles["cycle"] == 12)
    ].iloc[0]
    assert spike["flag_soh_capacity_outlier"]
    assert spike["SOH_raw"] > 1.4
    assert 0.99 < spike["SOH_clean"] < 1.02

    for battery_id in (2, 3):
        zero_ir = cleaned.cycles.loc[
            (cleaned.cycles["battery_id"] == battery_id)
            & (cleaned.cycles["cycle"] == 12)
        ].iloc[0]
        assert zero_ir["flag_ir_invalid"]
        assert zero_ir["IR_raw"] == 0
        assert zero_ir["IR_clean"] > 0

    assert raw.equals(raw_copy), "raw data must remain untouched"
    assert not cleaned.cycles[["SOH_clean", "SOH_smooth_robust", "IR_clean"]].isna().any().any()


def test_features_preserve_test_censoring() -> None:
    summary, raw = load_raw_data()
    cleaned = clean_cycle_data(summary, raw)
    features = extract_battery_features(cleaned.cycles)

    assert len(features) == 49
    assert features.loc[features["prediction_test"] == 0, "SOH_200"].notna().all()
    assert features.loc[features["prediction_test"] == 1, "SOH_200"].isna().all()
    assert np.isfinite(features.loc[features["prediction_test"] == 1, "SOH_150"]).all()
