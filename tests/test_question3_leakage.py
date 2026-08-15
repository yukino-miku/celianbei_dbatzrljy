"""Leakage and boundary checks for question three."""

import numpy as np

from celianbei_dbatzrljy.config import PROJECT_ROOT
from celianbei_dbatzrljy.question3_models import (
    FUTURE_CYCLES,
    base_curve_predictions,
    load_question3_data,
    strategy_template,
)


def test_test_batteries_end_at_cycle_150():
    data = load_question3_data(PROJECT_ROOT)
    assert data.test["battery_id"].nunique() == 9
    assert data.test["cycle"].max() == 150
    assert not (data.test["cycle"] > 150).any()


def test_training_pseudotest_boundary_is_150_to_200():
    data = load_question3_data(PROJECT_ROOT)
    battery_id = int(data.train["battery_id"].iloc[0])
    group = data.train.loc[data.train["battery_id"] == battery_id]
    template = strategy_template(data.train, group["strategy"].iloc[0], battery_id)
    predictions = base_curve_predictions(group.loc[group["cycle"] <= 150], template)
    assert set(predictions) == {
        "individual_quadratic", "local_linear", "strategy_template", "template_individual"
    }
    assert all(len(values) == len(FUTURE_CYCLES) == 50 for values in predictions.values())


def test_lobo_strategy_template_excludes_target_battery():
    data = load_question3_data(PROJECT_ROOT)
    strategy = data.train["strategy"].iloc[0]
    members = data.train.loc[data.train["strategy"] == strategy, "battery_id"].unique()
    target = int(members[0])
    peers = data.train.loc[(data.train["strategy"] == strategy) & (data.train["battery_id"] != target)]
    template = strategy_template(data.train, strategy, target)
    expected = peers.groupby("cycle")["SOH_smooth_robust"].mean().to_numpy()
    assert np.allclose(template["template_soh"], expected)


def test_global_id_is_not_an_engineered_feature():
    data = load_question3_data(PROJECT_ROOT)
    assert "global_id" not in data.features.columns
    assert data.features["prediction_test"].value_counts().to_dict() == {0: 40, 1: 9}
