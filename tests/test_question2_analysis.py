"""Integrity checks for the question-two analysis design."""

import numpy as np

from celianbei_dbatzrljy.config import PROJECT_ROOT
from celianbei_dbatzrljy.question2_stats import (
    add_stress_index,
    collinearity_diagnostics,
    load_question2_data,
)


def test_question2_reuses_only_non_test_question1_batteries():
    data = load_question2_data(PROJECT_ROOT)
    assert len(data.batteries) == 40
    assert data.batteries["battery_id"].nunique() == 40
    assert set(data.strategies["strategy_code"]) == {f"S{i}" for i in range(1, 10)}


def test_s2_missing_c1_is_not_imputed():
    data = load_question2_data(PROJECT_ROOT)
    s2 = data.strategies.loc[data.strategies["strategy_code"] == "S2"].iloc[0]
    assert np.isnan(s2["C1"])
    assert data.strategies["C1"].notna().sum() == 8


def test_soc_weighted_stress_matches_definition():
    data = load_question2_data(PROJECT_ROOT)
    result = add_stress_index(data.strategies, 2.0)
    s8 = result.loc[result["strategy_code"] == "S8"].iloc[0]
    expected = 31 / 80 * 3.7**2 + (80 - 31) / 80 * 5.9**2
    assert np.isclose(s8["soc_weighted_stress"], expected)
    assert np.isnan(result.loc[result["strategy_code"] == "S2", "soc_weighted_stress"]).all()


def test_collinearity_is_reported_for_full_and_dataset3_scopes():
    data = load_question2_data(PROJECT_ROOT)
    diagnostics, correlations = collinearity_diagnostics(data.strategies)
    assert set(diagnostics["scope"]) == {"all_parameterized_strategies", "dataset3_only"}
    assert diagnostics.groupby("scope").size().eq(3).all()
    assert len(correlations) == 2 * 2 * 3 * 3
