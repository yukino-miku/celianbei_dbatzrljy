from __future__ import annotations

import numpy as np
import pandas as pd

from celianbei_dbatzrljy.config import PROJECT_ROOT
from celianbei_dbatzrljy.unified_q3q4 import fit_policy_informed_quadratic
from celianbei_dbatzrljy.unified_quadratic import (
    analytic_eol,
    quadratic_derived,
    quadratic_soh,
    stable_to_native,
)
from celianbei_dbatzrljy.unified_q3q4 import PolicyPrior


def test_analytic_eol_is_the_first_quadratic_crossing() -> None:
    a, d1, d2 = 1.0, 0.01, 0.002
    eol = float(analytic_eol(a, d1, d2))
    assert eol > 0
    assert np.isclose(float(quadratic_soh(eol, a, d1, d2)), 0.8, atol=1e-10)
    assert float(quadratic_soh(eol - 1e-4, a, d1, d2)) > 0.8


def test_stable_parameter_round_trip_and_nonnegative_projection() -> None:
    d1, d2 = stable_to_native(np.array([0.03, 0.001]), np.array([0.01, 0.004]))
    assert np.all(d1 >= 0) and np.all(d2 >= 0)
    r100, _, acceleration = quadratic_derived(d1, d2)
    assert np.all(r100 >= acceleration)
    assert np.allclose(acceleration, 2 * d2)


def test_test_batteries_contain_no_cycles_after_150() -> None:
    cycles = pd.read_csv(PROJECT_ROOT / "data/processed/question1/cleaned_cycle_data.csv")
    test = cycles.loc[cycles["prediction_test"] == 1]
    assert test["battery_id"].nunique() == 9
    assert test["cycle"].max() == 150
    assert not (test["cycle"] > 150).any()


def test_policy_fit_respects_physical_bounds() -> None:
    cycle = np.arange(1, 151)
    early = pd.DataFrame(
        {"cycle": cycle, "SOH_smooth_robust": quadratic_soh(cycle, 1.0, 0.004, 0.001)}
    )
    prior = PolicyPrior(0.003, 0.0012, 0.001, 0.0005, "unit_test", 3)
    parameters, _ = fit_policy_informed_quadratic(early, prior, 0.3)
    assert 0.85 <= parameters[0] <= 1.15
    assert parameters[1] >= 0 and parameters[2] >= 0
    future = quadratic_soh(np.arange(151, 501), *parameters)
    assert np.all(np.diff(future) <= 1e-10)


def test_formal_manifest_records_required_bootstrap_sizes() -> None:
    manifest_path = PROJECT_ROOT / "outputs/unified_quadratic_v2/manifest.json"
    if not manifest_path.exists():
        return
    import json

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    assert manifest["baseline_preserved"] is True
    assert manifest["strategy_bootstrap_samples"] >= 1000
    assert manifest["optimization_bootstrap_samples"] >= 1000
    assert manifest["main_eol_definition"].startswith("first n")
