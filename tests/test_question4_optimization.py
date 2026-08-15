"""Integrity and physical-reasonableness checks for question four."""

import numpy as np

from celianbei_dbatzrljy.question4_models import (
    add_trust_domain_flags,
    charge_time_cross_validation,
    fit_main_models,
    generate_parameter_grid,
    load_question4_data,
    parameterized_scope,
    pareto_front,
    proxy_cross_validation,
    select_main_stress_p,
)


def test_physical_two_phase_time_is_additive_and_minus_is_audited():
    strategies, _ = load_question4_data()
    s8 = strategies.query("strategy_code == 'S8'").iloc[0]
    expected = 60 * ((31 / 100) / 3.7 + ((80 - 31) / 100) / 5.9)
    assert np.isclose(s8.T_ideal_physical, expected)
    assert 9.9 < s8.T_ideal_physical < 10.2
    assert abs(s8.T_ideal_as_written_minus) < 0.1


def test_continuous_optimization_uses_only_six_parameterized_dataset3_strategies():
    strategies, batteries = load_question4_data()
    local = parameterized_scope(strategies, "dataset3_only")
    assert len(batteries) == 40
    assert set(local.strategy_code) == {"S4", "S5", "S6", "S7", "S8", "S9"}
    assert strategies.query("strategy_code == 'S2'")["C1"].isna().all()


def test_main_domain_is_inside_hull_and_local_trust_region():
    strategies, _ = load_question4_data()
    observed = parameterized_scope(strategies, "dataset3_only")
    classified, metadata = add_trust_domain_flags(generate_parameter_grid(strategies), observed)
    main = classified.query("inside_main_domain")
    assert len(main) == metadata["n_main_domain_grid_points"]
    assert main.inside_convex_hull.all()
    assert main.inside_trust_region.all()
    assert len(main) > 1000


def test_selected_proxies_are_physical_and_pareto_points_are_nondominated():
    strategies, _ = load_question4_data()
    charge_validation = charge_time_cross_validation(strategies)
    proxy_validation = proxy_cross_validation(strategies)
    p, _ = select_main_stress_p(proxy_validation)
    charge, responses = fit_main_models(strategies, charge_validation, p)
    assert charge.name in set(charge_validation.model)
    assert p in {2.0, 2.5, 3.0}
    assert responses["SOH200"].transform == "soh_harm_log"
    assert responses["slope50_200"].transform == "degradation_log"

    example = parameterized_scope(strategies, "dataset3_only")
    example = example.assign(
        predicted_charge_time=charge.predict(example),
        predicted_degradation_rate=-responses["slope50_200"].predict(example),
    )
    front = pareto_front(example)
    for row in front.itertuples():
        dominated = example[
            (example.predicted_charge_time <= row.predicted_charge_time)
            & (example.predicted_degradation_rate <= row.predicted_degradation_rate)
            & (
                (example.predicted_charge_time < row.predicted_charge_time)
                | (example.predicted_degradation_rate < row.predicted_degradation_rate)
            )
        ]
        assert dominated.empty
