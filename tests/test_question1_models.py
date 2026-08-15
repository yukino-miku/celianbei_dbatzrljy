import numpy as np

from celianbei_dbatzrljy.models import fit_degradation_model, predict_model


def test_all_candidate_models_are_monotone_by_construction() -> None:
    cycle = np.arange(1, 201, dtype=float)
    values = 1.005 - 0.012 * (cycle / 100.0) - 0.004 * (cycle / 100.0) ** 2

    for model in ("linear", "quadratic", "power", "exponential"):
        fit = fit_degradation_model(model, cycle, values)
        assert fit.success
        grid = np.linspace(0, 1000, 1001)
        prediction = predict_model(model, grid, fit.parameters)
        assert np.all(np.diff(prediction) <= 1e-8)


def test_quadratic_model_recovers_bounded_eol_on_synthetic_curve() -> None:
    cycle = np.arange(1, 201, dtype=float)
    values = 1.0 - 0.010 * (cycle / 100.0) - 0.006 * (cycle / 100.0) ** 2
    fit = fit_degradation_model("quadratic", cycle, values)

    assert fit.success
    assert np.isfinite(fit.eol_cycle)
    assert fit.eol_cycle > 200
    assert abs(predict_model("quadratic", np.array([fit.eol_cycle]), fit.parameters)[0] - 0.8) < 1e-7
