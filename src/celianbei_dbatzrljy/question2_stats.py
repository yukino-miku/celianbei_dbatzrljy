"""Robust strategy and charging-parameter statistics for question two."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from math import comb

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut
from statsmodels.api import OLS, add_constant
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .config import RANDOM_SEED


RESPONSES = {
    "SOH_150": "SOH150",
    "SOH_200": "SOH200",
    "SOH_slope_50_200_per100": "slope_50_200",
    "SOH_curvature_per100sq": "curvature",
    "eol_point": "quadratic_eol",
}
PRIMARY_RESPONSES = ("SOH_200", "SOH_slope_50_200_per100", "eol_point")
PARAMETERS = ("C1", "Q1", "C2")


@dataclass(frozen=True)
class Question2Data:
    batteries: pd.DataFrame
    strategies: pd.DataFrame
    mapping: pd.DataFrame


def load_question2_data(project_root) -> Question2Data:
    """Reuse question-one results and attach untouched strategy metadata."""
    q1 = pd.read_csv(project_root / "outputs/question1/tables/battery_question1_results.csv")
    summary = pd.read_csv(project_root / "data/raw/battery_summary.csv")
    mapping = pd.read_csv(project_root / "outputs/question1/tables/strategy_mapping.csv")
    meta_columns = [
        "battery_id",
        "policy",
        "C1",
        "Q1",
        "C2",
        "initial_capacity",
        "mean_chargetime",
        "dataset_id",
        "prediction_test",
    ]
    meta = summary[meta_columns].rename(
        columns={
            "policy": "strategy",
            "dataset_id": "dataset_id_summary",
            "prediction_test": "prediction_test_summary",
        }
    )
    batteries = q1.merge(meta, on=["battery_id", "strategy"], validate="one_to_one")
    if len(batteries) != 40 or batteries["prediction_test_summary"].ne(0).any():
        raise ValueError("Question-two battery table must contain exactly 40 non-test batteries")
    if not (batteries["dataset_id"] == batteries["dataset_id_summary"]).all():
        raise ValueError("dataset_id differs between question-one results and raw summary")
    batteries = batteries.drop(columns=["dataset_id_summary", "prediction_test_summary"])
    batteries["is_newstructure"] = batteries["strategy"].str.contains("NEWSTRUCTURE").astype(int)

    parameter_uniqueness = batteries.groupby("strategy")[["C1", "Q1", "C2", "dataset_id"]].nunique(dropna=False)
    if parameter_uniqueness.max().max() != 1:
        raise ValueError("A strategy maps to more than one parameter or dataset value")

    aggregations: dict[str, tuple[str, str]] = {
        "n_training_batteries": ("battery_id", "size"),
        "mean_chargetime": ("mean_chargetime_observed", "mean"),
    }
    for response in RESPONSES:
        aggregations[f"{response}_mean"] = (response, "mean")
        aggregations[f"{response}_median"] = (response, "median")
        aggregations[f"{response}_std"] = (response, "std")
    strategies = (
        batteries.groupby(
            ["strategy", "strategy_code", "C1", "Q1", "C2", "dataset_id", "is_newstructure"],
            dropna=False,
            sort=False,
        )
        .agg(**aggregations)
        .reset_index()
    )
    strategies = strategies.sort_values(
        "strategy_code", key=lambda x: x.str[1:].astype(int)
    ).reset_index(drop=True)
    return Question2Data(batteries=batteries, strategies=strategies, mapping=mapping)


def _eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    overall = float(np.mean(values))
    total = float(np.sum((values - overall) ** 2))
    if total <= 0:
        return 0.0
    between = sum(
        len(group) * (float(np.mean(group)) - overall) ** 2
        for label in np.unique(labels)
        for group in [values[labels == label]]
    )
    return float(between / total)


def _permutation_eta_p(
    values: np.ndarray, labels: np.ndarray, *, samples: int, seed: int
) -> tuple[float, float]:
    observed = _eta_squared(values, labels)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(samples):
        exceed += _eta_squared(values, rng.permutation(labels)) >= observed - 1e-15
    return observed, float((exceed + 1) / (samples + 1))


def strategy_difference_tests(
    batteries: pd.DataFrame, *, permutation_samples: int = 19_999
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels = batteries["strategy_code"].to_numpy()
    for index, (response, label) in enumerate(RESPONSES.items()):
        frame = batteries[["strategy_code", response]].dropna()
        groups = [g[response].to_numpy(dtype=float) for _, g in frame.groupby("strategy_code")]
        welch = anova_oneway(groups, use_var="unequal", welch_correction=True)
        kruskal = stats.kruskal(*groups)
        eta2, permutation_p = _permutation_eta_p(
            frame[response].to_numpy(dtype=float),
            frame["strategy_code"].to_numpy(),
            samples=permutation_samples,
            seed=RANDOM_SEED + index,
        )
        n = len(frame)
        k = frame["strategy_code"].nunique()
        epsilon2 = max(0.0, float((kruskal.statistic - k + 1) / (n - k)))
        medians = frame.groupby("strategy_code")[response].median()
        overall_mad = 1.4826 * np.median(
            np.abs(frame[response] - np.median(frame[response]))
        )
        rows.append(
            {
                "response": response,
                "response_label": label,
                "n_batteries": n,
                "n_strategies": k,
                "welch_statistic": float(welch.statistic),
                "welch_df_num": float(welch.df_num),
                "welch_df_denom": float(welch.df_denom),
                "welch_p": float(welch.pvalue),
                "kruskal_h": float(kruskal.statistic),
                "kruskal_p": float(kruskal.pvalue),
                "permutation_eta2": eta2,
                "permutation_p": permutation_p,
                "kruskal_epsilon2": epsilon2,
                "median_range": float(medians.max() - medians.min()),
                "median_range_over_mad": (
                    float((medians.max() - medians.min()) / overall_mad)
                    if overall_mad > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _cliffs_delta(first: np.ndarray, second: np.ndarray) -> float:
    comparisons = np.sign(first[:, None] - second[None, :])
    return float(comparisons.mean())


def _hedges_g(first: np.ndarray, second: np.ndarray) -> float:
    n1, n2 = len(first), len(second)
    pooled_denominator = n1 + n2 - 2
    if pooled_denominator <= 0:
        return np.nan
    pooled = np.sqrt(
        ((n1 - 1) * np.var(first, ddof=1) + (n2 - 1) * np.var(second, ddof=1))
        / pooled_denominator
    )
    if pooled <= 0:
        return np.nan
    correction = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return float(correction * (np.mean(first) - np.mean(second)) / pooled)


def _exact_pairwise_permutation_p(first: np.ndarray, second: np.ndarray) -> float:
    combined = np.concatenate([first, second])
    n_first = len(first)
    observed = abs(float(np.mean(first) - np.mean(second)))
    total = comb(len(combined), n_first)
    exceed = 0
    for selection in combinations(range(len(combined)), n_first):
        mask = np.zeros(len(combined), dtype=bool)
        mask[list(selection)] = True
        statistic = abs(float(np.mean(combined[mask]) - np.mean(combined[~mask])))
        exceed += statistic >= observed - 1e-15
    return float(exceed / total)


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    order = np.argsort(p_values.to_numpy(dtype=float))
    sorted_p = p_values.to_numpy(dtype=float)[order]
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, sorted_p * (len(sorted_p) - np.arange(len(sorted_p))))
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return pd.Series(adjusted, index=p_values.index)


def pairwise_strategy_tests(
    batteries: pd.DataFrame,
    global_tests: pd.DataFrame,
    *,
    bootstrap_samples: int = 5_000,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(RANDOM_SEED + 100)
    significant = set(global_tests.loc[global_tests["permutation_p"] < 0.05, "response"])
    for response in significant:
        groups = {
            code: group[response].dropna().to_numpy(dtype=float)
            for code, group in batteries.groupby("strategy_code")
        }
        for first_code, second_code in combinations(sorted(groups), 2):
            first, second = groups[first_code], groups[second_code]
            bootstrap_difference = np.empty(bootstrap_samples)
            for iteration in range(bootstrap_samples):
                bootstrap_difference[iteration] = np.median(
                    rng.choice(first, len(first), replace=True)
                ) - np.median(rng.choice(second, len(second), replace=True))
            rows.append(
                {
                    "response": response,
                    "strategy_1": first_code,
                    "strategy_2": second_code,
                    "n_1": len(first),
                    "n_2": len(second),
                    "mean_difference_1_minus_2": float(np.mean(first) - np.mean(second)),
                    "median_difference_1_minus_2": float(np.median(first) - np.median(second)),
                    "median_difference_ci95_low": float(np.quantile(bootstrap_difference, 0.025)),
                    "median_difference_ci95_high": float(np.quantile(bootstrap_difference, 0.975)),
                    "exact_permutation_p": _exact_pairwise_permutation_p(first, second),
                    "cliffs_delta": _cliffs_delta(first, second),
                    "hedges_g": _hedges_g(first, second),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["holm_p_within_response"] = result.groupby("response", group_keys=False)[
        "exact_permutation_p"
    ].apply(_holm_adjust)
    return result.sort_values(["response", "holm_p_within_response"]).reset_index(drop=True)


def _standardize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    mean = frame.mean()
    scale = frame.std(ddof=0).replace(0, np.nan)
    return (frame - mean) / scale, mean, scale


def collinearity_diagnostics(strategies: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    correlations: list[dict[str, object]] = []
    scopes = {
        "all_parameterized_strategies": strategies.dropna(subset=["C1"]).copy(),
        "dataset3_only": strategies.loc[(strategies["dataset_id"] == 3) & strategies["C1"].notna()].copy(),
    }
    for scope, frame in scopes.items():
        x = frame[list(PARAMETERS)].astype(float)
        z, _, _ = _standardize(x)
        design = add_constant(z).to_numpy(dtype=float)
        condition = float(np.linalg.cond(design))
        corr = x.corr(method="pearson")
        max_abs_corr = float(
            corr.where(~np.eye(len(corr), dtype=bool)).abs().max().max()
        )
        for i, predictor in enumerate(PARAMETERS, start=1):
            rows.append(
                {
                    "scope": scope,
                    "n_strategies": len(frame),
                    "predictor": predictor,
                    "VIF": float(variance_inflation_factor(design, i)),
                    "condition_number": condition,
                    "max_abs_parameter_correlation": max_abs_corr,
                }
            )
        for method in ("pearson", "spearman"):
            matrix = x.corr(method=method)
            for row_name in PARAMETERS:
                for column_name in PARAMETERS:
                    correlations.append(
                        {
                            "scope": scope,
                            "method": method,
                            "parameter_1": row_name,
                            "parameter_2": column_name,
                            "correlation": float(matrix.loc[row_name, column_name]),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(correlations)


def _fit_standardized_ols(
    frame: pd.DataFrame, predictors: list[str], response: str
):
    clean = frame.dropna(subset=[*predictors, response]).copy()
    x, _, _ = _standardize(clean[predictors].astype(float))
    y = clean[response].astype(float)
    y_scale = float(y.std(ddof=0))
    y_standardized = (y - y.mean()) / y_scale if y_scale > 0 else y * np.nan
    result = OLS(y_standardized, add_constant(x)).fit()
    return clean, x, y_standardized, result


def parameter_ols_results(strategies: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes = {
        "all_unadjusted": (strategies.dropna(subset=["C1"]).copy(), list(PARAMETERS)),
        "all_batch_adjusted": (
            strategies.dropna(subset=["C1"]).assign(
                dataset_2=lambda x: (x["dataset_id"] == 2).astype(float),
                dataset_3=lambda x: (x["dataset_id"] == 3).astype(float),
            ),
            [*PARAMETERS, "dataset_2", "dataset_3"],
        ),
        "dataset3_only": (
            strategies.loc[(strategies["dataset_id"] == 3) & strategies["C1"].notna()].copy(),
            list(PARAMETERS),
        ),
    }
    for scope, (frame, predictors) in scopes.items():
        for response in RESPONSES:
            clean, x, y, result = _fit_standardized_ols(frame, predictors, f"{response}_median")
            for coefficient in ["const", *predictors]:
                rows.append(
                    {
                        "scope": scope,
                        "response": response,
                        "n_strategies": len(clean),
                        "n_predictors": len(predictors),
                        "residual_df": float(result.df_resid),
                        "coefficient": coefficient,
                        "standardized_beta": float(result.params[coefficient]),
                        "standard_error": float(result.bse[coefficient]),
                        "p_value_descriptive_only": float(result.pvalues[coefficient]),
                        "ci95_low_descriptive_only": float(result.conf_int().loc[coefficient, 0]),
                        "ci95_high_descriptive_only": float(result.conf_int().loc[coefficient, 1]),
                        "r_squared": float(result.rsquared),
                        "adjusted_r_squared": float(result.rsquared_adj),
                        "condition_number": float(np.linalg.cond(add_constant(x).to_numpy(dtype=float))),
                    }
                )
    return pd.DataFrame(rows)


def _loo_predictions(x: np.ndarray, y: np.ndarray, *, alpha: float | None = None) -> np.ndarray:
    predictions = np.empty(len(y), dtype=float)
    for train_index, test_index in LeaveOneOut().split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]
        x_mean = x_train.mean(axis=0)
        x_scale = x_train.std(axis=0)
        x_scale[x_scale == 0] = 1.0
        y_mean, y_scale = y_train.mean(), y_train.std()
        if y_scale == 0:
            predictions[test_index] = y_mean
            continue
        zx_train = (x_train - x_mean) / x_scale
        zx_test = (x_test - x_mean) / x_scale
        zy_train = (y_train - y_mean) / y_scale
        if alpha is None:
            parameters = np.linalg.lstsq(
                np.column_stack([np.ones(len(zx_train)), zx_train]), zy_train, rcond=None
            )[0]
            prediction_z = np.column_stack([np.ones(len(zx_test)), zx_test]) @ parameters
        else:
            model = Ridge(alpha=alpha).fit(zx_train, zy_train)
            prediction_z = model.predict(zx_test)
        predictions[test_index] = prediction_z * y_scale + y_mean
    return predictions


def ridge_parameter_results(strategies: pd.DataFrame) -> pd.DataFrame:
    frame = strategies.dropna(subset=["C1"]).copy()
    x = frame[list(PARAMETERS)].to_numpy(dtype=float)
    alphas = np.logspace(-4, 3, 80)
    rows: list[dict[str, object]] = []
    for response in RESPONSES:
        y = frame[f"{response}_median"].to_numpy(dtype=float)
        scores = []
        for alpha in alphas:
            predictions = _loo_predictions(x, y, alpha=float(alpha))
            scores.append(np.sqrt(mean_squared_error(y, predictions)))
        best_alpha = float(alphas[int(np.argmin(scores))])
        z_x, _, _ = _standardize(pd.DataFrame(x, columns=PARAMETERS))
        y_scale = np.std(y)
        z_y = (y - np.mean(y)) / y_scale
        model = Ridge(alpha=best_alpha).fit(z_x, z_y)
        predictions = _loo_predictions(x, y, alpha=best_alpha)
        for predictor, coefficient in zip(PARAMETERS, model.coef_):
            rows.append(
                {
                    "response": response,
                    "n_strategies": len(frame),
                    "best_alpha": best_alpha,
                    "predictor": predictor,
                    "standardized_beta": float(coefficient),
                    "loo_rmse": float(np.sqrt(mean_squared_error(y, predictions))),
                    "loo_mae": float(mean_absolute_error(y, predictions)),
                }
            )
    return pd.DataFrame(rows)


def _linear_model_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    design = np.column_stack([np.ones(len(x)), x])
    parameters = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ parameters
    residual = y - fitted
    rss = max(float(np.sum(residual**2)), np.finfo(float).tiny)
    n = len(y)
    k = design.shape[1]
    aic = n * np.log(rss / n) + 2 * k
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n > k + 1 else np.nan
    predictions = _loo_predictions(x, y)
    return {
        "n": n,
        "k_including_intercept": k,
        "fit_r_squared": float(1 - rss / np.sum((y - np.mean(y)) ** 2)),
        "AICc": float(aicc),
        "loo_rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        "loo_mae": float(mean_absolute_error(y, predictions)),
    }


def stress_p_search(
    strategies: pd.DataFrame, *, p_grid: np.ndarray | None = None
) -> tuple[pd.DataFrame, float]:
    frame = strategies.dropna(subset=["C1"]).copy()
    p_grid = np.linspace(1.0, 3.0, 41) if p_grid is None else p_grid
    rows: list[dict[str, object]] = []
    for p in p_grid:
        stress = (
            frame["Q1"] / 80.0 * frame["C1"] ** p
            + (80.0 - frame["Q1"]) / 80.0 * frame["C2"] ** p
        ).to_numpy(dtype=float)
        for scope, scoped_frame, scoped_stress in (
            ("all_parameterized", frame, stress),
            (
                "dataset3_only",
                frame.loc[frame["dataset_id"] == 3],
                stress[frame["dataset_id"].to_numpy() == 3],
            ),
        ):
            for response in PRIMARY_RESPONSES:
                y = scoped_frame[f"{response}_median"].to_numpy(dtype=float)
                metrics = _linear_model_metrics(scoped_stress[:, None], y)
                rows.append(
                    {
                        "scope": scope,
                        "p": float(p),
                        "response": response,
                        **metrics,
                        "normalized_loo_rmse": float(metrics["loo_rmse"] / np.std(y, ddof=0)),
                    }
                )
    result = pd.DataFrame(rows)
    full = result.loc[result["scope"] == "all_parameterized"]
    combined = full.groupby("p")["normalized_loo_rmse"].mean()
    shared_p = float(combined.idxmin())
    combined_rows = pd.DataFrame(
        {
            "scope": "all_parameterized",
            "p": combined.index,
            "response": "combined_primary_responses",
            "normalized_loo_rmse": combined.values,
        }
    )
    result = pd.concat([result, combined_rows], ignore_index=True, sort=False)
    result["selected_shared_p"] = np.isclose(result["p"], shared_p)
    return result, shared_p


def add_stress_index(strategies: pd.DataFrame, p: float) -> pd.DataFrame:
    result = strategies.copy()
    result["stress_p"] = p
    result["soc_weighted_stress"] = (
        result["Q1"] / 80.0 * result["C1"] ** p
        + (80.0 - result["Q1"]) / 80.0 * result["C2"] ** p
    )
    result["phase1_exposure"] = result["C1"] * result["Q1"] / 80.0
    result["phase2_exposure"] = result["C2"] * (80.0 - result["Q1"]) / 80.0
    return result


def _exact_spearman_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rank_x = stats.rankdata(x).astype(float)
    rank_y = stats.rankdata(y).astype(float)
    rank_x -= rank_x.mean()
    rank_y -= rank_y.mean()
    denominator = float(np.linalg.norm(rank_x) * np.linalg.norm(rank_y))
    if denominator <= 0:
        return np.nan, np.nan
    observed = float(rank_x @ rank_y / denominator)
    permuted_y = np.asarray(list(permutations(rank_y)), dtype=float)
    permuted_rho = permuted_y @ rank_x / denominator
    p_value = float(np.mean(np.abs(permuted_rho) >= abs(observed) - 1e-15))
    return observed, p_value


def univariate_associations(
    strategies_with_stress: pd.DataFrame, *, bootstrap_samples: int = 5_000
) -> pd.DataFrame:
    frame = strategies_with_stress.dropna(subset=["C1"]).copy()
    predictors = [*PARAMETERS, "phase1_exposure", "phase2_exposure", "soc_weighted_stress"]
    rng = np.random.default_rng(RANDOM_SEED + 200)
    rows: list[dict[str, object]] = []
    for predictor in predictors:
        for response in RESPONSES:
            clean = frame[[predictor, f"{response}_median"]].dropna()
            x = clean[predictor].to_numpy(dtype=float)
            y = clean[f"{response}_median"].to_numpy(dtype=float)
            rho, exact_p = _exact_spearman_p(x, y)
            boot = []
            for _ in range(bootstrap_samples):
                sampled = rng.integers(0, len(clean), len(clean))
                if np.unique(x[sampled]).size < 2 or np.unique(y[sampled]).size < 2:
                    continue
                value = stats.spearmanr(x[sampled], y[sampled]).statistic
                if np.isfinite(value):
                    boot.append(float(value))
            boot_array = np.asarray(boot)
            rows.append(
                {
                    "predictor": predictor,
                    "response": response,
                    "n_strategies": len(clean),
                    "spearman_rho": rho,
                    "exact_permutation_p": exact_p,
                    "bootstrap_valid_n": len(boot_array),
                    "rho_ci95_low": float(np.quantile(boot_array, 0.025)),
                    "rho_ci95_high": float(np.quantile(boot_array, 0.975)),
                    "bootstrap_positive_fraction": float(np.mean(boot_array > 0)),
                    "bootstrap_same_sign_fraction": float(
                        np.mean(np.sign(boot_array) == np.sign(rho))
                    ),
                }
            )
    return pd.DataFrame(rows)


def stress_models(strategies_with_stress: pd.DataFrame) -> pd.DataFrame:
    frame = strategies_with_stress.dropna(subset=["soc_weighted_stress"]).copy()
    rows: list[dict[str, object]] = []
    scopes = {
        "all_unadjusted": (frame, ["soc_weighted_stress"]),
        "all_batch_adjusted": (
            frame.assign(
                dataset_2=lambda x: (x["dataset_id"] == 2).astype(float),
                dataset_3=lambda x: (x["dataset_id"] == 3).astype(float),
            ),
            ["soc_weighted_stress", "dataset_2", "dataset_3"],
        ),
        "dataset3_only": (frame.loc[frame["dataset_id"] == 3], ["soc_weighted_stress"]),
    }
    for scope, (scope_frame, predictors) in scopes.items():
        for response in RESPONSES:
            clean, x, y, fit = _fit_standardized_ols(
                scope_frame, predictors, f"{response}_median"
            )
            raw_x = clean[predictors].to_numpy(dtype=float)
            raw_y = clean[f"{response}_median"].to_numpy(dtype=float)
            predictions = _loo_predictions(raw_x, raw_y)
            rho, exact_p = _exact_spearman_p(
                clean["soc_weighted_stress"].to_numpy(dtype=float), raw_y
            )
            rows.append(
                {
                    "scope": scope,
                    "response": response,
                    "n_strategies": len(clean),
                    "n_predictors": len(predictors),
                    "stress_standardized_beta": float(fit.params["soc_weighted_stress"]),
                    "stress_standard_error": float(fit.bse["soc_weighted_stress"]),
                    "stress_p_descriptive_only": float(fit.pvalues["soc_weighted_stress"]),
                    "spearman_rho_unadjusted": rho,
                    "spearman_exact_p": exact_p,
                    "fit_r_squared": float(fit.rsquared),
                    "loo_rmse": float(np.sqrt(mean_squared_error(raw_y, predictions))),
                    "loo_mae": float(mean_absolute_error(raw_y, predictions)),
                    "condition_number": float(
                        np.linalg.cond(add_constant(x).to_numpy(dtype=float))
                    ),
                }
            )
    return pd.DataFrame(rows)


def parameter_model_comparison(
    strategies_with_stress: pd.DataFrame,
) -> pd.DataFrame:
    frame = strategies_with_stress.dropna(subset=["C1"]).copy()
    candidates = {
        "C1_Q1_C2": list(PARAMETERS),
        "phase_exposure": ["phase1_exposure", "phase2_exposure"],
        "shared_p_stress": ["soc_weighted_stress"],
    }
    rows: list[dict[str, object]] = []
    for response in RESPONSES:
        y = frame[f"{response}_median"].to_numpy(dtype=float)
        for model, predictors in candidates.items():
            metrics = _linear_model_metrics(frame[predictors].to_numpy(dtype=float), y)
            rows.append(
                {
                    "response": response,
                    "model": model,
                    "predictors": "+".join(predictors),
                    **metrics,
                    "normalized_loo_rmse": float(metrics["loo_rmse"] / np.std(y, ddof=0)),
                }
            )
    return pd.DataFrame(rows)


def grouped_bootstrap_coefficients(
    batteries: pd.DataFrame,
    *,
    samples: int = 1_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hierarchical strategy-first bootstrap for standardized parameter coefficients."""
    rng = np.random.default_rng(RANDOM_SEED + 300)
    parameterized = batteries.dropna(subset=["C1"]).copy()
    scopes = {
        "all_unadjusted": (parameterized, list(PARAMETERS), False),
        "all_batch_adjusted": (parameterized, [*PARAMETERS, "dataset_2", "dataset_3"], True),
        "dataset3_only": (
            parameterized.loc[parameterized["dataset_id"] == 3], list(PARAMETERS), False
        ),
    }
    rows: list[dict[str, object]] = []
    attempts: dict[tuple[str, str], int] = {}
    for scope, (scope_frame, predictors, add_batch) in scopes.items():
        strategy_codes = scope_frame["strategy_code"].drop_duplicates().to_numpy()
        grouped = {code: g for code, g in scope_frame.groupby("strategy_code")}
        for response in PRIMARY_RESPONSES:
            valid = 0
            attempts[(scope, response)] = samples
            for iteration in range(samples):
                sampled_codes = rng.choice(strategy_codes, len(strategy_codes), replace=True)
                strategy_rows = []
                for draw_index, code in enumerate(sampled_codes):
                    group = grouped[code]
                    sampled_group = group.iloc[rng.integers(0, len(group), len(group))]
                    row = {
                        "bootstrap_cluster": f"{code}_{draw_index}",
                        "C1": float(group["C1"].iloc[0]),
                        "Q1": float(group["Q1"].iloc[0]),
                        "C2": float(group["C2"].iloc[0]),
                        "dataset_id": int(group["dataset_id"].iloc[0]),
                        response: float(sampled_group[response].median()),
                    }
                    strategy_rows.append(row)
                sample_frame = pd.DataFrame(strategy_rows)
                if add_batch:
                    sample_frame["dataset_2"] = (sample_frame["dataset_id"] == 2).astype(float)
                    sample_frame["dataset_3"] = (sample_frame["dataset_id"] == 3).astype(float)
                x = sample_frame[predictors].astype(float)
                z_x, _, _ = _standardize(x)
                if z_x.isna().any().any() or np.linalg.matrix_rank(add_constant(z_x)) < len(predictors) + 1:
                    continue
                y = sample_frame[response].to_numpy(dtype=float)
                y_scale = np.std(y)
                if y_scale <= 0:
                    continue
                fit = OLS((y - np.mean(y)) / y_scale, add_constant(z_x)).fit()
                valid += 1
                for predictor in predictors:
                    rows.append(
                        {
                            "scope": scope,
                            "response": response,
                            "iteration": iteration,
                            "predictor": predictor,
                            "standardized_beta": float(fit.params[predictor]),
                        }
                    )
            attempts[(scope, response)] = valid
    distribution = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    for (scope, response, predictor), group in distribution.groupby(
        ["scope", "response", "predictor"]
    ):
        values = group["standardized_beta"].to_numpy(dtype=float)
        summaries.append(
            {
                "scope": scope,
                "response": response,
                "predictor": predictor,
                "requested_bootstrap_samples": samples,
                "valid_bootstrap_samples": attempts[(scope, response)],
                "valid_fraction": attempts[(scope, response)] / samples,
                "beta_median": float(np.median(values)),
                "beta_ci95_low": float(np.quantile(values, 0.025)),
                "beta_ci95_high": float(np.quantile(values, 0.975)),
                "positive_fraction": float(np.mean(values > 0)),
                "same_sign_fraction": float(max(np.mean(values > 0), np.mean(values < 0))),
            }
        )
    return distribution, pd.DataFrame(summaries)


def conclusion_matrix(associations: pd.DataFrame) -> pd.DataFrame:
    predictors = [*PARAMETERS, "soc_weighted_stress"]
    rows: list[dict[str, object]] = []
    for predictor in predictors:
        subset = associations.loc[
            (associations["predictor"] == predictor)
            & associations["response"].isin(PRIMARY_RESPONSES)
        ].set_index("response")
        oriented = {
            "SOH_200": -float(subset.loc["SOH_200", "spearman_rho"]),
            "SOH_slope_50_200_per100": -float(
                subset.loc["SOH_slope_50_200_per100", "spearman_rho"]
            ),
            "eol_point": -float(subset.loc["eol_point", "spearman_rho"]),
        }
        signs = np.sign(list(oriented.values()))
        observed_consistent = bool(np.all(signs == signs[0]) and signs[0] != 0)
        observed_support = sum(
            subset.loc[response, "exact_permutation_p"] < 0.10
            for response in ("SOH_200", "SOH_slope_50_200_per100")
        )
        eol_support = bool(subset.loc["eol_point", "exact_permutation_p"] < 0.10)
        sign_stability = min(
            float(subset.loc[response, "bootstrap_same_sign_fraction"])
            for response in PRIMARY_RESPONSES
        )
        if observed_consistent and observed_support == 2 and sign_stability >= 0.80:
            classification = "cross_metric_stable_association"
        elif eol_support and observed_support == 0:
            classification = "eol_model_sensitive"
        elif observed_consistent:
            classification = "directionally_consistent_but_limited"
        else:
            classification = "mixed_or_weak"
        rows.append(
            {
                "predictor": predictor,
                "rho_harm_SOH200": oriented["SOH_200"],
                "rho_harm_slope": oriented["SOH_slope_50_200_per100"],
                "rho_harm_eol": oriented["eol_point"],
                "observed_metric_support_count_p_lt_0_10": observed_support,
                "eol_support_p_lt_0_10": eol_support,
                "minimum_bootstrap_same_sign_fraction": sign_stability,
                "classification": classification,
                "interpretation_scope": "observational association; parameters are collinear and batch-confounded",
            }
        )
    return pd.DataFrame(rows)


def s8_diagnostic(strategies_with_stress: pd.DataFrame) -> pd.DataFrame:
    frame = strategies_with_stress.dropna(subset=["soc_weighted_stress"]).copy()
    metrics = {
        "soc_weighted_stress": False,
        "SOH_200_median": True,
        "SOH_slope_50_200_per100_median": True,
        "eol_point_median": True,
    }
    for metric, higher_is_better in metrics.items():
        frame[f"rank_{metric}"] = frame[metric].rank(
            ascending=higher_is_better, method="min"
        )
    return frame[
        [
            "strategy_code",
            "strategy",
            "dataset_id",
            "C1",
            "Q1",
            "C2",
            "soc_weighted_stress",
            "SOH_200_median",
            "SOH_slope_50_200_per100_median",
            "eol_point_median",
            *[f"rank_{metric}" for metric in metrics],
        ]
    ].sort_values("strategy_code", key=lambda x: x.str[1:].astype(int))
