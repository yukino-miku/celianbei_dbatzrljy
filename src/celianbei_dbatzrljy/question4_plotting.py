"""Paper-quality candidate figures for question four."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import seaborn as sns

from .plotting import configure_style


COLORS = {"fast_charge": "#DC2626", "knee": "#F59E0B", "longevity": "#059669"}
LABELS = {"fast_charge": "fast-charge", "knee": "knee / balanced", "longevity": "longevity"}


def _save(fig: plt.Figure, root: Path, stem: str) -> None:
    for format_name in ("png", "svg"):
        directory = root / format_name
        directory.mkdir(parents=True, exist_ok=True)
        kwargs = {"dpi": 320} if format_name == "png" else {}
        fig.savefig(directory / f"{stem}.{format_name}", facecolor="white", bbox_inches="tight", **kwargs)
    plt.close(fig)


def _annotate_strategies(ax, frame: pd.DataFrame, x: str, y: str) -> None:
    for row in frame.itertuples():
        ax.annotate(row.strategy_code, (getattr(row, x), getattr(row, y)), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)


def _mark_recommendations(ax, recommendations: pd.DataFrame, x: str, y: str) -> None:
    for row in recommendations.itertuples():
        color = COLORS[row.recommendation_type]
        ax.scatter(getattr(row, x), getattr(row, y), s=125, marker="*", color=color,
                   edgecolor="white", linewidth=0.8, zorder=6, label=LABELS[row.recommendation_type])


def _surface(ax, slice_frame: pd.DataFrame, x: str, y: str, z: str, title: str, colorbar_label: str) -> None:
    table = slice_frame.pivot(index=y, columns=x, values=z).sort_index().sort_index(axis=1)
    x_values = table.columns.to_numpy(float)
    y_values = table.index.to_numpy(float)
    masked = np.ma.masked_invalid(table.to_numpy(float))
    contours = ax.contourf(x_values, y_values, masked, levels=14, cmap="viridis")
    plt.colorbar(contours, ax=ax, label=colorbar_label)
    ax.set(title=title, xlabel=x, ylabel=y)


def make_question4_plots(results: dict[str, pd.DataFrame], figure_root: Path) -> list[str]:
    configure_style()
    stems: list[str] = []
    strategies = results["all_strategy_inputs"].copy()
    strategies["degradation_rate"] = -strategies["SOH_slope_50_200_per100_median"]
    observed = results["observed_strategy_surrogate_predictions"]
    grid = results["evaluated_grid"]
    feasible = results["optimization_grid_trusted"]
    front = results["pareto_candidates_main"]
    recommendations = results["recommended_strategies"]

    # 1. Existing strategies: observed time versus observed SOH200.
    fig, ax = plt.subplots(figsize=(8.6, 5.7))
    sns.scatterplot(data=strategies, x="mean_chargetime", y="SOH_200_median", hue="dataset_id",
                    size="n_training_batteries", sizes=(70, 180), palette="viridis", ax=ax)
    _annotate_strategies(ax, strategies, "mean_chargetime", "SOH_200_median")
    ax.set(title="Observed strategies: charge time and SOH at cycle 200",
           xlabel="mean observed charge time (min)", ylabel="median SOH200")
    stem = "q4_01_existing_time_soh200"; _save(fig, figure_root, stem); stems.append(stem)

    # 2. Existing strategies: observed time versus observed degradation rate.
    fig, ax = plt.subplots(figsize=(8.6, 5.7))
    sns.scatterplot(data=strategies, x="mean_chargetime", y="degradation_rate", hue="dataset_id",
                    size="n_training_batteries", sizes=(70, 180), palette="viridis", ax=ax)
    _annotate_strategies(ax, strategies, "mean_chargetime", "degradation_rate")
    ax.set(title="Observed strategies: charge time and early degradation",
           xlabel="mean observed charge time (min)", ylabel="SOH loss / 100 cycles (50–200)")
    stem = "q4_02_existing_time_degradation"; _save(fig, figure_root, stem); stems.append(stem)

    # 3. Existing strategies: observed time versus model-based EOL.
    fig, ax = plt.subplots(figsize=(8.6, 5.7))
    sns.scatterplot(data=strategies, x="mean_chargetime", y="eol_point_median", hue="dataset_id",
                    size="n_training_batteries", sizes=(70, 180), palette="viridis", ax=ax)
    _annotate_strategies(ax, strategies, "mean_chargetime", "eol_point_median")
    ax.set(title="Observed strategies: charge time and conditional EOL estimate",
           xlabel="mean observed charge time (min)", ylabel="median estimated EOL (cycles)")
    stem = "q4_03_existing_time_eol"; _save(fig, figure_root, stem); stems.append(stem)

    # 4. Existing strategies, trusted candidates and main Pareto front.
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    ax.scatter(feasible.predicted_charge_time, feasible.predicted_degradation_rate, s=8,
               color="#CBD5E1", alpha=0.35, label="trusted surrogate candidates")
    ax.plot(front.predicted_charge_time, front.predicted_degradation_rate, color="#2563EB", lw=2.3,
            label="model Pareto front")
    ax.scatter(observed.mean_chargetime, -observed.SOH_slope_50_200_per100_median, s=80,
               marker="D", color="#111827", label="observed strategies")
    label_offsets = {
        "S4": (-26, 15), "S5": (8, -18), "S6": (8, 15),
        "S7": (-26, -18), "S8": (8, 8), "S9": (8, 8),
    }
    for row in observed.itertuples():
        ax.annotate(
            row.strategy_code,
            (row.mean_chargetime, -row.SOH_slope_50_200_per100_median),
            xytext=label_offsets[row.strategy_code],
            textcoords="offset points",
            fontsize=8,
        )
    ax.set(title="Trusted candidates and the model-derived Pareto front",
           xlabel="charge time (min)", ylabel="predicted SOH loss / 100 cycles")
    ax.legend(fontsize=8)
    stem = "q4_04_pareto_existing_candidates"; _save(fig, figure_root, stem); stems.append(stem)

    # 5. Three representative recommendations on the Pareto front.
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    ax.plot(front.predicted_charge_time, front.predicted_degradation_rate, color="#2563EB", lw=2.2)
    _mark_recommendations(ax, recommendations, "predicted_charge_time", "predicted_degradation_rate")
    ax.set(title="Fast-charge, knee and longevity recommendations (model candidates)",
           xlabel="predicted charge time (min)", ylabel="predicted SOH loss / 100 cycles")
    ax.legend(fontsize=8)
    stem = "q4_05_recommendations_on_pareto"; _save(fig, figure_root, stem); stems.append(stem)

    # 6. Pareto points in parameter space.
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    scatter = ax.scatter(front.Q1, front.C2, c=front.C1, s=28, cmap="viridis", alpha=0.78)
    plt.colorbar(scatter, ax=ax, label="C1 (C-rate)")
    for row in recommendations.itertuples():
        ax.scatter(row.Q1, row.C2, marker="*", s=150, color=COLORS[row.recommendation_type],
                   edgecolor="white")
    ax.set(title="Pareto solutions in charging-parameter space", xlabel="Q1 (%SOC)", ylabel="C2 (C-rate)")
    stem = "q4_06_parameter_space_pareto"; _save(fig, figure_root, stem); stems.append(stem)

    # 7. Q1-C2 degradation surface at the knee C1 slice, masked to the main trust domain.
    knee = recommendations.query("recommendation_type == 'knee'").iloc[0]
    c1_slice = grid[np.isclose(grid.C1, knee.C1) & grid.inside_main_domain]
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    _surface(ax, c1_slice, "Q1", "C2", "predicted_degradation_rate",
             f"Predicted degradation at C1={knee.C1:.2f} C (trusted cells only)",
             "SOH loss / 100 cycles")
    ax.scatter(observed.Q1, observed.C2, s=55, facecolor="white", edgecolor="#111827")
    _annotate_strategies(ax, observed, "Q1", "C2")
    stem = "q4_07_q1_c2_degradation_surface"; _save(fig, figure_root, stem); stems.append(stem)

    # 8. C1-C2 degradation surface at the knee Q1 slice.
    q1_slice = grid[np.isclose(grid.Q1, knee.Q1) & grid.inside_main_domain]
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    _surface(ax, q1_slice, "C1", "C2", "predicted_degradation_rate",
             f"Predicted degradation at Q1={knee.Q1:.0f}% SOC (trusted cells only)",
             "SOH loss / 100 cycles")
    ax.scatter(observed.C1, observed.C2, s=55, facecolor="white", edgecolor="#111827")
    _annotate_strategies(ax, observed, "C1", "C2")
    stem = "q4_08_c1_c2_degradation_surface"; _save(fig, figure_root, stem); stems.append(stem)

    # 9. Charge-time response surface on the same Q1 slice.
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    _surface(ax, q1_slice, "C1", "C2", "predicted_charge_time",
             f"Physical/empirical charge-time surface at Q1={knee.Q1:.0f}% SOC",
             "predicted time (min)")
    ax.scatter(observed.C1, observed.C2, s=55, facecolor="white", edgecolor="#111827")
    stem = "q4_09_charge_time_surface"; _save(fig, figure_root, stem); stems.append(stem)

    # 10. Geometric knee identification in normalized objective space.
    normalized = front.copy()
    normalized["time_norm"] = (front.predicted_charge_time - front.predicted_charge_time.min()) / max(
        front.predicted_charge_time.max() - front.predicted_charge_time.min(), 1e-12)
    normalized["degradation_norm"] = (front.predicted_degradation_rate - front.predicted_degradation_rate.min()) / max(
        front.predicted_degradation_rate.max() - front.predicted_degradation_rate.min(), 1e-12)
    knee_front = normalized.loc[(normalized.C1 == knee.C1) & (normalized.Q1 == knee.Q1) & (normalized.C2 == knee.C2)].iloc[0]
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.plot(normalized.time_norm, normalized.degradation_norm, color="#2563EB", lw=2)
    ax.scatter(0, 0, marker="X", s=100, color="#111827", label="ideal point")
    ax.plot([0, knee_front.time_norm], [0, knee_front.degradation_norm], ls="--", color="#F59E0B")
    ax.scatter(knee_front.time_norm, knee_front.degradation_norm, marker="*", s=180, color="#F59E0B",
               edgecolor="white", label="minimum distance knee")
    ax.set(title="Knee identification by distance to the normalized ideal point",
           xlabel="normalized charge-time objective", ylabel="normalized degradation objective")
    ax.legend()
    stem = "q4_10_knee_geometry"; _save(fig, figure_root, stem); stems.append(stem)

    # 11. Arrows from nearest experimental strategies to recommendations.
    comparison = results["recommendation_nearest_strategy_comparison"]
    fig, ax = plt.subplots(figsize=(9.0, 5.9))
    ax.scatter(observed.predicted_charge_time, observed.predicted_degradation_rate, marker="D", s=75,
               color="#111827", label="existing strategy, evaluated by proxy")
    for comparison_row in comparison.itertuples():
        baseline = observed.query("strategy_code == @comparison_row.nearest_strategy_code").iloc[0]
        recommendation = recommendations.query(
            "recommendation_type == @comparison_row.recommendation_type"
        ).iloc[0]
        ax.annotate("", xy=(recommendation.predicted_charge_time, recommendation.predicted_degradation_rate),
                    xytext=(baseline.predicted_charge_time, baseline.predicted_degradation_rate),
                    arrowprops={"arrowstyle": "->", "color": COLORS[comparison_row.recommendation_type], "lw": 2})
        ax.scatter(recommendation.predicted_charge_time, recommendation.predicted_degradation_rate, marker="*",
                   s=140, color=COLORS[comparison_row.recommendation_type], edgecolor="white")
        ax.annotate(f"{baseline.strategy_code}→{LABELS[comparison_row.recommendation_type]}",
                    (recommendation.predicted_charge_time, recommendation.predicted_degradation_rate),
                    xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.set(title="Model-predicted changes from nearest experimental strategies",
           xlabel="predicted charge time (min)", ylabel="predicted SOH loss / 100 cycles")
    ax.legend(fontsize=8)
    stem = "q4_11_existing_to_recommendation_arrows"; _save(fig, figure_root, stem); stems.append(stem)

    # 12. Stress-exponent sensitivity of Pareto fronts.
    p_fronts = results["stress_p_sensitivity_pareto"]
    fig, ax = plt.subplots(figsize=(9.0, 5.9))
    for p, group in p_fronts.groupby("sensitivity_p"):
        ax.plot(group.predicted_charge_time, group.predicted_degradation_rate, lw=2, label=f"p={p:g}")
    ax.set(title="Pareto-front sensitivity to the SOC-stress exponent",
           xlabel="predicted charge time (min)", ylabel="predicted SOH loss / 100 cycles")
    ax.legend()
    stem = "q4_12_stress_p_pareto_sensitivity"; _save(fig, figure_root, stem); stems.append(stem)

    # 13. Bootstrap recommendation distribution in parameter space.
    bootstrap = results["bootstrap_recommendations"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    for ax, x, y in zip(axes, ("C1", "Q1", "Q1"), ("C2", "C1", "C2")):
        for label, group in bootstrap.groupby("recommendation_type"):
            ax.scatter(group[x], group[y], s=12, alpha=0.16, color=COLORS[label], label=LABELS[label])
        ax.set(xlabel=x, ylabel=y)
    axes[0].legend(fontsize=7)
    fig.suptitle("Strategy-level bootstrap distribution of recommended parameters", y=1.01)
    stem = "q4_13_bootstrap_recommendation_distribution"; _save(fig, figure_root, stem); stems.append(stem)

    # 14. Domain-sensitivity recommendations.
    domain = results["domain_sensitivity_recommendations"]
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    sns.scatterplot(data=domain, x="predicted_charge_time", y="predicted_degradation_rate", hue="domain",
                    style="recommendation_type", s=100, ax=ax)
    ax.set(title="Recommended points under alternative trust-domain restrictions",
           xlabel="predicted charge time (min)", ylabel="predicted SOH loss / 100 cycles")
    ax.legend(fontsize=7, ncol=2)
    stem = "q4_14_domain_sensitivity"; _save(fig, figure_root, stem); stems.append(stem)

    # 15. Normalized performance of the three main recommendations.
    performance = recommendations.set_index("recommendation_type")[[
        "predicted_charge_time", "predicted_degradation_rate", "predicted_SOH200", "predicted_EOL"
    ]].copy()
    performance["charge_speed"] = 1 - (performance.predicted_charge_time - feasible.predicted_charge_time.min()) / max(
        feasible.predicted_charge_time.max() - feasible.predicted_charge_time.min(), 1e-12)
    performance["low_degradation"] = 1 - (performance.predicted_degradation_rate - feasible.predicted_degradation_rate.min()) / max(
        feasible.predicted_degradation_rate.max() - feasible.predicted_degradation_rate.min(), 1e-12)
    performance["SOH200"] = (performance.predicted_SOH200 - feasible.predicted_SOH200.min()) / max(
        feasible.predicted_SOH200.max() - feasible.predicted_SOH200.min(), 1e-12)
    performance["EOL"] = (performance.predicted_EOL - feasible.predicted_EOL.min()) / max(
        feasible.predicted_EOL.max() - feasible.predicted_EOL.min(), 1e-12)
    long = performance[["charge_speed", "low_degradation", "SOH200", "EOL"]].reset_index().melt(
        id_vars="recommendation_type", var_name="metric", value_name="normalized_score"
    )
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    sns.barplot(data=long, x="metric", y="normalized_score", hue="recommendation_type",
                palette=COLORS, ax=ax)
    ax.set(title="Normalized model performance of representative recommendations",
           xlabel="higher is better", ylabel="normalized score", ylim=(0, 1.08))
    ax.legend(fontsize=8)
    stem = "q4_15_recommendation_performance"; _save(fig, figure_root, stem); stems.append(stem)

    # 16. Marginal benefit along the Pareto front.
    marginal = results["pareto_marginal_benefit_curve"].replace([np.inf, -np.inf], np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    axes[0].plot(marginal.predicted_charge_time, marginal.degradation_reduction_per_added_minute,
                 color="#2563EB", lw=1.5)
    axes[0].set(title="Local degradation reduction per added minute", xlabel="predicted charge time (min)",
                ylabel="SOH-loss reduction / min")
    axes[1].plot(marginal.predicted_charge_time, marginal.EOL_gain_per_added_minute,
                 color="#059669", lw=1.5)
    axes[1].set(title="Conditional EOL gain per added minute", xlabel="predicted charge time (min)",
                ylabel="predicted cycles / min")
    fig.suptitle("Marginal benefits along the model Pareto front", y=1.02)
    stem = "q4_16_marginal_benefit"; _save(fig, figure_root, stem); stems.append(stem)

    # 17. Three-dimensional experimental coverage and optimized candidates.
    fig = plt.figure(figsize=(9.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    sample = feasible.iloc[::max(len(feasible) // 800, 1)]
    ax.scatter(sample.C1, sample.Q1, sample.C2, s=7, color="#CBD5E1", alpha=0.18,
               label="trusted optimization cells")
    ax.scatter(observed.C1, observed.Q1, observed.C2, s=80, marker="D", color="#111827",
               label="observed strategies")
    for row in recommendations.itertuples():
        ax.scatter(row.C1, row.Q1, row.C2, s=170, marker="*", color=COLORS[row.recommendation_type],
                   label=LABELS[row.recommendation_type])
    ax.set(xlabel="C1 (C-rate)", ylabel="Q1 (%SOC)", zlabel="C2 (C-rate)",
           title="Experimental coverage and model-recommended points")
    ax.legend(fontsize=7, loc="best")
    stem = "q4_17_experimental_coverage"; _save(fig, figure_root, stem); stems.append(stem)
    return stems
