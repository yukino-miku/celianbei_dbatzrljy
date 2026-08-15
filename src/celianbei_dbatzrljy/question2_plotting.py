"""Paper-quality candidate figures for question two."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import seaborn as sns

from .plotting import configure_style


PRIMARY = [
    ("SOH_200", "SOH at cycle 200", "SOH"),
    ("SOH_slope_50_200_per100", "Robust slope, cycles 50–200", "SOH / 100 cycles"),
    ("eol_point", "Quadratic estimated EOL", "cycles"),
]
PARAMETERS = ["C1", "Q1", "C2"]


def _save(fig: plt.Figure, figure_root: Path, stem: str) -> None:
    png = figure_root / "png"
    svg = figure_root / "svg"
    png.mkdir(parents=True, exist_ok=True)
    svg.mkdir(parents=True, exist_ok=True)
    fig.savefig(png / f"{stem}.png", dpi=320, facecolor="white")
    fig.savefig(svg / f"{stem}.svg", facecolor="white")
    plt.close(fig)


def _annotate(ax, frame: pd.DataFrame, x: str, y: str) -> None:
    for row in frame.itertuples():
        ax.annotate(row.strategy_code, (getattr(row, x), getattr(row, y)), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)


def make_question2_plots(
    batteries: pd.DataFrame,
    strategies: pd.DataFrame,
    results: dict[str, pd.DataFrame],
    figure_root: Path,
) -> list[str]:
    configure_style()
    stems: list[str] = []
    strategy_order = [f"S{i}" for i in range(1, 10)]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0))
    for ax, (column, title, ylabel) in zip(axes, PRIMARY):
        sns.boxplot(data=batteries, x="strategy_code", y=column, order=strategy_order, ax=ax, color="#DCEAF7",
                    width=.58, fliersize=0)
        sns.stripplot(data=batteries, x="strategy_code", y=column, order=strategy_order, ax=ax, color="#1F4E79",
                      size=4.5, jitter=.16, alpha=.82)
        ax.set(title=title, xlabel="strategy", ylabel=ylabel)
    fig.suptitle("Strategy-level distributions (each point is one training battery)", y=1.02)
    stem = "q2_01_strategy_primary_distributions"; _save(fig, figure_root, stem); stems.append(stem)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))
    for ax, (column, title, ylabel) in zip(axes, PRIMARY):
        y = f"{column}_median"
        size = 45 + 35 * strategies["n_training_batteries"]
        ax.scatter(strategies["mean_chargetime"], strategies[y], s=size, c=strategies["dataset_id"],
                   cmap="viridis", edgecolor="white", linewidth=.8)
        _annotate(ax, strategies, "mean_chargetime", y)
        ax.set(title=title, xlabel="Mean charge time", ylabel=ylabel)
    fig.suptitle("Charge-time and degradation trade-offs; colour denotes dataset_id", y=1.02)
    stem = "q2_02_chargetime_degradation_relations"; _save(fig, figure_root, stem); stems.append(stem)

    parameterized = strategies.dropna(subset=["C1"])
    fig, axes = plt.subplots(3, 3, figsize=(14.2, 12.0))
    for row, parameter in enumerate(PARAMETERS):
        for col, (response, title, ylabel) in enumerate(PRIMARY):
            ax = axes[row, col]; y = f"{response}_median"
            ax.scatter(parameterized[parameter], parameterized[y], c=parameterized["dataset_id"],
                       cmap="viridis", s=70, edgecolor="white")
            _annotate(ax, parameterized, parameter, y)
            ax.set(xlabel=parameter, ylabel=ylabel, title=f"{parameter} vs {title}")
    fig.suptitle("Charging parameters and strategy-median degradation metrics", y=1.005)
    stem = "q2_03_parameter_response_scatter"; _save(fig, figure_root, stem); stems.append(stem)

    correlations = results["parameter_correlations"]
    vif = results["collinearity"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    for ax, scope in zip(axes[:2], ["all_parameterized_strategies", "dataset3_only"]):
        table = correlations.query("scope == @scope and method == 'pearson'").pivot(
            index="parameter_1", columns="parameter_2", values="correlation")
        sns.heatmap(table, annot=True, vmin=-1, vmax=1, cmap="vlag", square=True, ax=ax,
                    cbar=ax is axes[1])
        ax.set(title=scope.replace("_", " "), xlabel="", ylabel="")
    sns.barplot(data=vif, x="predictor", y="VIF", hue="scope", ax=axes[2], palette="Set2")
    axes[2].axhline(5, color="#B91C1C", ls="--", lw=1); axes[2].set(title="VIF diagnostic", xlabel="", ylabel="VIF")
    axes[2].legend(fontsize=7)
    stem = "q2_04_parameter_collinearity"; _save(fig, figure_root, stem); stems.append(stem)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    stress_frame = strategies.dropna(subset=["soc_weighted_stress"])
    for ax, (response, title, ylabel) in zip(axes, PRIMARY):
        y = f"{response}_median"
        sns.regplot(data=stress_frame, x="soc_weighted_stress", y=y, ax=ax, ci=None,
                    scatter_kws={"s": 70, "color": "#2563EB"}, line_kws={"color": "#B91C1C"})
        _annotate(ax, stress_frame, "soc_weighted_stress", y)
        ax.set(title=title, xlabel=f"SOC-weighted stress (p={stress_frame.stress_p.iloc[0]:.2f})", ylabel=ylabel)
    stem = "q2_05_stress_primary_relations"; _save(fig, figure_root, stem); stems.append(stem)

    search = results["stress_p_search"]
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for response, group in search.query("scope == 'all_parameterized'").groupby("response"):
        ax.plot(group["p"], group["normalized_loo_rmse"], marker="o", ms=3, label=response)
    selected = search.loc[search["selected_shared_p"], "p"].iloc[0]
    ax.axvline(selected, color="#B91C1C", ls="--", label=f"shared p={selected:.2f}")
    ax.set(title="Stress exponent selected by leave-one-strategy-out prediction", xlabel="p",
           ylabel="Normalized LOOCV RMSE"); ax.legend()
    stem = "q2_06_stress_exponent_selection"; _save(fig, figure_root, stem); stems.append(stem)

    stress_models = results["stress_models"]
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.7), sharey=True)
    for ax, (response, title, _) in zip(axes, PRIMARY):
        part = stress_models.query("response == @response")
        ax.bar(part["scope"], part["stress_standardized_beta"], color=["#2563EB", "#F59E0B", "#10B981"])
        ax.axhline(0, color="#334155", lw=.8); ax.tick_params(axis="x", rotation=24)
        ax.set(title=title, ylabel="standardized stress coefficient")
    stem = "q2_07_batch_sensitivity_stress"; _save(fig, figure_root, stem); stems.append(stem)

    boot = results["bootstrap_summary"].query("predictor in @PARAMETERS")
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.4), sharey=True)
    scope_markers = {"all_unadjusted": "o", "all_batch_adjusted": "s", "dataset3_only": "^"}
    for ax, (response, title, _) in zip(axes, PRIMARY):
        part = boot.query("response == @response")
        for offset, (scope, marker) in enumerate(scope_markers.items()):
            g = part.query("scope == @scope").set_index("predictor").reindex(PARAMETERS)
            y = np.arange(3) + (offset - 1) * .18
            ax.errorbar(g["beta_median"], y, xerr=[g["beta_median"]-g["beta_ci95_low"],
                        g["beta_ci95_high"]-g["beta_median"]], fmt=marker, capsize=3, label=scope)
        ax.axvline(0, color="#334155", lw=.8); ax.set(title=title, yticks=np.arange(3), yticklabels=PARAMETERS,
                                                      xlabel="bootstrap standardized beta")
    axes[-1].legend(fontsize=7, loc="best")
    stem = "q2_08_grouped_bootstrap_coefficients"; _save(fig, figure_root, stem); stems.append(stem)

    heat_columns = ["C1", "Q1", "C2", "mean_chargetime", "SOH_200_median",
                    "SOH_slope_50_200_per100_median", "eol_point_median"]
    heat = strategies.set_index("strategy_code")[heat_columns].apply(lambda c: (c-c.mean())/c.std(ddof=0))
    fig, ax = plt.subplots(figsize=(11.5, 6.2)); sns.heatmap(heat, cmap="vlag", center=0, annot=True,
                                                            fmt=".1f", ax=ax)
    ax.set(title="Standardized strategy parameter and degradation profile", xlabel="", ylabel="strategy")
    stem = "q2_09_strategy_standardized_heatmap"; _save(fig, figure_root, stem); stems.append(stem)

    comparison = results["model_comparison"]
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    sns.pointplot(data=comparison, x="response", y="normalized_loo_rmse", hue="model", dodge=.25,
                  markers=["o", "s", "^"], ax=ax)
    ax.set(title="Leave-one-strategy-out comparison of parsimonious models", xlabel="response",
           ylabel="Normalized LOOCV RMSE"); ax.tick_params(axis="x", rotation=18)
    stem = "q2_10_parameter_model_comparison"; _save(fig, figure_root, stem); stems.append(stem)

    ds3 = strategies.query("dataset_id == 3 and C1 == C1")
    triangulation = mtri.Triangulation(ds3["Q1"], ds3["C2"])
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    contours = ax.tricontourf(triangulation, ds3["SOH_200_median"], levels=10, cmap="viridis")
    fig.colorbar(contours, ax=ax, label="median SOH200")
    ax.scatter(ds3["Q1"], ds3["C2"], c="white", edgecolor="#111827", s=65)
    _annotate(ax, ds3, "Q1", "C2")
    ax.set(title="Dataset 3 interpolation inside the sampled parameter hull", xlabel="Q1 (%SOC)", ylabel="C2 (C-rate)")
    stem = "q2_11_dataset3_local_soh200_surface"; _save(fig, figure_root, stem); stems.append(stem)

    s8 = results["s8_diagnostic"].set_index("strategy_code")
    rank_columns = [c for c in s8.columns if c.startswith("rank_")]
    fig, ax = plt.subplots(figsize=(9.4, 5.3)); s8[rank_columns].T.plot(ax=ax, marker="o", alpha=.45,
                                                                      legend=False, color="#94A3B8")
    s8.loc[["S8"], rank_columns].T.plot(ax=ax, marker="o", lw=3, color="#DC2626", label="S8")
    ax.invert_yaxis(); ax.set(title="S8 rank profile (rank 1 = highest stress or worst outcome)",
                              xlabel="metric", ylabel="rank"); ax.tick_params(axis="x", rotation=20); ax.legend()
    stem = "q2_12_s8_rank_profile"; _save(fig, figure_root, stem); stems.append(stem)

    pairwise = results["pairwise"]
    if not pairwise.empty and (pairwise["response"] == "SOH_200").any():
        matrix = pd.DataFrame(np.nan, index=strategies.strategy_code, columns=strategies.strategy_code)
        for row in pairwise.query("response == 'SOH_200'").itertuples():
            matrix.loc[row.strategy_1, row.strategy_2] = row.cliffs_delta
            matrix.loc[row.strategy_2, row.strategy_1] = -row.cliffs_delta
        np.fill_diagonal(matrix.values, 0)
        fig, ax = plt.subplots(figsize=(8.0, 6.8)); sns.heatmap(matrix, cmap="vlag", center=0, vmin=-1, vmax=1,
                                                               annot=True, fmt=".2f", ax=ax)
        ax.set(title="Pairwise Cliff's delta for SOH200 (row minus column)", xlabel="", ylabel="")
        stem = "q2_13_pairwise_soh200_effects"; _save(fig, figure_root, stem); stems.append(stem)

    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    sns.boxplot(data=batteries, x="strategy_code", y="SOH_curvature_per100sq", order=strategy_order,
                color="#E0E7FF", fliersize=0, ax=ax)
    sns.stripplot(data=batteries, x="strategy_code", y="SOH_curvature_per100sq", order=strategy_order,
                  color="#4338CA", ax=ax)
    ax.axhline(0, color="#334155", lw=.8); ax.set(title="Observed quadratic curvature by strategy",
                                                  xlabel="strategy", ylabel="SOH / 100 cycles²")
    stem = "q2_14_strategy_curvature"; _save(fig, figure_root, stem); stems.append(stem)

    coefficients = results["parameter_ols"].query("coefficient in @PARAMETERS and scope in ['all_unadjusted','dataset3_only']")
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    for ax, (response, title, _) in zip(axes, PRIMARY):
        part = coefficients.query("response == @response")
        sns.barplot(data=part, x="coefficient", y="standardized_beta", hue="scope", ax=ax, palette="Set2")
        ax.axhline(0, color="#334155", lw=.8); ax.set(title=title, xlabel="", ylabel="standardized beta")
    axes[-1].legend(fontsize=7)
    stem = "q2_15_full_vs_dataset3_parameter_coefficients"; _save(fig, figure_root, stem); stems.append(stem)
    return stems
