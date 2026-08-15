"""Paper-quality candidate figures for question one."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from .config import EOL_THRESHOLD, FIGURE_PNG_DIR, FIGURE_SVG_DIR, MAX_EOL_CYCLE
from .models import predict_model


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "Arial"],
            "axes.unicode_minus": False,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFCFE",
            "axes.edgecolor": "#334155",
            "grid.color": "#D9E2EC",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.75,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURE_PNG_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_SVG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PNG_DIR / f"{stem}.png", dpi=320, facecolor="white")
    fig.savefig(FIGURE_SVG_DIR / f"{stem}.svg", facecolor="white")
    plt.close(fig)


def _strategy_tools(mapping: pd.DataFrame):
    order = mapping["strategy_code"].tolist()
    strategy_to_code = dict(zip(mapping["strategy"], mapping["strategy_code"]))
    palette_values = sns.color_palette("colorblind", n_colors=len(order))
    palette = dict(zip(order, palette_values))
    return order, strategy_to_code, palette


def _with_codes(frame: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    if "strategy_code" in frame.columns:
        return frame.copy()
    return frame.merge(mapping, on="strategy", how="left", validate="many_to_one")


def plot_all_observed_curves(cleaned: pd.DataFrame, mapping: pd.DataFrame) -> None:
    order, strategy_to_code, palette = _strategy_tools(mapping)
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    for _, group in cleaned.groupby("battery_id"):
        code = strategy_to_code[group["strategy"].iloc[0]]
        is_test = bool(group["prediction_test"].iloc[0])
        ax.plot(
            group["cycle"],
            group["SOH_smooth_robust"],
            color=palette[code],
            linewidth=1.0 if not is_test else 1.3,
            alpha=0.52 if not is_test else 0.85,
            linestyle="--" if is_test else "-",
        )
    handles = [Line2D([0], [0], color=palette[code], lw=2, label=code) for code in order]
    handles.extend(
        [
            Line2D([0], [0], color="#334155", lw=1.5, linestyle="-", label="训练电池"),
            Line2D([0], [0], color="#334155", lw=1.5, linestyle="--", label="测试电池(仅1-150)"),
        ]
    )
    ax.legend(handles=handles, ncol=4, frameon=True, loc="lower left")
    ax.set(title="全部电池的稳健平滑 SOH 轨迹", xlabel="循环次数", ylabel="SOH")
    ax.set_xlim(1, 200)
    _save(fig, "q1_01_all_battery_soh_curves")


def plot_strategy_mean_curves(cleaned: pd.DataFrame, mapping: pd.DataFrame) -> None:
    order, _, palette = _strategy_tools(mapping)
    train = _with_codes(cleaned.loc[cleaned["prediction_test"] == 0], mapping)
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    for code in order:
        group = train.loc[train["strategy_code"] == code]
        stats = group.groupby("cycle")["SOH_smooth_robust"].agg(["mean", "std", "count"])
        sem = stats["std"] / np.sqrt(stats["count"])
        low = stats["mean"] - 1.96 * sem
        high = stats["mean"] + 1.96 * sem
        ax.plot(stats.index, stats["mean"], color=palette[code], lw=2.0, label=code)
        ax.fill_between(stats.index, low, high, color=palette[code], alpha=0.10, linewidth=0)
    ax.legend(ncol=3, frameon=True)
    ax.set(
        title="各策略训练电池的平均 SOH 曲线与点态 95% 置信带",
        xlabel="循环次数",
        ylabel="SOH",
    )
    ax.set_xlim(1, 200)
    _save(fig, "q1_02_strategy_mean_soh_ci")


def plot_strategy_facets(cleaned: pd.DataFrame, mapping: pd.DataFrame) -> None:
    order, _, palette = _strategy_tools(mapping)
    coded = _with_codes(cleaned, mapping)
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 10.0), sharex=True, sharey=True)
    for ax, code in zip(axes.flat, order):
        group = coded.loc[coded["strategy_code"] == code]
        for _, battery in group.groupby("battery_id"):
            is_test = bool(battery["prediction_test"].iloc[0])
            ax.plot(
                battery["cycle"],
                battery["SOH_smooth_robust"],
                color=palette[code],
                alpha=0.75,
                lw=1.25,
                linestyle="--" if is_test else "-",
            )
        ax.set_title(f"{code}  (n={group['battery_id'].nunique()})")
        ax.set_xlim(1, 200)
    for ax in axes[-1, :]:
        ax.set_xlabel("循环次数")
    for ax in axes[:, 0]:
        ax.set_ylabel("SOH")
    fig.suptitle("各充电策略的电池 SOH 轨迹小多图（虚线为测试电池）", y=1.01, fontsize=15)
    fig.tight_layout()
    _save(fig, "q1_03_strategy_soh_small_multiples")


def plot_cleaning_examples(cleaned: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    battery1 = cleaned.loc[cleaned["battery_id"] == 1]
    axes[0].plot(battery1["cycle"], battery1["SOH_raw"], color="#94A3B8", lw=1.1, label="原始 SOH")
    axes[0].plot(
        battery1["cycle"],
        battery1["SOH_smooth_official"],
        color="#F97316",
        lw=1.3,
        alpha=0.9,
        label="附件 SOH_smooth",
    )
    axes[0].plot(battery1["cycle"], battery1["SOH_clean"], color="#2563EB", lw=1.0, label="局部清洗 SOH")
    axes[0].plot(
        battery1["cycle"],
        battery1["SOH_smooth_robust"],
        color="#0F766E",
        lw=2.0,
        label="重新稳健平滑",
    )
    flagged = battery1.loc[battery1["flag_soh_capacity_outlier"]]
    axes[0].scatter(flagged["cycle"], flagged["SOH_raw"], s=45, color="#DC2626", zorder=5, label="异常尖峰")
    axes[0].set(title="电池 1：SOH 尖峰的局部清洗", xlabel="循环次数", ylabel="SOH", ylim=(0.97, 1.46))
    axes[0].legend(frameon=True, loc="upper right")

    ir_examples = cleaned.loc[cleaned["battery_id"].isin([2, 3]) & cleaned["cycle"].between(5, 20)]
    for battery_id, group in ir_examples.groupby("battery_id"):
        axes[1].plot(group["cycle"], group["IR_raw"], marker="o", ms=3, lw=1, label=f"电池 {battery_id} 原始")
        axes[1].plot(group["cycle"], group["IR_clean"], lw=2, linestyle="--", label=f"电池 {battery_id} 清洗")
    axes[1].set(title="电池 2/3：零内阻的局部插补", xlabel="循环次数", ylabel="内阻 (Ω)")
    axes[1].legend(frameon=True, ncol=2)
    fig.tight_layout()
    _save(fig, "q1_04_cleaning_before_after")


def _distribution_plot(
    data: pd.DataFrame,
    mapping: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    stem: str,
) -> None:
    order, _, palette = _strategy_tools(mapping)
    coded = _with_codes(data, mapping)
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    sns.boxplot(
        data=coded,
        x="strategy_code",
        y=column,
        order=order,
        hue="strategy_code",
        palette=palette,
        dodge=False,
        width=0.62,
        showfliers=False,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=coded,
        x="strategy_code",
        y=column,
        order=order,
        color="#111827",
        size=4.5,
        jitter=0.13,
        alpha=0.78,
        ax=ax,
    )
    ax.set(title=title, xlabel="充电策略代码", ylabel=ylabel)
    _save(fig, stem)


def plot_feature_distributions(features: pd.DataFrame, mapping: pd.DataFrame) -> None:
    train = features.loc[features["prediction_test"] == 0]
    _distribution_plot(train, mapping, "SOH_150", "各策略第 150 次循环附近的 SOH", "SOH$_{150}$", "q1_05_soh150_by_strategy")
    _distribution_plot(train, mapping, "SOH_200", "各策略第 200 次循环附近的 SOH", "SOH$_{200}$", "q1_06_soh200_by_strategy")
    _distribution_plot(
        train,
        mapping,
        "SOH_slope_50_200_per100",
        "各策略 50-200 次循环的稳健 SOH 衰减速率",
        "SOH 变化 / 100 循环",
        "q1_07_slope_50_200_by_strategy",
    )


def plot_lifetime_distribution(results: pd.DataFrame, mapping: pd.DataFrame) -> None:
    _distribution_plot(
        results,
        mapping,
        "eol_point",
        "各策略的 80% SOH 估计循环寿命（仅显示有界估计）",
        "估计循环寿命",
        "q1_08_estimated_life_by_strategy",
    )


def _scatter_relation(
    results: pd.DataFrame,
    mapping: pd.DataFrame,
    x: str,
    xlabel: str,
    stem: str,
) -> None:
    order, _, palette = _strategy_tools(mapping)
    coded = _with_codes(results, mapping)
    valid = coded[[x, "eol_point", "strategy_code"]].dropna()
    rho, pvalue = spearmanr(valid[x], valid["eol_point"]) if len(valid) >= 3 else (np.nan, np.nan)
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    for code in order:
        group = valid.loc[valid["strategy_code"] == code]
        ax.scatter(group[x], group["eol_point"], s=42, alpha=0.82, color=palette[code], label=code)
    ax.set(
        title=f"{xlabel}与估计寿命的关系（Spearman ρ={rho:.2f}, p={pvalue:.3g}）",
        xlabel=xlabel,
        ylabel="估计循环寿命",
    )
    ax.legend(ncol=3, frameon=True)
    _save(fig, stem)


def plot_lifetime_relations(results: pd.DataFrame, mapping: pd.DataFrame) -> None:
    _scatter_relation(
        results,
        mapping,
        "SOH_slope_50_200_per100",
        "50-200 次循环 SOH 变化 / 100 循环",
        "q1_09_slope_vs_estimated_life",
    )
    _scatter_relation(results, mapping, "SOH_200", "SOH$_{200}$", "q1_10_soh200_vs_estimated_life")


def plot_model_extrapolations(
    cleaned: pd.DataFrame,
    fits: pd.DataFrame,
    results: pd.DataFrame,
) -> None:
    valid = results.dropna(subset=["eol_point"]).sort_values("eol_point")
    if len(valid) < 3:
        return
    selected_rows = valid.iloc[[0, len(valid) // 2, -1]]
    labels = ["较短寿命", "中等寿命", "较长寿命"]
    model_colors = {
        "linear": "#2563EB",
        "quadratic": "#DC2626",
        "power": "#7C3AED",
        "exponential": "#059669",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9))
    for ax, (_, selected), label in zip(axes, selected_rows.iterrows(), labels):
        battery_id = int(selected["battery_id"])
        observed = cleaned.loc[cleaned["battery_id"] == battery_id]
        battery_fits = fits.loc[(fits["battery_id"] == battery_id) & (fits["cutoff"] == 200)]
        valid_eols = battery_fits["eol_cycle"].dropna()
        x_end = min(MAX_EOL_CYCLE, max(500.0, float(valid_eols.max()) * 1.08 if len(valid_eols) else 1000.0))
        grid = np.linspace(1, x_end, 700)
        ax.scatter(observed["cycle"], observed["SOH_clean"], s=8, color="#94A3B8", alpha=0.45, label="清洗观测")
        ax.plot(observed["cycle"], observed["SOH_smooth_robust"], color="#111827", lw=2.0, label="稳健平滑")
        for _, row in battery_fits.iterrows():
            if not isinstance(row["parameters_json"], str):
                continue
            parameters = np.asarray(json.loads(row["parameters_json"]), dtype=float)
            prediction = predict_model(row["model"], grid, parameters)
            ax.plot(grid, prediction, color=model_colors[row["model"]], lw=1.5, label=row["model"])
            if np.isfinite(row["eol_cycle"]):
                ax.scatter([row["eol_cycle"]], [EOL_THRESHOLD], color=model_colors[row["model"]], s=28, zorder=6)
        ax.axhline(EOL_THRESHOLD, color="#B91C1C", linestyle="--", lw=1.2)
        ax.axvline(200, color="#64748B", linestyle=":", lw=1.1)
        ax.set(title=f"{label}：电池 {battery_id}", xlabel="循环次数", ylabel="SOH", ylim=(0.76, 1.03))
    handles, labels_legend = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels_legend, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=6, frameon=True, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("典型电池的候选退化模型：观测、外推、80% SOH 阈值与 EOL", y=1.02, fontsize=14)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, "q1_11_typical_model_extrapolations")


def plot_validation_errors(validation: pd.DataFrame) -> None:
    long = validation.melt(
        id_vars=["battery_id", "model", "cutoff"],
        value_vars=["MAE", "RMSE", "MaxAE"],
        var_name="metric",
        value_name="error",
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), sharey=False)
    colors = {100: "#F59E0B", 150: "#2563EB"}
    for ax, metric in zip(axes, ["MAE", "RMSE", "MaxAE"]):
        subset = long.loc[long["metric"] == metric]
        sns.boxplot(
            data=subset,
            x="model",
            y="error",
            hue="cutoff",
            palette=colors,
            showfliers=False,
            ax=ax,
        )
        ax.set(title=metric, xlabel="候选模型", ylabel="SOH 误差")
        ax.tick_params(axis="x", rotation=20)
        if ax is not axes[0] and ax.get_legend() is not None:
            ax.get_legend().remove()
    axes[0].legend(title="拟合截断", frameon=True)
    fig.suptitle("前 100/150 次循环拟合后的时间截断验证误差", y=1.02, fontsize=14)
    fig.tight_layout()
    _save(fig, "q1_12_truncation_validation_errors")


def plot_lifetime_stability(stability: pd.DataFrame, selected_model: str) -> None:
    selected = stability.loc[stability["model"] == selected_model]
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    for _, row in selected.iterrows():
        values = [row["eol_100"], row["eol_150"], row["eol_200"]]
        ax.plot([100, 150, 200], values, color="#64748B", alpha=0.42, lw=1.0, marker="o", ms=3)
    median = selected[["eol_100", "eol_150", "eol_200"]].median()
    ax.plot([100, 150, 200], median.values, color="#B91C1C", lw=2.8, marker="o", label="跨电池中位数")
    ax.set(
        title=f"{selected_model} 模型的寿命预测稳定性",
        xlabel="用于拟合的最大循环次数",
        ylabel="80% SOH 估计循环寿命",
        xticks=[100, 150, 200],
    )
    ax.legend(frameon=True)
    _save(fig, "q1_13_lifetime_stability_100_150_200")


def plot_auxiliary_trends(cleaned: pd.DataFrame, mapping: pd.DataFrame) -> None:
    order, _, palette = _strategy_tools(mapping)
    train = _with_codes(cleaned.loc[cleaned["prediction_test"] == 0], mapping)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    variables = [("Tavg", "平均温度 (°C)"), ("IR_clean", "平均内阻 (Ω)"), ("chargetime", "平均充电时间 (min)")]
    for ax, (column, ylabel) in zip(axes, variables):
        for code in order:
            group = train.loc[train["strategy_code"] == code]
            mean_curve = group.groupby("cycle")[column].mean()
            ax.plot(mean_curve.index, mean_curve.values, color=palette[code], lw=1.5, label=code)
        ax.set(xlabel="循环次数", ylabel=ylabel)
    axes[0].legend(ncol=3, frameon=True)
    fig.suptitle("温度、内阻与充电时间的策略平均观测轨迹（描述性）", y=1.02, fontsize=14)
    fig.tight_layout()
    _save(fig, "q1_14_auxiliary_strategy_trends")


def plot_model_selection(summary: pd.DataFrame) -> None:
    ordered = summary.sort_values("selection_score")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = ["#0F766E" if selected else "#94A3B8" for selected in ordered["selected"]]
    ax.barh(ordered["model"], ordered["selection_score"], color=colors)
    ax.invert_yaxis()
    ax.set(title="候选退化模型的综合选择得分（越低越优）", xlabel="加权排名得分", ylabel="候选模型")
    _save(fig, "q1_15_model_selection_score")


def generate_all_figures(
    cleaned: pd.DataFrame,
    features: pd.DataFrame,
    validation: pd.DataFrame,
    fits: pd.DataFrame,
    stability: pd.DataFrame,
    model_summary: pd.DataFrame,
    results: pd.DataFrame,
    mapping: pd.DataFrame,
    selected_model: str,
) -> None:
    configure_style()
    plot_all_observed_curves(cleaned, mapping)
    plot_strategy_mean_curves(cleaned, mapping)
    plot_strategy_facets(cleaned, mapping)
    plot_cleaning_examples(cleaned)
    plot_feature_distributions(features, mapping)
    plot_lifetime_distribution(results, mapping)
    plot_lifetime_relations(results, mapping)
    plot_model_extrapolations(cleaned, fits, results)
    plot_validation_errors(validation)
    plot_lifetime_stability(stability, selected_model)
    plot_auxiliary_trends(cleaned, mapping)
    plot_model_selection(model_summary)
