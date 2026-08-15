"""Paper-quality candidate figures for question three."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .plotting import configure_style
from .question3_models import strategy_template


def _save(fig: plt.Figure, root: Path, stem: str) -> None:
    for format_name in ("png", "svg"):
        directory = root / format_name; directory.mkdir(parents=True, exist_ok=True)
        kwargs = {"dpi": 320} if format_name == "png" else {}
        fig.savefig(directory / f"{stem}.{format_name}", facecolor="white", **kwargs)
    plt.close(fig)


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values); return x, np.arange(1, len(x)+1)/len(x)


def make_question3_plots(
    train_cycles: pd.DataFrame, test_cycles: pd.DataFrame, pseudotest: pd.DataFrame,
    battery_metrics: pd.DataFrame, model_summary: pd.DataFrame, strategy_metrics: pd.DataFrame,
    deviations_train: pd.DataFrame, test_predictions: pd.DataFrame, deviations_test: pd.DataFrame,
    ensemble_weights: pd.DataFrame, eol_validation: pd.DataFrame, test_eol: pd.DataFrame,
    eol_curves: pd.DataFrame, figure_root: Path,
) -> list[str]:
    configure_style(); stems: list[str] = []
    selected = pseudotest.query("model == 'adaptive_ensemble'")
    palette = sns.color_palette("colorblind", 9)

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for _, g in selected.groupby("battery_id"):
        ax.plot(g["cycle"], g["actual_soh"], color="#94A3B8", lw=.8, alpha=.45)
        ax.plot(g["cycle"], g["predicted_soh"], color="#2563EB", lw=.8, alpha=.38)
    actual_mean = selected.groupby("cycle")["actual_soh"].mean()
    predicted_mean = selected.groupby("cycle")["predicted_soh"].mean()
    ax.plot(actual_mean.index, actual_mean, color="#111827", lw=2.5, label="actual mean")
    ax.plot(predicted_mean.index, predicted_mean, color="#DC2626", lw=2.5, ls="--", label="predicted mean")
    ax.set(title="LOBO pseudo-test: actual and predicted SOH, cycles 151–200", xlabel="cycle", ylabel="SOH")
    ax.legend(); stem="q3_01_pseudotest_all_actual_vs_predicted"; _save(fig,figure_root,stem); stems.append(stem)

    representative = (battery_metrics.query("model == 'adaptive_ensemble'").sort_values("RMSE")
                      .iloc[[0, 7, 15, 23, 31, -1]]["battery_id"].tolist())
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharex=True, sharey=True)
    for ax, battery_id in zip(axes.flat, representative):
        observed = train_cycles.query("battery_id == @battery_id")
        future = selected.query("battery_id == @battery_id")
        ax.plot(observed.query("cycle <= 150")["cycle"], observed.query("cycle <= 150")["SOH_clean"],
                color="#334155", lw=1.2, label="observed 1–150")
        ax.plot(future["cycle"], future["actual_soh"], color="#10B981", lw=1.5, label="actual 151–200")
        ax.plot(future["cycle"], future["predicted_soh"], color="#DC2626", lw=1.6, ls="--", label="predicted")
        rmse = battery_metrics.query("battery_id == @battery_id and model == 'adaptive_ensemble'")["RMSE"].iloc[0]
        ax.set(title=f"Battery {battery_id}, RMSE={rmse:.4f}", xlabel="cycle", ylabel="SOH")
    axes[0,0].legend(fontsize=7); stem="q3_02_representative_pseudotests"; _save(fig,figure_root,stem); stems.append(stem)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    sns.boxplot(data=battery_metrics, x="model", y="MAE", ax=axes[0], color="#DBEAFE", fliersize=2)
    sns.boxplot(data=battery_metrics, x="model", y="RMSE", ax=axes[1], color="#DCFCE7", fliersize=2)
    for ax in axes: ax.tick_params(axis="x", rotation=25); ax.set_xlabel("")
    axes[0].set_title("Battery-level MAE"); axes[1].set_title("Battery-level RMSE")
    stem="q3_03_model_error_distributions"; _save(fig,figure_root,stem); stems.append(stem)

    heat = strategy_metrics.pivot(index="strategy_code", columns="model", values="RMSE_mean")
    fig, ax = plt.subplots(figsize=(10.5, 6.0)); sns.heatmap(heat, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax)
    ax.set(title="Mean pseudo-test RMSE by strategy and model", xlabel="model", ylabel="strategy")
    stem="q3_04_strategy_prediction_errors"; _save(fig,figure_root,stem); stems.append(stem)

    cycle200 = selected.query("cycle == 200")
    fig, ax = plt.subplots(figsize=(6.6, 6.0)); sns.scatterplot(data=cycle200, x="actual_soh", y="predicted_soh",
                                                               hue="strategy_code", s=65, ax=ax, palette=palette)
    low = min(cycle200.actual_soh.min(), cycle200.predicted_soh.min()); high=max(cycle200.actual_soh.max(),cycle200.predicted_soh.max())
    ax.plot([low,high],[low,high], color="#111827", ls="--", lw=1); ax.set_aspect("equal", adjustable="box")
    ax.set(title="Cycle-200 SOH: prediction vs observation", xlabel="actual SOH200", ylabel="predicted SOH200")
    ax.legend(ncol=2, fontsize=7); stem="q3_05_cycle200_scatter"; _save(fig,figure_root,stem); stems.append(stem)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for model, group in battery_metrics.groupby("model"):
        x,y=_ecdf(group["RMSE"].to_numpy(float)); ax.step(x,y,where="post",label=model)
    ax.set(title="ECDF of battery-level pseudo-test RMSE", xlabel="RMSE", ylabel="cumulative fraction"); ax.legend(fontsize=7)
    stem="q3_06_model_error_ecdf"; _save(fig,figure_root,stem); stems.append(stem)

    example_id = int(deviations_train.sort_values("deviation_score").iloc[len(deviations_train)//2]["battery_id"])
    example = train_cycles.query("battery_id == @example_id"); strategy=example["strategy"].iloc[0]
    template = strategy_template(train_cycles, strategy, example_id)
    pred_template = pseudotest.query("battery_id == @example_id and model == 'strategy_template'")
    pred_corrected = pseudotest.query("battery_id == @example_id and model == 'template_individual'")
    fig, ax = plt.subplots(figsize=(10.2, 5.8)); ax.plot(template.cycle,template.template_soh,color="#94A3B8",lw=2,label="LOBO strategy template")
    ax.plot(example.query("cycle <= 150").cycle,example.query("cycle <= 150").SOH_smooth_robust,color="#111827",label="individual observed")
    ax.plot(pred_template.cycle,pred_template.predicted_soh,color="#F59E0B",ls="--",label="template future")
    ax.plot(pred_corrected.cycle,pred_corrected.predicted_soh,color="#2563EB",lw=2,label="template + individual correction")
    ax.plot(pred_corrected.cycle,pred_corrected.actual_soh,color="#10B981",label="actual future")
    ax.axvline(150,color="#64748B",ls=":"); ax.set(title=f"Template and individual correction: battery {example_id}",xlabel="cycle",ylabel="SOH"); ax.legend()
    stem="q3_07_template_individual_correction"; _save(fig,figure_root,stem); stems.append(stem)

    fig, axes = plt.subplots(3,3,figsize=(14.0,10.5),sharex=True,sharey=True)
    for ax,(battery_id,g) in zip(axes.flat,test_cycles.groupby("battery_id",sort=True)):
        pred=test_predictions.query("battery_id == @battery_id")
        ax.plot(g.cycle,g.SOH_clean,color="#334155",lw=1.3); ax.plot(pred.cycle,pred.predicted_soh,color="#DC2626",lw=1.7)
        ax.axvline(150,color="#94A3B8",ls=":"); ax.set_title(f"B{battery_id} / {g.strategy_code.iloc[0]}")
    fig.supxlabel("cycle"); fig.supylabel("SOH"); fig.suptitle("Nine test batteries: observed 1–150 and forecast 151–200")
    stem="q3_08_test_battery_forecasts"; _save(fig,figure_root,stem); stems.append(stem)

    fig, axes = plt.subplots(3,3,figsize=(14.0,10.5),sharex=True,sharey=True)
    for ax,(battery_id,g) in zip(axes.flat,test_cycles.groupby("battery_id",sort=True)):
        pred=test_predictions.query("battery_id == @battery_id")
        ax.plot(g.cycle,g.SOH_smooth_robust,color="#334155",lw=1.2); ax.plot(pred.cycle,pred.predicted_soh,color="#2563EB",lw=1.7)
        ax.fill_between(pred.cycle,pred.prediction_interval_95_low,pred.prediction_interval_95_high,color="#60A5FA",alpha=.25)
        ax.set_title(f"B{battery_id}: {pred.deviation_flag.iloc[0]}")
    fig.supxlabel("cycle"); fig.supylabel("SOH"); fig.suptitle("Test forecasts with LOBO-calibrated 95% prediction intervals")
    stem="q3_09_test_prediction_intervals"; _save(fig,figure_root,stem); stems.append(stem)

    fig, axes = plt.subplots(3,3,figsize=(14.0,12.8),sharey=True, constrained_layout=True)
    for ax,(battery_id,g) in zip(axes.flat,test_cycles.groupby("battery_id",sort=True)):
        curve=eol_curves.query("battery_id == @battery_id"); row=test_eol.query("battery_id == @battery_id").iloc[0]
        ax.plot(g.cycle,g.SOH_smooth_robust,color="#334155",lw=1.0); ax.plot(curve.cycle,curve.predicted_soh,color="#DC2626",lw=1.4)
        ax.axhline(.8,color="#111827",ls="--"); ax.axvline(row.selected_eol_point,color="#F59E0B",ls=":")
        ax.set_xlim(0,min(2500,max(500,row.selected_eol_point*1.1))); ax.set_title(f"B{battery_id}: EOL≈{row.selected_eol_point:.0f}")
    fig.supxlabel("cycle"); fig.supylabel("SOH"); fig.suptitle(
        "Conditional quadratic EOL extrapolations (display capped at 2500 cycles)", fontsize=16
    )
    stem="q3_10_test_eol_extrapolations"; _save(fig,figure_root,stem); stems.append(stem)

    eol_wide=eol_validation.pivot(index="battery_id",columns="scheme",values="estimated_eol")
    fig, ax=plt.subplots(figsize=(7.0,6.2)); ax.scatter(eol_wide.quadratic_150,eol_wide.quadratic_predicted_200,c="#2563EB",s=55)
    bounds=[np.nanmin(eol_wide.values),np.nanmax(eol_wide.values)]; ax.plot(bounds,bounds,ls="--",color="#111827")
    ax.set(title="EOL stability after adding predicted cycles 151–200",xlabel="EOL from cycles 1–150",ylabel="EOL after forecast to 200")
    stem="q3_11_eol150_vs_predicted200"; _save(fig,figure_root,stem); stems.append(stem)

    test_weights=ensemble_weights.query("scope == 'test_forecast'")
    wide=test_weights.pivot(index="battery_id",columns="component",values="adaptive_weight")
    fig, ax=plt.subplots(figsize=(10.0,5.5)); wide.plot(kind="bar",stacked=True,ax=ax,colormap="Set2")
    ax.set(title="Adaptive ensemble contributions for test batteries",xlabel="battery_id",ylabel="weight"); ax.legend(fontsize=7,ncol=2)
    stem="q3_12_test_ensemble_weights"; _save(fig,figure_root,stem); stems.append(stem)

    fig, ax=plt.subplots(figsize=(9.5,5.4)); combined=pd.concat([deviations_train.assign(scope="training LOBO"),deviations_test.assign(scope="test")])
    sns.stripplot(data=combined,x="strategy_code",y="deviation_score",hue="scope",dodge=True,ax=ax,palette=["#94A3B8","#DC2626"])
    ax.axhline(1.25,color="#F59E0B",ls="--",lw=1); ax.axhline(2,color="#B91C1C",ls="--",lw=1)
    ax.set(title="Deviation from same-strategy early-cycle peers",xlabel="strategy",ylabel="standardized deviation score")
    stem="q3_13_test_group_deviation"; _save(fig,figure_root,stem); stems.append(stem)
    return stems
