"""Paper-quality candidate figures for the unified quadratic refactor."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


COLORS = {
    "blue": "#2563EB", "teal": "#0F766E", "orange": "#EA580C",
    "red": "#DC2626", "purple": "#7C3AED", "slate": "#475569",
}


def _configure() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "Arial"],
            "axes.unicode_minus": False, "axes.titlesize": 13, "axes.titleweight": "bold",
            "axes.labelsize": 10.5, "legend.fontsize": 8.3, "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5, "figure.facecolor": "white", "axes.facecolor": "#FBFCFE",
            "grid.color": "#DCE3EB", "grid.linewidth": 0.6, "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, root: Path, question: str, stem: str) -> None:
    for extension, dpi in (("png", 320), ("svg", None)):
        destination = root / question / extension
        destination.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination / f"{stem}.{extension}", dpi=dpi, facecolor="white")
    plt.close(fig)


def _strategy_palette(codes: pd.Series) -> dict[str, tuple[float, ...]]:
    ordered = sorted(codes.dropna().unique(), key=lambda x: int(str(x)[1:]))
    return dict(zip(ordered, sns.color_palette("colorblind", len(ordered))))


def generate_unified_figures(results: dict) -> None:
    """Generate 23 numbered figures as both 320-dpi PNG and SVG."""
    _configure()
    root = Path(results["figure_root"])
    battery = results["battery_parameters"]
    draws = results["parameter_draws"]
    cutoff = results["cutoff_stability"]
    strategy = results["strategy_table"]
    comparison = results["strategy_comparison"]

    # Q1-01: native and stable parameter distributions.
    long = battery.melt(id_vars=["battery_id"], value_vars=["d1", "d2", "R100", "A"], var_name="parameter", value_name="value")
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    sns.boxplot(data=long, x="parameter", y="value", hue="parameter", legend=False, palette="Set2", ax=ax)
    sns.stripplot(data=long, x="parameter", y="value", color="#334155", size=3, alpha=.55, ax=ax)
    ax.set(title="二次退化参数的跨电池分布", xlabel="参数", ylabel="每 100 次循环尺度的 SOH 损失")
    _save(fig, root, "question1", "q1_01_parameter_distributions")

    # Q1-02: compensation with boundary markers.
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    palette = _strategy_palette(battery["strategy_code"])
    for code, group in battery.groupby("strategy_code"):
        ax.scatter(group["d1"], group["d2"], s=48, alpha=.8, label=code, color=palette[code])
    ax.axvline(1e-8, color="#94A3B8", lw=1); ax.axhline(1e-8, color="#94A3B8", lw=1)
    ax.legend(ncol=3); ax.set(title="原生参数 $d_1$–$d_2$ 补偿与零边界", xlabel="$d_1$", ylabel="$d_2$")
    _save(fig, root, "question1", "q1_02_d1_d2_compensation")

    # Q1-03: cutoff stability comparison.
    stable_long = []
    for name in ("d1", "d2", "R100", "A", "eol_cycle"):
        for early in (100, 150):
            column = f"{name}_relative_change_{early}_200"
            stable_long.append(pd.DataFrame({"parameter": name, "cutoff": str(early), "relative_change": cutoff[column].clip(upper=3)}))
    stable_long = pd.concat(stable_long, ignore_index=True)
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    sns.boxplot(data=stable_long, x="parameter", y="relative_change", hue="cutoff", showfliers=False, ax=ax)
    ax.set(title="使用前 100/150 次相对于前 200 次的参数稳定性", xlabel="参数", ylabel="相对变化（截断于 3 以便显示）")
    _save(fig, root, "question1", "q1_03_cutoff_parameter_stability")

    # Q1-04: phase slopes by battery.
    phase_columns = [c for c in battery.columns if c.startswith("SOH_slope_")]
    phase_matrix = battery.set_index("battery_id")[phase_columns]
    fig, ax = plt.subplots(figsize=(9.0, 8.0))
    sns.heatmap(phase_matrix, cmap="RdBu_r", center=0, robust=True, ax=ax, cbar_kws={"label": "SOH/100 cycles"})
    ax.set(title="各电池分阶段稳健 SOH 斜率", xlabel="循环窗口", ylabel="battery_id")
    _save(fig, root, "question1", "q1_04_phase_slope_heatmap")

    # Q1 supplement: stable parameter scatter requested explicitly.
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    for code, group in battery.groupby("strategy_code"):
        ax.scatter(group["R100"], group["A"], s=48, alpha=.8, label=code, color=palette[code])
    ax.legend(ncol=3); ax.set(title="稳定重参数化 $R_{100}$–$A$", xlabel="$R_{100}=d_1+2d_2$", ylabel="$A=2d_2$")
    _save(fig, root, "question1", "q1_s01_R100_A_scatter")

    # Q2-05: standardized strategy response heatmap.
    heat_columns = ["d1", "d2", "R100", "A", "SOH_200", "quadratic_eol"]
    heat = strategy.set_index("strategy_code")[heat_columns]
    z = (heat - heat.mean()) / heat.std(ddof=0).replace(0, 1)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    sns.heatmap(z, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set(title="策略级退化参数与寿命指标（列内标准化）", xlabel="指标", ylabel="策略")
    _save(fig, root, "question2", "q2_05_strategy_response_heatmap")

    # Q2-06: LOSO performance.
    combined = comparison.loc[comparison["response"] == "combined"].copy()
    combined["label"] = combined.apply(lambda r: r["model"] if pd.isna(r["p"]) else f"{r['model']} p={r['p']:g}", axis=1)
    best_by_label = combined.sort_values("normalized_loso_rmse").groupby("label", as_index=False).first().sort_values("normalized_loso_rmse")
    fig, ax = plt.subplots(figsize=(10.0, 6.2))
    sns.barplot(data=best_by_label, y="label", x="normalized_loso_rmse", hue="model", dodge=False, ax=ax)
    ax.axvline(1, color=COLORS["red"], ls="--", lw=1); ax.set(title="策略参数桥接模型的 LOSO 归一化 RMSE", xlabel="两个响应的平均归一化 RMSE", ylabel="候选模型")
    _save(fig, root, "question2", "q2_06_strategy_model_loso")

    # Q2-07: selected model predicted versus observed.
    selection = comparison.loc[comparison["response"] == "combined"].query("model != 'summed_stress_benchmark'").sort_values("normalized_loso_rmse").iloc[0]
    predictions = results["strategy_loso"]
    mask = (predictions["model"] == selection["model"]) & np.isclose(predictions["alpha"], selection["alpha"])
    mask &= predictions["p"].isna() if pd.isna(selection["p"]) else np.isclose(predictions["p"], selection["p"])
    selected_prediction = predictions.loc[mask]
    responses = selected_prediction["response"].unique()
    fig, axes = plt.subplots(1, len(responses), figsize=(5.3 * len(responses), 4.8))
    axes = np.atleast_1d(axes)
    for ax, response in zip(axes, responses):
        group = selected_prediction.loc[selected_prediction["response"] == response]
        ax.scatter(group["actual"], group["predicted"], color=COLORS["blue"], s=55)
        low, high = min(group[["actual", "predicted"]].min()), max(group[["actual", "predicted"]].max())
        ax.plot([low, high], [low, high], "--", color="#64748B")
        for row in group.itertuples(): ax.annotate(row.strategy_code, (row.actual, row.predicted), xytext=(3, 3), textcoords="offset points", fontsize=8)
        ax.set(title=response, xlabel="观测策略中位数", ylabel="LOSO 预测")
    fig.suptitle("选定策略桥接模型：观测–留一预测", y=1.02)
    _save(fig, root, "question2", "q2_07_selected_model_observed_predicted")

    # Q2-08: grouped bootstrap coefficients.
    coef = results["coefficient_summary"].copy()
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    coef["label"] = coef["response"] + " ← " + coef["predictor"]
    y = np.arange(len(coef))
    ax.errorbar(coef["beta_median"], y, xerr=[coef["beta_median"]-coef["beta_ci95_low"], coef["beta_ci95_high"]-coef["beta_median"]], fmt="o", color=COLORS["purple"], capsize=3)
    ax.axvline(0, color="#64748B", ls="--"); ax.set_yticks(y, coef["label"]); ax.set(title="策略优先分组 bootstrap 的标准化系数", xlabel="系数中位数与 95% 区间", ylabel="")
    _save(fig, root, "question2", "q2_08_bootstrap_coefficients")

    # Q2-09: separate phase exposures across p.
    exposure = combined.loc[combined["model"].isin(["separate_phase_exposure", "summed_stress_benchmark"])]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    sns.lineplot(data=exposure, x="p", y="normalized_loso_rmse", hue="model", marker="o", ax=ax)
    ax.set(title="两阶段暴露与求和 stress benchmark 的指数敏感性", xlabel="指数 p", ylabel="LOSO 归一化 RMSE")
    _save(fig, root, "question2", "q2_09_phase_exposure_p_search")

    # Q2-10: mechanism association ablation.
    mechanism = results["mechanism"]
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    sns.barplot(data=mechanism, x="response", y="normalized_lobo_rmse", hue="model", ax=ax)
    ax.tick_params(axis="x", rotation=18); ax.set(title="早期 IR/温度/充电时间特征的增量预测检验", xlabel="后续退化响应", ylabel="电池 LOBO 归一化 RMSE")
    _save(fig, root, "question2", "q2_10_early_health_mechanism_ablation")

    # Q2-11: S3/S9 exact parameter contrast.
    structure = results["recreation"]
    pair = strategy.loc[strategy["strategy_code"].isin(["S3", "S9"])].melt(id_vars="strategy_code", value_vars=["R100", "A", "SOH_200", "quadratic_eol"], var_name="response", value_name="value")
    pair["value_z"] = pair.groupby("response")["value"].transform(lambda x: (x-x.mean())/(x.std(ddof=0) or 1))
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    sns.barplot(data=pair, x="response", y="value_z", hue="strategy_code", ax=ax)
    ax.set(title="相同 $(C_1,Q_1,C_2)$ 的 S3/S9 结构–批次对照", xlabel="响应", ylabel="两点内标准化差异")
    _save(fig, root, "question2", "q2_11_S3_S9_structure_contrast")

    # Q2-12: evidence matrix.
    evidence = results["evidence"].copy()
    map_value = {"unsupported": 0, "directional_but_limited": 1, "robust": 2}
    evidence["score"] = evidence["classification"].map(map_value)
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    sns.barplot(data=evidence, y="evidence_channel", x="score", hue="classification", dodge=False,
                palette={"unsupported":"#CBD5E1", "directional_but_limited":"#F59E0B", "robust":"#0F766E"}, ax=ax)
    ax.set_xticks([0,1,2], ["不支持","方向性有限","稳健"]); ax.set(title="七通道证据审计", xlabel="证据等级", ylabel="")
    _save(fig, root, "question2", "q2_12_evidence_matrix")

    # Q2 supplements: explicit parameter-response and exposure-response views.
    parameterized = strategy.loc[strategy["main_dataset3"]].copy()
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.0))
    for ax, (predictor, response) in zip(axes.flat, [(p, r) for r in ("R100", "A") for p in ("C1", "Q1", "C2")]):
        sns.regplot(data=parameterized, x=predictor, y=response, ci=None, scatter_kws={"s":55}, line_kws={"lw":1.2}, ax=ax)
        for row in parameterized.itertuples(): ax.annotate(row.strategy_code, (getattr(row,predictor),getattr(row,response)), xytext=(3,3), textcoords="offset points", fontsize=8)
        ax.set_title(f"{predictor} → {response}")
    fig.suptitle("dataset 3 充电参数与二次退化响应（仅描述性趋势）", y=1.01)
    _save(fig, root, "question2", "q2_s01_raw_parameters_vs_responses")

    exposure_best = comparison.loc[(comparison.response=="combined")&(comparison.model=="separate_phase_exposure")].sort_values("normalized_loso_rmse").iloc[0]
    p_value=float(exposure_best.p); q=parameterized.Q1/80; parameterized["E1"]=q*parameterized.C1**p_value; parameterized["E2"]=(1-q)*parameterized.C2**p_value
    fig, axes = plt.subplots(2,2,figsize=(10.0,8.0))
    for ax,(predictor,response) in zip(axes.flat,[("E1","R100"),("E2","R100"),("E1","A"),("E2","A")]):
        sns.regplot(data=parameterized,x=predictor,y=response,ci=None,scatter_kws={"s":60},ax=ax)
        for row in parameterized.itertuples(): ax.annotate(row.strategy_code,(getattr(row,predictor),getattr(row,response)),xytext=(3,3),textcoords="offset points",fontsize=8)
        ax.set_title(f"{predictor}(p={p_value:g}) → {response}")
    fig.suptitle("两阶段暴露分别对应基础速率与加速度",y=1.01)
    _save(fig, root, "question2", "q2_s02_phase_exposures_vs_responses")

    exposure_coef=results["exposure_coefficient_summary"].copy(); exposure_coef["label"]=exposure_coef.response+" ← "+exposure_coef.predictor
    fig,ax=plt.subplots(figsize=(7.5,4.8)); y=np.arange(len(exposure_coef)); ax.errorbar(exposure_coef.beta_median,y,xerr=[exposure_coef.beta_median-exposure_coef.beta_ci95_low,exposure_coef.beta_ci95_high-exposure_coef.beta_median],fmt="o",color=COLORS["teal"],capsize=3); ax.axvline(0,color="#64748B",ls="--"); ax.set_yticks(y,exposure_coef.label); ax.set(title=f"E1/E2 标准化系数分组 bootstrap（p={p_value:g}）",xlabel="系数中位数与95%区间",ylabel="")
    _save(fig, root, "question2", "q2_s03_exposure_coefficient_forest")

    fig,ax=plt.subplots(figsize=(10.0,4.8)); ax.axis("off")
    nodes=[(.08,.55,"充电策略\nC1,Q1,C2"),(.38,.75,"early IR / Tavg"),(.38,.35,"early charge time"),(.72,.55,"later degradation\nslopes / SOH200")]
    for x,y,label in nodes: ax.text(x,y,label,ha="center",va="center",bbox=dict(boxstyle="round,pad=.5",fc="#EFF6FF",ec="#2563EB"),fontsize=11)
    for start,end in [((.16,.55),(.29,.72)),((.16,.55),(.29,.38)),((.47,.72),(.63,.58)),((.47,.38),(.63,.52))]: ax.annotate("",xy=end,xytext=start,arrowprops=dict(arrowstyle="->",lw=1.6,color="#475569"))
    m=results["mechanism"].pivot(index="response",columns="model",values="normalized_lobo_rmse"); gain=(m.M1_plus_initial_capacity-m.M2_plus_early_health).median(); ax.text(.5,.08,f"策略级留一验证：M2 相对 M1 的中位归一化 RMSE 改善 = {gain:.3f}\n仅为时间有序关联，不是因果中介",ha="center",fontsize=10,color=COLORS["red"] if gain<=0 else COLORS["teal"])
    ax.set_title("充电策略–早期健康状态–后续退化的关联路径检验")
    _save(fig, root, "question2", "q2_s04_early_health_path")

    match=results["parameter_matches"].copy(); fig,axes=plt.subplots(1,3,figsize=(12.0,4.5),sharey=True)
    for ax,(target,group) in zip(axes,match.groupby("target_parameter")):
        group=group.sort_values("match_rank"); labels=group.strategy_low+"–"+group.strategy_high; ax.bar(labels,group.nuisance_standardized_distance,color=["#0F766E" if q=="good" else "#F59E0B" if q=="moderate" else "#CBD5E1" for q in group.match_quality]); ax.tick_params(axis="x",rotation=25); ax.set_title(f"目标 {target}"); ax.set_xlabel("策略对")
    axes[0].set_ylabel("非目标参数标准化距离"); fig.suptitle("局部准对照匹配质量（灰色仅作弱证据）",y=1.01)
    _save(fig, root, "question2", "q2_s05_parameter_specific_matches")

    # Q3-13: pseudo-test point cloud.
    pseudo = results["combined_predictions"]
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    sample = pseudo.sort_values(["model", "battery_id", "cycle"]).groupby(
        ["model", "battery_id"], group_keys=False
    ).nth(list(range(0, 50, 5))).reset_index()
    sns.scatterplot(data=sample, x="actual_soh", y="predicted_soh", hue="model", s=24, alpha=.65, ax=ax)
    low, high = sample[["actual_soh","predicted_soh"]].min().min(), sample[["actual_soh","predicted_soh"]].max().max()
    ax.plot([low,high],[low,high],"--",color="#64748B"); ax.set(title="40 块训练电池伪测试：真实与预测 SOH", xlabel="真实 SOH（151–200）", ylabel="预测 SOH")
    _save(fig, root, "question3", "q3_13_pseudotest_actual_predicted")

    # Q3-14: per-battery RMSE distributions.
    metrics = []
    for (battery_id, model), group in pseudo.groupby(["battery_id","model"]):
        metrics.append({"battery_id":battery_id,"model":model,"RMSE":np.sqrt(np.mean((group.predicted_soh-group.actual_soh)**2))})
    metrics = pd.DataFrame(metrics)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    sns.boxplot(data=metrics, x="model", y="RMSE", hue="model", legend=False, ax=ax); sns.stripplot(data=metrics, x="model", y="RMSE", color="#334155", alpha=.45, size=3, ax=ax)
    ax.tick_params(axis="x", rotation=15); ax.set(title="短期预测模型的逐电池 RMSE", xlabel="模型", ylabel="RMSE")
    _save(fig, root, "question3", "q3_14_model_rmse_distribution")

    # Q3-15: penalty and feature ablation.
    diag_path = Path(results["output_root"])/"question3/tables/policy_penalty_lobo_diagnostics.csv"
    diag = pd.read_csv(diag_path).groupby("penalty")["future_RMSE"].agg(["mean", lambda x:x.quantile(.9)]).reset_index()
    diag.columns=["penalty","mean","p90"]
    fig, axes = plt.subplots(1,2,figsize=(11.0,4.5))
    axes[0].plot(diag.penalty,diag["mean"],"o-",label="mean"); axes[0].plot(diag.penalty,diag.p90,"s--",label="p90"); axes[0].set_xscale("symlog",linthresh=.01); axes[0].legend(); axes[0].set(title="策略先验惩罚的 LOBO 选择",xlabel="$\\lambda$",ylabel="未来 RMSE")
    ablation=results["ablation"]; sns.barplot(data=ablation,y="ablation",x="pointwise_RMSE",hue="stable_improvement",dodge=False,ax=axes[1]); axes[1].set(title="前 150 次特征消融",xlabel="点态 RMSE",ylabel="")
    _save(fig, root, "question3", "q3_15_penalty_and_feature_ablation")

    # Q3-16: nine test forecasts.
    test_predictions = results["test_predictions"]
    cycles = results["cycles"]
    fig, axes = plt.subplots(3,3,figsize=(13.2,10.0),sharex=True,sharey=True)
    for ax,(battery_id,pred) in zip(axes.flat,test_predictions.groupby("battery_id")):
        observed=cycles.loc[cycles.battery_id==battery_id]
        ax.plot(observed.cycle,observed.SOH_smooth_robust,color="#334155",lw=1.5,label="观测")
        ax.plot(pred.cycle,pred.predicted_soh,color=COLORS["blue"],lw=1.7,label="预测")
        ax.fill_between(pred.cycle,pred.prediction_interval_95_low,pred.prediction_interval_95_high,color=COLORS["blue"],alpha=.15)
        ax.axvline(150,color="#94A3B8",ls="--",lw=.8); ax.set_title(f"Battery {battery_id} ({pred.strategy_code.iloc[0]})")
    axes[0,0].legend(); fig.supxlabel("循环次数"); fig.supylabel("SOH"); fig.suptitle("9 块测试电池：前 150 次观测与 151–200 次预测",y=.995)
    _save(fig, root, "question3", "q3_16_test_forecast_panels")

    # Q3-17: EOL changes and uncertainty.
    eol=results["test_eol"].sort_values("battery_id")
    fig,ax=plt.subplots(figsize=(10.0,5.2)); y=np.arange(len(eol)); ax.errorbar(eol.final_quadratic_eol,y,xerr=[eol.final_quadratic_eol-eol.eol_ci95_low,eol.eol_ci95_high-eol.final_quadratic_eol],fmt="o",color=COLORS["purple"],capsize=3,label="补充预测后")
    ax.scatter(eol.eol_from_cycle150_only,y,marker="x",color=COLORS["orange"],label="仅前150次"); ax.set_yticks(y,eol.battery_id.astype(str)); ax.legend(); ax.set(title="测试电池统一二次 EOL：截断变化与 bootstrap 区间",xlabel="估计循环寿命",ylabel="battery_id")
    _save(fig, root, "question3", "q3_17_test_eol_uncertainty")

    # Q3 supplements: representative policy curves and final quadratic EOL panels.
    policy=pseudo.loc[pseudo.model=="policy_informed_quadratic"]
    representative=metrics.loc[metrics.model=="policy_informed_quadratic"].sort_values("RMSE").iloc[[0,len(metrics.loc[metrics.model=="policy_informed_quadratic"])//2,-1]].battery_id.tolist()
    fig,axes=plt.subplots(1,3,figsize=(13.0,4.5),sharey=False)
    for ax,battery_id in zip(axes,representative):
        observed=cycles.loc[cycles.battery_id==battery_id]; pred=policy.loc[policy.battery_id==battery_id]
        ax.plot(observed.loc[observed.cycle<=150,"cycle"],observed.loc[observed.cycle<=150,"SOH_smooth_robust"],color="#334155",label="fit input")
        ax.plot(observed.loc[observed.cycle>150,"cycle"],observed.loc[observed.cycle>150,"SOH_clean"],color=COLORS["teal"],label="true future")
        ax.plot(pred.cycle,pred.predicted_soh,color=COLORS["orange"],ls="--",label="policy quadratic")
        ax.axvline(150,color="#94A3B8",ls=":"); ax.set_title(f"Battery {battery_id}")
    axes[0].legend(fontsize=7); fig.supxlabel("循环次数"); fig.supylabel("SOH"); fig.suptitle("policy-informed quadratic 伪测试代表曲线",y=1.01)
    _save(fig, root, "question3", "q3_s01_policy_quadratic_pseudotest_curves")

    eol_curves=results["test_eol_curves"]; fig,axes=plt.subplots(3,3,figsize=(13.0,10.0),sharex=False,sharey=True)
    for ax,(battery_id,curve) in zip(axes.flat,eol_curves.groupby("battery_id")):
        observed=cycles.loc[cycles.battery_id==battery_id]; ax.plot(observed.cycle,observed.SOH_smooth_robust,color="#334155",lw=1.2,label="observed 1-150"); ax.plot(curve.cycle,curve.predicted_soh,color=COLORS["purple"],lw=1.4,label="final quadratic"); ax.axhline(.8,color=COLORS["red"],ls="--",lw=.9); ax.axvline(curve.eol.iloc[0],color="#94A3B8",ls=":",lw=.8); ax.set_title(f"Battery {battery_id}: EOL={curve.eol.iloc[0]:.0f}")
    axes[0,0].legend(fontsize=7); fig.supxlabel("循环次数"); fig.supylabel("SOH"); fig.suptitle("9 块测试电池最终统一二次曲线与 80% 交点",y=.995)
    _save(fig, root, "question3", "q3_s02_test_final_quadratic_eol_panels")

    # Q4-18: candidate time-life and dual fronts.
    candidates=results["candidates"]; pareto=results["pareto_median"]; robust=results["pareto_robust"]; rec=results["recommendations"]
    fig,ax=plt.subplots(figsize=(8.5,5.8)); ax.scatter(candidates.predicted_charge_time,candidates.eol_median,s=8,alpha=.12,color="#64748B",label="可信域候选")
    ax.plot(pareto.predicted_charge_time,pareto.eol_median,color=COLORS["blue"],lw=2,label="中位寿命 Pareto"); ax.plot(robust.predicted_charge_time,robust.eol_p10,color=COLORS["orange"],lw=2,label="P10 稳健 Pareto")
    ax.scatter(rec.predicted_charge_time,rec.eol_median,s=90,color=COLORS["red"],marker="*",label="代表推荐"); ax.legend(); ax.set(title="充电时间–统一二次寿命的双 Pareto 前沿",xlabel="预测充电时间 / min",ylabel="估计 EOL / cycles")
    _save(fig, root, "question4", "q4_18_dual_pareto_fronts")

    # Q4-19: recommendation SOH curves.
    curves=results["recommendation_curves"]
    fig,ax=plt.subplots(figsize=(8.8,5.6))
    for row in rec.itertuples():
        group=curves.loc[(np.isclose(curves.C1,row.C1))&(np.isclose(curves.Q1,row.Q1))&(np.isclose(curves.C2,row.C2))]
        ax.plot(group.cycle,group.predicted_soh,lw=2,label=f"{row.recommendation_type}: ({row.C1:.2f},{row.Q1:.0f},{row.C2:.2f})")
        ax.axvline(row.eol_median,ls=":",lw=.8,color="#94A3B8")
    ax.axhline(.8,color=COLORS["red"],ls="--",label="SOH=0.8"); ax.legend(); ax.set(title="三类推荐策略的完整统一二次 SOH 曲线",xlabel="循环次数",ylabel="SOH")
    _save(fig, root, "question4", "q4_19_recommended_full_curves")

    # Q4-20: parameter-space Pareto projection.
    fig,axes=plt.subplots(1,2,figsize=(11.0,4.8)); sc=axes[0].scatter(candidates.Q1,candidates.C2,c=candidates.eol_median,s=12,cmap="viridis"); axes[0].scatter(pareto.Q1,pareto.C2,color=COLORS["red"],s=14,label="Pareto"); axes[0].legend(); axes[0].set(title="$Q_1$–$C_2$ 可信域",xlabel="$Q_1$ / %SOC",ylabel="$C_2$ / C"); fig.colorbar(sc,ax=axes[0],label="EOL")
    sc2=axes[1].scatter(candidates.C1,candidates.C2,c=candidates.predicted_charge_time,s=12,cmap="magma_r"); axes[1].scatter(pareto.C1,pareto.C2,color="#22C55E",s=14); axes[1].set(title="$C_1$–$C_2$ 参数覆盖",xlabel="$C_1$ / C",ylabel="$C_2$ / C"); fig.colorbar(sc2,ax=axes[1],label="min")
    _save(fig, root, "question4", "q4_20_parameter_space_pareto")

    # Q4-21: p sensitivity fronts.
    pfront=results["p_fronts"]
    fig,ax=plt.subplots(figsize=(8.5,5.6)); sns.lineplot(data=pfront,x="predicted_charge_time",y="eol_median",hue="p",palette="viridis",estimator=None,units="p",ax=ax); ax.set(title="两阶段暴露指数 p 的 Pareto 敏感性（点预测）",xlabel="预测充电时间 / min",ylabel="估计 EOL")
    _save(fig, root, "question4", "q4_21_exponent_pareto_sensitivity")

    # Q4-22: bootstrap recommendation parameter distribution.
    boot=results["bootstrap_recommendations"]
    fig,axes=plt.subplots(1,3,figsize=(12.0,4.5))
    for ax,column,label in zip(axes,["C1","Q1","C2"],["$C_1$ / C","$Q_1$ / %SOC","$C_2$ / C"]):
        sns.violinplot(data=boot,x="recommendation_type",y=column,hue="recommendation_type",legend=False,inner="quart",ax=ax); ax.tick_params(axis="x",rotation=20); ax.set(xlabel="",ylabel=label)
    fig.suptitle("1000 次分组 bootstrap 的推荐参数分布",y=1.01)
    _save(fig, root, "question4", "q4_22_bootstrap_recommendation_distribution")

    # Q4-23: trust-domain sensitivity recommendations.
    domain=results["domain_recommendations"]
    fig,ax=plt.subplots(figsize=(8.8,5.6)); sns.scatterplot(data=domain,x="predicted_charge_time",y="eol_median",hue="domain",style="recommendation_type",s=85,ax=ax); ax.set(title="不同可信域约束下的代表方案漂移",xlabel="预测充电时间 / min",ylabel="估计 EOL")
    _save(fig, root, "question4", "q4_23_trust_domain_sensitivity")

    # Q4 supplements: old versus new front and credible recommendation region.
    old=results.get("old_pareto",pd.DataFrame())
    if not old.empty:
        fig,ax=plt.subplots(figsize=(8.4,5.5)); ax.plot(old.predicted_charge_time,old.predicted_EOL,color="#94A3B8",lw=2,label="old SOC-stress EOL surrogate"); ax.plot(pareto.predicted_charge_time,pareto.eol_median,color=COLORS["blue"],lw=2,label="new quadratic median EOL"); ax.plot(robust.predicted_charge_time,robust.eol_p10,color=COLORS["orange"],lw=2,label="new quadratic P10"); ax.legend(); ax.set(title="旧 SOC-stress 与新 quadratic Pareto 前沿",xlabel="预测充电时间 / min",ylabel="模型估计 EOL（定义依赖模型）")
        _save(fig,root,"question4","q4_s01_old_stress_vs_new_quadratic_pareto")

    fig,ax=plt.subplots(figsize=(8.2,5.5)); ax.scatter(candidates.C1,candidates.Q1,s=5,color="#CBD5E1",alpha=.25,label="可信域网格"); sns.scatterplot(data=boot,x="C1",y="Q1",hue="recommendation_type",s=18,alpha=.2,ax=ax); ax.scatter(rec.C1,rec.Q1,marker="*",s=160,color=COLORS["red"],label="主推荐点"); ax.set(title="bootstrap 推荐区域而非单一精确点",xlabel="$C_1$ / C",ylabel="$Q_1$ / %SOC")
    _save(fig,root,"question4","q4_s02_bootstrap_recommendation_regions")
