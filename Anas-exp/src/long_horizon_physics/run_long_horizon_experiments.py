"""
Master Runner: run_long_horizon_experiments.py
Package: long_horizon_physics
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Executes all experimental investigations for the Long-Horizon Reservoir Forecasting Research:
1. Literature foundations & research hypothesis matrix.
2. 13.85 MMSTB model disagreement & horizon error compounding analysis.
3. Long-term asymptotic stress testing (t -> 50 years).
4. Leave-reservoir-out candidate evaluations on Cases 1-70 (authoritative scorer).
5. Multi-origin & multi-horizon cross-validation (2003, 2004, 2005 x 4Q, 12Q, 20Q).
6. January 2008 forward deployment on Field Case 0 across official scenarios.
7. Renders 5 publication figures and syncs to artifact brain directory.
"""
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
import time
import argparse
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ensemble.data import (
    load_uncertainty,
    resolve_shared_dir,
    TRAIN_CASES,
    YEAR_DAYS,
)
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
)
from long_horizon_physics.literature_foundations import (
    build_research_hypothesis_matrix,
)
from long_horizon_physics.disagreement_analysis import (
    run_horizon_disagreement_audit,
    run_horizon_error_compounding_backtest,
)
from long_horizon_physics.experimental_evaluation import (
    evaluate_candidates_on_pseudo_fields,
    run_multi_origin_candidate_benchmark,
    run_long_term_asymptotic_stress_tests,
    EVAL_PSEUDO_FIELDS,
)
from long_horizon_physics.deployment_2008_forecaster import (
    run_deployment_2008_comparison,
)

TABLE_DIR = Path("outputs/research/long_horizon_physics/tables")
FIG_DIR = Path("outputs/research/long_horizon_physics/figures")
BRAIN_DIR = Path("/Users/tengkuanas/.gemini/antigravity-cli/brain/4b7e1d00-9795-42a6-b5bb-c629e9cf25cb")


def plot_figure_1_hypotheses_concept(save_path: Path):
    """Figure 1: Multiphase Coupled Physics Formulation vs Independent Base Models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Left: Paradigm comparison diagram
    ax = axes[0]
    ax.axis("off")
    ax.set_title("A: Traditional Independent Models vs Multiphase Coupled Representation", fontsize=11, fontweight="bold")
    
    box_props = dict(boxstyle="round,pad=0.5", facecolor="#ebf5fb", edgecolor="#2980b9", lw=1.5)
    box_props_c = dict(boxstyle="round,pad=0.5", facecolor="#e8f8f5", edgecolor="#16a085", lw=1.5)
    
    text_trad = (
        "CURRENT BASELINE (UNCOUPLED):\n"
        "- Oil rate: q_o(t) = f_1(t, u)  [Ridge Exponential DCA]\n"
        "- Water rate: q_w(t) = f_2(t, u)  [Independent Water Model]\n"
        "PROBLEM: Rates diverge independently;\n"
        "water cut f_w = q_w / (q_o + q_w) can violate material balance."
    )
    text_coup = (
        "NEW MULTIPHASE COUPLING (PHASE 2):\n"
        "1. Total Liquid Rate: q_L(t) = q_o(t) + q_w(t)  [Mild decline]\n"
        "2. Water Cut / WOR: f_w(t) in [0, 1)  [Logistic / Koval]\n"
        "3. Exact Physical Coupling:\n"
        "   q_o(t) = q_L(t) * [1 - f_w(t)]\n"
        "   q_w(t) = q_L(t) * f_w(t)\n"
        "BENEFIT: Strict mass conservation, bounded water cut."
    )
    
    ax.text(0.05, 0.70, text_trad, transform=ax.transAxes, fontsize=9.5, bbox=box_props, verticalalignment="top", family="monospace")
    ax.text(0.05, 0.32, text_coup, transform=ax.transAxes, fontsize=9.5, bbox=box_props_c, verticalalignment="top", family="monospace")
    
    # Right: Fractional flow curve
    ax2 = axes[1]
    QL = np.linspace(0, 50, 200)
    for K_K in [2.0, 4.0, 8.0]:
        fw = (K_K * (QL / 50.0)) / (1.0 + (K_K - 1.0) * (QL / 50.0))
        ax2.plot(QL, fw, lw=2.2, label=f"Koval Factor K_K = {K_K:.1f}")
    ax2.set_xlabel("Cumulative Produced Liquid Q_L (MMSTB)", fontweight="bold")
    ax2.set_ylabel("Water Cut f_w(t)", fontweight="bold")
    ax2.set_title("B: Koval Fractional Flow Progression (SPE-173315-MS)", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, loc="lower right")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_figure_2_disagreement_expansion(df_traj: pd.DataFrame, df_milestones: pd.DataFrame, save_path: Path):
    """Figure 2: Base Model Divergence Across 20 Years and Error Compounding."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    years = df_traj["years_ahead"].values
    
    # Left: 80-quarter Cumulative Oil Trajectories
    ax1 = axes[0]
    ax1.plot(years, df_traj["et_cum"] / 1e6, label="Recursive ExtraTrees (61.7 MMSTB)", color="#27ae60", lw=2.5)
    ax1.plot(years, df_traj["exp_cum"] / 1e6, label="Learned Exponential DCA (47.9 MMSTB)", color="#2980b9", lw=2.5)
    ax1.plot(years, df_traj["hyp_cum"] / 1e6, label="Learned Hyperbolic DCA (52.1 MMSTB)", color="#e67e22", lw=2.5)
    ax1.plot(years, df_traj["hyb_cum"] / 1e6, label="Hybrid Dynamic (50.8 MMSTB)", color="#8e44ad", lw=2.5)
    ax1.fill_between(years, df_traj["exp_cum"] / 1e6, df_traj["et_cum"] / 1e6, color="#f39c12", alpha=0.15, label="13.85 MMSTB Range")
    
    ax1.set_xlabel("Forecast Horizon (Years Beyond 2008)", fontweight="bold")
    ax1.set_ylabel("20-Year Cumulative Oil (MMSTB)", fontweight="bold")
    ax1.set_title("A: The 13.85 MMSTB Base-Model Range on Field Case 0", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, loc="upper left")
    
    # Right: Disagreement Spread Expansion Over Time
    ax2 = axes[1]
    ax2.plot(years, df_traj["spread_cum"] / 1e6, color="#c0392b", lw=2.5, label="Inter-Model Spread (MMSTB)")
    
    for _, row in df_milestones.iterrows():
        yr = row["years_ahead"]
        sp = row["spread_cum_oil_stb"] / 1e6
        ax2.scatter(yr, sp, color="#c0392b", s=40, zorder=5)
        ax2.annotate(f"{sp:.2f}M", (yr, sp), textcoords="offset points", xytext=(-5, 8), fontsize=8, fontweight="bold")
        
    ax2.set_xlabel("Forecast Horizon (Years Ahead)", fontweight="bold")
    ax2.set_ylabel("Cumulative Oil Spread (MMSTB)", fontweight="bold")
    ax2.set_title("B: Model Disagreement Expansion Over Horizon", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, loc="upper left")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_figure_3_candidate_benchmark(summary_df: pd.DataFrame, save_path: Path):
    """Figure 3: Macro, Oil, and Water NRMSE of Challenger Models vs Frozen Baseline."""
    fig, ax = plt.subplots(figsize=(14, 5.5))
    
    df_sorted = summary_df.sort_values("macro_nrmse_mean").reset_index(drop=True)
    models = df_sorted["model_name"].tolist()
    labels = [m.replace("Challenger_", "").replace("Frozen_Ensemble_", "").replace("_", " ") for m in models]
    
    x = np.arange(len(models))
    width = 0.26
    
    rects1 = ax.bar(x - width, df_sorted["macro_nrmse_mean"], width, label="Macro NRMSE", color="#2980b9")
    rects2 = ax.bar(x, df_sorted["oil_nrmse_mean"], width, label="Oil NRMSE", color="#27ae60")
    rects3 = ax.bar(x + width, df_sorted["water_nrmse_mean"], width, label="Water NRMSE", color="#c0392b")
    
    base_macro = df_sorted.loc[df_sorted["model_name"] == "Frozen_Ensemble_Baseline", "macro_nrmse_mean"].values[0]
    ax.axhline(base_macro, color="#2980b9", linestyle="--", alpha=0.7, label=f"Frozen Baseline ({base_macro:.4f})")
    
    ax.set_ylabel("Cumulative Increment NRMSE (12Q)", fontweight="bold")
    ax.set_title("Figure 3: Challenger Forecasting Models vs Frozen Baseline (Leave-Reservoir-Out Validation)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="upper left")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_figure_4_multi_origin_robustness(multi_df: pd.DataFrame, save_path: Path):
    """Figure 4: Robustness Across Origins (2003, 2004, 2005) and Horizons (4Q, 12Q, 20Q)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    origins = ["2003-01-01", "2004-01-01", "2005-01-01"]
    titles = ["Origin: Jan 2003 (Early Decline)", "Origin: Jan 2004 (Mid Maturity)", "Origin: Jan 2005 (Late Maturity)"]
    
    key_models = [
        "Frozen_Ensemble_Baseline",
        "H4_Selective_Water_Correction",
        "H5_Modified_Arps_Dlim",
        "H8_Constrained_Tail_Hybrid",
    ]
    
    for i, orig in enumerate(origins):
        ax = axes[i]
        sub = multi_df[multi_df["origin"] == orig]
        horizons = sorted(sub["horizon_quarters"].unique())
        
        for m in key_models:
            m_sub = sub[sub["model_name"] == m].sort_values("horizon_quarters")
            if len(m_sub) > 0:
                lbl = m.replace("H4_", "").replace("H5_", "").replace("H8_", "").replace("Frozen_Ensemble_", "").replace("_", " ")
                ax.plot(m_sub["horizon_quarters"], m_sub["macro_nrmse_mean"], marker="o", lw=2, label=lbl)
                
        ax.set_title(titles[i], fontsize=10, fontweight="bold")
        ax.set_xlabel("Forecast Horizon (Quarters)", fontweight="bold")
        if i == 0:
            ax.set_ylabel("Macro Increment NRMSE", fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        if i == 0:
            ax.legend(frameon=True, fontsize=8)
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_figure_5_deployment_trajectories(df_scenarios: pd.DataFrame, df_summary: pd.DataFrame, save_path: Path):
    """Figure 5: 20-Year Forward Deployment Trajectories on Field Case 0."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Left: Boxplot of 20-Year Cumulative Oil EUR Across Scenarios
    ax1 = axes[0]
    models = df_summary["model_name"].tolist()
    labels = [m.replace("Challenger_", "").replace("Frozen_Ensemble_", "").replace("_", "\n") for m in models]
    
    data = [df_scenarios[df_scenarios["model_name"] == m]["eur_oil_stb"].values / 1e6 for m in models]
    
    ax1.boxplot(data, tick_labels=labels, patch_artist=True,
                boxprops=dict(facecolor="#aed6f1", color="#1b4f72"),
                medianprops=dict(color="#922b21", lw=2))
    ax1.set_ylabel("20-Year Cumulative Oil EUR (MMSTB)", fontweight="bold")
    ax1.set_title("A: Scenario Uncertainty Envelope on Field Case 0", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    
    # Right: Terminal Oil Rate Comparison
    ax2 = axes[1]
    term_rates = df_summary["mean_terminal_oil_rate_stbd"].values
    bars = ax2.bar(labels, term_rates, color=["#2980b9", "#27ae60", "#e67e22", "#8e44ad"])
    ax2.set_ylabel("Terminal Oil Rate at 2028 (STB/d)", fontweight="bold")
    ax2.set_title("B: Plausible Terminal Production Rate at Year 20", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    for b in bars:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2.0, h + 10, f"{h:,.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run Long-Horizon Reservoir Forecasting Science Experiments")
    parser.add_argument("--force", action="store_true", help="Force recomputation of all stages")
    args = parser.parse_args()
    
    print("=" * 80)
    print("EXXONMOBIL DATAWORKS CHALLENGE 2026")
    print("LONG-HORIZON RESERVOIR FORECASTING RESEARCH")
    print("=" * 80)
    t_start = time.perf_counter()
    
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    BRAIN_DIR.mkdir(parents=True, exist_ok=True)
    
    shared_dir = resolve_shared_dir()
    unc_df = load_uncertainty(shared_dir)
    raw_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(raw_df)
    
    # -------------------------------------------------------------------------
    # STAGE 1: Research Hypothesis Matrix
    # -------------------------------------------------------------------------
    print("\n[Stage 1] Building Research Hypothesis Matrix...")
    hyp_df = build_research_hypothesis_matrix()
    hyp_path = TABLE_DIR / "research_hypothesis_matrix.csv"
    hyp_df.to_csv(hyp_path, index=False)
    print(f"  Saved hypothesis matrix with {len(hyp_df)} candidates: {hyp_path}")
    
    # -------------------------------------------------------------------------
    # STAGE 2: Disagreement & Error Compounding Analysis
    # -------------------------------------------------------------------------
    milestones_path = TABLE_DIR / "base_models_horizon_milestones_audit.csv"
    traj_path = TABLE_DIR / "base_models_80q_trajectories.csv"
    disagree_findings_path = TABLE_DIR / "base_models_disagreement_findings.json"
    backtest_path = TABLE_DIR / "horizon_error_compounding_backtest.csv"
    
    if milestones_path.exists() and traj_path.exists() and backtest_path.exists() and not args.force:
        print("\n[Stage 2] Loading cached Disagreement & Error Compounding Analysis...")
        df_milestones = pd.read_csv(milestones_path)
        df_traj = pd.read_csv(traj_path)
        df_backtest = pd.read_csv(backtest_path)
    else:
        print("\n[Stage 2] Executing Disagreement & Error Compounding Analysis...")
        t0 = time.perf_counter()
        df_milestones, df_traj, findings = run_horizon_disagreement_audit(
            case_dict=case_dict, unc_df=unc_df, inference_origin=pd.Timestamp("2008-01-01")
        )
        df_milestones.to_csv(milestones_path, index=False)
        df_traj.to_csv(traj_path, index=False)
        with open(disagree_findings_path, "w") as f:
            json.dump(findings, f, indent=2)
            
        print("  Running retrospective horizon error compounding backtest (4Q to 20Q)...")
        df_backtest = run_horizon_error_compounding_backtest(
            case_dict=case_dict, unc_df=unc_df, test_cases=EVAL_PSEUDO_FIELDS, origin=pd.Timestamp("2003-01-01")
        )
        df_backtest.to_csv(backtest_path, index=False)
        print(f"  Completed in {time.perf_counter() - t0:.2f}s. Saved milestones, trajectories, and backtest.")
        
    # -------------------------------------------------------------------------
    # STAGE 3: Long-Term Asymptotic Stress Tests
    # -------------------------------------------------------------------------
    stress_path = TABLE_DIR / "asymptotic_stress_test_results.csv"
    if stress_path.exists() and not args.force:
        print("\n[Stage 3] Loading cached Asymptotic Stress Tests...")
        df_stress = pd.read_csv(stress_path)
    else:
        print("\n[Stage 3] Executing 50-Year Asymptotic Stress Tests...")
        t0 = time.perf_counter()
        df_stress = run_long_term_asymptotic_stress_tests()
        df_stress.to_csv(stress_path, index=False)
        print(f"  Completed in {time.perf_counter() - t0:.2f}s. Saved stress tests.")
        
    # -------------------------------------------------------------------------
    # STAGE 4: Leave-Reservoir-Out Candidate Benchmark (Cases 1-70)
    # -------------------------------------------------------------------------
    case_bench_path = TABLE_DIR / "candidate_models_case_details.csv"
    summ_bench_path = TABLE_DIR / "candidate_models_pseudo_field_benchmark.csv"
    if case_bench_path.exists() and summ_bench_path.exists() and not args.force:
        print("\n[Stage 4] Loading cached Candidate Models Pseudo-Field Benchmark...")
        case_bench_df = pd.read_csv(case_bench_path)
        summary_bench_df = pd.read_csv(summ_bench_path)
    else:
        print("\n[Stage 4] Benchmarking Challenger Models vs Frozen Baseline (Origin 2005, 12Q)...")
        t0 = time.perf_counter()
        case_bench_df, summary_bench_df = evaluate_candidates_on_pseudo_fields(
            case_dict=case_dict,
            unc_df=unc_df,
            pseudo_fields=EVAL_PSEUDO_FIELDS,
            origin=pd.Timestamp("2005-01-01"),
            horizon_quarters=12,
        )
        case_bench_df.to_csv(case_bench_path, index=False)
        summary_bench_df.to_csv(summ_bench_path, index=False)
        print(f"  Completed in {time.perf_counter() - t0:.2f}s. Saved benchmark results.")
        
    # -------------------------------------------------------------------------
    # STAGE 5: Multi-Origin & Multi-Horizon Cross-Validation
    # -------------------------------------------------------------------------
    multi_orig_path = TABLE_DIR / "candidate_models_multi_origin_benchmark.csv"
    if multi_orig_path.exists() and not args.force:
        print("\n[Stage 5] Loading cached Multi-Origin Benchmark...")
        df_multi_orig = pd.read_csv(multi_orig_path)
    else:
        print("\n[Stage 5] Running Multi-Origin Benchmark (2003, 2004, 2005 x 4Q, 12Q, 20Q)...")
        t0 = time.perf_counter()
        df_multi_orig = run_multi_origin_candidate_benchmark(
            case_dict=case_dict, unc_df=unc_df, pseudo_fields=EVAL_PSEUDO_FIELDS
        )
        df_multi_orig.to_csv(multi_orig_path, index=False)
        print(f"  Completed in {time.perf_counter() - t0:.2f}s. Saved multi-origin benchmark.")
        
    # -------------------------------------------------------------------------
    # STAGE 6: January 2008 Forward Deployment on Field Case 0
    # -------------------------------------------------------------------------
    deploy_case_path = TABLE_DIR / "deployment_2008_scenarios_comparison.csv"
    deploy_summ_path = TABLE_DIR / "deployment_2008_summary_comparison.csv"
    if deploy_case_path.exists() and deploy_summ_path.exists() and not args.force:
        print("\n[Stage 6] Loading cached Deployment 2008 Comparison...")
        df_deploy_cases = pd.read_csv(deploy_case_path)
        df_deploy_summ = pd.read_csv(deploy_summ_path)
    else:
        print("\n[Stage 6] Running 20-Year (2008-2028) Deployment Comparison on Field Case 0...")
        t0 = time.perf_counter()
        df_deploy_cases, df_deploy_summ, deploy_findings = run_deployment_2008_comparison(
            case_dict=case_dict, unc_df=unc_df, inference_origin=pd.Timestamp("2008-01-01"), horizon_quarters=80
        )
        df_deploy_cases.to_csv(deploy_case_path, index=False)
        df_deploy_summ.to_csv(deploy_summ_path, index=False)
        print(f"  Completed in {time.perf_counter() - t0:.2f}s. Saved deployment comparisons.")
        
    # -------------------------------------------------------------------------
    # STAGE 7: Render Publication Figures
    # -------------------------------------------------------------------------
    print("\n[Stage 7] Rendering 5 Publication Figures...")
    f1_path = FIG_DIR / "fig1_multiphase_physics_formulation.png"
    f2_path = FIG_DIR / "fig2_base_models_disagreement_expansion.png"
    f3_path = FIG_DIR / "fig3_candidate_models_pseudo_field_benchmark.png"
    f4_path = FIG_DIR / "fig4_multi_origin_robustness_benchmark.png"
    f5_path = FIG_DIR / "fig5_deployment_2008_comparison_trajectories.png"
    
    plot_figure_1_hypotheses_concept(f1_path)
    plot_figure_2_disagreement_expansion(df_traj, df_milestones, f2_path)
    plot_figure_3_candidate_benchmark(summary_bench_df, f3_path)
    plot_figure_4_multi_origin_robustness(df_multi_orig, f4_path)
    plot_figure_5_deployment_trajectories(df_deploy_cases, df_deploy_summ, f5_path)
    
    # Sync figures to BRAIN_DIR for report embedding
    for f in [f1_path, f2_path, f3_path, f4_path, f5_path]:
        shutil.copy2(f, BRAIN_DIR / f.name)
    print("  All 5 figures rendered and synced to artifact brain directory.")
    
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f"ALL LONG-HORIZON EXPERIMENTS COMPLETED IN {elapsed:.2f} SECONDS")
    print("=" * 80)


if __name__ == "__main__":
    main()
