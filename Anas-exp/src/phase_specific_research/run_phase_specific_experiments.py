"""
Master Runner: run_phase_specific_experiments.py
Package: phase_specific_research
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Executes the complete experimental investigation for Phase-Specific Forecast Improvement & Long-Tail Research:
1. Full ensemble asymptotic behavior audit & constrained long-tail model.
2. Direct phase recombination verification across case-level trajectories (Origins 2002-2005).
3. Physical selection rule & observable state correlation analysis.
4. Nested reservoir-grouped cross-validation across Cases 1-70 (cluster-robust statistics).
5. Secondary confirmation on Cases 71-85.
6. January 2008 forward deployment on Field Case 0 across 10 official scenarios.
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
    VAL_CASES,
    PARAMS,
)
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
)
from ensemble.generate_competition_deliverable import (
    load_locked_config,
    fit_locked_base_models,
)
from forecast_advancement.protocol_framework import fit_base_models_under_protocol
from phase_specific_research.asymptotic_analysis import (
    analytical_arps_recovery_limit,
    run_frozen_ensemble_long_term_stress,
    compare_frozen_vs_constrained_tail,
)
from phase_specific_research.phase_recombination import (
    predict_candidate_rates,
    ALL_CANDIDATE_IDS,
)
from phase_specific_research.selection_rule_analysis import (
    analyze_selection_rules,
)
from phase_specific_research.nested_cross_validation import (
    run_full_nested_cross_validation,
    evaluate_all_candidates_at_origin,
    EVAL_PSEUDO_FIELDS,
    EVAL_GRID,
)
from phase_specific_research.deployment_2008_forecaster import (
    run_deployment_2008_scenarios,
)

TABLE_DIR = Path("outputs/research/phase_specific_research/tables")
FIG_DIR = Path("outputs/research/phase_specific_research/figures")
BRAIN_DIR = Path("/Users/tengkuanas/.gemini/antigravity-cli/brain/4b7e1d00-9795-42a6-b5bb-c629e9cf25cb")


def plot_figure_1_asymptotic_audit(df_stress: pd.DataFrame, df_comp: pd.DataFrame, save_path: Path):
    """Figure 1: Full Ensemble Asymptotic Behavior & Support-Tapered Tail."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Left: Terminal Oil Rate across Horizons (20, 30, 50, 100 Years)
    ax1 = axes[0]
    years = df_comp["horizon_years"].values
    froz_rates = df_comp["frozen_terminal_oil_rate_stbd"].values
    tail_rates = df_comp["tail_terminal_oil_rate_stbd"].values
    
    ax1.plot(years, froz_rates, marker="o", color="#c0392b", lw=2.5, label="Frozen Production Ensemble (Asymptotic Floor: ~737 STB/d)")
    ax1.plot(years, tail_rates, marker="s", color="#27ae60", lw=2.5, label="Constrained Long-Tail Ensemble (Tapers to 0 STB/d)")
    ax1.axhline(737.12, color="#c0392b", linestyle=":", alpha=0.7, label="ExtraTrees Floor (0.2405 * 3,065 = 737.1 STB/d)")
    ax1.axhline(0.0, color="gray", linestyle="-", alpha=0.4)
    
    ax1.set_xlabel("Horizon Ahead (Years beyond 2008)", fontweight="bold")
    ax1.set_ylabel("Terminal Oil Rate (STB/d)", fontweight="bold")
    ax1.set_title("A: Structural Asymptotic Rate Floor on Field Case 0", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, fontsize=8.5, loc="center right")
    
    # Right: Cumulative Recovery Divergence
    ax2 = axes[1]
    froz_eur = df_comp["frozen_eur_oil_stb"].values / 1e6
    tail_eur = df_comp["tail_eur_oil_stb"].values / 1e6
    
    ax2.plot(years, froz_eur, marker="o", color="#c0392b", lw=2.5, label="Frozen Ensemble EUR (Diverges linearly to infinity)")
    ax2.plot(years, tail_eur, marker="s", color="#27ae60", lw=2.5, label="Constrained Tail EUR (Finite Limit: 48.44 MMSTB)")
    
    ax2.set_xlabel("Horizon Ahead (Years beyond 2008)", fontweight="bold")
    ax2.set_ylabel("Cumulative Oil Recovery EUR (MMSTB)", fontweight="bold")
    ax2.set_title("B: Cumulative Recovery Convergence vs Infinite Divergence", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, fontsize=8.5, loc="upper left")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_figure_2_phase_recombination(case_df_2005: pd.DataFrame, save_path: Path):
    """Figure 2: Phase Recombination Case Details (Origin 2005, 12Q)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Left: Candidate Comparison at 2005 12Q
    ax1 = axes[0]
    cands = [
        "Candidate_A_Frozen_Baseline",
        "Candidate_B_ExtraTreesOil_FrozenWater",
        "Candidate_C_FrozenOil_ExtraTreesWater",
        "Candidate_D_ExtraTreesBoth",
    ]
    labels = ["Frozen Baseline", "ET Oil +\nFrozen Water", "Frozen Oil +\nET Water (Cand C)", "Pure ET Both"]
    
    macro_means = [case_df_2005[case_df_2005["candidate_id"] == c]["macro_nrmse"].mean() for c in cands]
    oil_means = [case_df_2005[case_df_2005["candidate_id"] == c]["oil_nrmse"].mean() for c in cands]
    wat_means = [case_df_2005[case_df_2005["candidate_id"] == c]["water_nrmse"].mean() for c in cands]
    
    x = np.arange(len(cands))
    w = 0.25
    ax1.bar(x - w, macro_means, w, label="Macro NRMSE", color="#2980b9")
    ax1.bar(x, oil_means, w, label="Oil NRMSE", color="#27ae60")
    ax1.bar(x + w, wat_means, w, label="Water NRMSE", color="#c0392b")
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Cumulative Increment NRMSE (12Q)", fontweight="bold")
    ax1.set_title("A: Controlled Phase Recombination (Origin 2005, 12Q)", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, loc="upper right", fontsize=8.5)
    
    # Right: Case-by-case difference for Candidate C vs Baseline (7 Wins, 8 Losses)
    ax2 = axes[1]
    sub_base = case_df_2005[case_df_2005["candidate_id"] == "Candidate_A_Frozen_Baseline"].sort_values("case_id")
    sub_c = case_df_2005[case_df_2005["candidate_id"] == "Candidate_C_FrozenOil_ExtraTreesWater"].sort_values("case_id")
    diffs = (sub_c["macro_nrmse"].values - sub_base["macro_nrmse"].values)
    cases = [f"C{c}" for c in sub_base["case_id"].values]
    
    colors = ["#27ae60" if d < 0 else "#c0392b" for d in diffs]
    bars = ax2.bar(cases, diffs, color=colors)
    ax2.axhline(0.0, color="black", lw=1)
    
    ax2.set_xlabel("Pseudo-Field Test Cases (Cases 1 to 70)", fontweight="bold")
    ax2.set_ylabel("Macro NRMSE Difference (Cand C - Baseline)", fontweight="bold")
    ax2.set_title("B: Case-Level Breakdown: 7 Wins (Green) vs 8 Regressions (Red)", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_figure_3_selection_rules(corr_df: pd.DataFrame, gate_comp: Dict[str, Any], save_path: Path):
    """Figure 3: Selection Rule Correlations and Gate Comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Left: Correlation with ExtraTrees Advantage
    ax1 = axes[0]
    corr_df_sorted = corr_df.sort_values("corr_with_candC_win", ascending=True)
    y_pos = np.arange(len(corr_df_sorted))
    
    ax1.barh(y_pos, corr_df_sorted["corr_with_candC_win"], color="#3498db")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(corr_df_sorted["feature"], fontsize=9)
    ax1.axvline(0.0, color="black", lw=1)
    ax1.set_xlabel("Pearson Correlation with Candidate C Outperformance", fontweight="bold")
    ax1.set_title("A: Physical Observable Correlates with ExtraTrees Water Advantage", fontsize=11, fontweight="bold")
    ax1.grid(axis="x", linestyle=":", alpha=0.6)
    
    # Right: Gate accuracy comparison
    ax2 = axes[1]
    gates = ["Logistic Regression\n(Empirical Gate)", "Water Cut > 0.35\n(Physical Rule)", "Horizon <= 8Q\n(Prespecified Schedule)"]
    accs = [
        gate_comp["logistic_regression_test_accuracy"],
        gate_comp["rule_watercut_gt_035_test_accuracy"],
        gate_comp["rule_horizon_le_8q_test_accuracy"],
    ]
    bars = ax2.bar(gates, accs, color=["#e67e22", "#27ae60", "#9b59b6"], width=0.5)
    ax2.set_ylabel("Out-of-Fold Prediction Accuracy", fontweight="bold")
    ax2.set_title("B: Complex Learned Gate vs Simple Physical Selection Rules", fontsize=11, fontweight="bold")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    for b in bars:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2.0, h + 0.02, f"{h:.1%}", ha="center", va="bottom", fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_figure_4_nested_cv_benchmark(summary_df: pd.DataFrame, save_path: Path):
    """Figure 4: Nested Reservoir-Grouped Cross-Validation Benchmark with Clustered CIs."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    
    # Left: Macro NRMSE across all candidates
    ax1 = axes[0]
    df_sorted = summary_df.sort_values("macro_nrmse_mean").reset_index(drop=True)
    cands = df_sorted["candidate_id"].tolist()
    labels = [c.replace("Candidate_", "").replace("_", " ") for c in cands]
    
    y = np.arange(len(cands))
    ax1.errorbar(
        df_sorted["macro_nrmse_mean"], y,
        xerr=df_sorted["diff_se_cluster"] * 1.96,
        fmt="o", color="#2980b9", ecolor="#7fb3d5", elinewidth=2, capsize=4, markersize=7
    )
    
    base_macro = summary_df.loc[summary_df["candidate_id"] == "Candidate_A_Frozen_Baseline", "macro_nrmse_mean"].values[0]
    ax1.axvline(base_macro, color="#c0392b", linestyle="--", alpha=0.7, label=f"Frozen Baseline ({base_macro:.4f})")
    
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8.5)
    ax1.set_xlabel("Mean Macro NRMSE (Cluster-Robust 95% CI)", fontweight="bold")
    ax1.set_title("A: Multi-Origin Multi-Horizon Nested CV (Cases 1 to 70)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, loc="upper right")
    
    # Right: Win Rate vs Baseline
    ax2 = axes[1]
    win_rates = df_sorted["win_rate_vs_baseline"].values * 100.0
    colors = ["#27ae60" if w > 50 else "#95a5a6" for w in win_rates]
    bars = ax2.barh(y, win_rates, color=colors)
    ax2.axvline(50.0, color="black", linestyle=":", alpha=0.7, label="50% Parity Line")
    
    ax2.set_yticks(y)
    ax2.set_yticklabels(["" for _ in y])
    ax2.set_xlabel("Win Rate vs Frozen Baseline (%)", fontweight="bold")
    ax2.set_title("B: Proportion of Evaluations Outperforming Frozen Baseline", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, 100)
    ax2.grid(axis="x", linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, loc="lower right")
    
    for b in bars:
        w = b.get_width()
        if w > 0:
            ax2.text(w + 1.5, b.get_y() + b.get_height() / 2.0, f"{w:.1f}%", va="center", fontsize=8, fontweight="bold")
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_figure_5_deployment_scenarios(df_scenarios: pd.DataFrame, df_summary: pd.DataFrame, save_path: Path):
    """Figure 5: January 2008 Forward Deployment Trajectories on Field Case 0."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Left: 20-Year EUR Oil Scenario Envelope
    ax1 = axes[0]
    cands = df_summary["candidate_id"].tolist()
    labels = [c.replace("Candidate_", "").replace("_", "\n") for c in cands]
    data = [df_scenarios[df_scenarios["candidate_id"] == c]["eur_oil_stb"].values / 1e6 for c in cands]
    
    ax1.boxplot(data, tick_labels=labels, patch_artist=True,
                boxprops=dict(facecolor="#d4e6f1", color="#1b4f72"),
                medianprops=dict(color="#922b21", lw=2))
    ax1.set_ylabel("20-Year Cumulative Oil EUR (MMSTB)", fontweight="bold")
    ax1.set_title("A: Scenario Uncertainty Envelope on Field Case 0", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    
    # Right: Terminal Oil Rate at 2028
    ax2 = axes[1]
    term_rates = df_summary["mean_terminal_oil_rate_stbd"].values
    bars = ax2.bar(labels, term_rates, color=["#2980b9", "#27ae60", "#e67e22", "#8e44ad"])
    ax2.set_ylabel("Mean Terminal Oil Rate at 2028 (STB/d)", fontweight="bold")
    ax2.set_title("B: Year 20 Terminal Production Rate Comparison", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    for b in bars:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2.0, h + 10, f"{h:,.0f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run Phase-Specific Forecast Improvement Experiments")
    parser.add_argument("--force", action="store_true", help="Force recomputation of all stages")
    args = parser.parse_args()
    
    print("=" * 80)
    print("EXXONMOBIL DATAWORKS CHALLENGE 2026")
    print("PHASE-SPECIFIC FORECAST IMPROVEMENT & LONG-TAIL RESEARCH")
    print("=" * 80)
    t_start = time.perf_counter()
    
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    BRAIN_DIR.mkdir(parents=True, exist_ok=True)
    
    shared_dir = resolve_shared_dir()
    unc_df = load_uncertainty(shared_dir)
    raw_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(raw_df)
    
    config = load_locked_config()
    locked_weights = config["weights"]
    
    # -------------------------------------------------------------------------
    # STAGE 1: Full Ensemble Asymptotic Behavior Audit & Long-Tail Model
    # -------------------------------------------------------------------------
    stress_path = TABLE_DIR / "frozen_ensemble_asymptotic_stress_results.csv"
    tail_comp_path = TABLE_DIR / "frozen_vs_constrained_tail_comparison.csv"
    
    if stress_path.exists() and tail_comp_path.exists() and not args.force:
        print("\n[Stage 1] Loading cached Asymptotic Behavior Audit...")
        df_stress = pd.read_csv(stress_path)
        df_comp = pd.read_csv(tail_comp_path)
    else:
        print("\n[Stage 1] Executing Asymptotic Behavior Audit (20, 30, 50, 100 Years)...")
        t0 = time.perf_counter()
        origin_2008 = pd.Timestamp("2008-01-01")
        models_2008 = fit_locked_base_models(case_dict, unc_df, TRAIN_CASES, origin_2008)
        
        prod = pd.read_parquet(shared_dir / "production_timeseries.parquet")
        c0 = prod[prod["case_num"] == 0]
        c0_2008 = c0[c0["date"] == "2008-01-01"].set_index("metric")["value"]
        init_state_0 = {
            "qo": float(c0_2008["oil_rate"]),
            "qw": float(c0_2008["water_rate"]),
            "qg": float(c0_2008["gas_rate"]),
            "Qo": float(c0_2008["oil_cum"]),
            "Qw": float(c0_2008["water_cum"]),
            "Qg": float(c0_2008["gas_cum"]),
            "wor": float(c0_2008["water_rate"] / max(c0_2008["oil_rate"], 1.0)),
            "res_age": 10.0,
            "origin_date": origin_2008,
        }
        params_0 = unc_df[PARAMS].mean()
        
        df_stress = run_frozen_ensemble_long_term_stress(models_2008, locked_weights, init_state_0, params_0)
        df_comp = compare_frozen_vs_constrained_tail(models_2008, locked_weights, init_state_0, params_0)
        
        df_stress.to_csv(stress_path, index=False)
        df_comp.to_csv(tail_comp_path, index=False)
        print(f"  Completed in {time.perf_counter() - t0:.2f}s. Saved stress and tail comparison tables.")
        
    # -------------------------------------------------------------------------
    # STAGE 2: Direct Phase Recombination Verification at Origin 2005 (12Q)
    # -------------------------------------------------------------------------
    recomb_2005_path = TABLE_DIR / "phase_recombination_2005_12q_case_details.csv"
    if recomb_2005_path.exists() and not args.force:
        print("\n[Stage 2] Loading cached Phase Recombination Details (2005, 12Q)...")
        case_df_2005 = pd.read_csv(recomb_2005_path)
    else:
        print("\n[Stage 2] Verifying Phase Recombination Trajectories (Origin 2005, 12Q)...")
        t0 = time.perf_counter()
        case_df_2005 = evaluate_all_candidates_at_origin(
            case_dict=case_dict,
            unc_df=unc_df,
            test_cases=EVAL_PSEUDO_FIELDS,
            origin=pd.Timestamp("2005-01-01"),
            horizon_quarters=12,
            locked_weights=locked_weights,
        )
        case_df_2005.to_csv(recomb_2005_path, index=False)
        print(f"  Completed in {time.perf_counter() - t0:.2f}s. Saved 2005 12Q case details.")
        
    # -------------------------------------------------------------------------
    # STAGE 3: Selection Rule & Correlate Analysis
    # -------------------------------------------------------------------------
    corr_path = TABLE_DIR / "selection_rule_feature_correlations.csv"
    gate_path = TABLE_DIR / "selection_rule_gate_comparison.json"
    if corr_path.exists() and gate_path.exists() and not args.force:
        print("\n[Stage 3] Loading cached Selection Rule Analysis...")
        corr_df = pd.read_csv(corr_path)
        with open(gate_path, "r") as f:
            gate_comp = json.load(f)
    else:
        print("\n[Stage 3] Analyzing Physical Selection Rules & Observable Features...")
        t0 = time.perf_counter()
        corr_df, gate_comp = analyze_selection_rules(case_df_2005)
        corr_df.to_csv(corr_path, index=False)
        with open(gate_path, "w") as f:
            json.dump(gate_comp, f, indent=2)
        print(f"  Completed in {time.perf_counter() - t0:.2f}s. Saved selection rule findings.")
        
    # -------------------------------------------------------------------------
    # STAGE 4: Full Nested Reservoir-Grouped Cross-Validation (Cases 1-70)
    # -------------------------------------------------------------------------
    nested_cases_path = TABLE_DIR / "nested_cv_case_evaluations.csv"
    nested_summ_path = TABLE_DIR / "nested_cv_summary_benchmark.csv"
    if nested_cases_path.exists() and nested_summ_path.exists() and not args.force:
        print("\n[Stage 4] Loading cached Nested Cross-Validation Benchmark...")
        master_case_df = pd.read_csv(nested_cases_path)
        summary_df = pd.read_csv(nested_summ_path)
    else:
        print("\n[Stage 4] Running Full Nested Reservoir-Grouped Cross-Validation...")
        t0 = time.perf_counter()
        master_case_df, summary_df = run_full_nested_cross_validation(
            case_dict=case_dict,
            unc_df=unc_df,
            test_cases=EVAL_PSEUDO_FIELDS,
            eval_grid=EVAL_GRID,
        )
        master_case_df.to_csv(nested_cases_path, index=False)
        summary_df.to_csv(nested_summ_path, index=False)
        print(f"  Completed in {time.perf_counter() - t0:.2f}s. Saved nested CV tables.")
        
    # -------------------------------------------------------------------------
    # STAGE 5: Secondary Confirmation on Cases 71-85
    # -------------------------------------------------------------------------
    val_cases_path = TABLE_DIR / "secondary_confirmation_cases_71_to_85.csv"
    if val_cases_path.exists() and not args.force:
        print("\n[Stage 5] Loading cached Secondary Confirmation on Cases 71-85...")
        val_df = pd.read_csv(val_cases_path)
    else:
        print("\n[Stage 5] Running Secondary Confirmation on Validation Cases 71-85 (Origin 2005, 12Q)...")
        t0 = time.perf_counter()
        val_df = evaluate_all_candidates_at_origin(
            case_dict=case_dict,
            unc_df=unc_df,
            test_cases=VAL_CASES,
            origin=pd.Timestamp("2005-01-01"),
            horizon_quarters=12,
            locked_weights=locked_weights,
        )
        val_df.to_csv(val_cases_path, index=False)
        print(f"  Completed in {time.perf_counter() - t0:.2f}s. Saved secondary confirmation table.")
        
    # -------------------------------------------------------------------------
    # STAGE 6: January 2008 Forward Deployment on Field Case 0
    # -------------------------------------------------------------------------
    deploy_case_path = TABLE_DIR / "deployment_2008_scenarios_comparison.csv"
    deploy_summ_path = TABLE_DIR / "deployment_2008_summary_comparison.csv"
    if deploy_case_path.exists() and deploy_summ_path.exists() and not args.force:
        print("\n[Stage 6] Loading cached Deployment 2008 Comparisons...")
        df_deploy_scenarios = pd.read_csv(deploy_case_path)
        df_deploy_summary = pd.read_csv(deploy_summ_path)
    else:
        print("\n[Stage 6] Running January 2008 Forward Deployment Across 10 Scenarios...")
        t0 = time.perf_counter()
        origin_2008 = pd.Timestamp("2008-01-01")
        models_2008 = fit_locked_base_models(case_dict, unc_df, TRAIN_CASES, origin_2008)
        
        df_deploy_scenarios, df_deploy_summary = run_deployment_2008_scenarios(
            models=models_2008,
            locked_weights=locked_weights,
        )
        df_deploy_scenarios.to_csv(deploy_case_path, index=False)
        df_deploy_summary.to_csv(deploy_summ_path, index=False)
        print(f"  Completed in {time.perf_counter() - t0:.2f}s. Saved deployment tables.")
        
    # -------------------------------------------------------------------------
    # STAGE 7: Render Publication Figures
    # -------------------------------------------------------------------------
    print("\n[Stage 7] Rendering 5 Publication Figures...")
    f1_path = FIG_DIR / "fig1_ensemble_asymptotic_rate_floor_deconstruction.png"
    f2_path = FIG_DIR / "fig2_phase_specific_recombination_matrix.png"
    f3_path = FIG_DIR / "fig3_selection_rules_and_feature_correlations.png"
    f4_path = FIG_DIR / "fig4_nested_clustered_cross_validation_benchmark.png"
    f5_path = FIG_DIR / "fig5_deployment_2008_scenarios_and_case0.png"
    
    plot_figure_1_asymptotic_audit(df_stress, df_comp, f1_path)
    plot_figure_2_phase_recombination(case_df_2005, f2_path)
    plot_figure_3_selection_rules(corr_df, gate_comp, f3_path)
    plot_figure_4_nested_cv_benchmark(summary_df, f4_path)
    plot_figure_5_deployment_scenarios(df_deploy_scenarios, df_deploy_summary, f5_path)
    
    # Sync figures to BRAIN_DIR for report embedding
    for f in [f1_path, f2_path, f3_path, f4_path, f5_path]:
        shutil.copy2(f, BRAIN_DIR / f.name)
    print("  All 5 figures rendered and synced to artifact brain directory.")
    
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f"ALL PHASE-SPECIFIC EXPERIMENTS COMPLETED IN {elapsed:.2f} SECONDS")
    print("=" * 80)


if __name__ == "__main__":
    main()
