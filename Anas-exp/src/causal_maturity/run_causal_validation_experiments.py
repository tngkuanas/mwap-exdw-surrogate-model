"""
Master Runner: run_causal_validation_experiments.py
Package: causal_maturity
Addresses: All Sections 1-10 — Complete Empirical Benchmark & Diagnostic Figures
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import time
import shutil
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ensemble.data import resolve_shared_dir, load_uncertainty, TRAIN_CASES, VAL_CASES
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
)
from causal_maturity.sustained_decline_detector import (
    SustainedDeclineDetector,
    extract_robust_case_maturity,
)
from causal_maturity.exact_time_varying_integration import (
    ExactDeclineIntegrator,
    ExactTimeVaryingDeclineForecaster,
    evaluate_raw_decline,
)
from causal_maturity.dual_protocol_evaluator import (
    run_cross_fitted_gate_benchmark,
)
from causal_maturity.deployment_2008_evaluator import (
    run_january_2008_research_deployment,
)

TABLE_DIR = Path("outputs/research/maturity_causal_validation/tables")
FIG_DIR = Path("outputs/research/maturity_causal_validation/figures")
REPRO_DIR = Path("outputs/research/maturity_causal_validation/reproducibility")
BRAIN_DIR = Path("/Users/tengkuanas/.gemini/antigravity-cli/brain/4b7e1d00-9795-42a6-b5bb-c629e9cf25cb")

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPRO_DIR.mkdir(parents=True, exist_ok=True)


def generate_baseline_reconciliation_table() -> pd.DataFrame:
    """Reconciles the exact sources of variation between earlier reported baseline scores."""
    records = [
        {
            "origin": "2004-01-01",
            "horizon_quarters": 16,
            "training_protocol": "Protocol A (Strict As-Of)",
            "dca_training_method": "Historical Fallback (horizon_quarters=0)",
            "base_models_training_horizon": "Pre-origin data strictly <= 2004-01-01",
            "macro_nrmse": 0.039083,
            "oil_nrmse": 0.039457,
            "water_nrmse": 0.038709,
            "historical_source": "DEPLOYMENT_FAITHFUL_VALIDATION_REPORT (Model E_asof)",
        },
        {
            "origin": "2004-01-01",
            "horizon_quarters": 16,
            "training_protocol": "Protocol B (Forward Supervised)",
            "dca_training_method": "Forward-Supervised Targets (horizon_quarters=80)",
            "base_models_training_horizon": "Training cases post-origin data through 2008",
            "macro_nrmse": 0.053032,
            "oil_nrmse": 0.062834,
            "water_nrmse": 0.043229,
            "historical_source": "PRODUCTION_ANCHORED_HYBRID_RESEARCH_REPORT (Model A)",
        },
        {
            "origin": "2005-01-01",
            "horizon_quarters": 12,
            "training_protocol": "Protocol A (Strict As-Of)",
            "dca_training_method": "Historical Fallback (horizon_quarters=0)",
            "base_models_training_horizon": "Pre-origin data strictly <= 2005-01-01",
            "macro_nrmse": 0.038780,
            "oil_nrmse": 0.036802,
            "water_nrmse": 0.040759,
            "historical_source": "DEPLOYMENT_FAITHFUL_VALIDATION_REPORT (Model E_asof)",
        },
        {
            "origin": "2005-01-01",
            "horizon_quarters": 12,
            "training_protocol": "Protocol B (Forward Supervised)",
            "dca_training_method": "Forward-Supervised Targets (horizon_quarters=80)",
            "base_models_training_horizon": "Training cases post-origin data through 2008",
            "macro_nrmse": 0.027557,
            "oil_nrmse": 0.050607,
            "water_nrmse": 0.004507,
            "historical_source": "PRODUCTION_ANCHORED_HYBRID_RESEARCH_REPORT (Model A)",
        },
    ]
    df_reconcile = pd.DataFrame(records)
    out_p = TABLE_DIR / "reconciled_baseline_evaluation.csv"
    df_reconcile.to_csv(out_p, index=False)
    print(f"Saved Reconciled Baseline Scores: {out_p}")
    return df_reconcile


def generate_figure1_sustained_decline_diagnostics(c0_maturity: Dict[str, Any], case_dict: Dict[int, pd.DataFrame]):
    """Figure 1: Robust sustained decline detector vs flawed 92% peak trigger."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left: Case 0 history with plateau, threshold, and true decline onset
    c0 = case_dict[1] # Use clean trajectory
    shared_dir = resolve_shared_dir()
    prod = pd.read_parquet(shared_dir / "production_timeseries.parquet")
    df0 = prod[prod["case_num"] == 0]
    piv0 = df0.pivot(index="date", columns="metric", values="value").ffill()
    piv0.index = pd.to_datetime(piv0.index)
    q_dates = pd.date_range("1998-01-01", "2008-01-01", freq="QS")
    q_piv0 = piv0.loc[piv0.index.isin(q_dates)]
    
    dates = q_piv0.index
    qo = q_piv0["oil_rate"].values
    
    ax1.plot(dates, qo, color="#1565c0", lw=2.5, label="Actual Oil Rate (Case 0)")
    ax1.axhline(c0_maturity["plateau_rate"], color="#2e7d32", ls="--", lw=1.8, label=f"Plateau Rate ({c0_maturity['plateau_rate']:,.0f} STB/d)")
    ax1.axhline(c0_maturity["plateau_rate"] * 0.92, color="#f57c00", ls=":", lw=1.5, label="92% Plateau Boundary")
    ax1.axhline(c0_maturity["plateau_rate"] * 0.90, color="#d32f2f", ls=":", lw=1.5, label="90% Decline Threshold")
    
    # Mark flawed trigger (startup 1998)
    ax1.scatter([dates[0]], [qo[0]], color="#c2185b", s=120, zorder=5, label="Flawed Iteration 10 Trigger (Startup t=0)")
    # Mark true onset
    onset_d = c0_maturity["decline_onset_date"]
    onset_val = qo[np.where(dates == onset_d)[0][0]]
    ax1.scatter([onset_d], [onset_val], color="#2e7d32", s=140, marker="*", zorder=6, label=f"True Decline Onset ({str(onset_d)[:10]})")
    
    ax1.axvspan(dates[0], pd.Timestamp("1998-10-01"), color="#e1bee7", alpha=0.3, label="Startup Ramp-up (Filtered)")
    ax1.axvspan(c0_maturity["plateau_start_date"], c0_maturity["plateau_end_date"], color="#c8e6c9", alpha=0.3, label="Plateau Window")
    ax1.axvspan(c0_maturity["decline_onset_date"], pd.Timestamp("2008-01-01"), color="#ffcdd2", alpha=0.3, label="True Sustained Decline")
    
    ax1.set_title("Field Case 0: Correct Physical Decline Detection vs Flawed Trigger", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Oil Rate (STB/d)")
    ax1.legend(loc="lower left", fontsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.4)
    
    # Right: Case-to-case variation in decline onset and decline depth across 70 training cases at 2004
    detector = SustainedDeclineDetector()
    depths = []
    q_declines = []
    cases_samp = list(range(1, 71))
    for c in cases_samp:
        df_c = case_dict[c]
        mat = extract_robust_case_maturity(df_c, pd.Timestamp("2004-01-01"))
        depths.append(mat["decline_depth"])
        q_declines.append(mat["quarters_in_decline"])
        
    ax2.scatter(depths, q_declines, color="#4527a0", alpha=0.75, s=60, edgecolors="none")
    ax2.set_title("Genuinely Varying Physical Maturity Across 70 Reservoirs at 2004-01-01", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Decline Depth (Fraction of Plateau Lost)")
    ax2.set_ylabel("Quarters in Sustained Decline")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.text(0.05, 0.90, f"Decline Depth: [{min(depths):.2f}, {max(depths):.2f}]\nQuarters in Decline: [{min(q_declines)}, {max(q_declines)}]\n(Not an origin-constant!)", 
             transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=9)
             
    plt.tight_layout()
    p = FIG_DIR / "fig1_sustained_decline_detector_diagnostics.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print(f"Generated Figure 1: {p}")


def generate_figure2_protocol_benchmark(df_sum_all: pd.DataFrame):
    """Figure 2: Protocol A vs Protocol B benchmark across candidate models."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    df_pB = df_sum_all[df_sum_all["protocol"] == "Protocol_B"].copy()
    models = [
        "Model_A_Frozen_Ensemble",
        "Model_B_Corrected_GP_Residual",
        "Fixed_Weight_Gate",
        "Calendar_Age_Gate",
        "Physical_Maturity_Gate",
        "Oil_Only_Maturity_Gate",
        "Model_F_Exact_Time_Varying_DCA",
    ]
    
    df_filt = df_pB[df_pB["model_name"].isin(models)].copy()
    origins = sorted(df_filt["origin"].unique())
    x = np.arange(len(models))
    width = 0.20
    
    colors = ["#1b5e20", "#1565c0", "#e65100", "#6a1b9a"]
    
    for i, orig in enumerate(origins):
        sub = df_filt[df_filt["origin"] == orig]
        scores = [sub[sub["model_name"] == m]["macro_nrmse_mean"].values[0] if len(sub[sub["model_name"] == m]) > 0 else np.nan for m in models]
        ax.bar(x + (i - len(origins)/2.0 + 0.5) * width, scores, width=width, label=f"Origin {orig}", color=colors[i % len(colors)], alpha=0.85)
        
    ax.set_xticks(x)
    clean_labels = [m.replace("_", " ").replace("Model ", "").replace("Gate", "") for m in models]
    ax.set_xticklabels(clean_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Macro Increment NRMSE")
    ax.set_title("Cross-Fitted Model Comparison Across Multiple Origins (Protocol B)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    
    plt.tight_layout()
    p = FIG_DIR / "fig2_protocol_a_vs_protocol_b_benchmark.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print(f"Generated Figure 2: {p}")


def generate_figure3_maturity_vs_calendar(df_case_level: pd.DataFrame):
    """Figure 3: Head-to-Head: Physical Maturity Gate vs Simple Calendar Age Gate."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    c_2004 = df_case_level[(df_case_level["origin"] == "2004-01-01") & (df_case_level["protocol"] == "Protocol_B")]
    cal_2004 = c_2004[c_2004["model_name"] == "Calendar_Age_Gate"].sort_values("case_id")["macro_nrmse"].values
    mat_2004 = c_2004[c_2004["model_name"] == "Physical_Maturity_Gate"].sort_values("case_id")["macro_nrmse"].values
    base_2004 = c_2004[c_2004["model_name"] == "Model_A_Frozen_Ensemble"].sort_values("case_id")["macro_nrmse"].values
    
    # Origin 2004 scatter
    ax1.scatter(cal_2004, mat_2004, color="#1565c0", s=80, alpha=0.8, edgecolors="k", label="Holdout Cases (71-85)")
    lims1 = [min(min(cal_2004), min(mat_2004))*0.9, max(max(cal_2004), max(mat_2004))*1.1]
    ax1.plot(lims1, lims1, color="red", ls="--", label="1:1 Parity Line")
    ax1.set_xlabel("Calendar Age Gate Macro NRMSE")
    ax1.set_ylabel("Physical Maturity Gate Macro NRMSE")
    ax1.set_title("Origin 2004 (16Q): Physical Maturity vs Calendar Age Gate", fontsize=10, fontweight="bold")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)
    diff_2004 = mat_2004 - cal_2004
    win_2004 = np.mean(mat_2004 < cal_2004) * 100
    ax1.text(0.05, 0.85, f"Mean Diff: {np.mean(diff_2004):+.5f}\nWin Rate: {win_2004:.1f}%", transform=ax1.transAxes,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=9)
             
    # Origin 2005 scatter
    c_2005 = df_case_level[(df_case_level["origin"] == "2005-01-01") & (df_case_level["protocol"] == "Protocol_B")]
    cal_2005 = c_2005[c_2005["model_name"] == "Calendar_Age_Gate"].sort_values("case_id")["macro_nrmse"].values
    mat_2005 = c_2005[c_2005["model_name"] == "Physical_Maturity_Gate"].sort_values("case_id")["macro_nrmse"].values
    
    ax2.scatter(cal_2005, mat_2005, color="#2e7d32", s=80, alpha=0.8, edgecolors="k", label="Holdout Cases (71-85)")
    lims2 = [min(min(cal_2005), min(mat_2005))*0.9, max(max(cal_2005), max(mat_2005))*1.1]
    ax2.plot(lims2, lims2, color="red", ls="--", label="1:1 Parity Line")
    ax2.set_xlabel("Calendar Age Gate Macro NRMSE")
    ax2.set_ylabel("Physical Maturity Gate Macro NRMSE")
    ax2.set_title("Origin 2005 (12Q): Physical Maturity vs Calendar Age Gate", fontsize=10, fontweight="bold")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)
    diff_2005 = mat_2005 - cal_2005
    win_2005 = np.mean(mat_2005 < cal_2005) * 100
    ax2.text(0.05, 0.85, f"Mean Diff: {np.mean(diff_2005):+.5f}\nWin Rate: {win_2005:.1f}%", transform=ax2.transAxes,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=9)
             
    plt.tight_layout()
    p = FIG_DIR / "fig3_physical_maturity_vs_calendar_age_gate.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print(f"Generated Figure 3: {p}")


def generate_figure4_exact_integration_diagnostics():
    """Figure 4: Exact numerical quadrature vs flawed approximation with floor crossings."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    d_min = 0.03
    d_base = 0.015
    decay_rates = np.array([0.4, 1.2, 3.0])
    c = np.array([0.06, -0.10, 0.05]) # Crosses d_min floor twice
    
    integrator = ExactDeclineIntegrator(d_base, decay_rates, c, d_min)
    t_years = np.linspace(0, 10, 501)
    
    raw_D = evaluate_raw_decline(t_years, d_base, decay_rates, c)
    integrand_D = np.maximum(d_min, raw_D)
    
    ax1.plot(t_years, raw_D, color="#e65100", ls="--", lw=2.0, label="Raw Decline D_raw(t)")
    ax1.axhline(d_min, color="red", ls=":", lw=1.8, label=f"Physical Floor D_min ({d_min}/yr)")
    ax1.plot(t_years, integrand_D, color="#1b5e20", lw=2.5, label="True Clipped Integrand max(D_min, D_raw)")
    ax1.fill_between(t_years, d_min, integrand_D, where=(raw_D < d_min), color="#ffcdd2", alpha=0.5, label="Floor Active Region")
    ax1.set_title("Decline Function with Multiple Floor Crossings", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Time (Years)")
    ax1.set_ylabel("Instantaneous Decline Rate (1/year)")
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.4)
    
    # Cumulative Integral Comparison
    exact_I = integrator.integrate_dense_array(t_years)
    flawed_I = integrator.flawed_previous_approximation(t_years)
    
    ax2.plot(t_years, exact_I, color="#1b5e20", lw=2.5, label="Exact Quadrature Integral I(t)")
    ax2.plot(t_years, flawed_I, color="#c2185b", ls="--", lw=2.0, label="Flawed Iteration 10 Approximation")
    ax2.set_title("Exact Quadrature vs Flawed Approximation", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Time (Years)")
    ax2.set_ylabel("Cumulative Integrated Decline I(t)")
    ax2.legend(loc="lower right", fontsize=8.5)
    ax2.grid(True, linestyle="--", alpha=0.4)
    
    max_err = np.max(np.abs(exact_I - flawed_I))
    ax2.text(0.05, 0.85, f"Max Integral Error: {max_err:.4f}\nExact Monotonicity: dq_o/dt < 0 Everywhere", transform=ax2.transAxes,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=9)
             
    plt.tight_layout()
    p = FIG_DIR / "fig4_exact_continuous_decline_integration.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print(f"Generated Figure 4: {p}")


def generate_figure5_deployment_scenarios(df_trajs: pd.DataFrame):
    """Figure 5: Official 2008 Deployment Simulation Trajectories Across 10 Scenarios."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    scenarios = [f"Case {i}" for i in range(1, 11)]
    cmap = plt.cm.tab10
    
    for i, sc in enumerate(scenarios):
        sub_frz = df_trajs[(df_trajs["scenario"] == sc) & (df_trajs["model_name"] == "Frozen_Production_Baseline")]
        sub_mat = df_trajs[(df_trajs["scenario"] == sc) & (df_trajs["model_name"] == "Model_D_Physical_Maturity_Hybrid")]
        if len(sub_frz) == 0:
            continue
        color = cmap(i % 10)
        ax1.plot(sub_frz["quarter_step"], sub_frz["cum_oil_stb"] / 1e6, color=color, lw=1.5, ls="--", alpha=0.7)
        ax1.plot(sub_mat["quarter_step"], sub_mat["cum_oil_stb"] / 1e6, color=color, lw=2.0, label=sc)
        
        # Water cut progression
        ax2.plot(sub_mat["quarter_step"], sub_mat["water_cut"], color=color, lw=1.8, label=sc)
        
    ax1.set_title("20-Year Cumulative Oil Trajectories: Model D (Solid) vs Frozen Baseline (Dashed)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Quarter Step (2008 to 2028)")
    ax1.set_ylabel("Cumulative Oil Recovery (MMstb)")
    ax1.legend(loc="lower right", ncol=2, fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.4)
    
    ax2.set_title("Water Cut Progression Across 10 Scenarios (Anchored to Field Case 0)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Quarter Step (2008 to 2028)")
    ax2.set_ylabel("Water Cut (fraction)")
    ax2.set_ylim([0.55, 1.0])
    ax2.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    p = FIG_DIR / "fig5_deployment_2008_scenarios_and_case0.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print(f"Generated Figure 5: {p}")


if __name__ == "__main__":
    t_start = time.time()
    print("=" * 80)
    print("STARTING CAUSAL MATURITY GATING & SCIENTIFIC VALIDATION MASTER SPRINT")
    print("=" * 80)
    
    # 1. Reconcile Frozen Baseline evaluations first
    df_reconcile = generate_baseline_reconciliation_table()
    
    # 2. Load dataset
    shared_dir = resolve_shared_dir()
    unc = load_uncertainty(shared_dir)
    raw_df = load_clean_simulation_quarterly()
    cdict = build_case_quarterly_trajectories(raw_df)
    
    # 3. Protocol B (Cross-Case Forward-Supervised) across 4 origins
    origins_to_test = [
        ("2004-01-01", 16),
        ("2005-01-01", 12),
        ("2006-01-01", 8),
        ("2007-01-01", 4),
    ]
    
    all_case_records = []
    all_summary_records = []
    
    for orig_str, hq in origins_to_test:
        c_df, s_df = run_cross_fitted_gate_benchmark(
            cdict, unc, orig_str, hq, protocol="Protocol_B"
        )
        if len(c_df) > 0:
            all_case_records.append(c_df)
            all_summary_records.append(s_df)
            
    # 4. Protocol A (Strict Historical As-Of) across feasible origins
    for orig_str, hq in [("2006-01-01", 8), ("2007-01-01", 4), ("2004-01-01", 16)]:
        c_df_A, s_df_A = run_cross_fitted_gate_benchmark(
            cdict, unc, orig_str, hq, protocol="Protocol_A"
        )
        if len(c_df_A) > 0:
            all_case_records.append(c_df_A)
            all_summary_records.append(s_df_A)
            
    df_case_level = pd.concat(all_case_records, ignore_index=True)
    df_summary = pd.concat(all_summary_records, ignore_index=True)
    
    # Save benchmark tables
    case_path = TABLE_DIR / "case_level_validation_details.csv"
    sum_b_path = TABLE_DIR / "protocol_b_cross_fitted_benchmark.csv"
    sum_a_path = TABLE_DIR / "protocol_a_strict_asof_benchmark.csv"
    
    df_case_level.to_csv(case_path, index=False)
    df_summary[df_summary["protocol"] == "Protocol_B"].to_csv(sum_b_path, index=False)
    df_summary[df_summary["protocol"] == "Protocol_A"].to_csv(sum_a_path, index=False)
    
    print(f"\nSaved Case Level Validation Details: {case_path}")
    print(f"Saved Protocol B Benchmark Summary: {sum_b_path}")
    print(f"Saved Protocol A Benchmark Summary: {sum_a_path}")
    
    # 5. January 2008 Deployment Simulation
    df_miles, df_trajs, c0_mat = run_january_2008_research_deployment(cdict, unc)
    miles_path = TABLE_DIR / "deployment_2008_milestones.csv"
    trajs_path = TABLE_DIR / "deployment_2008_trajectories.csv"
    df_miles.to_csv(miles_path, index=False)
    df_trajs.to_csv(trajs_path, index=False)
    print(f"Saved 2008 Deployment Milestones: {miles_path}")
    print(f"Saved 2008 Deployment Trajectories: {trajs_path}")
    
    # 6. Generate Diagnostic Figures
    generate_figure1_sustained_decline_diagnostics(c0_mat, cdict)
    generate_figure2_protocol_benchmark(df_summary)
    generate_figure3_maturity_vs_calendar(df_case_level)
    generate_figure4_exact_integration_diagnostics()
    generate_figure5_deployment_scenarios(df_trajs)
    
    # Copy figures to brain directory for report embedding
    for f in FIG_DIR.glob("*.png"):
        shutil.copy2(f, BRAIN_DIR / f.name)
        print(f"Copied {f.name} to brain directory.")
        
    print(f"\nALL EXPERIMENTS COMPLETED IN {time.time()-t_start:.1f} SECONDS.")
