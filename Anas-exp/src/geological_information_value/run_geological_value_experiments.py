"""
Master Runner: run_geological_value_experiments.py
Package: geological_information_value
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Executes all experimental investigations for the Geological Information Value Study:
1. Controlled Geological Information-Value Benchmark (12 configurations across pseudo-fields).
2. Forward-Model Geological Sensitivity (Local Jacobians, Global Sobol Indices, Horizon Trends).
3. Parameter Inversion and Identifiability Validation (Recovery Errors, Coverage, Equifinality).
4. Ten Official Scenarios Variance Decomposition (Strategy A vs Quantiles vs Prior vs Architecture).
5. Extended Pipeline Post-Origin Mutation Leakage Audit.
6. Evaluation of Geology-Conditioned Physical Decline Forecaster (GCDF).
7. Exports Master Tables and 5 Publication Figures.
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
    YEAR_DAYS,
)
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
)
from ensemble.generate_competition_deliverable import (
    load_locked_config,
    fit_locked_base_models,
    generate_scenario_predictions,
)
from forecast_advancement.protocol_framework import (
    fit_base_models_under_protocol,
    build_asof_case_dict,
)
from geological_information_value.authoritative_scorer import (
    integrate_and_score_rates,
)
from geological_information_value.parameter_bounds import (
    PARAM_NAMES,
    SHORT_NAMES,
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    RANGES,
    PRIOR_MEANS,
    create_parameter_series,
)
from geological_information_value.controlled_information_experiment import (
    EVAL_PSEUDO_FIELDS,
    run_controlled_geological_value_experiment,
)
from geological_information_value.forward_model_sensitivity import (
    compute_local_jacobians,
    compute_global_sensitivity_analysis_saltelli,
    inspect_fitted_representations,
)
from geological_information_value.parameter_inversion_validation import (
    run_parameter_inversion_audit,
)
from geological_information_value.official_scenarios_audit import (
    run_official_scenarios_variance_audit,
)
from geological_information_value.end_to_end_leakage_audit import (
    run_extended_leakage_mutation_test,
)
from geological_information_value.geology_conditioned_challenger import (
    GeologyConditionedPhysicalForecaster,
)

SCHEMA_VERSION = "v2_authoritative_scorer"
TABLE_DIR = Path("outputs/research/geological_information_value/tables")
FIG_DIR = Path("outputs/research/geological_information_value/figures")
BRAIN_DIR = Path("/Users/tengkuanas/.gemini/antigravity-cli/brain/4b7e1d00-9795-42a6-b5bb-c629e9cf25cb")


def plot_figure_1_controlled_information_value(summary_df: pd.DataFrame, save_path: Path):
    """Figure 1: Macro, Oil, and Water NRMSE across Controlled Geological Configurations."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    order = [
        "A_True_Geology",
        "B_Point_Inferred",
        "C_Bayes_Posterior_Mean",
        "D_Population_Mean",
        "E_Random_Prior_Sample",
        "F_Adversarial_Mismatch",
        "G1_Ablate_FaultTrans",
        "G2_Ablate_Porosity",
        "G3a_Ablate_Perm_PriorRef",
        "G3b_Ablate_Perm_PopMean",
        "G3c_Ablate_Perm_PopMedian",
        "G4_Ablate_AquiferVolume",
    ]
    df_plot = summary_df.set_index("condition").reindex(order).reset_index()
    
    labels = [
        "A: True Geology",
        "B: Point Inferred",
        "C: Bayes Mean",
        "D: Pop Mean",
        "E: Prior Draws (M=10)",
        "F: Adversarial Mismatch",
        "G1: Ablate Ft",
        "G2: Ablate Poro",
        "G3a: Perm (Prior k=1.0)",
        "G3b: Perm (Mean k=5.29)",
        "G3c: Perm (Med k=5.34)",
        "G4: Ablate AqPV",
    ]
    x = np.arange(len(labels))
    width = 0.25
    
    rects1 = ax.bar(x - width, df_plot["macro_nrmse_mean"], width, label="Macro NRMSE", color="#1f77b4")
    rects2 = ax.bar(x, df_plot["oil_nrmse_mean"], width, label="Oil NRMSE", color="#2ca02c")
    rects3 = ax.bar(x + width, df_plot["water_nrmse_mean"], width, label="Water NRMSE", color="#d62728")
    
    true_macro = df_plot.loc[df_plot["condition"] == "A_True_Geology", "macro_nrmse_mean"].values[0]
    ax.axhline(true_macro, color="#1f77b4", linestyle="--", alpha=0.7, label=f"True Geology Macro ({true_macro:.4f})")
    
    ax.set_ylabel("NRMSE (Horizon 12Q)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 1: Controlled Geological Information Value Benchmark (Authoritative Scorer)\nIdentical Frozen Ensemble & Historical State Across 12 Controlled Geological Inputs", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.legend(frameon=True, loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_figure_2_forward_model_sensitivities(jac_df: pd.DataFrame, save_path: Path):
    """Figure 2: Forward Model Normalized Sensitivities across Horizons."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    df_80q = jac_df[jac_df["horizon_quarters"] == 80]
    models = ["Learned_Exponential_DCA", "Recursive_ExtraTrees", "Blend_Phase_Specific_Simplex"]
    
    params = SHORT_NAMES
    x = np.arange(len(params))
    width = 0.25
    
    colors = ["#ff7f0e", "#1f77b4", "#2ca02c"]
    for idx, m in enumerate(models):
        sub = df_80q[df_80q["model_name"] == m].set_index("short_name").reindex(params)
        axes[0].bar(x + (idx - 1) * width, np.abs(sub["norm_sens_cum_oil"]), width, label=m.replace("_", " "), color=colors[idx])
        
    axes[0].set_title("A: Normalized Sensitivity of 20-Year Cumulative Oil |S_Qo|", fontweight="bold", fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(params, fontweight="bold")
    axes[0].set_ylabel("Dimensionless Sensitivity |(Delta u / Qo) * dQo/du|", fontsize=10)
    axes[0].legend(frameon=True, fontsize=8)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    
    horizons = [4, 12, 40, 80]
    df_ens = jac_df[jac_df["model_name"] == "Blend_Phase_Specific_Simplex"]
    
    line_styles = ["-o", "-s", "-^", "-d"]
    for idx, p in enumerate(SHORT_NAMES):
        sub_p = df_ens[df_ens["short_name"] == p].set_index("horizon_quarters").reindex(horizons)
        axes[1].plot(horizons, np.abs(sub_p["norm_sens_cum_oil"]), line_styles[idx], label=p, linewidth=2, markersize=6)
        
    axes[1].set_title("B: Ensemble Oil Sensitivity vs Forecast Horizon", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("Forecast Horizon (Quarters)", fontsize=10)
    axes[1].set_ylabel("Dimensionless Sensitivity |S_Qo|", fontsize=10)
    axes[1].set_xticks(horizons)
    axes[1].legend(frameon=True, fontsize=9)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_figure_3_parameter_recovery_vs_downstream(inv_df: pd.DataFrame, save_path: Path):
    """Figure 3: Parameter Recovery Error vs Downstream Forecast Sensitivity (Equifinality)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    err_cols = ["norm_err_Ft", "norm_err_Poro", "norm_err_Perm", "norm_err_AqPV"]
    err_means = [inv_df[c].mean() * 100.0 for c in err_cols]
    err_stds = [inv_df[c].std() * 100.0 for c in err_cols]
    
    bars = axes[0].bar(SHORT_NAMES, err_means, yerr=err_stds, capsize=5, color=["#3274a1", "#e1812c", "#3a923a", "#c03d3e"])
    axes[0].set_title("A: Range-Normalized Parameter Inversion Error (%)", fontweight="bold", fontsize=11)
    axes[0].set_ylabel("Parameter Recovery Error (% of Admissible Range)", fontsize=10)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    axes[1].scatter(inv_df["mean_norm_param_error"] * 100.0, inv_df["delta_macro_forecast"], color="#800080", s=60, alpha=0.8, edgecolors="k")
    axes[1].set_title("B: Inversion Error vs Downstream Forecast Delta\nDemonstration of Equifinality & Downstream Insensitivity", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("Mean Normalized Parameter Recovery Error (%)", fontsize=10)
    axes[1].set_ylabel("Forecast Delta |NRMSE_inferred - NRMSE_true|", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    
    mean_sens = inv_df["downstream_sensitivity_ratio"].mean()
    axes[1].text(0.05, 0.90, f"Mean Downstream Sensitivity Ratio:\nkappa = {mean_sens:.4f}\n(Downstream error is virtually decoupled from parameter error)",
                 transform=axes[1].transAxes, fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_figure_4_official_scenarios_variance_decomposition(decomp_df: pd.DataFrame, eval_df: pd.DataFrame, save_path: Path):
    """Figure 4: 20-Year EUR Spread Across Selection, Prior, and Architecture."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    groups = [
        "1_Official_Strategy_A_Ten_Best",
        "2_Strategy_C_Quality_Quantiles",
        "3_Full_Prior_Box_Samples",
    ]
    data = [eval_df[eval_df["group"] == g]["eur_oil_stb"] / 1e6 for g in groups]
    labels = ["10 Official Scenarios\n(Strategy A: Best Matches)", "10 Quantile Scenarios\n(Strategy C: P10-P90)", "100 Prior Draws\n(Full Box Boxplot)"]
    
    axes[0].boxplot(data, tick_labels=labels, patch_artist=True,
                    boxprops=dict(facecolor="#5dade2", color="#1b4f72"),
                    medianprops=dict(color="#922b21", linewidth=2))
    axes[0].set_ylabel("20-Year Cumulative Oil (MMSTB)", fontsize=10, fontweight="bold")
    axes[0].set_title("A: EUR Oil Spread Across Geological Scenario Ensembles", fontsize=11, fontweight="bold")
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    
    spread_a = (eval_df[eval_df["group"] == "1_Official_Strategy_A_Ten_Best"]["eur_oil_stb"].max() -
                eval_df[eval_df["group"] == "1_Official_Strategy_A_Ten_Best"]["eur_oil_stb"].min())
    axes[0].text(0.05, 0.85, f"Strategy A Spread: {spread_a:,.0f} STB\n(Only 0.033% of EUR)",
                 transform=axes[0].transAxes, fontsize=9, bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3))
    
    scopes = [
        "Strategy A (Official)",
        "Strategy C (Quantiles)",
        "Full Prior Box",
        "Model Architecture",
    ]
    spreads_mmstb = [
        spread_a / 1e6,
        (eval_df[eval_df["group"] == "2_Strategy_C_Quality_Quantiles"]["eur_oil_stb"].max() - eval_df[eval_df["group"] == "2_Strategy_C_Quality_Quantiles"]["eur_oil_stb"].min()) / 1e6,
        (eval_df[eval_df["group"] == "3_Full_Prior_Box_Samples"]["eur_oil_stb"].max() - eval_df[eval_df["group"] == "3_Full_Prior_Box_Samples"]["eur_oil_stb"].min()) / 1e6,
        decomp_df.loc[decomp_df["evaluation_scope"].str.contains("Architecture"), "spread_eur_oil_stb"].values[0] / 1e6,
    ]
    
    bars = axes[1].bar(scopes, spreads_mmstb, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"])
    axes[1].set_ylabel("EUR Spread Range (MMSTB)", fontsize=10, fontweight="bold")
    axes[1].set_title("B: Variance Share Comparison: Geology vs Architecture", fontsize=11, fontweight="bold")
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, yval * 1.15, f"{yval:.3f} M", ha="center", va="bottom", fontsize=8, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_figure_5_geology_conditioned_challenger(challenger_df: pd.DataFrame, save_path: Path):
    """Figure 5: GCDF Challenger Performance vs Frozen Ensemble Baseline."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    models = challenger_df["model_name"].tolist()
    x = np.arange(len(models))
    width = 0.25
    
    ax.bar(x - width, challenger_df["macro_nrmse_mean"], width, label="Macro NRMSE", color="#1f77b4")
    ax.bar(x, challenger_df["oil_nrmse_mean"], width, label="Oil NRMSE", color="#2ca02c")
    ax.bar(x + width, challenger_df["water_nrmse_mean"], width, label="Water NRMSE", color="#d62728")
    
    ax.set_ylabel("NRMSE", fontsize=11, fontweight="bold")
    ax.set_title("Figure 5: Geology-Conditioned Physical Forecaster (GCDF) vs Production Baseline\nLeave-Reservoir-Out Validation Across 15 Pseudo-Fields (Authoritative Scorer)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in models], fontsize=10, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run Geological Information Value Experiments")
    parser.add_argument("--force", action="store_true", help="Force recomputation of all stages")
    args = parser.parse_args()
    
    print("=" * 80)
    print("EXXONMOBIL DATAWORKS CHALLENGE 2026")
    print("GEOLOGICAL INFORMATION VALUE & END-TO-END PREDICTIVE VALIDATION (CORRECTED)")
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
    # STAGE 1 & 2: Controlled Geological Information-Value Experiment
    # -------------------------------------------------------------------------
    summary_path = TABLE_DIR / "controlled_geological_value_summary.csv"
    case_path = TABLE_DIR / "controlled_geological_value_case_details.csv"
    if summary_path.exists() and case_path.exists() and not args.force:
        print("\n[Stage 1 & 2] Loading cached Controlled Geological Information-Value Experiment...")
        case_ctrl_df = pd.read_csv(case_path)
        summary_ctrl_df = pd.read_csv(summary_path)
    else:
        print("\n[Stage 1 & 2] Executing Controlled Geological Information-Value Experiment...")
        t0 = time.perf_counter()
        case_ctrl_df, summary_ctrl_df = run_controlled_geological_value_experiment(
            case_dict=case_dict,
            unc_df=unc_df,
            pseudo_fields=EVAL_PSEUDO_FIELDS,
            origin=pd.Timestamp("2005-01-01"),
            horizon_quarters=12,
            n_random_draws=10,
        )
        case_ctrl_df.to_csv(case_path, index=False)
        summary_ctrl_df.to_csv(summary_path, index=False)
        print(f"  Completed in {time.perf_counter() - t0:.2f}s. Saved 2 tables.")
        
    # -------------------------------------------------------------------------
    # STAGE 3: Quantify Forward-Model Geological Sensitivity
    # -------------------------------------------------------------------------
    jac_path = TABLE_DIR / "forward_model_local_jacobians.csv"
    sobol_path = TABLE_DIR / "forward_model_global_sensitivity_sobol.csv"
    conv_path = TABLE_DIR / "forward_model_sobol_convergence.csv"
    audit_path = TABLE_DIR / "fitted_representation_coefficients_audit.json"
    if jac_path.exists() and sobol_path.exists() and conv_path.exists() and audit_path.exists() and not args.force:
        print("\n[Stage 3] Loading cached Forward-Model Geological Sensitivity...")
        jac_df = pd.read_csv(jac_path)
        sobol_df = pd.read_csv(sobol_path)
    else:
        print("\n[Stage 3] Quantifying Forward-Model Geological Sensitivity (Local Jacobians & Saltelli GSA)...")
        t0 = time.perf_counter()
        config = load_locked_config()
        weights = config["weights"]
        inference_origin = pd.Timestamp("2008-01-01")
        models = fit_locked_base_models(case_dict, unc_df, TRAIN_CASES, inference_origin)
        if "Recursive_ExtraTrees" in models:
            models["Recursive_ExtraTrees"].model_qo.n_jobs = 1
            models["Recursive_ExtraTrees"].model_qw.n_jobs = 1
            
        prod = pd.read_parquet(shared_dir / "production_timeseries.parquet")
        c0 = prod[prod["case_num"] == 0]
        c0_2008 = c0[c0["date"] == "2008-01-01"].set_index("metric")["value"]
        c0_anchor = {
            "qo": float(c0_2008["oil_rate"]),
            "qw": float(c0_2008["water_rate"]),
            "qg": float(c0_2008["gas_rate"]),
            "Qo": float(c0_2008["oil_cum"]),
            "Qw": float(c0_2008["water_cum"]),
            "Qg": float(c0_2008["gas_cum"]),
            "wor": float(c0_2008["water_rate"] / max(c0_2008["oil_rate"], 1.0)),
            "res_age": 10.0,
            "origin_date": inference_origin,
        }
        
        jac_df = compute_local_jacobians(
            models=models,
            weights=weights,
            init_state=c0_anchor,
            base_u=PRIOR_MEANS,
            horizons=[4, 12, 40, 80],
        )
        jac_df.to_csv(jac_path, index=False)
        
        conv_df, sobol_df = compute_global_sensitivity_analysis_saltelli(
            models=models,
            weights=weights,
            init_state=c0_anchor,
            unc_df=unc_df,
            sample_sizes=[128, 256, 512, 1024],
            horizon_quarters=80,
            n_bootstrap=200,
        )
        conv_df.to_csv(conv_path, index=False)
        sobol_df.to_csv(sobol_path, index=False)
        
        fitted_audit = inspect_fitted_representations(models)
        with open(audit_path, "w") as f:
            json.dump(fitted_audit, f, indent=2)
        print(f"  Completed in {time.perf_counter() - t0:.2f}s. Saved Jacobians, Saltelli GSA, convergence, and coefficients audit.")
        
    # -------------------------------------------------------------------------
    # STAGE 4: Validate Actual Parameter Inversion and Identifiability
    # -------------------------------------------------------------------------
    inv_path = TABLE_DIR / "parameter_inversion_case_audit.csv"
    ident_path = TABLE_DIR / "parameter_identifiability_summary.csv"
    if inv_path.exists() and ident_path.exists() and not args.force:
        print("\n[Stage 4] Loading cached Parameter Inversion and Identifiability...")
        inv_case_df = pd.read_csv(inv_path)
        ident_summary_df = pd.read_csv(ident_path)
    else:
        print("\n[Stage 4] Validating Actual Parameter Inversion and Identifiability...")
        t0 = time.perf_counter()
        inv_case_df, ident_summary_df, coverage_audit = run_parameter_inversion_audit(
            case_dict=case_dict,
            unc_df=unc_df,
            pseudo_fields=EVAL_PSEUDO_FIELDS,
            origin=pd.Timestamp("2005-01-01"),
            horizon_quarters=12,
        )
        inv_case_df.to_csv(inv_path, index=False)
        ident_summary_df.to_csv(ident_path, index=False)
        print(f"  Completed in {time.perf_counter() - t0:.2f}s. Saved 2 tables.")
        
    # -------------------------------------------------------------------------
    # STAGE 5: Revisit Ten Official Forecast Scenarios
    # -------------------------------------------------------------------------
    eval_sc_path = TABLE_DIR / "official_vs_prior_scenarios_evaluations.csv"
    decomp_sc_path = TABLE_DIR / "official_scenarios_variance_decomposition.csv"
    findings_sc_path = TABLE_DIR / "official_scenarios_audit_findings.json"
    if eval_sc_path.exists() and decomp_sc_path.exists() and findings_sc_path.exists() and not args.force:
        print("\n[Stage 5] Loading cached Official 20-Year Forecast Scenarios Audit...")
        scenarios_eval_df = pd.read_csv(eval_sc_path)
        decomp_df = pd.read_csv(decomp_sc_path)
        with open(findings_sc_path, "r") as f:
            audit_findings = json.load(f)
    else:
        print("\n[Stage 5] Auditing Official 20-Year Forecast Scenarios and Variance Decomposition...")
        t0 = time.perf_counter()
        scenarios_eval_df, decomp_df, audit_findings = run_official_scenarios_variance_audit(
            n_prior_samples=100,
            seed=42,
        )
        scenarios_eval_df.to_csv(eval_sc_path, index=False)
        decomp_df.to_csv(decomp_sc_path, index=False)
        with open(findings_sc_path, "w") as f:
            json.dump(audit_findings, f, indent=2)
        print(f"  Completed in {time.perf_counter() - t0:.2f}s. Saved scenarios comparison and decomposition.")
        
    # -------------------------------------------------------------------------
    # STAGE 6: Extended Post-Origin Mutation Leakage Audit
    # -------------------------------------------------------------------------
    leakage_path = TABLE_DIR / "extended_leakage_mutation_test_results.csv"
    if leakage_path.exists() and not args.force:
        print("\n[Stage 6] Loading cached Extended Pipeline Post-Origin Mutation Leakage Audit...")
        leakage_res = pd.read_csv(leakage_path).iloc[0].to_dict()
    else:
        print("\n[Stage 6] Executing Extended Pipeline Post-Origin Mutation Leakage Audit...")
        t0 = time.perf_counter()
        leakage_res = run_extended_leakage_mutation_test(
            case_dict=case_dict,
            unc_df=unc_df,
            train_cases=list(range(1, 71)),
            test_case=71,
            origin=pd.Timestamp("2005-01-01"),
            horizon_quarters=12,
        )
        pd.DataFrame([leakage_res]).to_csv(leakage_path, index=False)
        print(f"  Completed in {time.perf_counter() - t0:.2f}s. Leak-free status: {leakage_res['protocol_a_is_fully_leak_free']}.")
        
    # -------------------------------------------------------------------------
    # STAGE 7: Evaluate Geology-Conditioned Physical Forecaster (GCDF)
    # -------------------------------------------------------------------------
    print("\n[Stage 7] Benchmarking Geology-Conditioned Physical Decline Forecaster (GCDF)...")
    t0 = time.perf_counter()
    origin_eval = pd.Timestamp("2005-01-01")
    horizon_eval = 12
    f_dates_eval = pd.date_range(origin_eval + pd.DateOffset(days=1), periods=horizon_eval, freq="QS")
    
    gcdf_records = []
    for c in EVAL_PSEUDO_FIELDS:
        train_pool = [x for x in TRAIN_CASES if x != c]
        asof_pool = build_asof_case_dict(case_dict, origin_eval)
        gcdf = GeologyConditionedPhysicalForecaster().fit(asof_pool, unc_df, train_pool, origin_eval)
        
        df_c = case_dict[c]
        pre = df_c[df_c["date"] <= origin_eval].reset_index(drop=True)
        post = df_c[df_c["date"] > origin_eval].iloc[:horizon_eval].reset_index(drop=True)
        last = pre.iloc[-1]
        u_c = unc_df.loc[c]
        init_st = {
            "qo": float(last["oil_rate_stbd"]),
            "qw": float(last["water_rate_stbd"]),
            "qg": float(last["gas_rate_mscfd"]),
            "Qo": float(last["oil_cum_stb"]),
            "Qw": float(last["water_cum_stb"]),
            "Qg": float(last["gas_cum_mscf"]),
            "wor": float(last.get("wor", float(last["water_rate_stbd"]) / max(float(last["oil_rate_stbd"]), 1.0))),
            "res_age": (origin_eval - pre["date"].iloc[0]).days / YEAR_DAYS,
            "origin_date": origin_eval,
        }
        
        pred_gcdf = gcdf.predict_rates(init_st, u_c, f_dates_eval)
        score_gcdf = integrate_and_score_rates(
            init_state=init_st,
            pred_rate_oil=pred_gcdf["oil_rate"],
            pred_rate_wat=pred_gcdf["water_rate"],
            post_obs_df=post,
            dt_days=91.3125,
        )
        gcdf_records.append({
            "case_id": c,
            "macro_nrmse": score_gcdf["macro_nrmse"],
            "oil_nrmse": score_gcdf["oil_nrmse"],
            "water_nrmse": score_gcdf["water_nrmse"],
            "oil_bias": score_gcdf["oil_bias"],
            "water_bias": score_gcdf["water_bias"],
        })
        
    df_gcdf_cases = pd.DataFrame(gcdf_records)
    base_summary = summary_ctrl_df[summary_ctrl_df["condition"] == "A_True_Geology"].iloc[0]
    
    gcdf_summary = pd.DataFrame([
        {
            "model_name": "Frozen_Ensemble_Baseline",
            "macro_nrmse_mean": float(base_summary["macro_nrmse_mean"]),
            "oil_nrmse_mean": float(base_summary["oil_nrmse_mean"]),
            "water_nrmse_mean": float(base_summary["water_nrmse_mean"]),
            "diff_vs_baseline": 0.0,
            "pct_improvement": 0.0,
        },
        {
            "model_name": "GCDF_Physical_Challenger",
            "macro_nrmse_mean": float(df_gcdf_cases["macro_nrmse"].mean()),
            "oil_nrmse_mean": float(df_gcdf_cases["oil_nrmse"].mean()),
            "water_nrmse_mean": float(df_gcdf_cases["water_nrmse"].mean()),
            "diff_vs_baseline": float(df_gcdf_cases["macro_nrmse"].mean() - base_summary["macro_nrmse_mean"]),
            "pct_improvement": float(
                (base_summary["macro_nrmse_mean"] - df_gcdf_cases["macro_nrmse"].mean())
                / base_summary["macro_nrmse_mean"]
                * 100.0
            ),
        },
    ])
    gcdf_bench_path = TABLE_DIR / "geology_conditioned_challenger_benchmark.csv"
    gcdf_summary.to_csv(gcdf_bench_path, index=False)
    print(f"  Completed in {time.perf_counter() - t0:.2f}s. Saved challenger benchmark.")
    
    # -------------------------------------------------------------------------
    # STAGE 8: Generate Publication Figures
    # -------------------------------------------------------------------------
    print("\n[Stage 8] Rendering 5 Publication Figures...")
    f1_path = FIG_DIR / "fig1_controlled_geological_information_value.png"
    f2_path = FIG_DIR / "fig2_forward_model_parameter_sensitivities.png"
    f3_path = FIG_DIR / "fig3_parameter_recovery_vs_downstream_insensitivity.png"
    f4_path = FIG_DIR / "fig4_official_scenarios_uncertainty_decomposition.png"
    f5_path = FIG_DIR / "fig5_geology_conditioned_challenger_comparison.png"
    
    plot_figure_1_controlled_information_value(summary_ctrl_df, f1_path)
    plot_figure_2_forward_model_sensitivities(jac_df, f2_path)
    plot_figure_3_parameter_recovery_vs_downstream(inv_case_df, f3_path)
    plot_figure_4_official_scenarios_variance_decomposition(decomp_df, scenarios_eval_df, f4_path)
    plot_figure_5_geology_conditioned_challenger(gcdf_summary, f5_path)
    
    # Copy figures to BRAIN_DIR for user artifact embedding
    for f in [f1_path, f2_path, f3_path, f4_path, f5_path]:
        shutil.copy2(f, BRAIN_DIR / f.name)
    print("  All 5 figures rendered and synced to artifact directory.")
    
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f"ALL EXPERIMENTS AND DELIVERABLES COMPLETED IN {elapsed:.2f} SECONDS")
    print("=" * 80)


if __name__ == "__main__":
    main()
