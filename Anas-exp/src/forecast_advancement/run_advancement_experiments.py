"""
Master Runner: run_advancement_experiments.py
Package: forecast_advancement
Addresses: Stages 1-6 End-to-End Orchestration and Visualization
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Executes:
1. Stage 1: Mandatory Leakage Mutation Test
2. Stage 2: Machine-Recomputed Baseline Across All Origins (Protocol A & B)
3. Stage 3 & 4: Conclusive Maturity Gating Benchmark & 2008 Deployment Gating Audit
4. Stage 5: Evaluation of New Mathematical Challengers (A, B, C)
5. Stage 6: Leave-Reservoir-Out Pseudo-Field Validation on Cases 1-70
6. Generates 7 Master CSV Tables and 5 Publication Figures
"""
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import time
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
    RS_SOLUTION,
)
from ensemble.generate_competition_deliverable import (
    load_locked_config,
    generate_scenario_predictions,
)
from forecast_advancement.protocol_framework import (
    run_data_mutation_leakage_test,
    build_asof_case_dict,
    fit_base_models_under_protocol,
)
from forecast_advancement.baseline_recomputation import (
    recompute_all_baseline_evaluations,
    FEASIBLE_ORIGINS,
)
from forecast_advancement.maturity_gating_decision import (
    run_gating_evaluation_experiment,
    compute_case_increment_nrmse_dual,
)
from forecast_advancement.recovery_state_forecaster import RecoveryStateForecaster
from forecast_advancement.constrained_liquid_dynamics import ConstrainedLiquidDynamicsForecaster
from forecast_advancement.functional_trajectory_forecaster import FunctionalTrajectoryForecaster
from forecast_advancement.end_to_end_pseudo_field_validator import run_end_to_end_pseudo_field_study

TABLE_DIR = Path("outputs/research/forecast_advancement/tables")
FIG_DIR = Path("outputs/research/forecast_advancement/figures")
BRAIN_DIR = Path("/Users/tengkuanas/.gemini/antigravity-cli/brain/4b7e1d00-9795-42a6-b5bb-c629e9cf25cb")

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_fig1_leakage_mutation(mut_res: Dict[str, Any], save_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ["Protocol A Oil Max Diff", "Protocol A Water Max Diff", "Protocol B Oil Max Diff", "Residual Forecaster Diff"]
    diffs = [
        max(mut_res["protocol_a_oil_max_diff"], 1e-16),
        max(mut_res["protocol_a_wat_max_diff"], 1e-16),
        mut_res["protocol_b_oil_max_diff"],
        max(mut_res["residual_forecaster_asof_diff"], 1e-16),
    ]
    colors = ["#2ecc71", "#2ecc71", "#e74c3c", "#3498db"]
    bars = ax.bar(metrics, diffs, color=colors, width=0.5)
    ax.set_yscale("log")
    ax.axhline(1e-12, color="gray", linestyle="--", label="Strict Deterministic Zero (< 1e-12)")
    ax.set_ylabel("Maximum Absolute Prediction Difference (STB/d)")
    ax.set_title("STAGE 1: POST-ORIGIN DATA-MUTATION LEAKAGE TEST\n(Protocol A Invariance vs Protocol B Future Sensitivity)")
    ax.legend()
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_fig2_recomputed_baseline(df_base: pd.DataFrame, save_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Macro NRMSE across origins
    p_a = df_base[df_base["protocol"] == "Protocol_A"].sort_values("origin")
    p_b = df_base[df_base["protocol"] == "Protocol_B"].sort_values("origin")
    
    x = np.arange(len(p_a))
    width = 0.35
    
    ax1.bar(x - width/2, p_a["macro_nrmse"], width, label="Protocol A (Strict Historical As-Of)", color="#2980b9")
    ax1.bar(x + width/2, p_b["macro_nrmse"], width, label="Protocol B (Forward-Supervised DCA)", color="#e67e22")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{r.origin}\n({r.horizon_quarters}Q)" for _, r in p_a.iterrows()])
    ax1.set_ylabel("Macro Cumulative Increment NRMSE")
    ax1.set_title("Frozen Baseline Recomputed Across Origins")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.5)
    
    # Oil vs Water breakdown
    ax2.plot(x, p_a["oil_nrmse"], "o-", label="Proto A Oil", color="#27ae60", linewidth=2)
    ax2.plot(x, p_a["water_nrmse"], "s--", label="Proto A Water", color="#2980b9", linewidth=2)
    ax2.plot(x, p_b["oil_nrmse"], "^-", label="Proto B Oil", color="#d35400", linewidth=2)
    ax2.plot(x, p_b["water_nrmse"], "d--", label="Proto B Water", color="#8e44ad", linewidth=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{r.origin}\n({r.horizon_quarters}Q)" for _, r in p_a.iterrows()])
    ax2.set_ylabel("Phase Increment NRMSE")
    ax2.set_title("Oil vs Water Error Breakdown")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_fig3_maturity_gating_decision(df_gate: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    origins = df_gate["origin"].unique()
    models = df_gate["model_name"].unique()
    
    x = np.arange(len(origins))
    n_models = len(models)
    width = 0.8 / n_models
    
    colors = ["#2c3e50", "#7f8c8d", "#e74c3c", "#f39c12", "#27ae60"]
    
    for i, m in enumerate(models):
        sub = df_gate[df_gate["model_name"] == m].set_index("origin").reindex(origins)
        ax.bar(x + (i - n_models/2 + 0.5) * width, sub["macro_nrmse_mean"], width, label=m, color=colors[i % len(colors)])
        
    ax.set_xticks(x)
    ax.set_xticklabels(origins)
    ax.set_ylabel("Macro NRMSE on Cases 71-85")
    ax.set_title("STAGE 4: MATURITY GATING CANDIDATE BENCHMARK\n(Nested 5-Fold Cross-Fitting Across Retrospective Origins)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_fig4_new_challengers(df_chall: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(df_chall["model_name"], df_chall["macro_nrmse_mean"], color=["#2c3e50", "#16a085", "#8e44ad", "#d35400"])
    ax.set_xlabel("Macro Increment NRMSE")
    ax.set_title("STAGE 5: NEW MATHEMATICAL & RESERVOIR-DOMAIN CHALLENGERS\n(Origin: 2005-01-01, 12-Quarter Horizon)")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va="center", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5, axis="x")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_fig5_pseudo_field_validation(df_pf: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(df_pf["approach_name"], df_pf["macro_nrmse_mean"], color=["#27ae60", "#2980b9", "#8e44ad", "#2c3e50", "#e67e22"])
    ax.set_xlabel("Mean Macro NRMSE Across 15 Leave-Out Pseudo-Fields")
    ax.set_title("STAGE 6: LEAVE-RESERVOIR-OUT PSEUDO-FIELD BENCHMARK (Cases 1-70)\n(Information Regimes & Challenger Comparison)")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va="center", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5, axis="x")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_all_advancement_experiments():
    t_global_start = time.perf_counter()
    print("=" * 80)
    print("EXXONMOBIL DATAWORKS 2026 — NEXT-GENERATION FORECAST RESEARCH")
    print("=" * 80)
    
    shared_dir = resolve_shared_dir()
    unc_df = load_uncertainty(shared_dir)
    clean_sim_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(clean_sim_df)
    
    # -------------------------------------------------------------------------
    # STAGE 1: Data-Mutation Leakage Acceptance Test
    # -------------------------------------------------------------------------
    print("\n--- STAGE 1: DATA-MUTATION LEAKAGE ACCEPTANCE TEST ---")
    mut_res = run_data_mutation_leakage_test(
        case_dict, unc_df, TRAIN_CASES, test_case=71, origin=pd.Timestamp("2005-01-01"), horizon_quarters=12
    )
    df_mut = pd.DataFrame([mut_res])
    df_mut.to_csv(TABLE_DIR / "leakage_mutation_test_results.csv", index=False)
    print(f"  Protocol A Oil Max Diff:       {mut_res['protocol_a_oil_max_diff']:.2e} (Zero Leakage: {mut_res['protocol_a_is_leak_free']})")
    print(f"  Protocol A Water Max Diff:     {mut_res['protocol_a_wat_max_diff']:.2e} (Zero Leakage: {mut_res['protocol_a_is_leak_free']})")
    print(f"  Protocol B Oil Max Diff:       {mut_res['protocol_b_oil_max_diff']:.4f} (Sensitivity: {mut_res['protocol_b_is_sensitive']})")
    print(f"  Residual Forecaster As-Of:     {mut_res['residual_forecaster_asof_diff']:.2e} (Zero Leakage: {mut_res['residual_forecaster_is_leak_free']})")
    assert mut_res["protocol_a_is_leak_free"], "PROTOCOL A LEAKAGE TEST FAILED!"
    
    # -------------------------------------------------------------------------
    # STAGE 2: Machine-Recomputed Baseline Across All Feasible Origins
    # -------------------------------------------------------------------------
    print("\n--- STAGE 2: RECOMPUTING FROZEN BASELINE ACROSS ALL ORIGINS ---")
    df_base_summary, df_base_cases = recompute_all_baseline_evaluations(case_dict, unc_df)
    df_base_summary.to_csv(TABLE_DIR / "recomputed_baseline_all_origins.csv", index=False)
    df_base_cases.to_csv(TABLE_DIR / "recomputed_baseline_case_level.csv", index=False)
    print(df_base_summary[["origin", "horizon_quarters", "protocol", "macro_nrmse", "oil_nrmse", "water_nrmse"]].to_string())
    
    # -------------------------------------------------------------------------
    # STAGE 3 & 4: Conclusive Maturity Gating Benchmark
    # -------------------------------------------------------------------------
    print("\n--- STAGE 4: CANDIDATE MATURITY GATING BENCHMARK (NESTED CV) ---")
    gating_summary_rows = []
    gating_case_rows = []
    
    for origin_str, horizon_q in [("2004-01-01", 16), ("2005-01-01", 12), ("2006-01-01", 8), ("2007-01-01", 4)]:
        # Protocol B evaluation
        summ_b, cases_b = run_gating_evaluation_experiment(
            case_dict, unc_df, origin_str, horizon_q, protocol="Protocol_B"
        )
        gating_summary_rows.append(summ_b)
        gating_case_rows.append(cases_b)
        
        # Protocol A evaluation
        summ_a, cases_a = run_gating_evaluation_experiment(
            case_dict, unc_df, origin_str, horizon_q, protocol="Protocol_A"
        )
        gating_summary_rows.append(summ_a)
        gating_case_rows.append(cases_a)
        
    df_gating_summary = pd.concat(gating_summary_rows, ignore_index=True)
    df_gating_cases = pd.concat(gating_case_rows, ignore_index=True)
    df_gating_summary.to_csv(TABLE_DIR / "maturity_gating_decision_benchmark.csv", index=False)
    df_gating_cases.to_csv(TABLE_DIR / "case_level_gating_details.csv", index=False)
    
    # -------------------------------------------------------------------------
    # STAGE 5: Evaluation of New Mathematical Challengers
    # -------------------------------------------------------------------------
    print("\n--- STAGE 5: NEW MATHEMATICAL & RESERVOIR-DOMAIN MODEL DEVELOPMENT ---")
    origin_eval = pd.Timestamp("2005-01-01")
    horizon_eval = 12
    f_dates_eval = pd.date_range(origin_eval + pd.DateOffset(days=1), periods=horizon_eval, freq="QS")
    
    # Train Challengers on Cases 1-70 strictly <= 2005-01-01
    print("  Fitting RecoveryStateForecaster...")
    m_state = RecoveryStateForecaster().fit(case_dict, unc_df, TRAIN_CASES, origin_eval)
    
    print("  Fitting ConstrainedLiquidDynamicsForecaster (Logistic)...")
    m_liq_log = ConstrainedLiquidDynamicsForecaster(fractional_flow_mode="logistic").fit(
        case_dict, unc_df, TRAIN_CASES, origin_eval
    )
    
    print("  Fitting FunctionalTrajectoryForecaster (Pre-2005 Rolling History Only)...")
    m_func = FunctionalTrajectoryForecaster().fit(
        case_dict, unc_df, TRAIN_CASES, origin_eval, window_quarters=12
    )
    
    # Baseline model
    m_base = fit_base_models_under_protocol(case_dict, unc_df, TRAIN_CASES, origin_eval, "Protocol_A")
    config = load_locked_config()
    weights = config["weights"]
    
    # Evaluate across held-out Cases 71-85
    chall_records = []
    for c in VAL_CASES:
        df_c = case_dict[c]
        pre = df_c[df_c["date"] <= origin_eval]
        post = df_c[df_c["date"] > origin_eval].iloc[:horizon_eval]
        if len(post) < horizon_eval:
            continue
        last = pre.iloc[-1]
        u_c = unc_df.loc[c]
        st = {
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
        
        # 1. Baseline
        _, aud = generate_scenario_predictions(m_base, weights, st, u_c, f_dates_eval, 1.0)
        m_b, no_b, nw_b = compute_case_increment_nrmse_dual(aud["qo_rate"], aud["qw_rate"], post, st)
        
        # 2. Recovery State
        p_state = m_state.predict_rates(df_c, u_c, origin_eval, f_dates_eval)
        m_s, no_s, nw_s = compute_case_increment_nrmse_dual(p_state["oil_rate"], p_state["water_rate"], post, st)
        
        # 3. Constrained Liquid Dynamics
        p_liq = m_liq_log.predict_rates(df_c, u_c, origin_eval, f_dates_eval)
        m_l, no_l, nw_l = compute_case_increment_nrmse_dual(p_liq["oil_rate"], p_liq["water_rate"], post, st)
        
        # 4. Functional Trajectory
        p_f = m_func.predict_rates(df_c, u_c, origin_eval, f_dates_eval)
        m_fn, no_fn, nw_fn = compute_case_increment_nrmse_dual(p_f["oil_rate"], p_f["water_rate"], post, st)
        
        chall_records.append({"case_id": c, "model": "Frozen_Ensemble", "macro": m_b, "oil": no_b, "water": nw_b})
        chall_records.append({"case_id": c, "model": "Recovery_State_Forecaster", "macro": m_s, "oil": no_s, "water": nw_s})
        chall_records.append({"case_id": c, "model": "Constrained_Liquid_Dynamics", "macro": m_l, "oil": no_l, "water": nw_l})
        chall_records.append({"case_id": c, "model": "Functional_Trajectory_Forecaster", "macro": m_fn, "oil": no_fn, "water": nw_fn})
        
    df_chall_cases = pd.DataFrame(chall_records)
    df_chall_summary = df_chall_cases.groupby("model").agg(
        macro_nrmse_mean=("macro", "mean"),
        oil_nrmse_mean=("oil", "mean"),
        water_nrmse_mean=("water", "mean"),
    ).reset_index().rename(columns={"model": "model_name"}).sort_values("macro_nrmse_mean")
    df_chall_summary.to_csv(TABLE_DIR / "new_mathematical_challengers_benchmark.csv", index=False)
    print(df_chall_summary.to_string())
    
    # -------------------------------------------------------------------------
    # STAGE 6: End-to-End Leave-Reservoir-Out Pseudo-Field Validation
    # -------------------------------------------------------------------------
    print("\n--- STAGE 6: LEAVE-RESERVOIR-OUT PSEUDO-FIELD VALIDATION ---")
    df_pf_summary, df_pf_cases = run_end_to_end_pseudo_field_study(case_dict, unc_df)
    df_pf_summary.to_csv(TABLE_DIR / "pseudo_field_validation_summary.csv", index=False)
    df_pf_cases.to_csv(TABLE_DIR / "pseudo_field_validation_case_details.csv", index=False)
    print(df_pf_summary.to_string())
    
    # -------------------------------------------------------------------------
    # GENERATE PUBLICATION FIGURES
    # -------------------------------------------------------------------------
    print("\n--- GENERATING PUBLICATION FIGURES ---")
    plot_fig1_leakage_mutation(mut_res, FIG_DIR / "fig1_leakage_mutation_proof.png")
    plot_fig2_recomputed_baseline(df_base_summary, FIG_DIR / "fig2_baseline_recomputed_all_origins.png")
    
    # Filter Protocol B for figure 3
    df_gating_proto_b = df_gating_summary[df_gating_summary["protocol"] == "Protocol_B"]
    plot_fig3_maturity_gating_decision(df_gating_proto_b, FIG_DIR / "fig3_maturity_gating_final_decision.png")
    plot_fig4_new_challengers(df_chall_summary, FIG_DIR / "fig4_new_challengers_physics_evaluation.png")
    plot_fig5_pseudo_field_validation(df_pf_summary, FIG_DIR / "fig5_pseudo_field_inversion_vs_forecasting.png")
    
    # Copy figures to brain directory for artifact presentation
    for f in FIG_DIR.glob("*.png"):
        dest = BRAIN_DIR / f.name
        dest.write_bytes(f.read_bytes())
        print(f"  Copied {f.name} to brain directory.")
        
    t_total = time.perf_counter() - t_global_start
    print(f"\nALL STAGES COMPLETED SUCCESSFULLY IN {t_total:.1f} SECONDS.")


if __name__ == "__main__":
    run_all_advancement_experiments()
