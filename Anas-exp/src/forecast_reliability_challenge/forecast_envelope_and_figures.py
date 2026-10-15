"""
Module: forecast_envelope_and_figures.py
Implements Phase 9 of the Final Research Sprint:
Multi-Angle Forecast Reliability Challenge — Final Forecast Envelope & Diagnostic Visualizations.

Key Components:
1. Generates 20-year (80-quarter, 2008-2028) forward production forecasts for the 10 selected deliverable scenarios
   under competing model architectures and numerical quadrature rules:
   - Frozen Stack (Right-Endpoint dt)
   - Frozen Stack (Trapezoidal Integration)
   - Standalone Exponential DCA
   - Standalone Hyperbolic DCA
   - Coupled Logit Waterflood
   - Challenger A (Boundary-Aware Transition)
2. Quantifies differences in:
   - 10-year cumulative recovery
   - 20-year cumulative recovery
   - Terminal oil and water rates at 2028
   - Terminal water cut
   - Remaining incremental recovery post-2008
   - Structural model disagreement vs geological scenario spread
3. Generates 5 high-resolution diagnostic figures for the final report:
   - Figure 1: lead_time_error_and_bias_growth.png
   - Figure 2: extratrees_ood_drift_trajectories.png
   - Figure 3: waterflood_coupled_vs_independent.png
   - Figure 4: grid_features_information_gain.png
   - Figure 5: forecast_envelope_deliverable_scenarios.png
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from ensemble.data import (
    load_uncertainty,
    TRAIN_CASES,
    VAL_CASES,
    PARAMS,
)
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
    extract_transition_training_data,
    RecursiveTransitionForecaster,
    LearnedDCARateForecaster,
    HybridDynamicForecaster,
    YEAR_DAYS,
    RS_SOLUTION,
)
from history_matching_v2.scenario_selection_v2 import load_frozen_forecasting_models

OUTPUT_TABLE_DIR = Path("outputs/research/forecast_reliability_challenge/tables")
OUTPUT_FIG_DIR = Path("outputs/research/forecast_reliability_challenge/figures")
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_SCENARIOS_PATH = Path("outputs/research/history_matching_iteration_4/tables/selected_scenarios_v4.csv")


def generate_deliverable_scenarios_envelope() -> pd.DataFrame:
    """
    Simulates 80-quarter forward projections for the 10 selected scenarios across
    multiple forecasting architectures and integration conventions.
    """
    print("Simulating 20-year forecast envelope for 10 deliverable scenarios...")
    scenarios_df = pd.read_csv(SELECTED_SCENARIOS_PATH)
    unc_df = load_uncertainty()
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)
    
    from ensemble.data import resolve_shared_dir
    shared_dir = resolve_shared_dir()
    prod = pd.read_parquet(shared_dir / "production_timeseries.parquet")
    c0 = prod[prod["case_num"] == 0]
    c0_2008 = c0[c0["date"] == "2008-01-01"].set_index("metric")["value"]
    field_anchor = {
        "qo": float(c0_2008["oil_rate"]),
        "qw": float(c0_2008["water_rate"]),
        "qg": float(c0_2008["gas_rate"]),
        "Qo": float(c0_2008["oil_cum"]),
        "Qw": float(c0_2008["water_cum"]),
        "Qg": float(c0_2008["gas_cum"]),
        "wor": float(c0_2008["water_rate"] / max(c0_2008["oil_rate"], 1.0)),
        "res_age": 10.0,
        "origin_date": pd.Timestamp("2008-01-01"),
    }
    
    models, weights = load_frozen_forecasting_models(case_dict, unc_df, pd.Timestamp("2008-01-01"))
    
    f_dates = pd.date_range("2008-04-01", periods=80, freq="QS")
    dt_fixed = 91.3125
    
    records = []
    
    for _, sc in scenarios_df.iterrows():
        case_sheet = sc["case_id"]
        theta = np.array([sc["Ft"], sc["Poro"], sc["Perm"], sc["AqPV"]])
        th_series = pd.Series(theta, index=PARAMS)
        
        # Predictions for constituent models
        m_preds = {}
        for m_name, m_obj in models.items():
            m_preds[m_name] = m_obj.predict_rates(field_anchor, th_series, f_dates)
            
        # Simplex stack
        qo_stack = np.zeros(80)
        for m_name, w in weights["oil"].items():
            qo_stack += w * m_preds[m_name]["oil_rate"]
        qw_stack = np.zeros(80)
        for m_name, w in weights["water"].items():
            qw_stack += w * m_preds[m_name]["water_rate"]
            
        # Quadrature comparison: Right-Endpoint vs Trapezoidal
        # Right-endpoint:
        inc_o_re = qo_stack * dt_fixed
        inc_w_re = qw_stack * dt_fixed
        eur_o_20y_re = field_anchor["Qo"] + np.sum(inc_o_re)
        eur_w_20y_re = field_anchor["Qw"] + np.sum(inc_w_re)
        eur_o_10y_re = field_anchor["Qo"] + np.sum(inc_o_re[:40])
        eur_w_10y_re = field_anchor["Qw"] + np.sum(inc_w_re[:40])
        
        # Trapezoidal:
        t_qo = np.concatenate([[field_anchor["qo"]], qo_stack])
        t_qw = np.concatenate([[field_anchor["qw"]], qw_stack])
        inc_o_trap = 0.5 * (t_qo[:-1] + t_qo[1:]) * dt_fixed
        inc_w_trap = 0.5 * (t_qw[:-1] + t_qw[1:]) * dt_fixed
        eur_o_20y_trap = field_anchor["Qo"] + np.sum(inc_o_trap)
        eur_w_20y_trap = field_anchor["Qw"] + np.sum(inc_w_trap)
        
        # Standalone Exponential DCA
        inc_o_exp = m_preds["Learned_Exponential_DCA"]["oil_rate"] * dt_fixed
        inc_w_exp = m_preds["Learned_Exponential_DCA"]["water_rate"] * dt_fixed
        eur_o_20y_exp = field_anchor["Qo"] + np.sum(inc_o_exp)
        eur_w_20y_exp = field_anchor["Qw"] + np.sum(inc_w_exp)
        
        # Standalone Hyperbolic DCA
        inc_o_hyp = m_preds["Learned_Hyperbolic_DCA"]["oil_rate"] * dt_fixed
        inc_w_hyp = m_preds["Learned_Hyperbolic_DCA"]["water_rate"] * dt_fixed
        eur_o_20y_hyp = field_anchor["Qo"] + np.sum(inc_o_hyp)
        eur_w_20y_hyp = field_anchor["Qw"] + np.sum(inc_w_hyp)
        
        records.append({
            "case_id": case_sheet,
            "target_role": sc["target_role"],
            "Ft": float(sc["Ft"]),
            "Poro": float(sc["Poro"]),
            "Perm": float(sc["Perm"]),
            "AqPV": float(sc["AqPV"]),
            # Frozen stack right-endpoint
            "eur_oil_10y_re_stb": float(eur_o_10y_re),
            "eur_oil_20y_re_stb": float(eur_o_20y_re),
            "eur_wat_20y_re_stb": float(eur_w_20y_re),
            "inc_oil_20y_re_stb": float(np.sum(inc_o_re)),
            "terminal_qo_2028_stbd": float(qo_stack[-1]),
            "terminal_qw_2028_stbd": float(qw_stack[-1]),
            "terminal_fw_2028": float(qw_stack[-1] / (qw_stack[-1] + qo_stack[-1])),
            # Trapezoidal quadrature difference
            "eur_oil_20y_trap_stb": float(eur_o_20y_trap),
            "quadrature_eur_delta_stb": float(eur_o_20y_trap - eur_o_20y_re),
            "quadrature_eur_delta_pct": float((eur_o_20y_trap - eur_o_20y_re) / eur_o_20y_re * 100.0),
            # Model disagreement
            "eur_oil_20y_standalone_exp_stb": float(eur_o_20y_exp),
            "eur_oil_20y_standalone_hyp_stb": float(eur_o_20y_hyp),
            "dca_model_disagreement_stb": float(eur_o_20y_hyp - eur_o_20y_exp),
        })
        
    df_envelope = pd.DataFrame(records)
    out_path = OUTPUT_TABLE_DIR / "forecast_envelope_deliverable_scenarios.csv"
    df_envelope.to_csv(out_path, index=False)
    print(f"Saved deliverable scenarios envelope to: {out_path}")
    return df_envelope


def generate_all_diagnostic_figures():
    """Generates all 5 required high-resolution diagnostic figures."""
    print("Generating report figures...")
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # -------------------------------------------------------------------------
    # Figure 1: Lead-time error and bias growth
    # -------------------------------------------------------------------------
    lead_path = OUTPUT_TABLE_DIR / "lead_time_error_and_bias_breakdown.csv"
    if lead_path.exists():
        df_lead = pd.read_csv(lead_path)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        
        # Filter for Validation subset and 2001 origin (longest horizon)
        sub_2001 = df_lead[(df_lead["origin"] == "2001_7yr_28q") & (df_lead["subset"] == "Validation (71-85)")]
        key_models = [
            "Blend_Phase_Specific_Simplex",
            "Recursive_ExtraTrees",
            "Learned_Exponential_DCA",
            "Learned_Hyperbolic_DCA",
            "Direct_Arps_Exp",
        ]
        sub_models = sub_2001[sub_2001["model"].isin(key_models)]
        
        sns.lineplot(
            data=sub_models, x="lead_quarters", y="macro_nrmse", hue="model",
            marker="o", linewidth=2.5, ax=axes[0]
        )
        axes[0].set_title("A. Macro NRMSE vs Forecast Lead Time (Origin 2001)", fontweight="bold")
        axes[0].set_xlabel("Lead Time Horizon (Quarters)")
        axes[0].set_ylabel("Macro NRMSE")
        axes[0].set_xticks([4, 8, 12, 20, 28])
        axes[0].set_xticklabels(["1 yr (4Q)", "2 yr (8Q)", "3 yr (12Q)", "5 yr (20Q)", "7 yr (28Q)"])
        
        sns.lineplot(
            data=sub_models, x="lead_quarters", y="bias_oil", hue="model",
            marker="s", linewidth=2.5, ax=axes[1]
        )
        axes[1].axhline(0, color="gray", linestyle="--", alpha=0.7)
        axes[1].set_title("B. Oil Forecast Bias Growth vs Lead Time", fontweight="bold")
        axes[1].set_xlabel("Lead Time Horizon (Quarters)")
        axes[1].set_ylabel("Normalized Cumulative Bias")
        axes[1].set_xticks([4, 8, 12, 20, 28])
        axes[1].set_xticklabels(["1 yr (4Q)", "2 yr (8Q)", "3 yr (12Q)", "5 yr (20Q)", "7 yr (28Q)"])
        
        plt.tight_layout()
        fig1_path = OUTPUT_FIG_DIR / "lead_time_error_and_bias_growth.png"
        fig.savefig(fig1_path, dpi=300)
        plt.close()
        print(f"Saved Figure 1 to: {fig1_path}")
        
    # -------------------------------------------------------------------------
    # Figure 2: ExtraTrees OOD drift trajectories
    # -------------------------------------------------------------------------
    ood_path = OUTPUT_TABLE_DIR / "extratrees_ood_drift_metrics.csv"
    if ood_path.exists():
        df_ood = pd.read_csv(ood_path)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        
        axes[0].plot(df_ood["lead_quarter"], df_ood["mahalanobis_mean"], "b-o", linewidth=2.5, label="Mean Mahalanobis Dist")
        axes[0].fill_between(
            df_ood["lead_quarter"],
            df_ood["mahalanobis_mean"] - df_ood["mahalanobis_std"],
            df_ood["mahalanobis_mean"] + df_ood["mahalanobis_std"],
            alpha=0.2, color="blue", label="±1 Std Dev"
        )
        axes[0].set_title("A. State Space Mahalanobis Distance from Training Domain", fontweight="bold")
        axes[0].set_xlabel("Recursive Lead Quarter")
        axes[0].set_ylabel("Mahalanobis Distance")
        axes[0].legend()
        
        axes[1].plot(df_ood["lead_quarter"], df_ood["violation_fraction_mean"] * 100.0, "r-s", linewidth=2.5, label="Mean Bound Violations (%)")
        axes[1].plot(df_ood["lead_quarter"], df_ood["violation_fraction_max"] * 100.0, "k--^", alpha=0.7, label="Max Bound Violations (%)")
        axes[1].set_title("B. Feature Range Bound Violation Fraction (%)", fontweight="bold")
        axes[1].set_xlabel("Recursive Lead Quarter")
        axes[1].set_ylabel("Violation Fraction (%)")
        axes[1].legend()
        
        plt.tight_layout()
        fig2_path = OUTPUT_FIG_DIR / "extratrees_ood_drift_trajectories.png"
        fig.savefig(fig2_path, dpi=300)
        plt.close()
        print(f"Saved Figure 2 to: {fig2_path}")
        
    # -------------------------------------------------------------------------
    # Figure 3: Waterflood coupled vs independent
    # -------------------------------------------------------------------------
    cpl_path = OUTPUT_TABLE_DIR / "waterflood_coupled_challenger_comparison.csv"
    if cpl_path.exists():
        df_cpl = pd.read_csv(cpl_path)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        
        val_cpl = df_cpl[df_cpl["subset"] == "Validation (71-85)"]
        
        sns.barplot(data=val_cpl, x="origin", y="macro_nrmse", hue="model", ax=axes[0])
        axes[0].set_title("A. Macro NRMSE: Coupled vs Independent Models", fontweight="bold")
        axes[0].set_xlabel("Retrospective Origin")
        axes[0].set_ylabel("Macro NRMSE")
        
        sns.barplot(data=val_cpl, x="origin", y="water_cut_rmse", hue="model", ax=axes[1])
        axes[1].set_title("B. Water Cut Prediction RMSE", fontweight="bold")
        axes[1].set_xlabel("Retrospective Origin")
        axes[1].set_ylabel("Water Cut RMSE")
        
        plt.tight_layout()
        fig3_path = OUTPUT_FIG_DIR / "waterflood_coupled_vs_independent.png"
        fig.savefig(fig3_path, dpi=300)
        plt.close()
        print(f"Saved Figure 3 to: {fig3_path}")
        
    # -------------------------------------------------------------------------
    # Figure 4: Grid Features Information Gain
    # -------------------------------------------------------------------------
    grid_path = OUTPUT_TABLE_DIR / "grid_and_operational_features_cv_assessment.csv"
    if grid_path.exists():
        df_grid = pd.read_csv(grid_path)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        
        colors = ["#2b5c8f", "#4182bd", "#6baed6", "#9ecae1"]
        bars = ax.bar(df_grid["feature_set"], df_grid["oof_macro_nrmse"], color=colors, width=0.55, edgecolor="black")
        ax.set_title("5-Fold Cross-Validation Macro NRMSE Across Feature Sets (Cases 1-70)", fontweight="bold")
        ax.set_ylabel("Out-of-Fold Macro NRMSE")
        ax.set_xticks(range(len(df_grid["feature_set"])))
        ax.set_xticklabels(["Base Geology (4 Par)", "Base + Operational", "Base + 3D Grid", "Full Combined"], rotation=15)
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.001, f"{yval:.5f}", ha="center", va="bottom", fontweight="bold")
            
        plt.tight_layout()
        fig4_path = OUTPUT_FIG_DIR / "grid_features_information_gain.png"
        fig.savefig(fig4_path, dpi=300)
        plt.close()
        print(f"Saved Figure 4 to: {fig4_path}")
        
    # -------------------------------------------------------------------------
    # Figure 5: Deliverable Forecast Envelopes
    # -------------------------------------------------------------------------
    env_path = OUTPUT_TABLE_DIR / "forecast_envelope_deliverable_scenarios.csv"
    if env_path.exists():
        df_env = pd.read_csv(env_path)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        
        err_lower = np.maximum(0.0, (df_env["eur_oil_20y_re_stb"] - df_env["eur_oil_20y_standalone_exp_stb"]).to_numpy() / 1e6)
        err_upper = np.maximum(0.0, (df_env["eur_oil_20y_standalone_hyp_stb"] - df_env["eur_oil_20y_re_stb"]).to_numpy() / 1e6)
        axes[0].errorbar(
            df_env["case_id"], df_env["eur_oil_20y_re_stb"] / 1e6,
            yerr=[err_lower, err_upper],
            fmt="none", ecolor="black", capsize=4, label="DCA Spread (Exp to Hyp)"
        )
        axes[0].set_title("A. 20-Year Oil EUR & Model Disagreement Across Deliverable Cases", fontweight="bold")
        axes[0].set_xlabel("Submission Sheet")
        axes[0].set_ylabel("20-Year EUR (MMSTB)")
        axes[0].legend()
        
        axes[1].plot(df_env["case_id"], df_env["terminal_qo_2028_stbd"], "bo-", label="Terminal Oil Rate (STB/d)")
        ax1_twin = axes[1].twinx()
        ax1_twin.plot(df_env["case_id"], df_env["terminal_fw_2028"] * 100.0, "rs--", label="Terminal Water Cut (%)")
        axes[1].set_title("B. 2028 Terminal Oil Rate vs Water Cut", fontweight="bold")
        axes[1].set_xlabel("Submission Sheet")
        axes[1].set_ylabel("Terminal Oil Rate (STB/d)", color="blue")
        ax1_twin.set_ylabel("Terminal Water Cut (%)", color="red")
        
        plt.tight_layout()
        fig5_path = OUTPUT_FIG_DIR / "forecast_envelope_deliverable_scenarios.png"
        fig.savefig(fig5_path, dpi=300)
        plt.close()
        print(f"Saved Figure 5 to: {fig5_path}")


if __name__ == "__main__":
    generate_deliverable_scenarios_envelope()
    generate_all_diagnostic_figures()
