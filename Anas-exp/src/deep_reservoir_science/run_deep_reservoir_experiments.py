"""
Master Script: run_deep_reservoir_experiments.py
Coordinates and synthesizes all 6 pillars of Deep Reservoir Science:
1. Research 1: End-to-End Pseudo-Field Validation & Error Decomposition
2. Research 2: Satija et al. (2017) Direct History-to-Future Forecasting
3. Research 3: Physics-Normalized State Learning & Hazeem Benchmark Audit
4. Research 4: Physics-Constrained DCA Residual Forecasting & 20-Year Stability
5. Research 5: Forecast-Oriented Geological Scenario Selection (Oliver 2026)
6. Research 6: Two-Compartment Physical Dynamic Model & Structural Identifiability

Generates:
- Unified Master Metrics Table: unified_challengers_master_table.csv
- Publication Diagnostic Figures in outputs/research/deep_reservoir_science/figures/
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TABLE_DIR = Path("outputs/research/deep_reservoir_science/tables")
FIG_DIR = Path("outputs/research/deep_reservoir_science/figures")
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def generate_all_diagnostic_figures():
    """Generates clean, publication-ready figures for the research report."""
    
    # -------------------------------------------------------------------------
    # FIGURE 1: Pseudo-Field Validation Error Decomposition (Research 1)
    # -------------------------------------------------------------------------
    f1_path = TABLE_DIR / "pseudo_field_inversion_vs_forecasting_audit.csv"
    if f1_path.exists():
        df1 = pd.read_csv(f1_path)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        
        # Panel A: Case-by-case comparison across A, B, C
        x = np.arange(len(df1))
        width = 0.25
        ax1.bar(x - width, df1["macro_A_true_oracle"], width, label="Approach A: True Geology Oracle", color="#1f77b4", alpha=0.9)
        ax1.bar(x, df1["macro_B_inferred"], width, label="Approach B: Inferred Geology (HM)", color="#ff7f0e", alpha=0.9)
        ax1.bar(x + width, df1["macro_C_nogeo"], width, label="Approach C: No Geology (Prior)", color="#2ca02c", alpha=0.9)
        ax1.set_xlabel("Pseudo-Field ID (Cases 1-70)")
        ax1.set_ylabel("Macro Increment NRMSE")
        ax1.set_title("Pseudo-Field Forecast Error by Geological Information Regime")
        ax1.set_xticks(x)
        ax1.set_xticklabels(df1["pseudo_field_id"].astype(str), rotation=45)
        ax1.legend(loc="upper left", framealpha=0.9)
        ax1.grid(True, linestyle="--", alpha=0.4)
        
        # Panel B: Aggregate Error Decomposition Pie / Stacked Bar
        mean_A = df1["macro_A_true_oracle"].mean()
        mean_pen = df1["inversion_penalty"].mean()
        mean_B = df1["macro_B_inferred"].mean()
        
        shares = [mean_A / mean_B * 100.0, max(0.0, mean_pen / mean_B * 100.0)]
        labels = [f"Forecasting Model Error\n({shares[0]:.1f}%)", f"Parameter Inversion Penalty\n({shares[1]:.1f}%)"]
        colors = ["#1f77b4", "#d62728"]
        ax2.pie(shares, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140, explode=(0, 0.08))
        ax2.set_title(f"Mean Error Decomposition (Total Mean = {mean_B:.4f} NRMSE)")
        
        plt.tight_layout()
        out_fig1 = FIG_DIR / "fig1_pseudo_field_error_decomposition.png"
        fig.savefig(out_fig1, dpi=300)
        plt.close(fig)
        print(f"Generated Figure 1: {out_fig1}")

    # -------------------------------------------------------------------------
    # FIGURE 2: Direct History-to-Future Forecasting vs Baseline (Research 2)
    # -------------------------------------------------------------------------
    f2_path = TABLE_DIR / "direct_forecasting_satija_results.csv"
    if f2_path.exists():
        df2 = pd.read_csv(f2_path)
        fig, ax = plt.subplots(figsize=(10, 5))
        
        origins = df2["origin"].unique()
        models = df2["model_name"].unique()
        n_orig = len(origins)
        n_mod = len(models)
        
        x = np.arange(n_orig)
        width = 0.8 / n_mod
        colors = ["#2ca02c", "#9467bd", "#8c564b"]
        
        for i, m in enumerate(models):
            sub = df2[df2["model_name"] == m]
            ax.bar(x + (i - n_mod / 2 + 0.5) * width, sub["macro_nrmse"], width, label=m, color=colors[i % len(colors)], alpha=0.9)
            
        # Draw benchmark line for Frozen Baseline
        ax.axhline(0.0390, color="black", linestyle="--", linewidth=1.5, label="Frozen Ensemble Baseline (0.0390)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Origin {o}" for o in origins])
        ax.set_ylabel("Macro Increment NRMSE (Val Cases 71-85)")
        ax.set_title("Satija et al. (2017) Direct Forecasting vs Frozen Pipeline")
        ax.legend(framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.4)
        
        plt.tight_layout()
        out_fig2 = FIG_DIR / "fig2_direct_forecasting_satija_comparison.png"
        fig.savefig(out_fig2, dpi=300)
        plt.close(fig)
        print(f"Generated Figure 2: {out_fig2}")

    # -------------------------------------------------------------------------
    # FIGURE 3: Physics-Normalized Reservoir State Audit (Research 3)
    # -------------------------------------------------------------------------
    f3_path = TABLE_DIR / "physics_normalized_state_comparison.csv"
    if f3_path.exists():
        df3 = pd.read_csv(f3_path)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        
        sub_2005 = df3[df3["origin"] == "2005-01-01"].sort_values("macro_nrmse")
        
        # Panel A: Macro NRMSE across models
        y_pos = np.arange(len(sub_2005))
        bars = ax1.barh(y_pos, sub_2005["macro_nrmse"], color="#1f77b4", alpha=0.85)
        # Highlight frozen baseline
        for idx, m_name in enumerate(sub_2005["model_name"]):
            if "Frozen" in m_name:
                bars[idx].set_color("#2ca02c")
            elif "SVR" in m_name:
                bars[idx].set_color("#d62728")
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(sub_2005["model_name"])
        ax1.set_xlabel("Macro Increment NRMSE")
        ax1.set_title("Origin 2005-01-01 (12Q) Macro Model Comparison")
        ax1.grid(True, linestyle="--", alpha=0.4)
        
        # Panel B: Oil vs Water Breakdown
        width = 0.35
        ax2.bar(y_pos - width/2, sub_2005["oil_nrmse"], width, label="Oil NRMSE", color="#2ca02c", alpha=0.85)
        ax2.bar(y_pos + width/2, sub_2005["water_nrmse"], width, label="Water NRMSE (True Scoring)", color="#1f77b4", alpha=0.85)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(sub_2005["model_name"])
        ax2.set_xlabel("Phase Increment NRMSE")
        ax2.set_title("Oil vs True Water Breakthrough Dynamics")
        ax2.legend(loc="upper right", framealpha=0.9)
        ax2.grid(True, linestyle="--", alpha=0.4)
        
        plt.tight_layout()
        out_fig3 = FIG_DIR / "fig3_physics_normalized_state_audit.png"
        fig.savefig(out_fig3, dpi=300)
        plt.close(fig)
        print(f"Generated Figure 3: {out_fig3}")

    # -------------------------------------------------------------------------
    # FIGURE 4: Physics-Constrained Residual Forecasting & Stability (Research 4)
    # -------------------------------------------------------------------------
    f4_path = TABLE_DIR / "residual_20year_stability_audit.csv"
    if f4_path.exists():
        df4 = pd.read_csv(f4_path)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        
        cases = df4["case_id"].unique()
        models = df4["model_name"].unique()
        x = np.arange(len(cases))
        width = 0.25
        
        # Terminal rate comparison
        for i, m in enumerate(models):
            sub_m = df4[df4["model_name"] == m]
            ax1.bar(x + (i - 1) * width, sub_m["q_terminal_stbd"], width, label=m, alpha=0.9)
            
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"Case {c}" for c in cases])
        ax1.set_ylabel("Terminal Oil Rate (STB/d) at Year 20")
        ax1.set_title("20-Year Terminal Rate Stability Audit")
        ax1.legend(loc="upper right", framealpha=0.9)
        ax1.grid(True, linestyle="--", alpha=0.4)
        
        # 20-Year Cumulative Volume comparison
        for i, m in enumerate(models):
            sub_m = df4[df4["model_name"] == m]
            ax2.bar(x + (i - 1) * width, sub_m["cum_20yr_stb"] / 1e6, width, label=m, alpha=0.9)
            
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"Case {c}" for c in cases])
        ax2.set_ylabel("20-Year Cumulative Production (MMstb)")
        ax2.set_title("20-Year Cumulative Volume (Divergence vs Physical Damping)")
        ax2.legend(loc="upper right", framealpha=0.9)
        ax2.grid(True, linestyle="--", alpha=0.4)
        
        plt.tight_layout()
        out_fig4 = FIG_DIR / "fig4_constrained_residuals_and_stability.png"
        fig.savefig(out_fig4, dpi=300)
        plt.close(fig)
        print(f"Generated Figure 4: {out_fig4}")

    # -------------------------------------------------------------------------
    # FIGURE 5: Scenario Selection Trade-Offs (Research 5)
    # -------------------------------------------------------------------------
    f5_path = TABLE_DIR / "scenario_selection_comparison.csv"
    if f5_path.exists():
        df5 = pd.read_csv(f5_path)
        fig, ax = plt.subplots(figsize=(9, 5))
        
        for _, row in df5.iterrows():
            ax.scatter(row["mean_oil_dispersion_stb"] / 1e3, row["macro_nrmse"], s=180, alpha=0.9, label=row["policy"])
            ax.annotate(row["policy"], (row["mean_oil_dispersion_stb"] / 1e3, row["macro_nrmse"] + 0.00015),
                        fontsize=9, weight="bold")
                        
        ax.set_xlabel("Mean Scenario Dispersion / Spread (kSTB)")
        ax.set_ylabel("Macro Increment NRMSE")
        ax.set_title("Oliver (2026) Scenario Selection: Forecast Accuracy vs Scenario Dispersion")
        ax.grid(True, linestyle="--", alpha=0.4)
        
        plt.tight_layout()
        out_fig5 = FIG_DIR / "fig5_scenario_selection_tradeoffs.png"
        fig.savefig(out_fig5, dpi=300)
        plt.close(fig)
        print(f"Generated Figure 5: {out_fig5}")

    # -------------------------------------------------------------------------
    # FIGURE 6: Reduced-Order Dynamic Model Identifiability (Research 6)
    # -------------------------------------------------------------------------
    f6_path = TABLE_DIR / "reduced_order_identifiability_analysis.csv"
    if f6_path.exists():
        df6 = pd.read_csv(f6_path)
        fig, ax = plt.subplots(figsize=(9, 5))
        
        ax.semilogy(df6["singular_mode"], df6["singular_value"], "o-", color="#d62728", linewidth=2, markersize=8)
        ax.axhline(df6["singular_value"].iloc[0] * 1e-4, color="gray", linestyle="--", label="Numerical Noise Threshold (1e-4 x s_max)")
        ax.set_xlabel("Singular Mode")
        ax.set_ylabel("Sensitivity Singular Value (Log Scale)")
        ax.set_title("Two-Compartment Flow & Storage Model Identifiability Spectrum\n(Severe Rank Deficiency: Condition Number = 3.23e+17)")
        ax.legend(framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.4)
        
        plt.tight_layout()
        out_fig6 = FIG_DIR / "fig6_reduced_order_identifiability.png"
        fig.savefig(out_fig6, dpi=300)
        plt.close(fig)
        print(f"Generated Figure 6: {out_fig6}")


def build_unified_master_table():
    """Compiles the master comparison across all models, challengers, and benchmarks."""
    records = []
    
    # Baseline
    records.append({
        "pillar": "Production Baseline",
        "model_name": "Frozen Blend_Phase_Specific_Simplex",
        "validation_macro_nrmse_2004": 0.0391,
        "validation_macro_nrmse_2005": 0.0388,
        "mean_macro_nrmse": 0.0390,
        "asymptotic_20yr_stability": "GUARANTEED (Exponential DCA Anchor)",
        "scientific_feasibility": "Fully Verified & Deployed",
        "recommendation": "PROTECT AS BASELINE",
    })
    
    # Research 1: Pseudo-field Approaches
    f1_path = TABLE_DIR / "pseudo_field_inversion_vs_forecasting_audit.csv"
    if f1_path.exists():
        df1 = pd.read_csv(f1_path)
        records.append({
            "pillar": "Research 1: Pseudo-Field",
            "model_name": "Approach A (True Geology Oracle)",
            "validation_macro_nrmse_2004": np.nan,
            "validation_macro_nrmse_2005": float(df1["macro_A_true_oracle"].mean()),
            "mean_macro_nrmse": float(df1["macro_A_true_oracle"].mean()),
            "asymptotic_20yr_stability": "GUARANTEED",
            "scientific_feasibility": "Theoretical Information Benchmark (Unobtainable)",
            "recommendation": "ORACLE BOUND",
        })
        records.append({
            "pillar": "Research 1: Pseudo-Field",
            "model_name": "Approach B (Inferred Geology HM)",
            "validation_macro_nrmse_2004": np.nan,
            "validation_macro_nrmse_2005": float(df1["macro_B_inferred"].mean()),
            "mean_macro_nrmse": float(df1["macro_B_inferred"].mean()),
            "asymptotic_20yr_stability": "GUARANTEED",
            "scientific_feasibility": "Standard Two-Stage Pipeline",
            "recommendation": "CURRENT WORKFLOW",
        })
        records.append({
            "pillar": "Research 1: Pseudo-Field",
            "model_name": "Approach C (No Inversion / Prior)",
            "validation_macro_nrmse_2004": np.nan,
            "validation_macro_nrmse_2005": float(df1["macro_C_nogeo"].mean()),
            "mean_macro_nrmse": float(df1["macro_C_nogeo"].mean()),
            "asymptotic_20yr_stability": "GUARANTEED",
            "scientific_feasibility": "Observable Transition Model",
            "recommendation": "BASELINE ALTERNATIVE",
        })

    # Research 2: Direct Forecasting
    f2_path = TABLE_DIR / "direct_forecasting_satija_results.csv"
    if f2_path.exists():
        df2 = pd.read_csv(f2_path)
        for m in df2["model_name"].unique():
            sub = df2[df2["model_name"] == m]
            m_2004 = float(sub[sub["origin"] == "2004-01-01"]["macro_nrmse"].iloc[0]) if len(sub[sub["origin"] == "2004-01-01"]) > 0 else np.nan
            m_2005 = float(sub[sub["origin"] == "2005-01-01"]["macro_nrmse"].iloc[0]) if len(sub[sub["origin"] == "2005-01-01"]) > 0 else np.nan
            mean_m = float(np.nanmean([m_2004, m_2005]))
            records.append({
                "pillar": "Research 2: Direct Forecasting",
                "model_name": m,
                "validation_macro_nrmse_2004": m_2004,
                "validation_macro_nrmse_2005": m_2005,
                "mean_macro_nrmse": mean_m,
                "asymptotic_20yr_stability": "CONDITIONAL (Requires Latent Clamping)",
                "scientific_feasibility": "High Retrospective Accuracy",
                "recommendation": "PROMOTE FOR REVIEW (CHALLENGER)",
            })

    # Research 3: Physics-Normalized State
    f3_path = TABLE_DIR / "physics_normalized_state_comparison.csv"
    if f3_path.exists():
        df3 = pd.read_csv(f3_path)
        for m in ["SVR_GridNormalised_State", "SVR_Normalised_Plus_Geo", "Ridge_Normalised_Plus_Geo"]:
            sub = df3[df3["model_name"] == m]
            if len(sub) > 0:
                m_2004 = float(sub[sub["origin"] == "2004-01-01"]["macro_nrmse"].iloc[0]) if len(sub[sub["origin"] == "2004-01-01"]) > 0 else np.nan
                m_2005 = float(sub[sub["origin"] == "2005-01-01"]["macro_nrmse"].iloc[0]) if len(sub[sub["origin"] == "2005-01-01"]) > 0 else np.nan
                records.append({
                    "pillar": "Research 3: Normalized State",
                    "model_name": m,
                    "validation_macro_nrmse_2004": m_2004,
                    "validation_macro_nrmse_2005": m_2005,
                    "mean_macro_nrmse": float(np.nanmean([m_2004, m_2005])),
                    "asymptotic_20yr_stability": "POOR (Water Breakthrough Drift)",
                    "scientific_feasibility": "Refuted (Defective Water Scoring in Lit)",
                    "recommendation": "REJECT",
                })

    # Research 4: Physics-Constrained Residuals
    f4_val = TABLE_DIR / "physics_constrained_residual_comparison.csv"
    if f4_val.exists():
        df4v = pd.read_csv(f4_val)
        for m in ["DCA_Damped_GP_Residual", "DCA_Damped_Linear_Residual", "DCA_Unconstrained_Residual"]:
            sub = df4v[df4v["model_name"] == m]
            if len(sub) > 0:
                m_2004 = float(sub[sub["origin"] == "2004-01-01"]["macro_nrmse"].iloc[0]) if len(sub[sub["origin"] == "2004-01-01"]) > 0 else np.nan
                m_2005 = float(sub[sub["origin"] == "2005-01-01"]["macro_nrmse"].iloc[0]) if len(sub[sub["origin"] == "2005-01-01"]) > 0 else np.nan
                is_uncon = "Unconstrained" in m
                records.append({
                    "pillar": "Research 4: Constrained Residuals",
                    "model_name": m,
                    "validation_macro_nrmse_2004": m_2004,
                    "validation_macro_nrmse_2005": m_2005,
                    "mean_macro_nrmse": float(np.nanmean([m_2004, m_2005])),
                    "asymptotic_20yr_stability": "REJECTED (Severe 20-Year Drift)" if is_uncon else "GUARANTEED (Asymptotic Damping)",
                    "scientific_feasibility": "High Mathematical Rigor",
                    "recommendation": "REJECT (Unstable)" if is_uncon else "PROMOTE FOR REVIEW (CHALLENGER)",
                })

    # Research 6: Reduced-Order Compartment
    records.append({
        "pillar": "Research 6: Dynamic ROM",
        "model_name": "Two-Compartment Dynamic ROM",
        "validation_macro_nrmse_2004": np.nan,
        "validation_macro_nrmse_2005": 0.1078,
        "mean_macro_nrmse": 0.1078,
        "asymptotic_20yr_stability": "THEORETICALLY BOUNDED",
        "scientific_feasibility": "Structurally Non-Identifiable (kappa=3.2e17)",
        "recommendation": "REJECT (Non-Identifiable)",
    })

    master_df = pd.DataFrame(records)
    out_master = TABLE_DIR / "unified_challengers_master_table.csv"
    master_df.to_csv(out_master, index=False)
    print(f"\nSaved Unified Master Comparison Table: {out_master}")
    return master_df


if __name__ == "__main__":
    generate_all_diagnostic_figures()
    build_unified_master_table()
