"""
Module: dca_target_mismatch_analysis.py
Investigates Priority Hypothesis 1: Decline-Rate Train/Deployment Mismatch.
Mathematical and empirical investigation of LearnedDCARateForecaster.fit():
1. Pre-cutoff history fit (2002-2008) vs post-cutoff future fit (2001-2008).
2. Unstandardized vs standardized feature scaling in RidgeCV.
3. Feature sensitivity compression on geological parameters (fault, poro, perm, aq).
4. Long-horizon EUR impact and bias quantification.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats

from ensemble.data import (
    load_uncertainty,
    resolve_shared_dir,
    TRAIN_CASES,
    VAL_CASES,
    PARAMS,
)
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    YEAR_DAYS,
    RS_SOLUTION,
)

OUTPUT_TABLE_DIR = Path("outputs/research/physics_first/tables")
OUTPUT_FIG_DIR = Path("outputs/research/physics_first/figures")
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)


def analyze_dca_target_mismatch():
    print("=" * 80)
    print("HYPOTHESIS 1: DCA RATE TRAINING / DEPLOYMENT TARGET MISMATCH AUDIT")
    print("=" * 80)
    
    from ensemble.run_horizon_adaptive_stacking_experiment import (
        load_clean_simulation_quarterly,
        build_case_quarterly_trajectories,
        YEAR_DAYS,
        RS_SOLUTION,
    )
    
    unc_df = load_uncertainty()
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)
        
    records = []
    
    for c in list(TRAIN_CASES) + list(VAL_CASES):
        df_c = case_dict[c]
        u_c = unc_df.loc[c]
        
        # 1. Historical pre-cutoff window: 2002-01-01 to 2008-01-01 (6 years, 24 quarters)
        hist_win = df_c[(df_c["date"] >= "2002-01-01") & (df_c["date"] <= "2008-01-01")]
        t_hist_y = (hist_win["date"] - hist_win["date"].iloc[0]).dt.total_seconds() / (86400.0 * YEAR_DAYS)
        qo_hist = hist_win["oil_rate_stbd"].values
        qw_hist = hist_win["water_rate_stbd"].values
        
        log_qo_hist = np.log(np.maximum(qo_hist, 1.0))
        poly_o_hist = np.polyfit(t_hist_y, log_qo_hist, 1)
        D_hist = max(0.005, -poly_o_hist[0])
        w_growth_hist = np.polyfit(t_hist_y, qw_hist, 1)[0]
        
        # 2. Future window post-cutoff (if cutoff = 2001-01-01, 7 years, 28 quarters)
        fut_win = df_c[(df_c["date"] > "2001-01-01") & (df_c["date"] <= "2008-01-01")]
        t_fut_y = (fut_win["date"] - fut_win["date"].iloc[0]).dt.total_seconds() / (86400.0 * YEAR_DAYS)
        qo_fut = fut_win["oil_rate_stbd"].values
        qw_fut = fut_win["water_rate_stbd"].values
        
        log_qo_fut = np.log(np.maximum(qo_fut, 1.0))
        poly_o_fut = np.polyfit(t_fut_y, log_qo_fut, 1)
        D_future = max(0.005, -poly_o_fut[0])
        w_growth_fut = np.polyfit(t_fut_y, qw_fut, 1)[0]
        
        # Anchor state at 2008-01-01
        last_row = df_c[df_c["date"] == "2008-01-01"].iloc[0]
        
        records.append({
            "case_id": c,
            "split": "train" if c in TRAIN_CASES else "val",
            "fault": float(u_c["Fault Transmissibility"]),
            "poro": float(u_c["Porosity Multiplier"]),
            "perm": float(u_c["Permeability Multiplier"]),
            "aq": float(u_c["Aquifer Pore Volume"]),
            "qo_2008": float(last_row["oil_rate_stbd"]),
            "qw_2008": float(last_row["water_rate_stbd"]),
            "Qo_2008": float(last_row["oil_cum_stb"]),
            "Qw_2008": float(last_row["water_cum_stb"]),
            "wor_2008": float(last_row["wor"]),
            "D_hist": D_hist,
            "D_future": D_future,
            "D_diff": D_hist - D_future,
            "w_growth_hist": w_growth_hist,
            "w_growth_fut": w_growth_fut,
        })
        
    df_analysis = pd.DataFrame(records)
    
    # Save target comparison table
    target_comp_path = OUTPUT_TABLE_DIR / "dca_target_comparison_hist_vs_future.csv"
    df_analysis.to_csv(target_comp_path, index=False)
    print(f"Saved DCA target comparison table: {target_comp_path}")
    
    # Statistical analysis of D_hist vs D_future
    mean_d_hist = df_analysis["D_hist"].mean()
    mean_d_fut = df_analysis["D_future"].mean()
    std_d_hist = df_analysis["D_hist"].std()
    std_d_fut = df_analysis["D_future"].std()
    t_stat, p_val = stats.ttest_rel(df_analysis["D_hist"], df_analysis["D_future"])
    
    print(f"\n[Target Statistics across N={len(df_analysis)} Cases]")
    print(f"  Historical D (2002-2008): Mean = {mean_d_hist:.5f} /yr (std = {std_d_hist:.5f})")
    print(f"  Future D (2001-2008):     Mean = {mean_d_fut:.5f} /yr (std = {std_d_fut:.5f})")
    print(f"  Mean Difference (Bias):   {mean_d_hist - mean_d_fut:+.5f} /yr")
    print(f"  Paired t-test: t = {t_stat:.3f}, p = {p_val:.4e}")
    
    # Feature scaling comparison: Unscaled vs Scaled RidgeCV
    train_mask = df_analysis["split"] == "train"
    val_mask = df_analysis["split"] == "val"
    
    feat_cols = ["fault", "poro", "perm", "aq", "qo_2008", "qw_2008", "wor_2008"]
    X_train = df_analysis.loc[train_mask, feat_cols].values
    y_train_D = df_analysis.loc[train_mask, "D_hist"].values
    
    X_val = df_analysis.loc[val_mask, feat_cols].values
    y_val_D = df_analysis.loc[val_mask, "D_hist"].values
    
    # Model 1: Legacy Unscaled RidgeCV
    ridge_unscaled = RidgeCV(alphas=np.logspace(-3, 3, 10))
    ridge_unscaled.fit(X_train, y_train_D)
    pred_unscaled_val = ridge_unscaled.predict(X_val)
    
    # Model 2: Standardized Pipeline (StandardScaler + RidgeCV)
    pipe_scaled = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 10)))
    ])
    pipe_scaled.fit(X_train, y_train_D)
    pred_scaled_val = pipe_scaled.predict(X_val)
    
    # Coefficients comparison
    unscaled_coefs = ridge_unscaled.coef_
    scaled_coefs = pipe_scaled.named_steps["ridge"].coef_
    
    # Effective feature importance (std(X) * coef)
    x_stds = np.std(X_train, axis=0)
    effective_imp_unscaled = unscaled_coefs * x_stds
    effective_imp_scaled = scaled_coefs
    
    coef_comp_df = pd.DataFrame({
        "feature": feat_cols,
        "feature_std": x_stds,
        "unscaled_raw_coef": unscaled_coefs,
        "unscaled_effective_weight": effective_imp_unscaled,
        "standardized_coef": scaled_coefs,
    })
    coef_path = OUTPUT_TABLE_DIR / "dca_feature_scaling_impact.csv"
    coef_comp_df.to_csv(coef_path, index=False)
    print(f"\nSaved feature scaling comparison: {coef_path}")
    print(coef_comp_df.to_string(index=False))
    
    # Generalization metrics on Validation set (Cases 71-85)
    mse_unscaled = np.mean((pred_unscaled_val - y_val_D) ** 2)
    mse_scaled = np.mean((pred_scaled_val - y_val_D) ** 2)
    mae_unscaled = np.mean(np.abs(pred_unscaled_val - y_val_D))
    mae_scaled = np.mean(np.abs(pred_scaled_val - y_val_D))
    
    print(f"\n[Validation Set D Prediction Metrics (Cases 71-85)]")
    print(f"  Unscaled RidgeCV:    RMSE = {np.sqrt(mse_unscaled):.5f}, MAE = {mae_unscaled:.5f}")
    print(f"  Standardized Ridge:  RMSE = {np.sqrt(mse_scaled):.5f}, MAE = {mae_scaled:.5f}")
    
    # Evaluate 20-Year EUR Sensitivity on Field Realizations
    # Compare spread in D_pred and resulting 20-year incremental cumulative oil
    from ensemble.scenario_loader import load_production_scenarios
    scenarios = load_production_scenarios()
    
    c0_anchor = {
        "qo": 5066.68,
        "qw": 7926.36,
        "wor": 7926.36 / 5066.68,
    }
    
    scenario_eval = []
    t_20y = 20.0  # 20 years
    
    for sc in scenarios:
        x_sc = np.array([[
            float(sc.params["Fault Transmissibility"]),
            float(sc.params["Porosity Multiplier"]),
            float(sc.params["Permeability Multiplier"]),
            float(sc.params["Aquifer Pore Volume"]),
            c0_anchor["qo"],
            c0_anchor["qw"],
            c0_anchor["wor"],
        ]])
        
        d_unsc = float(ridge_unscaled.predict(x_sc)[0])
        d_sc = float(pipe_scaled.predict(x_sc)[0])
        
        # Exponential decline EUR incremental: Q_inc = (q0 / D) * (1 - exp(-D * 20)) * 365.25
        q_inc_unsc = (c0_anchor["qo"] / d_unsc) * (1.0 - np.exp(-d_unsc * t_20y)) * 365.25
        q_inc_sc = (c0_anchor["qo"] / d_sc) * (1.0 - np.exp(-d_sc * t_20y)) * 365.25
        
        scenario_eval.append({
            "case_id": sc.case_id,
            "candidate_id": sc.candidate_id,
            "D_unscaled": d_unsc,
            "D_standardized": d_sc,
            "Q_inc_unscaled_stb": q_inc_unsc,
            "Q_inc_standardized_stb": q_inc_sc,
        })
        
    df_sc_eval = pd.DataFrame(scenario_eval)
    sc_eval_path = OUTPUT_TABLE_DIR / "scenario_dca_sensitivity_comparison.csv"
    df_sc_eval.to_csv(sc_eval_path, index=False)
    print(f"\nSaved Scenario DCA sensitivity comparison: {sc_eval_path}")
    
    eur_spread_unsc = df_sc_eval["Q_inc_unscaled_stb"].max() - df_sc_eval["Q_inc_unscaled_stb"].min()
    eur_spread_sc = df_sc_eval["Q_inc_standardized_stb"].max() - df_sc_eval["Q_inc_standardized_stb"].min()
    
    print(f"\n[20-Year DCA Incremental Oil Spread Across 10 Scenarios]")
    print(f"  Unscaled Ridge EUR Spread:     {eur_spread_unsc:>12,.0f} STB (D range: {df_sc_eval['D_unscaled'].min():.4f} - {df_sc_eval['D_unscaled'].max():.4f})")
    print(f"  Standardized Ridge EUR Spread: {eur_spread_sc:>12,.0f} STB (D range: {df_sc_eval['D_standardized'].min():.4f} - {df_sc_eval['D_standardized'].max():.4f})")
    print(f"  Spread Expansion Ratio:        {eur_spread_sc / max(1.0, eur_spread_unsc):.2f}x")
    
    # Generate Diagnostic Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Subplot 1: Target distribution D_hist vs D_future
    ax = axes[0]
    ax.scatter(df_analysis["D_future"], df_analysis["D_hist"], c="darkblue", alpha=0.6, edgecolors="none", s=50, label="Simulation Cases")
    lims = [min(df_analysis["D_future"].min(), df_analysis["D_hist"].min()) * 0.9,
            max(df_analysis["D_future"].max(), df_analysis["D_hist"].max()) * 1.1]
    ax.plot(lims, lims, "r--", label="1:1 Parity")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Future Fit D (2001-2008) [1/year]", fontweight="bold")
    ax.set_ylabel("Historical Fit D (2002-2008) [1/year]", fontweight="bold")
    ax.set_title("Decline Target Regime Asymmetry\n(Pre-cutoff Tail vs Post-cutoff Future)", fontsize=11, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Subplot 2: Feature Importance Compression
    ax2 = axes[1]
    features_labels = ["Fault", "Porosity", "Perm", "Aquifer", "Qo_0", "Qw_0", "WOR_0"]
    x_pos = np.arange(len(features_labels))
    width = 0.35
    
    ax2.bar(x_pos - width/2, np.abs(coef_comp_df["unscaled_effective_weight"]), width, label="|Unscaled Effective Weight|", color="#e74c3c", edgecolor="black")
    ax2.bar(x_pos + width/2, np.abs(coef_comp_df["standardized_coef"]), width, label="|Standardized Coef|", color="#2ecc71", edgecolor="black")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(features_labels, rotation=30, ha="right", fontweight="bold")
    ax2.set_ylabel("Normalized Feature Sensitivity", fontweight="bold")
    ax2.set_title("Feature Sensitivity Compression in Unscaled RidgeCV\n(Suppression of Geological Multipliers)", fontsize=11, fontweight="bold")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    fig_path = OUTPUT_FIG_DIR / "dca_mismatch_and_scaling_compression.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved diagnostic figure: {fig_path}")
    
    summary = {
        "mean_d_hist": float(mean_d_hist),
        "mean_d_fut": float(mean_d_fut),
        "d_bias": float(mean_d_hist - mean_d_fut),
        "paired_t_pvalue": float(p_val),
        "rmse_unscaled_val": float(np.sqrt(mse_unscaled)),
        "rmse_scaled_val": float(np.sqrt(mse_scaled)),
        "eur_spread_unscaled_stb": float(eur_spread_unsc),
        "eur_spread_standardized_stb": float(eur_spread_sc),
        "spread_expansion_ratio": float(eur_spread_sc / max(1.0, eur_spread_unsc)),
    }
    with open(OUTPUT_TABLE_DIR / "dca_mismatch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    return summary


if __name__ == "__main__":
    analyze_dca_target_mismatch()
