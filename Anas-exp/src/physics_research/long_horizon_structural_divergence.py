"""
Module: long_horizon_structural_divergence.py
Investigates Priority Hypothesis 4: Structural Forecast Non-Identifiability Beyond 2008.
Grounding in Kennedy & O'Hagan (2001) Discrepancy Framework:
1. Decomposes forecast uncertainty into Model Discrepancy Variance vs Parameter Sensitivity Variance.
2. Evaluates the 4 base model architectures across the 80-quarter forecast horizon.
3. Quantifies inter-model divergence vs intra-scenario parameter spread over time.
4. Analyzes why parameter identifiability vanishes into terminal tail dynamics as t -> 20 years.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    RecursiveTransitionForecaster,
    LearnedDCARateForecaster,
    HybridDynamicForecaster,
    YEAR_DAYS,
    RS_SOLUTION,
)
from ensemble.scenario_loader import load_production_scenarios

OUTPUT_TABLE_DIR = Path("outputs/research/physics_first/tables")
OUTPUT_FIG_DIR = Path("outputs/research/physics_first/figures")
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)


def analyze_structural_divergence():
    print("=" * 80)
    print("HYPOTHESIS 4: LONG-HORIZON STRUCTURAL DIVERGENCE (KENNEDY & O'HAGAN 2001)")
    print("=" * 80)
    
    unc_df = load_uncertainty()
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)
    
    shared_dir = resolve_shared_dir()
    prod = pd.read_parquet(shared_dir / "production_timeseries.parquet")
    c0_2008 = prod[(prod["case_num"] == 0) & (prod["date"] == "2008-01-01")].set_index("metric")["value"]
    
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
    
    from ensemble.generate_competition_deliverable import fit_locked_base_models
    models = fit_locked_base_models(case_dict, unc_df, TRAIN_CASES, pd.Timestamp("2008-01-01"))
    scenarios = load_production_scenarios()
    
    forecast_dates_80q = pd.date_range("2008-04-01", periods=80, freq="QS")
    dt_days = 91.3125
    
    # Store predictions for each model and each scenario across all 80 quarters
    # Dimensions: (n_models, n_scenarios, 80)
    model_names = [
        "Recursive_ExtraTrees",
        "Learned_Exponential_DCA",
        "Learned_Hyperbolic_DCA",
        "Hybrid_Dynamic",
    ]
    
    pred_oil_cum = {m: np.zeros((len(scenarios), 80)) for m in model_names}
    pred_wat_cum = {m: np.zeros((len(scenarios), 80)) for m in model_names}
    
    # Evaluate individual models
    for s_idx, sc in enumerate(scenarios):
        u_c = sc.params
        
        # 1. ExtraTrees
        r_et = models["Recursive_ExtraTrees"].predict_rates(field_anchor, u_c, forecast_dates_80q)
        pred_oil_cum["Recursive_ExtraTrees"][s_idx] = field_anchor["Qo"] + np.cumsum(r_et["oil_rate"] * dt_days)
        pred_wat_cum["Recursive_ExtraTrees"][s_idx] = field_anchor["Qw"] + np.cumsum(r_et["water_rate"] * dt_days)
        
        # 2. Exp DCA
        r_exp = models["Learned_Exponential_DCA"].predict_rates(field_anchor, u_c, forecast_dates_80q)
        pred_oil_cum["Learned_Exponential_DCA"][s_idx] = field_anchor["Qo"] + np.cumsum(r_exp["oil_rate"] * dt_days)
        pred_wat_cum["Learned_Exponential_DCA"][s_idx] = field_anchor["Qw"] + np.cumsum(r_exp["water_rate"] * dt_days)
        
        # 3. Hyp DCA
        r_hyp = models["Learned_Hyperbolic_DCA"].predict_rates(field_anchor, u_c, forecast_dates_80q)
        pred_oil_cum["Learned_Hyperbolic_DCA"][s_idx] = field_anchor["Qo"] + np.cumsum(r_hyp["oil_rate"] * dt_days)
        pred_wat_cum["Learned_Hyperbolic_DCA"][s_idx] = field_anchor["Qw"] + np.cumsum(r_hyp["water_rate"] * dt_days)
        
        # 4. Hybrid
        r_hyb = models["Hybrid_Dynamic"].predict_rates(field_anchor, u_c, forecast_dates_80q)
        pred_oil_cum["Hybrid_Dynamic"][s_idx] = field_anchor["Qo"] + np.cumsum(r_hyb["oil_rate"] * dt_days)
        pred_wat_cum["Hybrid_Dynamic"][s_idx] = field_anchor["Qw"] + np.cumsum(r_hyb["water_rate"] * dt_days)
        
    # Variance decomposition per quarter k in [1..80]:
    # Parametric variance: Var_s(E_m[Q_m,s(k)])
    # Model structural variance: Var_m(E_s[Q_m,s(k)])
    quarters = np.arange(1, 81)
    years = quarters * 0.25
    
    var_param_list = []
    var_model_list = []
    ratio_list = []
    
    records_quarterly = []
    
    for k in range(80):
        # 2D matrix: models (rows) x scenarios (cols)
        mat_k = np.array([pred_oil_cum[m][:, k] for m in model_names])
        
        # Mean across models for each scenario: shape (n_scenarios,)
        mean_over_models = np.mean(mat_k, axis=0)
        var_param_k = float(np.var(mean_over_models))
        
        # Mean across scenarios for each model: shape (n_models,)
        mean_over_scenarios = np.mean(mat_k, axis=1)
        var_model_k = float(np.var(mean_over_scenarios))
        
        total_var = var_param_k + var_model_k
        ratio_k = var_model_k / max(1.0, var_param_k)
        
        var_param_list.append(var_param_k)
        var_model_list.append(var_model_k)
        ratio_list.append(ratio_k)
        
        records_quarterly.append({
            "quarter": k + 1,
            "years_ahead": years[k],
            "var_parameter": var_param_k,
            "std_parameter_stb": np.sqrt(var_param_k),
            "var_model_structural": var_model_k,
            "std_model_stb": np.sqrt(var_model_k),
            "ratio_model_to_param": ratio_k,
            "model_variance_share_pct": (var_model_k / max(1e-6, total_var)) * 100.0,
            "et_mean_cum_stb": float(mean_over_scenarios[0]),
            "exp_mean_cum_stb": float(mean_over_scenarios[1]),
            "hyp_mean_cum_stb": float(mean_over_scenarios[2]),
            "hyb_mean_cum_stb": float(mean_over_scenarios[3]),
        })
        
    df_var = pd.DataFrame(records_quarterly)
    var_path = OUTPUT_TABLE_DIR / "structural_vs_parameter_variance_decomposition.csv"
    df_var.to_csv(var_path, index=False)
    print(f"Saved structural variance decomposition table: {var_path}")
    
    # Milestone Horizon Audit
    q4 = df_var[df_var["quarter"] == 4].iloc[0]    # 1 year
    q20 = df_var[df_var["quarter"] == 20].iloc[0]  # 5 years
    q40 = df_var[df_var["quarter"] == 40].iloc[0]  # 10 years
    q80 = df_var[df_var["quarter"] == 80].iloc[0]  # 20 years
    
    print("\n[Uncertainty Decomposition Across Time Horizons]")
    print(f"  Horizon  1 Year  (4Q):  Param Std = {q4['std_parameter_stb']:>9,.0f} STB | Model Std = {q4['std_model_stb']:>9,.0f} STB | Model/Param Ratio = {q4['ratio_model_to_param']:>6.1f}x ({q4['model_variance_share_pct']:.1f}% Model)")
    print(f"  Horizon  5 Years (20Q): Param Std = {q20['std_parameter_stb']:>9,.0f} STB | Model Std = {q20['std_model_stb']:>9,.0f} STB | Model/Param Ratio = {q20['ratio_model_to_param']:>6.1f}x ({q20['model_variance_share_pct']:.1f}% Model)")
    print(f"  Horizon 10 Years (40Q): Param Std = {q40['std_parameter_stb']:>9,.0f} STB | Model Std = {q40['std_model_stb']:>9,.0f} STB | Model/Param Ratio = {q40['ratio_model_to_param']:>6.1f}x ({q40['model_variance_share_pct']:.1f}% Model)")
    print(f"  Horizon 20 Years (80Q): Param Std = {q80['std_parameter_stb']:>9,.0f} STB | Model Std = {q80['std_model_stb']:>9,.0f} STB | Model/Param Ratio = {q80['ratio_model_to_param']:>6.1f}x ({q80['model_variance_share_pct']:.1f}% Model)")
    
    # Diagnostic Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Subplot 1: Model Structural Discrepancy vs Parameter Uncertainty Std Dev over 80Q
    ax = axes[0]
    ax.plot(years, df_var["std_model_stb"] / 1e6, label="Model Structural Uncertainty (sigma_model)", color="#e74c3c", lw=2.5)
    ax.plot(years, df_var["std_parameter_stb"] / 1e6, label="Parameter Uncertainty (sigma_param)", color="#3498db", lw=2.5, linestyle="--")
    ax.set_xlabel("Forecast Horizon (Years Beyond 2008)", fontweight="bold")
    ax.set_ylabel("Standard Deviation (MMSTB)", fontweight="bold")
    ax.set_title("Forecast Uncertainty Expansion over 20-Year Horizon\n(Model Form vs Geological Parameters)", fontsize=11, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Subplot 2: 20-Year Cumulative Production Trajectories by Base Architecture
    ax2 = axes[1]
    palette = {
        "Recursive_ExtraTrees": "#2ecc71",
        "Learned_Exponential_DCA": "#3498db",
        "Learned_Hyperbolic_DCA": "#e67e22",
        "Hybrid_Dynamic": "#9b59b6",
    }
    
    for m in model_names:
        # Plot mean across scenarios + envelope
        m_arr = pred_oil_cum[m] / 1e6
        m_mean = np.mean(m_arr, axis=0)
        m_min = np.min(m_arr, axis=0)
        m_max = np.max(m_arr, axis=0)
        ax2.plot(years, m_mean, label=m.replace("_", " "), color=palette[m], lw=2.0)
        ax2.fill_between(years, m_min, m_max, color=palette[m], alpha=0.15)
        
    ax2.axvline(x=10.0, color="gray", linestyle=":", label="10-Year Horizon (Case 1)")
    ax2.axvline(x=20.0, color="black", linestyle=":", label="20-Year Horizon (Cases 2-10)")
    ax2.set_xlabel("Forecast Horizon (Years Beyond 2008)", fontweight="bold")
    ax2.set_ylabel("Cumulative Oil Production (MMSTB)", fontweight="bold")
    ax2.set_title("Base Model Trajectory Divergence (80 Quarters)\n(Kennedy & O'Hagan Simulator Discrepancy delta(t))", fontsize=11, fontweight="bold")
    ax2.legend(frameon=True, fontsize=8.5)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    fig_path = OUTPUT_FIG_DIR / "structural_divergence_and_uncertainty_growth.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved diagnostic figure: {fig_path}")
    
    summary = {
        "q80_param_std_stb": float(q80["std_parameter_stb"]),
        "q80_model_std_stb": float(q80["std_model_stb"]),
        "q80_ratio_model_to_param": float(q80["ratio_model_to_param"]),
        "q80_model_variance_share_pct": float(q80["model_variance_share_pct"]),
        "models_terminal_oil_mmstb": {
            "Recursive_ExtraTrees": float(q80["et_mean_cum_stb"] / 1e6),
            "Learned_Exponential_DCA": float(q80["exp_mean_cum_stb"] / 1e6),
            "Learned_Hyperbolic_DCA": float(q80["hyp_mean_cum_stb"] / 1e6),
            "Hybrid_Dynamic": float(q80["hyb_mean_cum_stb"] / 1e6),
        }
    }
    with open(OUTPUT_TABLE_DIR / "structural_divergence_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    return summary


if __name__ == "__main__":
    analyze_structural_divergence()
