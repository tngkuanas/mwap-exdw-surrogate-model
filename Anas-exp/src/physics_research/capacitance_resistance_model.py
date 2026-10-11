"""
Module: capacitance_resistance_model.py
Investigates Priority Hypothesis 3: Physics-Informed Waterflood State-Space Model (CRM).
Formulates, fits, and evaluates a Capacitance-Resistance Model (Sayarpour et al., 2009)
for reservoir waterflood surveillance and long-horizon forecasting.
Compares CRM-T (Field-Level Tank CRM) against the frozen production ensemble
on held-out Validation Cases (71-85).
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

from ensemble.data import (
    load_uncertainty,
    resolve_shared_dir,
    TRAIN_CASES,
    VAL_CASES,
    PARAMS,
)
from ensemble.metrics import compute_increment_nrmse
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
    YEAR_DAYS,
    RS_SOLUTION,
)

OUTPUT_TABLE_DIR = Path("outputs/research/physics_first/tables")
OUTPUT_FIG_DIR = Path("outputs/research/physics_first/figures")
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)


class FieldLevelCRM:
    """
    Field-Level Capacitance-Resistance Model (CRM-T) with logistic fractional flow coupling.
    Governing differential equation for total liquid rate q_L = q_o + q_w:
        tau * dq_L/dt + q_L = f_inj * I(t)
    Under balanced voidage replacement I(t) approx q_L(t_anchor), this yields
    a pressure-supported relaxation dynamics with characteristic time constant tau.
    Oil rate is partitioned via Gentil-Koval logit fractional flow:
        logit(f_w(t)) = logit(f_w,0) + beta_w * (t - t_0)
        q_o(t) = q_L(t) * (1 - f_w(t))
        q_w(t) = q_L(t) * f_w(t)
    """
    def __init__(self):
        self.tau_years = 5.0
        self.beta_w = 0.15
        self.qL_inf_ratio = 1.05
        
    def fit_from_case(self, hist_df: pd.DataFrame):
        """Calibrates tau and beta_w from historical trajectory."""
        dates = hist_df["date"]
        t_years = (dates - dates.iloc[0]).dt.total_seconds() / (86400.0 * YEAR_DAYS)
        qo = hist_df["oil_rate_stbd"].values
        qw = hist_df["water_rate_stbd"].values
        qL = qo + qw
        fw = np.clip(qw / np.maximum(qL, 1.0), 0.001, 0.999)
        logit_fw = np.log(fw / (1.0 - fw))
        
        # Fit fractional flow progression slope beta_w
        # post-breakthrough filter
        valid_idx = np.where(fw > 0.05)[0]
        if len(valid_idx) >= 5:
            poly = np.polyfit(t_years.iloc[valid_idx], logit_fw[valid_idx], 1)
            self.beta_w = max(0.01, float(poly[0]))
        else:
            self.beta_w = 0.15
            
        # Fit liquid relaxation: qL(t) = qL0 + (qL_inf - qL0) * (1 - exp(-t/tau))
        qL0 = qL[0]
        def loss(p):
            tau, qL_inf = p
            if tau <= 0.1:
                return 1e9
            pred_qL = qL0 + (qL_inf - qL0) * (1.0 - np.exp(-t_years.values / tau))
            return np.mean((pred_qL - qL) ** 2)
            
        res = minimize(loss, [5.0, qL[-1]], bounds=[(0.5, 30.0), (qL0 * 0.5, qL0 * 2.5)], method="L-BFGS-B")
        if res.success:
            self.tau_years = float(res.x[0])
            self.qL_inf_ratio = float(res.x[1] / max(1.0, qL[-1]))
        return self

    def predict_forecast(
        self,
        init_state: Dict[str, float],
        forecast_dates: pd.DatetimeIndex,
    ) -> Dict[str, np.ndarray]:
        t_years = (pd.DatetimeIndex(forecast_dates) - pd.Timestamp(init_state["origin_date"])).total_seconds() / (86400.0 * YEAR_DAYS)
        q0_o = init_state["qo"]
        q0_w = init_state["qw"]
        q0_L = q0_o + q0_w
        
        # 1. Total liquid rate dynamics under voidage maintenance
        # In mature waterflood, total liquid rate asymptotically plateaus at separator capacity
        qL_inf = q0_L * self.qL_inf_ratio
        qL_t = q0_L + (qL_inf - q0_L) * (1.0 - np.exp(-t_years / self.tau_years))
        
        # 2. Water cut progression
        fw0 = np.clip(q0_w / max(1.0, q0_L), 0.001, 0.999)
        logit_fw0 = np.log(fw0 / (1.0 - fw0))
        logit_fwt = logit_fw0 + self.beta_w * t_years
        fwt = 1.0 / (1.0 + np.exp(-logit_fwt))
        
        # 3. Phase rate partitioning
        qo_t = qL_t * (1.0 - fwt)
        qw_t = qL_t * fwt
        qg_t = RS_SOLUTION * qo_t
        
        dt_days = 91.3125
        cum_o = init_state["Qo"] + np.cumsum(qo_t * dt_days)
        cum_w = init_state["Qw"] + np.cumsum(qw_t * dt_days)
        cum_g = init_state["Qg"] + RS_SOLUTION * (cum_o - init_state["Qo"])
        
        return {
            "oil_rate": qo_t,
            "water_rate": qw_t,
            "gas_rate": qg_t,
            "cum_oil": cum_o,
            "cum_water": cum_w,
            "cum_gas": cum_g,
        }


def evaluate_crm_against_baseline():
    print("=" * 80)
    print("HYPOTHESIS 3: CAPACITANCE-RESISTANCE MODEL (CRM) BENCHMARK ON HELD-OUT CASES")
    print("=" * 80)
    
    unc_df = load_uncertainty()
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)
    
    from ensemble.generate_competition_deliverable import (
        fit_locked_base_models,
        generate_scenario_predictions,
    )
    
    # Fit frozen base models on training set (Cases 1-70) up to 2001-01-01
    # to evaluate strictly on held-out Validation Cases 71-85 over 2001-2008 (28 quarters)
    cutoff = pd.Timestamp("2001-01-01")
    models_frozen = fit_locked_base_models(case_dict, unc_df, TRAIN_CASES, cutoff)
    
    weights = {
        "oil": {
            "Recursive_ExtraTrees": 0.240466,
            "Learned_Exponential_DCA": 0.617902,
            "Learned_Hyperbolic_DCA": 0.0,
            "Hybrid_Dynamic": 0.141631,
        },
        "water": {
            "Recursive_ExtraTrees": 0.497170,
            "Learned_Exponential_DCA": 0.251415,
            "Learned_Hyperbolic_DCA": 0.251415,
            "Hybrid_Dynamic": 0.0,
        }
    }
    
    # Fit field-level CRM on training cases
    # Global CRM parameters estimated by averaging fits on train cases
    tau_list = []
    beta_list = []
    for c in TRAIN_CASES:
        df_c = case_dict[c]
        pre_c = df_c[df_c["date"] <= cutoff]
        crm_c = FieldLevelCRM().fit_from_case(pre_c)
        tau_list.append(crm_c.tau_years)
        beta_list.append(crm_c.beta_w)
        
    global_crm = FieldLevelCRM()
    global_crm.tau_years = float(np.median(tau_list))
    global_crm.beta_w = float(np.median(beta_list))
    print(f"Global CRM Calibrated Parameters from Train Cases: tau = {global_crm.tau_years:.2f} yrs, beta_w = {global_crm.beta_w:.4f} /yr")
    
    # Evaluate on held-out Validation Cases (71-85)
    horizon_q = 28
    results_records = []
    
    for c in VAL_CASES:
        df_c = case_dict[c]
        pre_c = df_c[df_c["date"] <= cutoff]
        post_c = df_c[df_c["date"] > cutoff].iloc[:horizon_q]
        
        last_row = pre_c.iloc[-1]
        u_c = unc_df.loc[c]
        f_dates = post_c["date"]
        
        init_state = {
            "qo": float(last_row["oil_rate_stbd"]),
            "qw": float(last_row["water_rate_stbd"]),
            "qg": float(last_row["gas_rate_mscfd"]),
            "Qo": float(last_row["oil_cum_stb"]),
            "Qw": float(last_row["water_cum_stb"]),
            "Qg": float(last_row["gas_cum_mscf"]),
            "wor": float(last_row["wor"]),
            "res_age": float(last_row["res_age_years"]),
            "origin_date": cutoff,
        }
        
        # Ground truth
        y_true_o = post_c["oil_cum_stb"].values
        y_true_w = post_c["water_cum_stb"].values
        y_true_g = post_c["gas_cum_mscf"].values
        
        # Ground truth increments
        true_inc_o = y_true_o - init_state["Qo"]
        true_inc_w = y_true_w - init_state["Qw"]
        true_inc_g = y_true_g - init_state["Qg"]
        
        # Model 1: Frozen Simplex Ensemble
        df_ens, aud_ens = generate_scenario_predictions(
            models=models_frozen,
            weights=weights,
            init_state=init_state,
            params=u_c,
            forecast_dates=f_dates,
            decline_sensitivity=1.0,
        )
        nrmse_o_ens = compute_increment_nrmse(aud_ens["cum_oil"] - y_true_o, true_inc_o)
        nrmse_w_ens = compute_increment_nrmse(aud_ens["cum_water"] - y_true_w, true_inc_w)
        nrmse_g_ens = compute_increment_nrmse(aud_ens["cum_gas"] - y_true_g, true_inc_g)
        macro_ens = (nrmse_o_ens + nrmse_w_ens + nrmse_g_ens) / 3.0
        
        # Model 2: Physics-Informed CRM
        crm_pred = global_crm.predict_forecast(init_state, f_dates)
        nrmse_o_crm = compute_increment_nrmse(crm_pred["cum_oil"] - y_true_o, true_inc_o)
        nrmse_w_crm = compute_increment_nrmse(crm_pred["cum_water"] - y_true_w, true_inc_w)
        nrmse_g_crm = compute_increment_nrmse(crm_pred["cum_gas"] - y_true_g, true_inc_g)
        macro_crm = (nrmse_o_crm + nrmse_w_crm + nrmse_g_crm) / 3.0
        
        results_records.append({
            "case_id": c,
            "ens_oil_nrmse": nrmse_o_ens,
            "ens_water_nrmse": nrmse_w_ens,
            "ens_macro_nrmse": macro_ens,
            "crm_oil_nrmse": nrmse_o_crm,
            "crm_water_nrmse": nrmse_w_crm,
            "crm_macro_nrmse": macro_crm,
            "crm_oil_diff": nrmse_o_crm - nrmse_o_ens,
            "crm_water_diff": nrmse_w_crm - nrmse_w_ens,
            "crm_macro_diff": macro_crm - macro_ens,
        })
        
    df_crm_eval = pd.DataFrame(results_records)
    table_path = OUTPUT_TABLE_DIR / "crm_vs_frozen_ensemble_validation.csv"
    df_crm_eval.to_csv(table_path, index=False)
    print(f"\nSaved CRM benchmark evaluation table: {table_path}")
    
    mean_macro_ens = df_crm_eval["ens_macro_nrmse"].mean()
    mean_macro_crm = df_crm_eval["crm_macro_nrmse"].mean()
    mean_oil_ens = df_crm_eval["ens_oil_nrmse"].mean()
    mean_oil_crm = df_crm_eval["crm_oil_nrmse"].mean()
    mean_wat_ens = df_crm_eval["ens_water_nrmse"].mean()
    mean_wat_crm = df_crm_eval["crm_water_nrmse"].mean()
    
    print("\n[Held-Out Validation Aggregate Metrics (Cases 71-85)]")
    print(f"  Frozen Ensemble: Macro NRMSE = {mean_macro_ens:.5f} (Oil = {mean_oil_ens:.5f}, Water = {mean_wat_ens:.5f})")
    print(f"  Field-Level CRM: Macro NRMSE = {mean_macro_crm:.5f} (Oil = {mean_oil_crm:.5f}, Water = {mean_wat_crm:.5f})")
    print(f"  Macro Performance Delta:       {mean_macro_crm - mean_macro_ens:+.5f} (CRM is {'Worse' if mean_macro_crm > mean_macro_ens else 'Better'})")
    
    # Diagnostic Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Subplot 1: Case-by-Case Macro NRMSE comparison
    ax = axes[0]
    cases = df_crm_eval["case_id"].values
    x = np.arange(len(cases))
    width = 0.35
    ax.bar(x - width/2, df_crm_eval["ens_macro_nrmse"], width, label="Frozen Simplex Ensemble", color="#2ecc71", edgecolor="black")
    ax.bar(x + width/2, df_crm_eval["crm_macro_nrmse"], width, label="Capacitance-Resistance Model (CRM)", color="#e74c3c", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in cases], rotation=45)
    ax.set_ylabel("Macro NRMSE", fontweight="bold")
    ax.set_title("Validation Case-by-Case Macro Error\n(Frozen Ensemble vs CRM)", fontsize=11, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Subplot 2: Water NRMSE Scatter Comparison
    ax2 = axes[1]
    ax2.scatter(df_crm_eval["ens_water_nrmse"], df_crm_eval["crm_water_nrmse"], c="#8e44ad", s=60, edgecolors="black", alpha=0.8, label="Validation Cases (71-85)")
    lims = [0, max(df_crm_eval["crm_water_nrmse"].max(), df_crm_eval["ens_water_nrmse"].max()) * 1.1]
    ax2.plot(lims, lims, "r--", label="1:1 Parity")
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel("Frozen Ensemble Water NRMSE", fontweight="bold")
    ax2.set_ylabel("CRM Water NRMSE", fontweight="bold")
    ax2.set_title("Water Recovery Trajectory Error\n(Logit Extrapolation Degradation)", fontsize=11, fontweight="bold")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    fig_path = OUTPUT_FIG_DIR / "crm_vs_ensemble_validation_comparison.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved diagnostic figure: {fig_path}")
    
    summary = {
        "mean_macro_ens": float(mean_macro_ens),
        "mean_macro_crm": float(mean_macro_crm),
        "macro_delta": float(mean_macro_crm - mean_macro_ens),
        "mean_oil_ens": float(mean_oil_ens),
        "mean_oil_crm": float(mean_oil_crm),
        "mean_wat_ens": float(mean_wat_ens),
        "mean_wat_crm": float(mean_wat_crm),
    }
    with open(OUTPUT_TABLE_DIR / "crm_benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    return summary


if __name__ == "__main__":
    evaluate_crm_against_baseline()
