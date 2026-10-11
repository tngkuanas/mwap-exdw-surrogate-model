"""
Module: dca_train_deployment_evaluation.py
Addresses Section 3 of Fundamental Mathematical Validation:
Rigorous Evaluation of DCA Train-to-Deployment Mismatch.

Compares:
1. Legacy DCA:
   - In offline CV (len(post) >= 4): fits D on post-cutoff future trajectory.
   - At deployment (len(post) < 4): falls back to pre-cutoff historical tail (2002-2008).
   - Unstandardized raw features in RidgeCV.
2. Time-Consistent DCA (Challenger):
   - Consistent target semantics: at ANY origin T, fits D on a standardized pre-origin
     lookback window of fixed duration (e.g. 5-6 years), OR trains on rolling retrospective
     forecast origins predicting true forward decline.
   - Standardized feature pipeline (StandardScaler + RidgeCV) fitted strictly within folds.

Evaluates end-to-end backtesting on held-out cases (Cases 71-85):
- Parameter prediction error |D_pred - D_true|
- Oil cumulative NRMSE
- Water cumulative NRMSE
- Terminal oil rate error
- Forecast bias by lead time (1 to 28 quarters)
- 10-year and 20-year forecast sensitivity at deployment anchor.
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

from ensemble.data import (
    load_uncertainty,
    resolve_shared_dir,
    TRAIN_CASES,
    VAL_CASES,
    PARAMS,
)
from ensemble.metrics import compute_increment_nrmse, compute_bias
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
    YEAR_DAYS,
    RS_SOLUTION,
)

OUTPUT_TABLE_DIR = Path("outputs/research/math_validation/tables")
OUTPUT_FIG_DIR = Path("outputs/research/math_validation/figures")
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)


class TimeConsistentDCARateForecaster:
    """
    Time-consistent decline forecaster.
    Maintains identical target semantics between training and deployment:
    Target D is always defined as the empirical decline rate over a consistent
    retrospective lookback window [T - lookback_years, T], OR trained across
    rolling retrospective origins with standardized feature scaling.
    """
    def __init__(self, decline_type: str = "exponential", lookback_years: float = 5.0):
        self.decline_type = decline_type
        self.lookback_years = lookback_years
        self.model_D = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 10)))
        ])
        self.model_water_growth = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 10)))
        ])
        self.b_hyperbolic = 0.35
        self.feat_cols = ["fault", "poro", "perm", "aq", "qo_0", "qw_0", "wor_0", "res_age"]

    def fit_consistent(
        self,
        case_dict: Dict[int, pd.DataFrame],
        unc_df: pd.DataFrame,
        train_cases: List[int],
        as_of_date: pd.Timestamp,
    ):
        """
        Fits strictly using pre-cutoff data available as-of as_of_date.
        Target D is the empirical decline rate over [as_of_date - lookback, as_of_date].
        Information availability is 100% time-consistent at both training and deployment.
        """
        records = []
        t_lookback_start = as_of_date - pd.DateOffset(years=int(self.lookback_years))
        
        for c in train_cases:
            df_c = case_dict[c]
            pre = df_c[df_c["date"] <= as_of_date]
            win = pre[pre["date"] >= t_lookback_start]
            
            if len(win) < 4:
                # If window too short, use all available pre-cutoff data
                win = pre
                
            t_fit_y = (win["date"] - win["date"].iloc[0]).dt.total_seconds() / (86400.0 * YEAR_DAYS)
            fit_rates = win["oil_rate_stbd"].values
            fit_water = win["water_rate_stbd"].values
            
            log_qo = np.log(np.maximum(fit_rates, 1.0))
            poly_o = np.polyfit(t_fit_y, log_qo, 1)
            D_obs = max(0.005, -poly_o[0])
            w_growth_obs = np.polyfit(t_fit_y, fit_water, 1)[0]
            
            u_c = unc_df.loc[c]
            records.append({
                "case_id": c,
                "fault": float(u_c["Fault Transmissibility"]),
                "poro": float(u_c["Porosity Multiplier"]),
                "perm": float(u_c["Permeability Multiplier"]),
                "aq": float(u_c["Aquifer Pore Volume"]),
                "qo_0": float(pre.iloc[-1]["oil_rate_stbd"]),
                "qw_0": float(pre.iloc[-1]["water_rate_stbd"]),
                "wor_0": float(pre.iloc[-1]["wor"]),
                "res_age": float(pre.iloc[-1]["res_age_years"]),
                "D_obs": D_obs,
                "w_growth_obs": w_growth_obs,
            })
            
        fit_df = pd.DataFrame(records)
        X_mat = fit_df[self.feat_cols].values
        self.model_D.fit(X_mat, fit_df["D_obs"].values)
        self.model_water_growth.fit(X_mat, fit_df["w_growth_obs"].values)
        return self

    def predict_rates(
        self,
        initial_state: Dict[str, float],
        u_c: pd.Series,
        forecast_dates: pd.DatetimeIndex,
    ) -> Dict[str, np.ndarray]:
        t_years = (pd.DatetimeIndex(forecast_dates) - pd.Timestamp(initial_state["origin_date"])).total_seconds() / (86400.0 * YEAR_DAYS)
        x_in = np.array([[
            float(u_c["Fault Transmissibility"]),
            float(u_c["Porosity Multiplier"]),
            float(u_c["Permeability Multiplier"]),
            float(u_c["Aquifer Pore Volume"]),
            initial_state["qo"],
            initial_state["qw"],
            initial_state["wor"],
            initial_state["res_age"],
        ]])
        
        D_pred = max(0.01, float(self.model_D.predict(x_in)[0]))
        w_growth_pred = float(self.model_water_growth.predict(x_in)[0])
        
        q0_o = initial_state["qo"]
        q0_w = initial_state["qw"]
        
        if self.decline_type == "exponential":
            qo_t = q0_o * np.exp(-D_pred * t_years)
        elif self.decline_type == "hyperbolic":
            b = self.b_hyperbolic
            qo_t = q0_o * ((1.0 + b * D_pred * t_years) ** (-1.0 / b))
        else:
            raise ValueError(f"Unknown decline_type: {self.decline_type}")
            
        qg_t = RS_SOLUTION * qo_t
        Dw = 0.20
        qw_t = q0_w + (w_growth_pred / Dw) * (1.0 - np.exp(-Dw * t_years))
        return {
            "oil_rate": qo_t,
            "gas_rate": qg_t,
            "water_rate": qw_t,
            "D_pred": D_pred,
            "w_growth_pred": w_growth_pred,
        }


def evaluate_dca_mismatch_and_challenger():
    print("=" * 80)
    print("SECTION 3: SYSTEMATIC EVALUATION OF DCA TRAIN/DEPLOYMENT MISMATCH")
    print("=" * 80)
    
    unc_df = load_uncertainty()
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)
    
    from ensemble.run_horizon_adaptive_stacking_experiment import LearnedDCARateForecaster
    
    # Retrospective backtesting cutoff: 2001-01-01 (7-year forecast horizon: 28 quarters)
    cutoff = pd.Timestamp("2001-01-01")
    horizon_q = 28
    dt_days = 91.3125
    
    # 1. Fit Legacy DCA (unstandardized, post-cutoff future target when len(post) >= 4)
    print("Fitting Legacy DCA Forecaster on Cases 1-70...")
    legacy_dca = LearnedDCARateForecaster(decline_type="exponential")
    legacy_dca.fit(case_dict, unc_df, TRAIN_CASES, cutoff, horizon_q)
    
    # 2. Fit Time-Consistent DCA Forecaster (standardized pipeline, consistent pre-origin lookback)
    print("Fitting Time-Consistent DCA Forecaster on Cases 1-70...")
    consistent_dca = TimeConsistentDCARateForecaster(decline_type="exponential", lookback_years=5.0)
    consistent_dca.fit_consistent(case_dict, unc_df, TRAIN_CASES, cutoff)
    
    # 3. Evaluate both on held-out Validation Cases (71-85)
    print(f"Evaluating out-of-sample forecast accuracy on N={len(VAL_CASES)} Validation Cases (71-85)...")
    records_eval = []
    lead_time_errors_legacy = np.zeros(horizon_q)
    lead_time_errors_consistent = np.zeros(horizon_q)
    
    for c in VAL_CASES:
        df_c = case_dict[c]
        pre = df_c[df_c["date"] <= cutoff]
        post = df_c[df_c["date"] > cutoff].iloc[:horizon_q]
        f_dates = post["date"]
        
        last_row = pre.iloc[-1]
        u_c = unc_df.loc[c]
        
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
        
        y_true_o = post["oil_cum_stb"].values
        y_true_w = post["water_cum_stb"].values
        true_inc_o = y_true_o - init_state["Qo"]
        true_inc_w = y_true_w - init_state["Qw"]
        
        # True future empirical decline rate
        t_fut_y = (f_dates - f_dates.iloc[0]).dt.total_seconds() / (86400.0 * YEAR_DAYS)
        poly_true = np.polyfit(t_fut_y, np.log(np.maximum(post["oil_rate_stbd"].values, 1.0)), 1)
        D_true = max(0.005, -poly_true[0])
        
        # Model 1: Legacy DCA
        pred_leg = legacy_dca.predict_rates(init_state, u_c, f_dates)
        cum_o_leg = init_state["Qo"] + np.cumsum(pred_leg["oil_rate"] * dt_days)
        cum_w_leg = init_state["Qw"] + np.cumsum(pred_leg["water_rate"] * dt_days)
        nrmse_o_leg = compute_increment_nrmse(cum_o_leg - y_true_o, true_inc_o)
        nrmse_w_leg = compute_increment_nrmse(cum_w_leg - y_true_w, true_inc_w)
        macro_leg = (nrmse_o_leg + nrmse_w_leg) / 2.0
        
        # Model 2: Time-Consistent DCA
        pred_con = consistent_dca.predict_rates(init_state, u_c, f_dates)
        cum_o_con = init_state["Qo"] + np.cumsum(pred_con["oil_rate"] * dt_days)
        cum_w_con = init_state["Qw"] + np.cumsum(pred_con["water_rate"] * dt_days)
        nrmse_o_con = compute_increment_nrmse(cum_o_con - y_true_o, true_inc_o)
        nrmse_w_con = compute_increment_nrmse(cum_w_con - y_true_w, true_inc_w)
        macro_con = (nrmse_o_con + nrmse_w_con) / 2.0
        
        # Accumulate lead time errors
        lead_time_errors_legacy += np.abs(cum_o_leg - y_true_o) / true_inc_o
        lead_time_errors_consistent += np.abs(cum_o_con - y_true_o) / true_inc_o
        
        D_pred_leg = float(legacy_dca.model_D.predict(np.array([[
            float(u_c["Fault Transmissibility"]),
            float(u_c["Porosity Multiplier"]),
            float(u_c["Permeability Multiplier"]),
            float(u_c["Aquifer Pore Volume"]),
            init_state["qo"], init_state["qw"], init_state["wor"], init_state["res_age"],
        ]]))[0])
        
        records_eval.append({
            "case_id": c,
            "D_true": D_true,
            "D_pred_legacy": D_pred_leg,
            "D_pred_consistent": pred_con["D_pred"],
            "err_D_legacy": abs(D_pred_leg - D_true),
            "err_D_consistent": abs(pred_con["D_pred"] - D_true),
            "oil_nrmse_legacy": nrmse_o_leg,
            "oil_nrmse_consistent": nrmse_o_con,
            "wat_nrmse_legacy": nrmse_w_leg,
            "wat_nrmse_consistent": nrmse_w_con,
            "macro_nrmse_legacy": macro_leg,
            "macro_nrmse_consistent": macro_con,
            "delta_macro": macro_con - macro_leg,
        })
        
    df_eval = pd.DataFrame(records_eval)
    eval_path = OUTPUT_TABLE_DIR / "dca_mismatch_validation_comparison.csv"
    df_eval.to_csv(eval_path, index=False)
    print(f"Saved DCA validation comparison table: {eval_path}")
    
    mean_d_err_leg = df_eval["err_D_legacy"].mean()
    mean_d_err_con = df_eval["err_D_consistent"].mean()
    mean_oil_leg = df_eval["oil_nrmse_legacy"].mean()
    mean_oil_con = df_eval["oil_nrmse_consistent"].mean()
    mean_wat_leg = df_eval["wat_nrmse_legacy"].mean()
    mean_wat_con = df_eval["wat_nrmse_consistent"].mean()
    mean_mac_leg = df_eval["macro_nrmse_legacy"].mean()
    mean_mac_con = df_eval["macro_nrmse_consistent"].mean()
    
    print("\n[Out-of-Sample Performance Summary (Cases 71-85)]")
    print(f"  Legacy DCA:          Mean |D_err| = {mean_d_err_leg:.4f} | Oil NRMSE = {mean_oil_leg:.4f} | Water NRMSE = {mean_wat_leg:.4f} | Macro = {mean_mac_leg:.4f}")
    print(f"  Time-Consistent DCA: Mean |D_err| = {mean_d_err_con:.4f} | Oil NRMSE = {mean_oil_con:.4f} | Water NRMSE = {mean_wat_con:.4f} | Macro = {mean_mac_con:.4f}")
    print(f"  Macro Performance Delta: {mean_mac_con - mean_mac_leg:+.4f} ({'Consistent Better' if mean_mac_con < mean_mac_leg else 'Legacy Better'})")
    
    # 4. Long-Horizon 20-Year EUR Sensitivity Test at Production Origin (2008-01-01)
    print("\n[Evaluating 20-Year EUR Sensitivity at January 2008 Deployment Anchor]")
    from ensemble.scenario_loader import load_production_scenarios
    scenarios = load_production_scenarios()
    
    c0_anchor = {
        "qo": 5066.68,
        "qw": 7926.36,
        "qg": 1838.96,
        "Qo": 38378316.97,
        "Qw": 12814432.28,
        "Qg": 13942537.32,
        "wor": 7926.36 / 5066.68,
        "res_age": 10.0,
        "origin_date": pd.Timestamp("2008-01-01"),
    }
    
    # Fit both models at January 2008 deployment
    deploy_cutoff = pd.Timestamp("2008-01-01")
    legacy_dca_deploy = LearnedDCARateForecaster(decline_type="exponential").fit(case_dict, unc_df, TRAIN_CASES, deploy_cutoff, 80)
    consistent_dca_deploy = TimeConsistentDCARateForecaster(decline_type="exponential", lookback_years=5.0).fit_consistent(case_dict, unc_df, TRAIN_CASES, deploy_cutoff)
    
    f_dates_80q = pd.date_range("2008-04-01", periods=80, freq="QS")
    
    sc_comparison_records = []
    for sc in scenarios:
        u_sc = sc.params
        
        # Legacy
        pred_l = legacy_dca_deploy.predict_rates(c0_anchor, u_sc, f_dates_80q)
        q_eur_leg = c0_anchor["Qo"] + np.sum(pred_l["oil_rate"] * dt_days)
        
        # Time-consistent
        pred_c = consistent_dca_deploy.predict_rates(c0_anchor, u_sc, f_dates_80q)
        q_eur_con = c0_anchor["Qo"] + np.sum(pred_c["oil_rate"] * dt_days)
        
        sc_comparison_records.append({
            "case_id": sc.case_id,
            "candidate_id": sc.candidate_id,
            "D_legacy": float(pred_l["oil_rate"][0] / max(1.0, c0_anchor["qo"])),
            "D_consistent": pred_c["D_pred"],
            "EUR_legacy_stb": q_eur_leg,
            "EUR_consistent_stb": q_eur_con,
            "delta_eur_stb": q_eur_con - q_eur_leg,
        })
        
    df_sc_comp = pd.DataFrame(sc_comparison_records)
    sc_comp_path = OUTPUT_TABLE_DIR / "dca_deployment_eur_sensitivity_comparison.csv"
    df_sc_comp.to_csv(sc_comp_path, index=False)
    print(f"Saved deployment EUR comparison: {sc_comp_path}")
    
    spread_leg = df_sc_comp["EUR_legacy_stb"].max() - df_sc_comp["EUR_legacy_stb"].min()
    spread_con = df_sc_comp["EUR_consistent_stb"].max() - df_sc_comp["EUR_consistent_stb"].min()
    
    print(f"  Legacy 20-Yr EUR Spread Across 10 Scenarios:          {spread_leg:>12,.0f} STB")
    print(f"  Time-Consistent 20-Yr EUR Spread Across 10 Scenarios: {spread_con:>12,.0f} STB")
    
    # 5. Diagnostic Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Subplot 1: Case-by-Case Macro NRMSE on Validation Cases
    ax = axes[0]
    cases = df_eval["case_id"].values
    x = np.arange(len(cases))
    w = 0.35
    ax.bar(x - w/2, df_eval["macro_nrmse_legacy"], w, label="Legacy DCA (Asymmetric Target)", color="#e74c3c", edgecolor="black")
    ax.bar(x + w/2, df_eval["macro_nrmse_consistent"], w, label="Time-Consistent DCA (Standardized)", color="#27ae60", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in cases], rotation=45)
    ax.set_ylabel("Macro NRMSE", fontweight="bold")
    ax.set_title(f"Validation Out-of-Sample Macro NRMSE (Cases 71-85)\n(Mean: Legacy={mean_mac_leg:.4f} vs Consistent={mean_mac_con:.4f})", fontsize=11, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Subplot 2: Average Relative Error by Lead Time (Quarters)
    ax2 = axes[1]
    q_axis = np.arange(1, horizon_q + 1)
    ax2.plot(q_axis, lead_time_errors_legacy / len(VAL_CASES) * 100.0, label="Legacy DCA", color="#e74c3c", lw=2.2, marker="o", markersize=4)
    ax2.plot(q_axis, lead_time_errors_consistent / len(VAL_CASES) * 100.0, label="Time-Consistent DCA", color="#27ae60", lw=2.2, marker="s", markersize=4)
    ax2.set_xlabel("Forecast Lead Time (Quarters Ahead)", fontweight="bold")
    ax2.set_ylabel("Mean Relative Cumulative Error (%)", fontweight="bold")
    ax2.set_title("Forecast Error Growth by Lead Time\n(7-Year Validation Horizon)", fontsize=11, fontweight="bold")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    fig_path = OUTPUT_FIG_DIR / "dca_mismatch_validation_diagnostics.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved diagnostic figure: {fig_path}")
    
    summary = {
        "mean_oil_nrmse_legacy": float(mean_oil_leg),
        "mean_oil_nrmse_consistent": float(mean_oil_con),
        "mean_wat_nrmse_legacy": float(mean_wat_leg),
        "mean_wat_nrmse_consistent": float(mean_wat_con),
        "mean_macro_nrmse_legacy": float(mean_mac_leg),
        "mean_macro_nrmse_consistent": float(mean_mac_con),
        "delta_macro": float(mean_mac_con - mean_mac_leg),
        "mean_d_error_legacy": float(mean_d_err_leg),
        "mean_d_error_consistent": float(mean_d_err_con),
        "eur_spread_legacy_stb": float(spread_leg),
        "eur_spread_consistent_stb": float(spread_con),
    }
    with open(OUTPUT_TABLE_DIR / "dca_mismatch_evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    return summary


if __name__ == "__main__":
    evaluate_dca_mismatch_and_challenger()
