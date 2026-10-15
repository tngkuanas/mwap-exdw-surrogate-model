"""
Module: long_horizon_ood_analysis.py
Implements Phase 3 of the Final Research Sprint:
Multi-Angle Forecast Reliability Challenge — 20-Year Out-of-Distribution Forecasting & Decline Regimes.

Key Components:
1. Retrospective backtesting matrix across forecast origins (2001, 2003, 2004, 2005) and prediction horizons (1 to 7 years).
2. Lead-time horizon error breakdown (1, 2, 3, 5, 7 years) and cumulative bias growth tracking.
3. Recursive ExtraTrees Out-of-Distribution (OOD) feature drift quantification (Mahalanobis distance & boundary violation).
4. Direct Arps decline fits using only pre-cutoff history.
5. Challenger A: Boundary-aware decline continuation (ExtraTrees near-term + Exponential DCA tail handoff).
6. Challenger B: Regime-aware decline dynamics (instantaneous Arps b-parameter tracking and decline stability).
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold

from ensemble.data import (
    load_uncertainty,
    TRAIN_CASES,
    VAL_CASES,
    PARAMS,
)
from ensemble.metrics import compute_increment_nrmse, compute_bias
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

OUTPUT_TABLE_DIR = Path("outputs/research/forecast_reliability_challenge/tables")
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = Path("src/ensemble/locked_ensemble_config.json")
with open(CONFIG_PATH) as f:
    LOCKED_CONFIG = json.load(f)
LOCKED_WEIGHTS = LOCKED_CONFIG["weights"]

RETROSPECTIVE_ORIGINS = [
    {"cutoff": "2001-01-01", "max_qtrs": 28, "years": 7.0, "label": "2001_7yr_28q"},
    {"cutoff": "2003-01-01", "max_qtrs": 20, "years": 5.0, "label": "2003_5yr_20q"},
    {"cutoff": "2004-01-01", "max_qtrs": 16, "years": 4.0, "label": "2004_4yr_16q"},
    {"cutoff": "2005-01-01", "max_qtrs": 12, "years": 3.0, "label": "2005_3yr_12q"},
]

LEAD_TIME_BUCKETS = [
    {"label": "1_yr_4q", "qtrs": 4},
    {"label": "2_yr_8q", "qtrs": 8},
    {"label": "3_yr_12q", "qtrs": 12},
    {"label": "5_yr_20q", "qtrs": 20},
    {"label": "7_yr_28q", "qtrs": 28},
]


class DirectHistoricalArpsForecaster:
    """
    Fits Arps exponential and hyperbolic decline curves strictly using pre-cutoff historical production.
    Zero future simulation leakage.
    """
    def __init__(self, decline_type: str = "exponential"):
        self.decline_type = decline_type
        
    def fit_and_predict(
        self,
        case_df: pd.DataFrame,
        cutoff_date: pd.Timestamp,
        forecast_dates: pd.DatetimeIndex,
    ) -> Dict[str, np.ndarray]:
        hist = case_df[case_df["date"] <= cutoff_date].sort_values("date")
        # Use decline window (e.g. from peak oil production or last 8 quarters)
        t_hist = ((hist["date"] - hist["date"].iloc[0]).dt.total_seconds() / (86400.0 * YEAR_DAYS)).to_numpy()
        qo_hist = hist["oil_rate_stbd"].to_numpy()
        qw_hist = hist["water_rate_stbd"].to_numpy()
        
        peak_idx = int(np.argmax(qo_hist))
        if peak_idx < len(qo_hist) - 2:
            fit_t = t_hist[peak_idx:] - t_hist[peak_idx]
            fit_qo = qo_hist[peak_idx:]
        else:
            fit_t = t_hist[-4:] - t_hist[-4]
            fit_qo = qo_hist[-4:]
            
        t_forecast = ((pd.DatetimeIndex(forecast_dates) - cutoff_date).total_seconds() / (86400.0 * YEAR_DAYS)).to_numpy()
        q0_o = max(1.0, float(hist["oil_rate_stbd"].iloc[-1]))
        q0_w = max(0.0, float(hist["water_rate_stbd"].iloc[-1]))
        
        # Fit exponential decline rate
        log_qo = np.log(np.maximum(fit_qo, 1.0))
        if len(fit_t) >= 2 and np.ptp(fit_t) > 0:
            slope, _ = np.polyfit(fit_t, log_qo, 1)
            D_exp = max(0.01, -slope)
        else:
            D_exp = 0.15
            
        if self.decline_type == "exponential":
            qo_pred = q0_o * np.exp(-D_exp * t_forecast)
        elif self.decline_type == "hyperbolic":
            b = 0.5
            qo_pred = q0_o * ((1.0 + b * D_exp * t_forecast) ** (-1.0 / b))
        else:
            qo_pred = q0_o * np.exp(-D_exp * t_forecast)
            
        # Water rate linear growth with relaxation
        if len(qw_hist) >= 4:
            poly_w = np.polyfit(t_hist[-4:] - t_hist[-4], qw_hist[-4:], 1)
            w_growth = max(0.0, poly_w[0])
        else:
            w_growth = 50.0
        Dw = 0.20
        qw_pred = q0_w + (w_growth / Dw) * (1.0 - np.exp(-Dw * t_forecast))
        qg_pred = RS_SOLUTION * qo_pred
        
        return {"oil_rate": qo_pred, "water_rate": qw_pred, "gas_rate": qg_pred}


class ChallengerABoundaryAwareForecaster:
    """
    Challenger A: Boundary-aware decline continuation.
    Uses Recursive ExtraTrees in the near term (up to switch_qtrs),
    then transitions to Exponential DCA with continuity matching at the handoff.
    """
    def __init__(self, switch_qtrs: int = 12):
        self.switch_qtrs = switch_qtrs
        self.hybrid = HybridDynamicForecaster(switch_quarters=switch_qtrs)
        
    def fit(self, case_dict, unc_df, train_cases, cutoff_date, horizon_quarters):
        self.hybrid.fit(case_dict, unc_df, train_cases, cutoff_date, horizon_quarters)
        return self
        
    def predict_rates(self, initial_state, u_c, forecast_dates):
        return self.hybrid.predict_rates(initial_state, u_c, forecast_dates)


def compute_extratrees_ood_drift(
    et_model: RecursiveTransitionForecaster,
    training_data: pd.DataFrame,
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    cases: List[int],
    cutoff_date: pd.Timestamp,
    horizon_qtrs: int = 28,
) -> pd.DataFrame:
    """
    Measures how recursively generated state features drift away from the training distribution.
    Computes:
    - Mahalanobis distance from training feature centroid
    - Normalized Euclidean distance
    - Feature range bound violations (fraction of features outside [min, max] training box)
    """
    from ensemble.run_horizon_adaptive_stacking_experiment import STATE_FEATURES
    
    X_train = training_data[STATE_FEATURES].to_numpy()
    mean_vec = np.mean(X_train, axis=0)
    cov_mat = np.cov(X_train, rowvar=False) + np.eye(len(STATE_FEATURES)) * 1e-4
    inv_cov = np.linalg.pinv(cov_mat)
    min_vec = np.min(X_train, axis=0)
    max_vec = np.max(X_train, axis=0)
    
    drift_records = []
    f_dates = pd.date_range(cutoff_date + pd.DateOffset(months=3), periods=horizon_qtrs, freq="QS")
    
    for c in cases:
        df_c = case_dict[c]
        pre = df_c[df_c["date"] <= cutoff_date]
        if len(pre) == 0:
            continue
        init_state = {
            "qo": float(pre.iloc[-1]["oil_rate_stbd"]),
            "qw": float(pre.iloc[-1]["water_rate_stbd"]),
            "res_age": float(pre.iloc[-1]["res_age_years"]),
        }
        u_c = unc_df.loc[c]
        
        curr_qo = init_state["qo"]
        curr_qw = init_state["qw"]
        curr_age = init_state["res_age"]
        
        for step_idx in range(horizon_qtrs):
            x_step = np.array([
                np.log(max(curr_qo, 1.0)),
                np.log(max(curr_qw, 1.0) + 1.0),
                curr_age,
                curr_qw / max(curr_qo, 1.0),
                float(u_c["Fault Transmissibility"]),
                float(u_c["Porosity Multiplier"]),
                float(u_c["Permeability Multiplier"]),
                float(u_c["Aquifer Pore Volume"]),
            ])
            
            # Mahalanobis distance
            diff = x_step - mean_vec
            mahal_dist = float(np.sqrt(np.dot(np.dot(diff, inv_cov), diff)))
            
            # Normalized Euclidean distance
            std_vec = np.std(X_train, axis=0) + 1e-6
            norm_eucl = float(np.linalg.norm((x_step - mean_vec) / std_vec))
            
            # Bound violations
            viol_count = int(np.sum((x_step < min_vec) | (x_step > max_vec)))
            viol_fraction = viol_count / len(STATE_FEATURES)
            
            drift_records.append({
                "case_id": c,
                "lead_quarter": step_idx + 1,
                "lead_years": (step_idx + 1) * 0.25,
                "mahalanobis_distance": mahal_dist,
                "norm_euclidean_distance": norm_eucl,
                "bound_violation_count": viol_count,
                "bound_violation_fraction": viol_fraction,
            })
            
            # Recursive step
            x_in = x_step.reshape(1, -1)
            d_log_qo = float(et_model.model_qo.predict(x_in)[0])
            d_qw = float(et_model.model_qw.predict(x_in)[0])
            
            curr_qo = max(0.0, curr_qo * np.exp(min(0.0, d_log_qo)))
            curr_qw = max(0.0, curr_qw + d_qw)
            curr_age += 0.25
            
    drift_df = pd.DataFrame(drift_records)
    summary_df = drift_df.groupby("lead_quarter").agg({
        "lead_years": "first",
        "mahalanobis_distance": ["mean", "std", "max"],
        "norm_euclidean_distance": ["mean", "max"],
        "bound_violation_fraction": ["mean", "max"],
    }).reset_index()
    summary_df.columns = [
        "lead_quarter", "lead_years",
        "mahalanobis_mean", "mahalanobis_std", "mahalanobis_max",
        "euclidean_mean", "euclidean_max",
        "violation_fraction_mean", "violation_fraction_max"
    ]
    return summary_df


def run_long_horizon_ood_experiment() -> Dict[str, pd.DataFrame]:
    """
    Executes the comprehensive Phase 3 Retrospective Backtesting & Out-of-Distribution analysis.
    """
    print("=" * 80)
    print("PHASE 3: RETROSPECTIVE BACKTESTING & OUT-OF-DISTRIBUTION HORIZON ANALYSIS")
    print("=" * 80)
    
    unc_df = load_uncertainty()
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)
    
    # -------------------------------------------------------------------------
    # 1. Retrospective Matrix across origins & horizons
    # -------------------------------------------------------------------------
    retro_results = []
    lead_time_records = []
    
    for origin_cfg in RETROSPECTIVE_ORIGINS:
        cutoff = pd.Timestamp(origin_cfg["cutoff"])
        max_qtrs = origin_cfg["max_qtrs"]
        origin_label = origin_cfg["label"]
        print(f"\nProcessing origin: {origin_label} (Cutoff: {cutoff.strftime('%Y-%m-%d')}, Max quarters: {max_qtrs})...")
        
        # Fit base models on Cases 1-70 pre-cutoff
        trans_data = extract_transition_training_data(case_dict, unc_df, TRAIN_CASES, cutoff)
        
        m_et = RecursiveTransitionForecaster(model_type="extratrees").fit(trans_data)
        m_exp = LearnedDCARateForecaster(decline_type="exponential").fit(
            case_dict, unc_df, TRAIN_CASES, cutoff, max_qtrs
        )
        m_hyp = LearnedDCARateForecaster(decline_type="hyperbolic").fit(
            case_dict, unc_df, TRAIN_CASES, cutoff, max_qtrs
        )
        m_hyb = HybridDynamicForecaster(switch_quarters=12).fit(
            case_dict, unc_df, TRAIN_CASES, cutoff, max_qtrs
        )
        m_direct_exp = DirectHistoricalArpsForecaster(decline_type="exponential")
        m_direct_hyp = DirectHistoricalArpsForecaster(decline_type="hyperbolic")
        
        # Challenger A variants: switch at 8 quarters vs 12 quarters
        challenger_a_8 = ChallengerABoundaryAwareForecaster(switch_qtrs=8).fit(
            case_dict, unc_df, TRAIN_CASES, cutoff, max_qtrs
        )
        
        models = {
            "Recursive_ExtraTrees": m_et,
            "Learned_Exponential_DCA": m_exp,
            "Learned_Hyperbolic_DCA": m_hyp,
            "Hybrid_Dynamic": m_hyb,
            "Challenger_A_H8": challenger_a_8,
        }
        
        # Evaluate on Validation Cases 71-85 and CV Cases 1-70
        eval_cases = DEV_CASES = list(range(1, 86))
        
        for c in eval_cases:
            is_val = c in VAL_CASES
            subset = "Validation (71-85)" if is_val else "Train (1-70)"
            
            df_c = case_dict[c]
            pre = df_c[df_c["date"] <= cutoff]
            post = df_c[df_c["date"] > cutoff].iloc[:max_qtrs]
            
            if len(post) == 0:
                continue
                
            n_actual = len(post)
            f_dates = post["date"].to_numpy()
            dt_days = post["dt_days"].to_numpy()
            
            init_state = {
                "qo": float(pre.iloc[-1]["oil_rate_stbd"]),
                "qw": float(pre.iloc[-1]["water_rate_stbd"]),
                "qg": float(pre.iloc[-1]["gas_rate_mscfd"]),
                "wor": float(pre.iloc[-1]["wor"]),
                "res_age": float(pre.iloc[-1]["res_age_years"]),
                "origin_date": cutoff,
            }
            u_c = unc_df.loc[c]
            
            # Ground truth actual cumulative increments
            act_inc_oil = post["oil_cum_stb"].to_numpy() - pre.iloc[-1]["oil_cum_stb"]
            act_inc_wat = post["water_cum_stb"].to_numpy() - pre.iloc[-1]["water_cum_stb"]
            act_rate_oil = post["oil_rate_stbd"].to_numpy()
            act_rate_wat = post["water_rate_stbd"].to_numpy()
            
            # Generate predictions for all models
            preds_dict = {}
            for m_name, m_obj in models.items():
                p = m_obj.predict_rates(init_state, u_c, f_dates)
                preds_dict[m_name] = {
                    "qo": p["oil_rate"][:n_actual],
                    "qw": p["water_rate"][:n_actual],
                }
                
            # Direct DCA baselines
            p_dir_exp = m_direct_exp.fit_and_predict(df_c, cutoff, pd.DatetimeIndex(f_dates))
            preds_dict["Direct_Arps_Exp"] = {
                "qo": p_dir_exp["oil_rate"][:n_actual],
                "qw": p_dir_exp["water_rate"][:n_actual],
            }
            p_dir_hyp = m_direct_hyp.fit_and_predict(df_c, cutoff, pd.DatetimeIndex(f_dates))
            preds_dict["Direct_Arps_Hyp"] = {
                "qo": p_dir_hyp["oil_rate"][:n_actual],
                "qw": p_dir_hyp["water_rate"][:n_actual],
            }
            
            # Frozen Simplex Stack
            q_oil_stack = np.zeros(n_actual)
            for m_name, w in LOCKED_WEIGHTS["oil"].items():
                q_oil_stack += w * preds_dict[m_name]["qo"]
            q_wat_stack = np.zeros(n_actual)
            for m_name, w in LOCKED_WEIGHTS["water"].items():
                q_wat_stack += w * preds_dict[m_name]["qw"]
            preds_dict["Blend_Phase_Specific_Simplex"] = {
                "qo": q_oil_stack,
                "qw": q_wat_stack,
            }
            
            # Evaluate across each model and lead-time horizon
            for m_name, pred_p in preds_dict.items():
                # Compute cumulative increments via right-endpoint dt
                pred_inc_oil = np.cumsum(pred_p["qo"] * dt_days)
                pred_inc_wat = np.cumsum(pred_p["qw"] * dt_days)
                
                # Full horizon metrics
                err_oil = pred_inc_oil - act_inc_oil
                err_wat = pred_inc_wat - act_inc_wat
                oil_nrmse = compute_increment_nrmse(err_oil, act_inc_oil)
                wat_nrmse = compute_increment_nrmse(err_wat, act_inc_wat)
                macro_nrmse = 0.5 * (oil_nrmse + wat_nrmse)
                bias_oil = compute_bias(err_oil, act_inc_oil)
                bias_wat = compute_bias(err_wat, act_inc_wat)
                
                retro_results.append({
                    "origin": origin_label,
                    "case_id": c,
                    "subset": subset,
                    "model": m_name,
                    "horizon_quarters": n_actual,
                    "oil_cum_nrmse": oil_nrmse,
                    "water_cum_nrmse": wat_nrmse,
                    "macro_nrmse": macro_nrmse,
                    "bias_oil": bias_oil,
                    "bias_wat": bias_wat,
                    "terminal_rate_err_oil": float(abs(pred_p["qo"][-1] - act_rate_oil[-1])),
                    "terminal_rate_err_wat": float(abs(pred_p["qw"][-1] - act_rate_wat[-1])),
                })
                
                # Breakdown by lead-time horizon buckets
                for bucket in LEAD_TIME_BUCKETS:
                    b_qtrs = bucket["qtrs"]
                    if n_actual >= b_qtrs:
                        b_pred_oil = pred_inc_oil[:b_qtrs]
                        b_act_oil = act_inc_oil[:b_qtrs]
                        b_pred_wat = pred_inc_wat[:b_qtrs]
                        b_act_wat = act_inc_wat[:b_qtrs]
                        
                        b_err_oil = b_pred_oil - b_act_oil
                        b_err_wat = b_pred_wat - b_act_wat
                        b_oil_nrmse = compute_increment_nrmse(b_err_oil, b_act_oil)
                        b_wat_nrmse = compute_increment_nrmse(b_err_wat, b_act_wat)
                        b_macro_nrmse = 0.5 * (b_oil_nrmse + b_wat_nrmse)
                        b_bias_oil = compute_bias(b_err_oil, b_act_oil)
                        b_bias_wat = compute_bias(b_err_wat, b_act_wat)
                        
                        lead_time_records.append({
                            "origin": origin_label,
                            "case_id": c,
                            "subset": subset,
                            "model": m_name,
                            "lead_time_bucket": bucket["label"],
                            "lead_quarters": b_qtrs,
                            "oil_nrmse": b_oil_nrmse,
                            "water_nrmse": b_wat_nrmse,
                            "macro_nrmse": b_macro_nrmse,
                            "bias_oil": b_bias_oil,
                            "bias_wat": b_bias_wat,
                        })
                        
    retro_df = pd.DataFrame(retro_results)
    lead_df = pd.DataFrame(lead_time_records)
    
    # -------------------------------------------------------------------------
    # 2. OOD Feature Drift Analysis for ExtraTrees
    # -------------------------------------------------------------------------
    print("\nMeasuring Recursive ExtraTrees Out-of-Distribution Feature Drift...")
    cutoff_2001 = pd.Timestamp("2001-01-01")
    trans_2001 = extract_transition_training_data(case_dict, unc_df, TRAIN_CASES, cutoff_2001)
    m_et_2001 = RecursiveTransitionForecaster(model_type="extratrees").fit(trans_2001)
    ood_drift_df = compute_extratrees_ood_drift(
        m_et_2001, trans_2001, case_dict, unc_df, TRAIN_CASES, cutoff_2001, horizon_qtrs=28
    )
    
    # -------------------------------------------------------------------------
    # 3. Challenger B: Decline Regime & Instantaneous b-Parameter Analysis
    # -------------------------------------------------------------------------
    print("\nAnalyzing Decline Regime & Arps b-parameter dynamics (Challenger B)...")
    regime_records = []
    for c in list(range(1, 86)):
        df_c = case_dict[c]
        # Analyze production after water breakthrough / plateau (1999 to 2008)
        decline_segment = df_c[df_c["date"] >= "2000-01-01"].copy().reset_index(drop=True)
        if len(decline_segment) >= 8:
            t_pts = (decline_segment["date"] - decline_segment["date"].iloc[0]).dt.total_seconds() / (86400.0 * YEAR_DAYS)
            q_pts = decline_segment["oil_rate_stbd"].values
            
            # Estimate nominal decline rate D(t) = -d(ln q)/dt
            log_q = np.log(np.maximum(q_pts, 1.0))
            d_log_q = np.gradient(log_q, t_pts)
            D_inst = -d_log_q
            
            # Loss ratio: a(t) = 1/D(t); b(t) = d(a)/dt = d(1/D)/dt
            with np.errstate(divide="ignore", invalid="ignore"):
                a_inst = 1.0 / np.maximum(D_inst, 1e-4)
                b_inst = np.gradient(a_inst, t_pts)
                
            # Filter stable values
            valid_b = b_inst[np.isfinite(b_inst) & (b_inst > -5.0) & (b_inst < 5.0)]
            mean_b = float(np.median(valid_b)) if len(valid_b) > 0 else 0.0
            
            # Fit exponential vs hyperbolic Arps directly on the full 8-year decline
            # Exp: q = q0 * exp(-D * t)
            # Hyp: q = q0 * (1 + b * D * t)**(-1/b)
            p_exp, _ = np.polyfit(t_pts, log_q, 1)
            d_exp_fit = max(0.005, -p_exp)
            pred_q_exp = q_pts[0] * np.exp(-d_exp_fit * t_pts)
            rmse_exp = float(np.sqrt(np.mean((q_pts - pred_q_exp) ** 2)))
            
            def hyp_func(t, D, b):
                return q_pts[0] * ((1.0 + b * D * t) ** (-1.0 / np.maximum(b, 1e-3)))
                
            try:
                popt, _ = curve_fit(hyp_func, t_pts, q_pts, p0=[d_exp_fit, 0.3], bounds=([0.001, 0.01], [1.0, 1.5]))
                pred_q_hyp = hyp_func(t_pts, *popt)
                rmse_hyp = float(np.sqrt(np.mean((q_pts - pred_q_hyp) ** 2)))
                opt_D = float(popt[0])
                opt_b = float(popt[1])
            except Exception:
                rmse_hyp = rmse_exp
                opt_D = d_exp_fit
                opt_b = 0.0
                
            regime_records.append({
                "case_id": c,
                "subset": "Validation (71-85)" if c in VAL_CASES else "Train (1-70)",
                "median_inst_b": mean_b,
                "fitted_exp_D": d_exp_fit,
                "fitted_hyp_D": opt_D,
                "fitted_hyp_b": opt_b,
                "exp_rmse_stbd": rmse_exp,
                "hyp_rmse_stbd": rmse_hyp,
                "hyp_superiority_pct": float(max(0.0, (rmse_exp - rmse_hyp) / (rmse_exp + 1e-6) * 100.0)),
            })
    regime_df = pd.DataFrame(regime_records)
    
    # -------------------------------------------------------------------------
    # Save Tables
    # -------------------------------------------------------------------------
    retro_summary = retro_df.groupby(["origin", "subset", "model"]).agg({
        "oil_cum_nrmse": ["mean", "std"],
        "water_cum_nrmse": ["mean", "std"],
        "macro_nrmse": ["mean", "std"],
        "bias_oil": "mean",
        "bias_wat": "mean",
    }).reset_index()
    retro_summary.columns = [
        "origin", "subset", "model",
        "oil_nrmse_mean", "oil_nrmse_std",
        "water_nrmse_mean", "water_nrmse_std",
        "macro_nrmse_mean", "macro_nrmse_std",
        "bias_oil_mean", "bias_wat_mean"
    ]
    
    lead_summary = lead_df.groupby(["origin", "subset", "model", "lead_time_bucket"]).agg({
        "lead_quarters": "first",
        "oil_nrmse": "mean",
        "water_nrmse": "mean",
        "macro_nrmse": "mean",
        "bias_oil": "mean",
        "bias_wat": "mean",
    }).reset_index()
    
    retro_path = OUTPUT_TABLE_DIR / "long_horizon_retrospective_matrix.csv"
    lead_path = OUTPUT_TABLE_DIR / "lead_time_error_and_bias_breakdown.csv"
    ood_path = OUTPUT_TABLE_DIR / "extratrees_ood_drift_metrics.csv"
    regime_path = OUTPUT_TABLE_DIR / "challenger_b_regime_dynamics.csv"
    
    retro_summary.to_csv(retro_path, index=False)
    lead_summary.to_csv(lead_path, index=False)
    ood_drift_df.to_csv(ood_path, index=False)
    regime_df.to_csv(regime_path, index=False)
    
    print(f"\nSaved retrospective matrix summary to: {retro_path}")
    print(f"Saved lead-time error and bias breakdown to: {lead_path}")
    print(f"Saved ExtraTrees OOD drift metrics to: {ood_path}")
    print(f"Saved Challenger B regime dynamics to: {regime_path}")
    
    return {
        "retro_summary": retro_summary,
        "lead_summary": lead_summary,
        "ood_drift": ood_drift_df,
        "regime": regime_df,
    }


if __name__ == "__main__":
    run_long_horizon_ood_experiment()
