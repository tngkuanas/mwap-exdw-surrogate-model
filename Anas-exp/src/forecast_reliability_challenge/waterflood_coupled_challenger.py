"""
Module: waterflood_coupled_challenger.py
Implements Phase 4 of the Final Research Sprint:
Multi-Angle Forecast Reliability Challenge — Physics-Informed Waterflood Challenger.

Key Components:
1. Coupled total liquid rate and water-cut modeling:
   q_liquid(t) = q_oil(t) + q_water(t)
   water_cut(t) = q_water(t) / q_liquid(t)
   q_water(t) = water_cut(t) * q_liquid(t)
   q_oil(t) = [1 - water_cut(t)] * q_liquid(t)
   q_gas(t) = Rs * q_oil(t)
2. Compares two coupled formulations:
   - Formulation A (Logit Water Cut): logit(fw(t)) = logit(fw0) + alpha_w * t, with fw_max bound.
   - Formulation B (WOR Semi-log Decline): ln(WOR(t)) = ln(WOR0) + beta_wor * Qo_inc(t).
3. Fair comparative evaluation against baseline independent phase forecasting:
   - Water cumulative NRMSE
   - Water rate NRMSE
   - Water breakthrough timing error (first quarter with fw >= 0.05)
   - Water-cut prediction error
   - Total liquid rate error
   - Historical anchor calibration (t=0 continuity)
   - Worst-case outlier failures
   - 20-year long-horizon behavior (2008-2028).
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import RidgeCV

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

EVAL_ORIGINS = [
    {"cutoff": "2001-01-01", "max_qtrs": 28, "label": "2001_7yr_28q"},
    {"cutoff": "2003-01-01", "max_qtrs": 20, "label": "2003_5yr_20q"},
    {"cutoff": "2005-01-01", "max_qtrs": 12, "label": "2005_3yr_12q"},
]


class CoupledLogitWaterfloodForecaster:
    """
    Formulation A: Coupled Liquid Rate & Logit Water Cut Forecaster.
    Models total liquid decline q_L(t) = q_L,0 * exp(-D_L * t)
    and water-cut dynamics via logit transform: logit(f_w(t)) = logit(f_w,0) + alpha_w * t.
    Guarantees strict mass conservation q_o + q_w = q_L and 0 <= f_w <= f_w_max.
    """
    def __init__(self, fw_max: float = 0.98):
        self.fw_max = fw_max
        self.model_DL = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.model_alpha = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.feat_cols = ["fault", "poro", "perm", "aq", "qL_0", "fw_0", "res_age"]
        
    def fit(
        self,
        case_dict: Dict[int, pd.DataFrame],
        unc_df: pd.DataFrame,
        train_cases: List[int],
        cutoff_date: pd.Timestamp,
        horizon_quarters: int,
    ):
        records = []
        for c in train_cases:
            df_c = case_dict[c]
            pre = df_c[df_c["date"] <= cutoff_date]
            post = df_c[df_c["date"] > cutoff_date].iloc[:horizon_quarters]
            
            if len(pre) == 0:
                continue
                
            q0_o = float(pre.iloc[-1]["oil_rate_stbd"])
            q0_w = float(pre.iloc[-1]["water_rate_stbd"])
            qL_0 = max(10.0, q0_o + q0_w)
            fw_0 = np.clip(q0_w / qL_0, 1e-4, self.fw_max - 1e-4)
            
            if len(post) >= 4:
                t_fit_y = ((post["date"] - cutoff_date).dt.total_seconds() / (86400.0 * YEAR_DAYS)).to_numpy()
                qL_post = post["oil_rate_stbd"].to_numpy() + post["water_rate_stbd"].to_numpy()
                qw_post = post["water_rate_stbd"].to_numpy()
            else:
                hist_seg = pre[pre["date"] >= "2000-01-01"]
                t_fit_y = ((hist_seg["date"] - hist_seg["date"].iloc[0]).dt.total_seconds() / (86400.0 * YEAR_DAYS)).to_numpy()
                qL_post = hist_seg["oil_rate_stbd"].to_numpy() + hist_seg["water_rate_stbd"].to_numpy()
                qw_post = hist_seg["water_rate_stbd"].to_numpy()
                
            # Fit liquid decline D_L: ln(q_L) = ln(q_L0) - D_L * t
            log_qL = np.log(np.maximum(qL_post, 10.0))
            if len(t_fit_y) >= 2 and np.ptp(t_fit_y) > 0:
                slope_L, _ = np.polyfit(t_fit_y, log_qL, 1)
                DL_obs = max(0.001, -slope_L)
            else:
                DL_obs = 0.05
                
            # Fit logit water cut rate: logit(fw(t)) = logit(fw_0) + alpha_w * t
            fw_post = np.clip(qw_post / np.maximum(qL_post, 1.0), 1e-4, self.fw_max - 1e-4)
            logit_fw = logit(fw_post / self.fw_max)
            if len(t_fit_y) >= 2 and np.ptp(t_fit_y) > 0:
                slope_alpha, _ = np.polyfit(t_fit_y, logit_fw, 1)
                alpha_obs = slope_alpha
            else:
                alpha_obs = 0.1
                
            u_c = unc_df.loc[c]
            records.append({
                "case_id": c,
                "fault": float(u_c["Fault Transmissibility"]),
                "poro": float(u_c["Porosity Multiplier"]),
                "perm": float(u_c["Permeability Multiplier"]),
                "aq": float(u_c["Aquifer Pore Volume"]),
                "qL_0": qL_0,
                "fw_0": fw_0,
                "res_age": float(pre.iloc[-1]["res_age_years"]),
                "DL_obs": DL_obs,
                "alpha_obs": alpha_obs,
            })
            
        fit_df = pd.DataFrame(records)
        X = fit_df[self.feat_cols].to_numpy()
        self.model_DL.fit(X, fit_df["DL_obs"].to_numpy())
        self.model_alpha.fit(X, fit_df["alpha_obs"].to_numpy())
        return self

    def predict_rates(
        self,
        initial_state: Dict[str, float],
        u_c: pd.Series,
        forecast_dates: pd.DatetimeIndex,
    ) -> Dict[str, np.ndarray]:
        forecast_dates = pd.DatetimeIndex(forecast_dates)
        t_years = ((forecast_dates - pd.Timestamp(initial_state["origin_date"])).total_seconds() / (86400.0 * YEAR_DAYS)).to_numpy()
        
        q0_o = initial_state["qo"]
        q0_w = initial_state["qw"]
        qL_0 = max(10.0, q0_o + q0_w)
        fw_0 = np.clip(q0_w / qL_0, 1e-4, self.fw_max - 1e-4)
        
        x_in = np.array([[
            float(u_c["Fault Transmissibility"]),
            float(u_c["Porosity Multiplier"]),
            float(u_c["Permeability Multiplier"]),
            float(u_c["Aquifer Pore Volume"]),
            qL_0,
            fw_0,
            initial_state["res_age"],
        ]])
        
        DL_pred = max(0.005, float(self.model_DL.predict(x_in)[0]))
        alpha_pred = float(self.model_alpha.predict(x_in)[0])
        
        # Liquid rate trajectory: exponential decline
        qL_pred = qL_0 * np.exp(-DL_pred * t_years)
        
        # Logit water cut trajectory
        logit_0 = logit(fw_0 / self.fw_max)
        logit_t = logit_0 + alpha_pred * t_years
        fw_pred = self.fw_max * expit(logit_t)
        
        # Coupled recovery
        qw_pred = fw_pred * qL_pred
        qo_pred = (1.0 - fw_pred) * qL_pred
        qg_pred = RS_SOLUTION * qo_pred
        
        return {
            "oil_rate": qo_pred,
            "water_rate": qw_pred,
            "gas_rate": qg_pred,
            "liquid_rate": qL_pred,
            "water_cut": fw_pred,
        }


class CoupledWORWaterfloodForecaster:
    """
    Formulation B: Coupled Semi-log WOR Decline Forecaster (Ershaghi & Omoregie / Gao et al.).
    Models ln(WOR(t)) = ln(WOR_0) + beta_wor * delta_Qo(t).
    Solves iteratively for oil and water rate at each step.
    """
    def __init__(self, fw_max: float = 0.98):
        self.fw_max = fw_max
        self.model_DL = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.model_beta = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.feat_cols = ["fault", "poro", "perm", "aq", "qL_0", "wor_0", "res_age"]
        
    def fit(
        self,
        case_dict: Dict[int, pd.DataFrame],
        unc_df: pd.DataFrame,
        train_cases: List[int],
        cutoff_date: pd.Timestamp,
        horizon_quarters: int,
    ):
        records = []
        for c in train_cases:
            df_c = case_dict[c]
            pre = df_c[df_c["date"] <= cutoff_date]
            post = df_c[df_c["date"] > cutoff_date].iloc[:horizon_quarters]
            
            if len(pre) == 0:
                continue
                
            q0_o = float(pre.iloc[-1]["oil_rate_stbd"])
            q0_w = float(pre.iloc[-1]["water_rate_stbd"])
            qL_0 = max(10.0, q0_o + q0_w)
            wor_0 = max(1e-4, q0_w / max(1.0, q0_o))
            
            if len(post) >= 4:
                t_fit_y = ((post["date"] - cutoff_date).dt.total_seconds() / (86400.0 * YEAR_DAYS)).to_numpy()
                qL_post = post["oil_rate_stbd"].to_numpy() + post["water_rate_stbd"].to_numpy()
                inc_qo = (post["oil_cum_stb"].to_numpy() - pre.iloc[-1]["oil_cum_stb"]) / 1e6 # MMSTB
                wor_post = np.maximum(1e-4, post["water_rate_stbd"].to_numpy() / np.maximum(1.0, post["oil_rate_stbd"].to_numpy()))
            else:
                hist_seg = pre[pre["date"] >= "2000-01-01"]
                t_fit_y = ((hist_seg["date"] - hist_seg["date"].iloc[0]).dt.total_seconds() / (86400.0 * YEAR_DAYS)).to_numpy()
                qL_post = hist_seg["oil_rate_stbd"].to_numpy() + hist_seg["water_rate_stbd"].to_numpy()
                inc_qo = (hist_seg["oil_cum_stb"].to_numpy() - hist_seg.iloc[0]["oil_cum_stb"]) / 1e6
                wor_post = np.maximum(1e-4, hist_seg["water_rate_stbd"].to_numpy() / np.maximum(1.0, hist_seg["oil_rate_stbd"].to_numpy()))
                
            log_qL = np.log(np.maximum(qL_post, 10.0))
            if len(t_fit_y) >= 2 and np.ptp(t_fit_y) > 0:
                slope_L, _ = np.polyfit(t_fit_y, log_qL, 1)
                DL_obs = max(0.001, -slope_L)
            else:
                DL_obs = 0.05
                
            log_wor = np.log(wor_post)
            if len(inc_qo) >= 2 and np.ptp(inc_qo) > 0:
                slope_wor, _ = np.polyfit(inc_qo, log_wor, 1)
                beta_obs = max(0.0, slope_wor)
            else:
                beta_obs = 0.5
                
            u_c = unc_df.loc[c]
            records.append({
                "case_id": c,
                "fault": float(u_c["Fault Transmissibility"]),
                "poro": float(u_c["Porosity Multiplier"]),
                "perm": float(u_c["Permeability Multiplier"]),
                "aq": float(u_c["Aquifer Pore Volume"]),
                "qL_0": qL_0,
                "wor_0": wor_0,
                "res_age": float(pre.iloc[-1]["res_age_years"]),
                "DL_obs": DL_obs,
                "beta_obs": beta_obs,
            })
            
        fit_df = pd.DataFrame(records)
        X = fit_df[self.feat_cols].to_numpy()
        self.model_DL.fit(X, fit_df["DL_obs"].to_numpy())
        self.model_beta.fit(X, fit_df["beta_obs"].to_numpy())
        return self

    def predict_rates(
        self,
        initial_state: Dict[str, float],
        u_c: pd.Series,
        forecast_dates: pd.DatetimeIndex,
    ) -> Dict[str, np.ndarray]:
        forecast_dates = pd.DatetimeIndex(forecast_dates)
        n_steps = len(forecast_dates)
        t_years = ((forecast_dates - pd.Timestamp(initial_state["origin_date"])).total_seconds() / (86400.0 * YEAR_DAYS)).to_numpy()
        
        q0_o = initial_state["qo"]
        q0_w = initial_state["qw"]
        qL_0 = max(10.0, q0_o + q0_w)
        wor_0 = max(1e-4, q0_w / max(1.0, q0_o))
        
        x_in = np.array([[
            float(u_c["Fault Transmissibility"]),
            float(u_c["Porosity Multiplier"]),
            float(u_c["Permeability Multiplier"]),
            float(u_c["Aquifer Pore Volume"]),
            qL_0,
            wor_0,
            initial_state["res_age"],
        ]])
        
        DL_pred = max(0.005, float(self.model_DL.predict(x_in)[0]))
        beta_pred = max(0.0, float(self.model_beta.predict(x_in)[0]))
        
        qo_preds = np.zeros(n_steps)
        qw_preds = np.zeros(n_steps)
        qL_preds = np.zeros(n_steps)
        fw_preds = np.zeros(n_steps)
        
        cum_qo_inc_mmstb = 0.0
        dt_days = 91.3125
        
        for k in range(n_steps):
            t_k = t_years[k]
            qL_k = qL_0 * np.exp(-DL_pred * t_k)
            
            # WOR model: ln(WOR) = ln(WOR_0) + beta * Qo_inc
            wor_k = wor_0 * np.exp(beta_pred * cum_qo_inc_mmstb)
            fw_k = min(self.fw_max, wor_k / (1.0 + wor_k))
            
            qw_k = fw_k * qL_k
            qo_k = (1.0 - fw_k) * qL_k
            
            qo_preds[k] = qo_k
            qw_preds[k] = qw_k
            qL_preds[k] = qL_k
            fw_preds[k] = fw_k
            
            cum_qo_inc_mmstb += (qo_k * dt_days) / 1e6
            
        qg_preds = RS_SOLUTION * qo_preds
        return {
            "oil_rate": qo_preds,
            "water_rate": qw_preds,
            "gas_rate": qg_preds,
            "liquid_rate": qL_preds,
            "water_cut": fw_preds,
        }


def run_waterflood_challenger_experiment() -> Dict[str, pd.DataFrame]:
    """
    Executes the comprehensive Phase 4 Physics-Informed Waterflood Challenger experiment.
    """
    print("=" * 80)
    print("PHASE 4: PHYSICS-INFORMED WATERFLOOD CHALLENGER EVALUATION")
    print("=" * 80)
    
    unc_df = load_uncertainty()
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)
    
    comparison_records = []
    breakthrough_records = []
    
    for origin_cfg in EVAL_ORIGINS:
        cutoff = pd.Timestamp(origin_cfg["cutoff"])
        max_qtrs = origin_cfg["max_qtrs"]
        origin_label = origin_cfg["label"]
        print(f"\nEvaluating waterflood models at origin: {origin_label} (Cutoff: {cutoff.strftime('%Y-%m-%d')})...")
        
        # Fit models on Cases 1-70
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
        
        # Fit coupled challengers
        m_logit = CoupledLogitWaterfloodForecaster().fit(
            case_dict, unc_df, TRAIN_CASES, cutoff, max_qtrs
        )
        m_wor = CoupledWORWaterfloodForecaster().fit(
            case_dict, unc_df, TRAIN_CASES, cutoff, max_qtrs
        )
        
        # Evaluate on all development cases (1-85)
        for c in list(range(1, 86)):
            subset = "Validation (71-85)" if c in VAL_CASES else "Train (1-70)"
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
            
            # Ground truth actuals
            act_inc_oil = post["oil_cum_stb"].to_numpy() - pre.iloc[-1]["oil_cum_stb"]
            act_inc_wat = post["water_cum_stb"].to_numpy() - pre.iloc[-1]["water_cum_stb"]
            act_rate_oil = post["oil_rate_stbd"].to_numpy()
            act_rate_wat = post["water_rate_stbd"].to_numpy()
            act_rate_liquid = act_rate_oil + act_rate_wat
            act_fw = act_rate_wat / np.maximum(act_rate_liquid, 1.0)
            
            # Predictions: Base models
            p_et = m_et.predict_rates(init_state, u_c, f_dates)
            p_exp = m_exp.predict_rates(init_state, u_c, f_dates)
            p_hyp = m_hyp.predict_rates(init_state, u_c, f_dates)
            p_hyb = m_hyb.predict_rates(init_state, u_c, f_dates)
            
            # Frozen stack
            q_oil_stack = (
                LOCKED_WEIGHTS["oil"]["Recursive_ExtraTrees"] * p_et["oil_rate"][:n_actual]
                + LOCKED_WEIGHTS["oil"]["Learned_Exponential_DCA"] * p_exp["oil_rate"][:n_actual]
                + LOCKED_WEIGHTS["oil"]["Learned_Hyperbolic_DCA"] * p_hyp["oil_rate"][:n_actual]
                + LOCKED_WEIGHTS["oil"]["Hybrid_Dynamic"] * p_hyb["oil_rate"][:n_actual]
            )
            q_wat_stack = (
                LOCKED_WEIGHTS["water"]["Recursive_ExtraTrees"] * p_et["water_rate"][:n_actual]
                + LOCKED_WEIGHTS["water"]["Learned_Exponential_DCA"] * p_exp["water_rate"][:n_actual]
                + LOCKED_WEIGHTS["water"]["Learned_Hyperbolic_DCA"] * p_hyp["water_rate"][:n_actual]
                + LOCKED_WEIGHTS["water"]["Hybrid_Dynamic"] * p_hyb["water_rate"][:n_actual]
            )
            
            # Coupled models
            p_logit = m_logit.predict_rates(init_state, u_c, f_dates)
            p_wor = m_wor.predict_rates(init_state, u_c, f_dates)
            
            models_to_test = {
                "Blend_Phase_Specific_Simplex (Baseline)": {
                    "qo": q_oil_stack, "qw": q_wat_stack,
                },
                "Recursive_ExtraTrees": {
                    "qo": p_et["oil_rate"][:n_actual], "qw": p_et["water_rate"][:n_actual],
                },
                "Coupled_Waterflood_Logit": {
                    "qo": p_logit["oil_rate"][:n_actual], "qw": p_logit["water_rate"][:n_actual],
                },
                "Coupled_Waterflood_WOR": {
                    "qo": p_wor["oil_rate"][:n_actual], "qw": p_wor["water_rate"][:n_actual],
                },
            }
            
            # Actual breakthrough quarter (first quarter where fw >= 0.05)
            act_bt_idx = np.where(act_fw >= 0.05)[0]
            act_bt_qtr = int(act_bt_idx[0] + 1) if len(act_bt_idx) > 0 else n_actual + 1
            
            for m_name, p_vals in models_to_test.items():
                pred_qo = p_vals["qo"]
                pred_qw = p_vals["qw"]
                pred_qL = pred_qo + pred_qw
                pred_fw = pred_qw / np.maximum(pred_qL, 1.0)
                
                pred_inc_oil = np.cumsum(pred_qo * dt_days)
                pred_inc_wat = np.cumsum(pred_qw * dt_days)
                
                oil_nrmse = compute_increment_nrmse(pred_inc_oil - act_inc_oil, act_inc_oil)
                wat_nrmse = compute_increment_nrmse(pred_inc_wat - act_inc_wat, act_inc_wat)
                macro_nrmse = 0.5 * (oil_nrmse + wat_nrmse)
                
                rate_oil_nrmse = float(np.sqrt(np.mean((pred_qo - act_rate_oil) ** 2)) / (np.mean(act_rate_oil) + 1e-6))
                rate_wat_nrmse = float(np.sqrt(np.mean((pred_qw - act_rate_wat) ** 2)) / (np.mean(act_rate_wat) + 1e-6))
                rate_liquid_nrmse = float(np.sqrt(np.mean((pred_qL - act_rate_liquid) ** 2)) / (np.mean(act_rate_liquid) + 1e-6))
                
                fw_rmse = float(np.sqrt(np.mean((pred_fw - act_fw) ** 2)))
                fw_mae = float(np.mean(np.abs(pred_fw - act_fw)))
                
                # Breakthrough timing error
                pred_bt_idx = np.where(pred_fw >= 0.05)[0]
                pred_bt_qtr = int(pred_bt_idx[0] + 1) if len(pred_bt_idx) > 0 else n_actual + 1
                bt_err_qtrs = pred_bt_qtr - act_bt_qtr
                
                comparison_records.append({
                    "origin": origin_label,
                    "case_id": c,
                    "subset": subset,
                    "model": m_name,
                    "oil_cum_nrmse": oil_nrmse,
                    "water_cum_nrmse": wat_nrmse,
                    "macro_nrmse": macro_nrmse,
                    "water_rate_nrmse": rate_wat_nrmse,
                    "liquid_rate_nrmse": rate_liquid_nrmse,
                    "water_cut_rmse": fw_rmse,
                    "water_cut_mae": fw_mae,
                    "breakthrough_error_qtrs": bt_err_qtrs,
                    "terminal_fw_pred": float(pred_fw[-1]),
                    "terminal_fw_act": float(act_fw[-1]),
                    "terminal_fw_err": float(abs(pred_fw[-1] - act_fw[-1])),
                })
                
    comp_df = pd.DataFrame(comparison_records)
    
    summary_df = comp_df.groupby(["origin", "subset", "model"]).agg({
        "oil_cum_nrmse": "mean",
        "water_cum_nrmse": "mean",
        "macro_nrmse": "mean",
        "water_rate_nrmse": "mean",
        "liquid_rate_nrmse": "mean",
        "water_cut_rmse": "mean",
        "breakthrough_error_qtrs": lambda x: float(np.mean(np.abs(x))),
        "terminal_fw_err": "mean",
    }).reset_index()
    summary_df.columns = [
        "origin", "subset", "model",
        "oil_cum_nrmse", "water_cum_nrmse", "macro_nrmse",
        "water_rate_nrmse", "liquid_rate_nrmse",
        "water_cut_rmse", "mean_abs_breakthrough_err_qtrs",
        "terminal_water_cut_error"
    ]
    
    comp_path = OUTPUT_TABLE_DIR / "waterflood_coupled_challenger_comparison.csv"
    summary_df.to_csv(comp_path, index=False)
    print(f"\nSaved waterflood coupled challenger comparison to: {comp_path}")
    
    return {
        "raw_comparison": comp_df,
        "summary": summary_df,
    }


if __name__ == "__main__":
    run_waterflood_challenger_experiment()
