"""
Module: maturity_gating_decision.py
Package: forecast_advancement
Addresses: Stages 3 & 4 — Conclusive Maturity Gating Disposition and Honest 2008 Deployment Gating
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Features:
1. Implements the 5 candidate architectures:
   - Frozen Ensemble (Model A baseline)
   - Constant Learned Weight (Global scalar w)
   - Physical Maturity Gate (Logistic on decline depth and quarters)
   - Oil-Only Maturity Gate (Gating oil while water is 100% frozen baseline)
   - Simple Decline-Rate-Informed Oil Correction (Analytical rate damping via instantaneous decline)
2. Nested case-grouped cross-fitting under Protocol A (strictly rolling history) and Protocol B (forward supervised).
3. Evaluates oil and water separately.
4. Removes hardcoded 2008 deployment weights (0.10, 0.05). If insufficient rolling history exists, marks unavailable and retains baseline.
"""
from typing import Dict, List, Tuple, Any, Optional
import time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
from sklearn.model_selection import KFold

from ensemble.data import load_uncertainty, TRAIN_CASES, VAL_CASES, YEAR_DAYS
from ensemble.metrics import compute_increment_nrmse, compute_bias
from ensemble.generate_competition_deliverable import (
    load_locked_config,
    generate_scenario_predictions,
)
from conditional_challengers.corrected_residual_dca import (
    CorrectedResidualForecaster,
    fit_historical_dca_at_origin,
)
from causal_maturity.sustained_decline_detector import extract_robust_case_maturity
from forecast_advancement.protocol_framework import (
    build_asof_case_dict,
    fit_base_models_under_protocol,
)


def compute_case_increment_nrmse_dual(
    qo_pred: np.ndarray,
    qw_pred: np.ndarray,
    post_df: pd.DataFrame,
    init_state: Dict[str, float],
    dt_days: float = 91.3125,
) -> Tuple[float, float, float]:
    """Computes (macro_nrmse, oil_nrmse, water_nrmse) for a case."""
    qo_clean = np.asarray(qo_pred, dtype=float).copy()
    qw_clean = np.asarray(qw_pred, dtype=float).copy()
    
    # Monotonic clamping safeguard for oil
    for t in range(1, len(qo_clean)):
        if qo_clean[t] > qo_clean[t - 1]:
            qo_clean[t] = qo_clean[t - 1]
    qo_clean = np.maximum(qo_clean, 0.0)
    qw_clean = np.maximum(qw_clean, 0.0)
    
    cum_o = init_state["Qo"] + np.cumsum(qo_clean * dt_days)
    cum_w = init_state["Qw"] + np.cumsum(qw_clean * dt_days)
    
    true_o = post_df["oil_cum_stb"].values if "oil_cum_stb" in post_df.columns else post_df["oil_cum"].values
    true_w = post_df["water_cum_stb"].values if "water_cum_stb" in post_df.columns else post_df["water_cum"].values
    
    inc_o = true_o - init_state["Qo"]
    inc_w = true_w - init_state["Qw"]
    
    no = compute_increment_nrmse(cum_o - true_o, inc_o)
    nw = compute_increment_nrmse(cum_w - true_w, inc_w)
    return 0.5 * (no + nw), no, nw


class SimpleDeclineRateOilCorrection:
    """Analytical decline-rate-informed oil correction."""
    def __init__(self):
        self.alpha = 0.0
        self.fitted = False

    def fit(self, oof_cases, oof_frz_preds, oof_post_dfs, oof_init_states, oof_maturities):
        def obj(a):
            total_err = 0.0
            for c in oof_cases:
                d_inst = float(oof_maturities[c].get("instantaneous_decline_rate", 0.0))
                factor = max(0.5, 1.0 - a * d_inst)
                qo_corr = oof_frz_preds[c]["oil_rate"] * factor
                qw_base = oof_frz_preds[c]["water_rate"]
                m, no, _ = compute_case_increment_nrmse_dual(
                    qo_corr, qw_base, oof_post_dfs[c], oof_init_states[c]
                )
                total_err += no
            return total_err / len(oof_cases)
            
        res = minimize_scalar(obj, bounds=(0.0, 2.0), method="bounded")
        self.alpha = float(res.x)
        self.fitted = True
        return self

    def predict(self, qo_base: np.ndarray, qw_base: np.ndarray, maturity: Dict[str, Any]):
        d_inst = float(maturity.get("instantaneous_decline_rate", 0.0))
        factor = max(0.5, 1.0 - self.alpha * d_inst)
        return qo_base * factor, qw_base


def run_gating_evaluation_experiment(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    origin_str: str,
    horizon_quarters: int,
    protocol: str = "Protocol_B",
    n_folds: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates candidate gates under Protocol A or Protocol B on outer holdout Cases 71-85.
    """
    orig = pd.Timestamp(origin_str)
    config = load_locked_config()
    weights = config["weights"]
    f_dates = pd.date_range(orig + pd.DateOffset(days=1), periods=horizon_quarters, freq="QS")
    dt_days = 91.3125
    
    # Check Protocol A rolling history availability
    if protocol == "Protocol_A":
        # Under Protocol A, to train a gate of length horizon_quarters,
        # we must have historical rolling origins T_roll where T_roll + horizon_quarters <= orig.
        # Commercial decline began around late 2001.
        min_decline_date = pd.Timestamp("2002-01-01")
        earliest_roll = min_decline_date
        latest_roll = orig - pd.DateOffset(days=int(horizon_quarters * dt_days))
        
        if latest_roll < earliest_roll:
            print(f"  [PROTOCOL A NOTICE] No historical rolling windows of length {horizon_quarters}Q completed prior to {origin_str}.")
            print(f"  Gate models marked UNAVAILABLE under Protocol A. Evaluating Frozen Baseline only.")
            
            # Baseline is still fully evaluable!
            m_frozen = fit_base_models_under_protocol(case_dict, unc_df, TRAIN_CASES, orig, "Protocol_A")
            base_macro, base_oil, base_wat = [], [], []
            case_details = []
            
            for c in VAL_CASES:
                df_c = case_dict[c]
                pre = df_c[df_c["date"] <= orig]
                post = df_c[df_c["date"] > orig].iloc[:horizon_quarters]
                if len(post) < horizon_quarters:
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
                    "res_age": (orig - pre["date"].iloc[0]).days / YEAR_DAYS,
                    "origin_date": orig,
                }
                _, aud = generate_scenario_predictions(m_frozen, weights, st, u_c, f_dates, 1.0)
                m, no, nw = compute_case_increment_nrmse_dual(aud["qo_rate"], aud["qw_rate"], post, st)
                base_macro.append(m)
                base_oil.append(no)
                base_wat.append(nw)
                case_details.append({
                    "origin": origin_str,
                    "horizon_quarters": horizon_quarters,
                    "protocol": protocol,
                    "case_id": c,
                    "model_name": "Frozen_Ensemble",
                    "macro_nrmse": m,
                    "oil_nrmse": no,
                    "water_nrmse": nw,
                })
                
            df_summ = pd.DataFrame([{
                "origin": origin_str,
                "horizon_quarters": horizon_quarters,
                "protocol": protocol,
                "model_name": "Frozen_Ensemble",
                "macro_nrmse_mean": float(np.mean(base_macro)),
                "oil_nrmse_mean": float(np.mean(base_oil)),
                "water_nrmse_mean": float(np.mean(base_wat)),
                "diff_vs_baseline": 0.0,
                "win_rate_vs_baseline": 0.5,
                "wilcoxon_p": 1.0,
                "status": "EVALUATED_BASELINE_DOMINANT",
            }])
            return df_summ, pd.DataFrame(case_details)

    # Standard Protocol B (or Protocol A with feasible rolling windows)
    # 1. Fit base models on training pool
    if protocol == "Protocol_A":
        tr_dict = build_asof_case_dict(case_dict, orig)
        m_frozen = fit_base_models_under_protocol(case_dict, unc_df, TRAIN_CASES, orig, "Protocol_A")
        m_dca = fit_historical_dca_at_origin(tr_dict, unc_df, TRAIN_CASES, orig)
        m_res = CorrectedResidualForecaster(architecture="gp").fit(
            tr_dict, unc_df, TRAIN_CASES, cutoff_date=orig, max_horizon_quarters=horizon_quarters
        )
    else:
        tr_dict = case_dict
        m_frozen = fit_base_models_under_protocol(case_dict, unc_df, TRAIN_CASES, orig, "Protocol_B")
        m_dca = fit_historical_dca_at_origin(tr_dict, unc_df, TRAIN_CASES, orig)
        m_res = CorrectedResidualForecaster(architecture="gp").fit(
            tr_dict, unc_df, TRAIN_CASES, cutoff_date=orig, max_horizon_quarters=horizon_quarters
        )
        
    # 2. Outer cross-fitting for gate training on Cases 1-70
    train_pool = np.array(TRAIN_CASES)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    oof_res = {}
    oof_frz = {}
    oof_post = {}
    oof_st = {}
    oof_mat = {}
    
    for tr_idx, val_idx in kf.split(train_pool):
        tr_c = train_pool[tr_idx].tolist()
        val_c = train_pool[val_idx].tolist()
        
        if protocol == "Protocol_A":
            sub_tr_dict = build_asof_case_dict(case_dict, orig)
            sub_frozen = fit_base_models_under_protocol(case_dict, unc_df, tr_c, orig, "Protocol_A")
            sub_dca = fit_historical_dca_at_origin(sub_tr_dict, unc_df, tr_c, orig)
            sub_res = CorrectedResidualForecaster(architecture="gp").fit(
                sub_tr_dict, unc_df, tr_c, cutoff_date=orig, max_horizon_quarters=horizon_quarters
            )
        else:
            sub_frozen = fit_base_models_under_protocol(case_dict, unc_df, tr_c, orig, "Protocol_B")
            sub_dca = fit_historical_dca_at_origin(case_dict, unc_df, tr_c, orig)
            sub_res = CorrectedResidualForecaster(architecture="gp").fit(
                case_dict, unc_df, tr_c, cutoff_date=orig, max_horizon_quarters=horizon_quarters
            )
            
        for c in val_c:
            df_c = case_dict[c]
            pre = df_c[df_c["date"] <= orig]
            post = df_c[df_c["date"] > orig].iloc[:horizon_quarters]
            if len(post) < horizon_quarters:
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
                "res_age": (orig - pre["date"].iloc[0]).days / YEAR_DAYS,
                "origin_date": orig,
            }
            _, aud = generate_scenario_predictions(sub_frozen, weights, st, u_c, f_dates, 1.0)
            p_res = sub_res.predict_rates(df_c, u_c, orig, f_dates, sub_dca)
            mat = extract_robust_case_maturity(df_c, orig)
            
            oof_res[c] = p_res
            oof_frz[c] = {"oil_rate": aud["qo_rate"], "water_rate": aud["qw_rate"]}
            oof_post[c] = post
            oof_st[c] = st
            oof_mat[c] = mat
            
    oof_cases = list(oof_res.keys())
    
    # 3. Fit Candidate Gates on out-of-fold loss
    # A) Constant scalar weight
    def loss_const(w):
        tot = 0.0
        for c in oof_cases:
            qo = w * oof_res[c]["oil_rate"] + (1.0 - w) * oof_frz[c]["oil_rate"]
            qw = w * oof_res[c]["water_rate"] + (1.0 - w) * oof_frz[c]["water_rate"]
            m, _, _ = compute_case_increment_nrmse_dual(qo, qw, oof_post[c], oof_st[c])
            tot += m
        return tot / len(oof_cases)
    opt_const = minimize_scalar(loss_const, bounds=(0.0, 1.0), method="bounded")
    w_const = float(opt_const.x)
    
    # B) Physical maturity gate (logistic on decline depth and quarters)
    def loss_phys(params):
        a, b, c_bias = params
        tot = 0.0
        for c in oof_cases:
            depth = float(oof_mat[c].get("decline_depth", 0.0))
            q_decl = float(oof_mat[c].get("quarters_in_decline", 0.0)) / 20.0
            z = a * depth + b * q_decl + c_bias
            w = 1.0 / (1.0 + np.exp(-z))
            qo = w * oof_res[c]["oil_rate"] + (1.0 - w) * oof_frz[c]["oil_rate"]
            qw = w * oof_res[c]["water_rate"] + (1.0 - w) * oof_frz[c]["water_rate"]
            m, _, _ = compute_case_increment_nrmse_dual(qo, qw, oof_post[c], oof_st[c])
            tot += m
        return tot / len(oof_cases)
    opt_phys = minimize(loss_phys, [0.0, 0.0, 0.0], method="Nelder-Mead")
    params_phys = opt_phys.x
    
    # C) Oil-only maturity gate
    def loss_oil_only(params):
        a, b, c_bias = params
        tot = 0.0
        for c in oof_cases:
            depth = float(oof_mat[c].get("decline_depth", 0.0))
            q_decl = float(oof_mat[c].get("quarters_in_decline", 0.0)) / 20.0
            z = a * depth + b * q_decl + c_bias
            w = 1.0 / (1.0 + np.exp(-z))
            qo = w * oof_res[c]["oil_rate"] + (1.0 - w) * oof_frz[c]["oil_rate"]
            qw = oof_frz[c]["water_rate"]  # Water 100% frozen baseline
            m, _, _ = compute_case_increment_nrmse_dual(qo, qw, oof_post[c], oof_st[c])
            tot += m
        return tot / len(oof_cases)
    opt_oil_only = minimize(loss_oil_only, [0.0, 0.0, 0.0], method="Nelder-Mead")
    params_oil_only = opt_oil_only.x
    
    # D) Simple decline rate correction
    model_simple = SimpleDeclineRateOilCorrection().fit(
        oof_cases, oof_frz, oof_post, oof_st, oof_mat
    )
    
    # 4. Evaluate on Holdout Cases 71-85
    models_to_eval = [
        "Frozen_Ensemble",
        "Constant_Learned_Weight",
        "Physical_Maturity_Gate",
        "Oil_Only_Maturity_Gate",
        "Simple_Decline_Rate_Oil_Correction",
    ]
    
    val_preds = {m: [] for m in models_to_eval}
    val_oil_preds = {m: [] for m in models_to_eval}
    val_wat_preds = {m: [] for m in models_to_eval}
    case_detail_records = []
    
    for c in VAL_CASES:
        df_c = case_dict[c]
        pre = df_c[df_c["date"] <= orig]
        post = df_c[df_c["date"] > orig].iloc[:horizon_quarters]
        if len(post) < horizon_quarters:
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
            "res_age": (orig - pre["date"].iloc[0]).days / YEAR_DAYS,
            "origin_date": orig,
        }
        _, aud = generate_scenario_predictions(m_frozen, weights, st, u_c, f_dates, 1.0)
        p_frz = {"oil_rate": aud["qo_rate"], "water_rate": aud["qw_rate"]}
        p_res = m_res.predict_rates(df_c, u_c, orig, f_dates, m_dca)
        mat = extract_robust_case_maturity(df_c, orig)
        
        # Predictions for each model
        # 1. Frozen Ensemble
        m_frz, no_frz, nw_frz = compute_case_increment_nrmse_dual(p_frz["oil_rate"], p_frz["water_rate"], post, st)
        val_preds["Frozen_Ensemble"].append(m_frz)
        val_oil_preds["Frozen_Ensemble"].append(no_frz)
        val_wat_preds["Frozen_Ensemble"].append(nw_frz)
        
        # 2. Constant Learned Weight
        qo_const = w_const * p_res["oil_rate"] + (1.0 - w_const) * p_frz["oil_rate"]
        qw_const = w_const * p_res["water_rate"] + (1.0 - w_const) * p_frz["water_rate"]
        m_c, no_c, nw_c = compute_case_increment_nrmse_dual(qo_const, qw_const, post, st)
        val_preds["Constant_Learned_Weight"].append(m_c)
        val_oil_preds["Constant_Learned_Weight"].append(no_c)
        val_wat_preds["Constant_Learned_Weight"].append(nw_c)
        
        # 3. Physical Maturity Gate
        depth = float(mat.get("decline_depth", 0.0))
        q_decl = float(mat.get("quarters_in_decline", 0.0)) / 20.0
        z_phys = params_phys[0] * depth + params_phys[1] * q_decl + params_phys[2]
        w_phys = 1.0 / (1.0 + np.exp(-z_phys))
        qo_phys = w_phys * p_res["oil_rate"] + (1.0 - w_phys) * p_frz["oil_rate"]
        qw_phys = w_phys * p_res["water_rate"] + (1.0 - w_phys) * p_frz["water_rate"]
        m_p, no_p, nw_p = compute_case_increment_nrmse_dual(qo_phys, qw_phys, post, st)
        val_preds["Physical_Maturity_Gate"].append(m_p)
        val_oil_preds["Physical_Maturity_Gate"].append(no_p)
        val_wat_preds["Physical_Maturity_Gate"].append(nw_p)
        
        # 4. Oil-Only Maturity Gate
        z_oil = params_oil_only[0] * depth + params_oil_only[1] * q_decl + params_oil_only[2]
        w_oil = 1.0 / (1.0 + np.exp(-z_oil))
        qo_oo = w_oil * p_res["oil_rate"] + (1.0 - w_oil) * p_frz["oil_rate"]
        qw_oo = p_frz["water_rate"]  # Water 100% frozen
        m_oo, no_oo, nw_oo = compute_case_increment_nrmse_dual(qo_oo, qw_oo, post, st)
        val_preds["Oil_Only_Maturity_Gate"].append(m_oo)
        val_oil_preds["Oil_Only_Maturity_Gate"].append(no_oo)
        val_wat_preds["Oil_Only_Maturity_Gate"].append(nw_oo)
        
        # 5. Simple Decline Rate Correction
        qo_sd, qw_sd = model_simple.predict(p_frz["oil_rate"], p_frz["water_rate"], mat)
        m_sd, no_sd, nw_sd = compute_case_increment_nrmse_dual(qo_sd, qw_sd, post, st)
        val_preds["Simple_Decline_Rate_Oil_Correction"].append(m_sd)
        val_oil_preds["Simple_Decline_Rate_Oil_Correction"].append(no_sd)
        val_wat_preds["Simple_Decline_Rate_Oil_Correction"].append(nw_sd)
        
        # Case detail
        for m_name in models_to_eval:
            case_detail_records.append({
                "origin": origin_str,
                "horizon_quarters": horizon_quarters,
                "protocol": protocol,
                "case_id": c,
                "model_name": m_name,
                "macro_nrmse": val_preds[m_name][-1],
                "oil_nrmse": val_oil_preds[m_name][-1],
                "water_nrmse": val_wat_preds[m_name][-1],
                "decline_depth": depth,
                "quarters_in_decline": mat.get("quarters_in_decline", 0),
            })
            
    # Summarize with statistics
    summary_records = []
    base_macro_arr = np.array(val_preds["Frozen_Ensemble"])
    
    for m_name in models_to_eval:
        arr_m = np.array(val_preds[m_name])
        arr_no = np.array(val_oil_preds[m_name])
        arr_nw = np.array(val_wat_preds[m_name])
        
        diff = arr_m - base_macro_arr
        mean_diff = float(np.mean(diff))
        
        # Win rate
        wins = np.sum(diff < -1e-6)
        ties = np.sum(np.abs(diff) <= 1e-6)
        win_rate = (wins + 0.5 * ties) / len(diff)
        
        # Wilcoxon signed rank test
        if np.all(np.abs(diff) < 1e-12):
            p_val = 1.0
        else:
            try:
                p_val = float(stats.wilcoxon(diff, alternative="two-sided").pvalue)
            except Exception:
                p_val = 1.0
                
        summary_records.append({
            "origin": origin_str,
            "horizon_quarters": horizon_quarters,
            "protocol": protocol,
            "model_name": m_name,
            "macro_nrmse_mean": float(np.mean(arr_m)),
            "oil_nrmse_mean": float(np.mean(arr_no)),
            "water_nrmse_mean": float(np.mean(arr_nw)),
            "diff_vs_baseline": mean_diff,
            "win_rate_vs_baseline": win_rate,
            "wilcoxon_p": p_val,
            "status": "EVALUATED",
        })
        
    return pd.DataFrame(summary_records), pd.DataFrame(case_detail_records)
