"""
Module: baseline_recomputation.py
Package: forecast_advancement
Addresses: Stage 2 — Machine-Recomputed Baseline Across All Feasible Origins Under Both Protocols
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Recomputes the frozen production baseline Blend_Phase_Specific_Simplex directly
at every feasible origin under:
1. Protocol A (Strict Historical As-Of): Truncated case_dict <= origin, historical fallback DCA.
2. Protocol B (Cross-Case Forward-Supervised): Forward targets post-origin on Cases 1-70.

Eliminates the arbitrary 1.5x horizon heuristic restriction.
Documents exact training time ranges and DCA targets.
"""
from typing import Dict, List, Tuple, Any
import time
import numpy as np
import pandas as pd

from ensemble.data import load_uncertainty, TRAIN_CASES, VAL_CASES, YEAR_DAYS
from ensemble.metrics import compute_increment_nrmse, compute_bias
from ensemble.generate_competition_deliverable import (
    load_locked_config,
    generate_scenario_predictions,
)
from forecast_advancement.protocol_framework import fit_base_models_under_protocol


FEASIBLE_ORIGINS = [
    ("2003-01-01", 12),
    ("2004-01-01", 16),
    ("2005-01-01", 12),
    ("2006-01-01", 8),
    ("2007-01-01", 4),
]


def evaluate_forecast_on_cases(
    models: Dict[str, Any],
    weights: Dict[str, Dict[str, float]],
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    val_cases: List[int],
    origin: pd.Timestamp,
    horizon_quarters: int,
    dt_days: float = 91.3125,
) -> Dict[str, Any]:
    """
    Evaluates predictions across validation cases, computing macro, oil, water NRMSE,
    bias, and case-level results.
    """
    f_dates = pd.date_range(origin + pd.DateOffset(days=1), periods=horizon_quarters, freq="QS")
    
    macro_list = []
    oil_list = []
    wat_list = []
    bo_list = []
    bw_list = []
    case_details = []
    
    for c in val_cases:
        df_c = case_dict[c]
        pre = df_c[df_c["date"] <= origin]
        post = df_c[df_c["date"] > origin].iloc[:horizon_quarters]
        if len(post) < horizon_quarters:
            continue
            
        last = pre.iloc[-1]
        u_c = unc_df.loc[c]
        init_st = {
            "qo": float(last["oil_rate_stbd"]),
            "qw": float(last["water_rate_stbd"]),
            "qg": float(last["gas_rate_mscfd"]),
            "Qo": float(last["oil_cum_stb"]),
            "Qw": float(last["water_cum_stb"]),
            "Qg": float(last["gas_cum_mscf"]),
            "wor": float(last.get("wor", float(last["water_rate_stbd"]) / max(float(last["oil_rate_stbd"]), 1.0))),
            "res_age": (origin - pre["date"].iloc[0]).days / YEAR_DAYS,
            "origin_date": origin,
        }
        
        _, aud = generate_scenario_predictions(models, weights, init_st, u_c, f_dates, 1.0)
        
        qo_pred = np.asarray(aud["qo_rate"], dtype=float).copy()
        qw_pred = np.asarray(aud["qw_rate"], dtype=float).copy()
        
        # Monotonic clamping safeguard
        for t in range(1, len(qo_pred)):
            if qo_pred[t] > qo_pred[t - 1]:
                qo_pred[t] = qo_pred[t - 1]
        qo_clean = np.maximum(qo_pred, 0.0)
        qw_clean = np.maximum(qw_pred, 0.0)
        
        cum_o = init_st["Qo"] + np.cumsum(qo_clean * dt_days)
        cum_w = init_st["Qw"] + np.cumsum(qw_clean * dt_days)
        
        true_o = post["oil_cum_stb"].values if "oil_cum_stb" in post.columns else post["oil_cum"].values
        true_w = post["water_cum_stb"].values if "water_cum_stb" in post.columns else post["water_cum"].values
        
        inc_o = true_o - init_st["Qo"]
        inc_w = true_w - init_st["Qw"]
        
        err_o = cum_o - true_o
        err_w = cum_w - true_w
        
        no = compute_increment_nrmse(err_o, inc_o)
        nw = compute_increment_nrmse(err_w, inc_w)
        bo = compute_bias(err_o, inc_o)
        bw = compute_bias(err_w, inc_w)
        macro = 0.5 * (no + nw)
        
        macro_list.append(macro)
        oil_list.append(no)
        wat_list.append(nw)
        bo_list.append(bo)
        bw_list.append(bw)
        
        case_details.append({
            "case_id": c,
            "macro_nrmse": macro,
            "oil_nrmse": no,
            "water_nrmse": nw,
            "oil_bias": bo,
            "water_bias": bw,
        })
        
    return {
        "macro_mean": float(np.mean(macro_list)),
        "macro_std": float(np.std(macro_list)),
        "oil_mean": float(np.mean(oil_list)),
        "wat_mean": float(np.mean(wat_list)),
        "oil_bias_mean": float(np.mean(bo_list)),
        "wat_bias_mean": float(np.mean(bw_list)),
        "n_cases": len(macro_list),
        "case_details": case_details,
    }


def recompute_all_baseline_evaluations(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    train_cases: List[int] = TRAIN_CASES,
    val_cases: List[int] = VAL_CASES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Directly runs baseline recomputation across all origins for Protocol A and Protocol B.
    Returns:
    - Summary DataFrame
    - Detailed case-level DataFrame
    """
    config = load_locked_config()
    weights = config["weights"]
    
    summary_rows = []
    case_rows = []
    
    for origin_str, horizon_q in FEASIBLE_ORIGINS:
        orig = pd.Timestamp(origin_str)
        
        for proto in ["Protocol_A", "Protocol_B"]:
            t0 = time.perf_counter()
            models = fit_base_models_under_protocol(
                case_dict, unc_df, train_cases, orig, protocol=proto, horizon_quarters=80
            )
            fit_time = time.perf_counter() - t0
            
            res = evaluate_forecast_on_cases(
                models, weights, case_dict, unc_df, val_cases, orig, horizon_q
            )
            
            if proto == "Protocol_A":
                dca_target_desc = f"Historical Pre-Origin Window (>= 2002-01-01 to {origin_str})"
                training_range = f"1998-01-01 to {origin_str}"
            else:
                dca_target_desc = f"Post-Origin Forward Simulators (80Q from {origin_str})"
                training_range = f"1998-01-01 to 2028-01-01 (training cases 1-70)"
                
            summary_rows.append({
                "origin": origin_str,
                "horizon_quarters": horizon_q,
                "protocol": proto,
                "model_name": "Blend_Phase_Specific_Simplex",
                "macro_nrmse": res["macro_mean"],
                "oil_nrmse": res["oil_mean"],
                "water_nrmse": res["wat_mean"],
                "oil_bias": res["oil_bias_mean"],
                "water_bias": res["wat_bias_mean"],
                "fit_time_sec": fit_time,
                "dca_training_target": dca_target_desc,
                "training_date_range": training_range,
                "n_eval_cases": res["n_cases"],
            })
            
            for cd in res["case_details"]:
                case_rows.append({
                    "origin": origin_str,
                    "horizon_quarters": horizon_q,
                    "protocol": proto,
                    "case_id": cd["case_id"],
                    "macro_nrmse": cd["macro_nrmse"],
                    "oil_nrmse": cd["oil_nrmse"],
                    "water_nrmse": cd["water_nrmse"],
                    "oil_bias": cd["oil_bias"],
                    "water_bias": cd["water_bias"],
                })
                
    df_summary = pd.DataFrame(summary_rows)
    df_cases = pd.DataFrame(case_rows)
    return df_summary, df_cases
