"""
Module: water_blend_engine.py
Package: conservative_water_research
Project: ExxonMobil DataWorks Challenge 2026

Engine for Conservative Water Blending Research:
1. Implements strict temporal and double-reservoir separation:
   - Outer held-out reservoir excluded from all outer base model training and inner calibration.
   - Inner target reservoirs excluded from their own inner model training (Leave-One-Out double exclusion).
   - Only completed inner windows ending on or before the outer evaluation origin are used.
   - Model fitting truncated at inner origin under Protocol A (Strict Historical As-Of).
2. Optimizes the actual cumulative forecasting loss:
   - Blends predicted water rates: w(fw_0) = gamma * clip((fw_0 - 0.30)/0.30, 0, 1).
   - Integrates using locked production cumulative convention (dt = 91.3125 days, monotonic accumulation).
   - Scores cumulative trajectory with authoritative cumulative-increment NRMSE scorer.
   - Selects gamma from {0.0, 0.10, 0.25, 0.50} using equally weighted reservoir-level mean losses.
   - Defaults to gamma=0.0 (explicit abstention) when calibration evidence is insufficient.
3. Supports exactly three frozen configurations:
   - Candidate_A_Frozen_Baseline
   - Candidate_Shrinkage_Blend (diagnostic control, reproduced unchanged)
   - Candidate_Corrected_Water_Blend (corrected conservative candidate)
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
import pandas as pd

from ensemble.generate_competition_deliverable import load_locked_config
from ensemble.run_horizon_adaptive_stacking_experiment import (
    RecursiveTransitionForecaster,
    LearnedDCARateForecaster,
    HybridDynamicForecaster,
    extract_transition_training_data,
    build_case_quarterly_trajectories,
    load_clean_simulation_quarterly,
    RS_SOLUTION,
    YEAR_DAYS,
)
from ensemble.data import (
    load_uncertainty,
    resolve_shared_dir,
    TRAIN_CASES,
    VAL_CASES,
)
from geological_information_value.authoritative_scorer import score_cumulative_forecast
from geological_information_value.parameter_bounds import PARAM_NAMES, create_parameter_series
from forecast_advancement.protocol_framework import build_asof_case_dict

DT_DAYS = 91.3125
CONFIG_PATH_DEFAULT = Path("src/ensemble/locked_ensemble_config.json")
ALLOWED_GAMMAS = [0.0, 0.10, 0.25, 0.50]

# Candidate configurations
CONFIG_BASELINE = "Candidate_A_Frozen_Baseline"
CONFIG_DIAGNOSTIC_SHRINKAGE = "Candidate_Shrinkage_Blend"
CONFIG_CORRECTED_BLEND = "Candidate_Corrected_Water_Blend"
EXACT_THREE_CANDIDATES = [CONFIG_BASELINE, CONFIG_DIAGNOSTIC_SHRINKAGE, CONFIG_CORRECTED_BLEND]


def get_valid_inner_asof_windows(eval_origin: pd.Timestamp) -> List[Tuple[str, int]]:
    """
    Returns only completed historical hindcast windows ending on or before eval_origin.
    Windows must start strictly before eval_origin and end strictly <= eval_origin.
    """
    all_candidate_windows = [
        ("2000-01-01", 4),
        ("2000-01-01", 8),
        ("2000-01-01", 12),
        ("2000-01-01", 16),
        ("2000-01-01", 20),
        ("2001-01-01", 4),
        ("2001-01-01", 8),
        ("2001-01-01", 12),
        ("2001-01-01", 16),
        ("2002-01-01", 4),
        ("2002-01-01", 8),
        ("2002-01-01", 12),
        ("2003-01-01", 4),
        ("2003-01-01", 8),
        ("2004-01-01", 4),
    ]
    valid = []
    for orig_str, h_q in all_candidate_windows:
        orig = pd.Timestamp(orig_str)
        if orig >= eval_origin:
            continue
        end_date = orig + pd.DateOffset(months=3 * h_q)
        if end_date <= eval_origin:
            valid.append((orig_str, h_q))
    return valid


def fit_models_protocol_a(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    train_cases: List[int],
    origin: pd.Timestamp,
    horizon_quarters: int = 20,
) -> Dict[str, Any]:
    """Fits base models strictly under Protocol A (truncated at origin)."""
    tr_dict = build_asof_case_dict(case_dict, origin)
    m_et = RecursiveTransitionForecaster(model_type="extratrees").fit(
        extract_transition_training_data(tr_dict, unc_df, train_cases, origin)
    )
    m_exp = LearnedDCARateForecaster(decline_type="exponential").fit(
        tr_dict, unc_df, train_cases, origin, horizon_quarters
    )
    m_hyp = LearnedDCARateForecaster(decline_type="hyperbolic").fit(
        tr_dict, unc_df, train_cases, origin, horizon_quarters
    )
    m_hyb = HybridDynamicForecaster(switch_quarters=12).fit(
        tr_dict, unc_df, train_cases, origin, horizon_quarters
    )
    return {
        "Recursive_ExtraTrees": m_et,
        "Learned_Exponential_DCA": m_exp,
        "Learned_Hyperbolic_DCA": m_hyp,
        "Hybrid_Dynamic": m_hyb,
        "train_cases": list(train_cases),
        "origin": origin,
    }


def precompute_isolated_inner_hindcasts(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    outer_held_out: int,
    eval_origin: pd.Timestamp,
    locked_weights: Dict[str, Dict[str, float]],
    inner_sample_cases: Optional[List[int]] = None,
    fit_tracker: Optional[Callable[[str, int, int, List[int]], None]] = None,
    inner_models_cache: Optional[Dict[Tuple[int, str], Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Generates inner hindcast database with double reservoir separation:
    - Excludes outer_held_out from all training and calibration.
    - For each inner target reservoir c_in, excludes c_in from its own inner model training.
    - Uses only completed windows ending <= eval_origin.
    - Truncates inner training data strictly at inner forecast origin (Protocol A).
    """
    if inner_sample_cases is None:
        inner_sample_cases = [c for c in [5, 15, 25, 35, 45, 55, 65] if c != outer_held_out]
    else:
        inner_sample_cases = [c for c in inner_sample_cases if c != outer_held_out]
        
    valid_windows = get_valid_inner_asof_windows(eval_origin)
    if len(valid_windows) == 0:
        return []
        
    unique_inner_origins = sorted(list(set(w[0] for w in valid_windows)))
    w_oil = locked_weights["oil"]
    w_wat = locked_weights["water"]
    
    hindcast_records = []
    
    # Fit inner models: for each inner target case c_in, train on TRAIN_CASES excluding c_outer and c_in
    for c_in in inner_sample_cases:
        df_c = case_dict[c_in]
        params = create_parameter_series(unc_df.loc[c_in, PARAM_NAMES].to_dict())
        train_pool_inner = [c for c in TRAIN_CASES if c != outer_held_out and c != c_in]
        
        # Verify double exclusion
        assert outer_held_out not in train_pool_inner, f"Leakage: outer {outer_held_out} in inner train pool!"
        assert c_in not in train_pool_inner, f"Leakage: inner target {c_in} in inner train pool!"
        
        # Fit models for this c_in across required inner origins (with optional fold-level caching)
        models_by_orig = {}
        for in_orig_str in unique_inner_origins:
            cache_key = (c_in, in_orig_str)
            if inner_models_cache is not None and cache_key in inner_models_cache:
                models_by_orig[in_orig_str] = inner_models_cache[cache_key]
            else:
                in_orig = pd.Timestamp(in_orig_str)
                if fit_tracker is not None:
                    fit_tracker(in_orig_str, outer_held_out, c_in, train_pool_inner)
                    
                m_fit = fit_models_protocol_a(
                    case_dict=case_dict,
                    unc_df=unc_df,
                    train_cases=train_pool_inner,
                    origin=in_orig,
                    horizon_quarters=20,
                )
                models_by_orig[in_orig_str] = m_fit
                if inner_models_cache is not None:
                    inner_models_cache[cache_key] = m_fit
            
        # Evaluate valid windows for c_in
        for orig_str, h_q in valid_windows:
            orig = pd.Timestamp(orig_str)
            hist_obs = df_c[df_c["date"] == orig]
            f_dates = pd.date_range(orig + pd.DateOffset(months=3), periods=h_q, freq="QS")
            true_fwd = df_c[df_c["date"].isin(f_dates)].sort_values("date")
            
            if len(hist_obs) == 0 or len(true_fwd) < h_q:
                continue
                
            obs = hist_obs.iloc[0]
            qo_0 = float(obs["oil_rate_stbd"])
            qw_0 = float(obs["water_rate_stbd"])
            fw_0 = float(qw_0 / max(qo_0 + qw_0, 1.0))
            t_days = (orig - df_c["date"].min()).days
            res_age_0 = float(t_days / YEAR_DAYS)
            
            init_state = {
                "qo": qo_0,
                "qw": qw_0,
                "qg": float(obs["gas_rate_mscfd"]),
                "Qo": float(obs["oil_cum_stb"]),
                "Qw": float(obs["water_cum_stb"]),
                "Qg": float(obs["gas_cum_mscf"]),
                "wor": float(qw_0 / max(qo_0, 1.0)),
                "res_age": res_age_0,
                "origin_date": orig,
            }
            
            models = models_by_orig[orig_str]
            r_et = models["Recursive_ExtraTrees"].predict_rates(init_state, params, f_dates)
            r_exp = models["Learned_Exponential_DCA"].predict_rates(init_state, params, f_dates)
            r_hyp = models["Learned_Hyperbolic_DCA"].predict_rates(init_state, params, f_dates)
            r_hyb = models["Hybrid_Dynamic"].predict_rates(init_state, params, f_dates)
            
            qo_frozen = (
                w_oil["Recursive_ExtraTrees"] * r_et["oil_rate"]
                + w_oil["Learned_Exponential_DCA"] * r_exp["oil_rate"]
                + w_oil["Learned_Hyperbolic_DCA"] * r_hyp["oil_rate"]
                + w_oil["Hybrid_Dynamic"] * r_hyb["oil_rate"]
            )
            qw_frozen = (
                w_wat["Recursive_ExtraTrees"] * r_et["water_rate"]
                + w_wat["Learned_Exponential_DCA"] * r_exp["water_rate"]
                + w_wat["Learned_Hyperbolic_DCA"] * r_hyp["water_rate"]
                + w_wat["Hybrid_Dynamic"] * r_hyb["water_rate"]
            )
            qw_et = r_et["water_rate"]
            
            hindcast_records.append({
                "case_id": c_in,
                "origin": orig_str,
                "horizon_quarters": h_q,
                "init_state": init_state,
                "fw_0": fw_0,
                "res_age": res_age_0,
                "qo_frozen": qo_frozen,
                "qw_frozen": qw_frozen,
                "qw_et": qw_et,
                "true_cum_oil": true_fwd["oil_cum_stb"].values,
                "true_cum_wat": true_fwd["water_cum_stb"].values,
                "true_rate_oil": true_fwd["oil_rate_stbd"].values,
                "true_rate_wat": true_fwd["water_rate_stbd"].values,
            })
            
    return hindcast_records


def calibrate_corrected_water_gamma(
    hindcast_records: List[Dict[str, Any]],
    allowed_gammas: List[float] = ALLOWED_GAMMAS,
    ramp_low: float = 0.30,
    ramp_high: float = 0.60,
) -> Dict[str, Any]:
    """
    Optimizes the actual forecasting loss across candidate gammas:
    - For each gamma, blends water rates: w = gamma * clip((fw_0 - 0.30)/0.30, 0, 1).
    - Integrates cumulative trajectory using production convention.
    - Scores cumulative trajectory with authoritative cumulative-increment scorer.
    - Selects gamma using equally weighted reservoir-level mean losses.
    - Defaults to gamma=0.0 when calibration evidence is insufficient.
    """
    if len(hindcast_records) < 5:
        return {
            "selected_gamma": 0.0,
            "mean_losses_by_gamma": {g: float("nan") for g in allowed_gammas},
            "status": "DEFAULT_INSUFFICIENT_EVIDENCE",
            "n_hindcasts": len(hindcast_records),
            "n_reservoirs": 0,
        }
        
    df_evals = []
    ramp_span = ramp_high - ramp_low
    
    for rec in hindcast_records:
        fw_0 = rec["fw_0"]
        c_id = rec["case_id"]
        qo_frozen = rec["qo_frozen"]
        qw_frozen = rec["qw_frozen"]
        qw_et = rec["qw_et"]
        init_state = rec["init_state"]
        true_co = rec["true_cum_oil"]
        true_cw = rec["true_cum_wat"]
        true_ro = rec["true_rate_oil"]
        true_rw = rec["true_rate_wat"]
        
        cum_o_base = np.maximum.accumulate(init_state["Qo"] + np.cumsum(np.maximum(qo_frozen, 0.0) * DT_DAYS))
        
        # Base weight fraction from ramp
        w_raw = np.clip((fw_0 - ramp_low) / ramp_span, 0.0, 1.0)
        
        row_losses = {"case_id": c_id, "origin": rec["origin"], "horizon_quarters": rec["horizon_quarters"]}
        for g in allowed_gammas:
            w_blend = g * w_raw
            qw_blend = np.maximum((1.0 - w_blend) * qw_frozen + w_blend * qw_et, 0.0)
            cum_w_blend = np.maximum.accumulate(init_state["Qw"] + np.cumsum(qw_blend * DT_DAYS))
            
            sc = score_cumulative_forecast(
                pred_cum_oil=cum_o_base,
                pred_cum_wat=cum_w_blend,
                true_cum_oil=true_co,
                true_cum_wat=true_cw,
                origin_cum_oil=init_state["Qo"],
                origin_cum_wat=init_state["Qw"],
                pred_rate_oil=qo_frozen,
                pred_rate_wat=qw_blend,
                true_rate_oil=true_ro,
                true_rate_wat=true_rw,
            )
            row_losses[g] = sc["macro_nrmse"]
            
        df_evals.append(row_losses)
        
    df_res = pd.DataFrame(df_evals)
    unique_cases = df_res["case_id"].unique()
    
    if len(unique_cases) < 3:
        return {
            "selected_gamma": 0.0,
            "mean_losses_by_gamma": {g: float("nan") for g in allowed_gammas},
            "status": "DEFAULT_TOO_FEW_RESERVOIRS",
            "n_hindcasts": len(df_evals),
            "n_reservoirs": len(unique_cases),
        }
        
    # Equal weighting across reservoirs: mean per reservoir, then mean across reservoirs
    res_means = df_res.groupby("case_id")[allowed_gammas].mean()
    agg_losses = res_means.mean(axis=0)
    
    best_gamma = 0.0
    min_loss = agg_losses[0.0]
    
    # Check if any non-zero gamma strictly beats gamma=0.0
    for g in allowed_gammas:
        if agg_losses[g] < min_loss - 1e-6:
            min_loss = agg_losses[g]
            best_gamma = g
            
    return {
        "selected_gamma": float(best_gamma),
        "mean_losses_by_gamma": {g: float(agg_losses[g]) for g in allowed_gammas},
        "status": "OPTIMIZED_LOSS",
        "n_hindcasts": len(df_evals),
        "n_reservoirs": len(unique_cases),
    }


def calibrate_unchanged_shrinkage_blend(hindcast_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Reproduces the existing Candidate_Shrinkage_Blend gate calibration unchanged:
    Minimizes linear surrogate: sum [ (1 - w_s)*loss_base + w_s*loss_et ]
    with gamma in [0.25, 0.40, 0.50, 0.65, 0.80] and ramp [0.30, 0.60].
    """
    if len(hindcast_records) < 5:
        return {"shrink_low": 0.30, "shrink_high": 0.60, "shrink_gamma": 0.50}
        
    records = []
    for rec in hindcast_records:
        qo_frozen = rec["qo_frozen"]
        qw_frozen = rec["qw_frozen"]
        qw_et = rec["qw_et"]
        init_state = rec["init_state"]
        true_co = rec["true_cum_oil"]
        true_cw = rec["true_cum_wat"]
        true_ro = rec["true_rate_oil"]
        true_rw = rec["true_rate_wat"]
        
        cum_o_base = np.maximum.accumulate(init_state["Qo"] + np.cumsum(np.maximum(qo_frozen, 0.0) * DT_DAYS))
        cum_w_base = np.maximum.accumulate(init_state["Qw"] + np.cumsum(np.maximum(qw_frozen, 0.0) * DT_DAYS))
        cum_w_et = np.maximum.accumulate(init_state["Qw"] + np.cumsum(np.maximum(qw_et, 0.0) * DT_DAYS))
        
        sc_base = score_cumulative_forecast(
            cum_o_base, cum_w_base, true_co, true_cw,
            init_state["Qo"], init_state["Qw"], qo_frozen, qw_frozen, true_ro, true_rw
        )
        sc_et = score_cumulative_forecast(
            cum_o_base, cum_w_et, true_co, true_cw,
            init_state["Qo"], init_state["Qw"], qo_frozen, qw_et, true_ro, true_rw
        )
        records.append({
            "fw_0": rec["fw_0"],
            "loss_base": sc_base["macro_nrmse"],
            "loss_et": sc_et["macro_nrmse"],
        })
        
    df_l = pd.DataFrame(records)
    fw_0 = df_l["fw_0"].values
    loss_base = df_l["loss_base"].values
    loss_et = df_l["loss_et"].values
    
    s_low = 0.30
    s_high = 0.60
    best_gamma = 0.50
    min_loss = float("inf")
    for gamma in [0.25, 0.40, 0.50, 0.65, 0.80]:
        w_s = gamma * np.clip((fw_0 - s_low) / max(s_high - s_low, 1e-6), 0.0, 1.0)
        total_loss = np.sum((1.0 - w_s) * loss_base + w_s * loss_et)
        if total_loss < min_loss:
            min_loss = total_loss
            best_gamma = gamma
            
    return {"shrink_low": float(s_low), "shrink_high": float(s_high), "shrink_gamma": float(best_gamma)}


def predict_exact_candidate_forecast(
    candidate_id: str,
    qo_frozen: np.ndarray,
    qw_frozen: np.ndarray,
    qw_et: np.ndarray,
    init_state: Dict[str, Any],
    calib_params: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """
    Generates predicted rate and cumulative trajectories for one of the exact 3 configurations.
    Enforces non-negativity and monotonic cumulative accumulation.
    """
    fw_0 = float(init_state["wor"] / (1.0 + init_state["wor"]))
    
    # Oil is frozen baseline across all 3 configurations
    qo = np.maximum(qo_frozen, 0.0)
    
    if candidate_id == CONFIG_BASELINE:
        qw = np.maximum(qw_frozen, 0.0)
        
    elif candidate_id == CONFIG_DIAGNOSTIC_SHRINKAGE:
        s_low = calib_params.get("shrink_low", 0.30)
        s_high = calib_params.get("shrink_high", 0.60)
        gamma = calib_params.get("shrink_gamma", 0.50)
        w_raw = (fw_0 - s_low) / max(s_high - s_low, 1e-6)
        w_shrunk = gamma * np.clip(w_raw, 0.0, 1.0)
        qw = np.maximum((1.0 - w_shrunk) * qw_frozen + w_shrunk * qw_et, 0.0)
        
    elif candidate_id == CONFIG_CORRECTED_BLEND:
        gamma_star = calib_params.get("selected_gamma", 0.0)
        w_raw = np.clip((fw_0 - 0.30) / 0.30, 0.0, 1.0)
        w_corr = gamma_star * w_raw
        qw = np.maximum((1.0 - w_corr) * qw_frozen + w_corr * qw_et, 0.0)
        
    else:
        raise ValueError(f"Unauthorized candidate_id: {candidate_id}. Design strictly limited to {EXACT_THREE_CANDIDATES}.")
        
    qg = RS_SOLUTION * qo
    cum_o = np.maximum.accumulate(init_state["Qo"] + np.cumsum(qo * DT_DAYS))
    cum_w = np.maximum.accumulate(init_state["Qw"] + np.cumsum(qw * DT_DAYS))
    cum_g = init_state["Qg"] + RS_SOLUTION * (cum_o - init_state["Qo"])
    
    return {
        "oil_rate": qo,
        "water_rate": qw,
        "gas_rate": qg,
        "cum_oil": cum_o,
        "cum_water": cum_w,
        "cum_gas": cum_g,
        "fw_0": fw_0,
    }
