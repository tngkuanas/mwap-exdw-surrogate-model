"""
Module: authoritative_scorer.py
Package: geological_information_value
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Authoritative, referee-consistent scoring engine for reservoir production forecasts.
Enforces the exact competition referee protocol:
1. Reconstructs forecast cumulative production using production-compatible integration:
   Q_pred(t_k) = Q_0 + sum_{j=1}^k max(q_j, 0) * dt_days
   with numerical monotonicity accumulation.
2. Computes cumulative prediction errors:
   e(t) = Q_pred(t) - Q_obs(t)
3. Computes true forward cumulative increments relative to forecast origin:
   Delta Q_obs(t) = Q_obs(t) - Q_0
4. Evaluates referee-standard increment NRMSE:
   NRMSE = RMS(e) / RMS(Delta Q_obs)
5. Evaluates normalized bias and terminal errors.
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from ensemble.metrics import compute_increment_nrmse, compute_bias

REFEREE_DT_DAYS_DEFAULT = 91.3125


def score_cumulative_forecast(
    pred_cum_oil: np.ndarray,
    pred_cum_wat: np.ndarray,
    true_cum_oil: np.ndarray,
    true_cum_wat: np.ndarray,
    origin_cum_oil: float,
    origin_cum_wat: float,
    pred_rate_oil: Optional[np.ndarray] = None,
    pred_rate_wat: Optional[np.ndarray] = None,
    true_rate_oil: Optional[np.ndarray] = None,
    true_rate_wat: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Computes referee-standard increment NRMSE and diagnostic metrics given cumulative series.
    
    Args:
        pred_cum_oil: (N,) predicted cumulative oil production in STB.
        pred_cum_wat: (N,) predicted cumulative water production in STB.
        true_cum_oil: (N,) true observed cumulative oil production in STB.
        true_cum_wat: (N,) true observed cumulative water production in STB.
        origin_cum_oil: scalar cumulative oil at forecast origin (t=0) in STB.
        origin_cum_wat: scalar cumulative water at forecast origin (t=0) in STB.
        pred_rate_oil: optional (N,) predicted oil rates in STB/d.
        pred_rate_wat: optional (N,) predicted water rates in STB/d.
        true_rate_oil: optional (N,) observed oil rates in STB/d.
        true_rate_wat: optional (N,) observed water rates in STB/d.
        
    Returns:
        Dict with keys:
            macro_nrmse, oil_nrmse, water_nrmse,
            oil_bias, water_bias,
            rmse_oil_stb, rmse_wat_stb,
            norm_inc_oil_stb, norm_inc_wat_stb,
            terminal_cum_err_oil_stb, terminal_cum_err_wat_stb,
            terminal_rate_err_oil_stbd, terminal_rate_err_wat_stbd,
            referee_convention_matched (bool)
    """
    pred_co = np.asarray(pred_cum_oil, dtype=float)
    pred_cw = np.asarray(pred_cum_wat, dtype=float)
    true_co = np.asarray(true_cum_oil, dtype=float)
    true_cw = np.asarray(true_cum_wat, dtype=float)
    
    assert len(pred_co) == len(true_co), f"Mismatched oil lengths: {len(pred_co)} vs {len(true_co)}"
    assert len(pred_cw) == len(true_cw), f"Mismatched water lengths: {len(pred_cw)} vs {len(true_cw)}"
    assert len(pred_co) > 0, "Empty forecast trajectory"
    
    # 1. Prediction errors
    err_oil = pred_co - true_co
    err_wat = pred_cw - true_cw
    
    # 2. Observed forward increments relative to forecast origin
    true_inc_oil = true_co - float(origin_cum_oil)
    true_inc_wat = true_cw - float(origin_cum_wat)
    
    # Safeguard: increments must be non-negative
    true_inc_oil = np.maximum(true_inc_oil, 0.0)
    true_inc_wat = np.maximum(true_inc_wat, 0.0)
    
    # 3. Increment NRMSE
    no = compute_increment_nrmse(err_oil, true_inc_oil)
    nw = compute_increment_nrmse(err_wat, true_inc_wat)
    macro = 0.5 * (no + nw)
    
    # 4. Normalized Bias
    bo = compute_bias(err_oil, true_inc_oil)
    bw = compute_bias(err_wat, true_inc_wat)
    
    # 5. Component RMS and Norms
    rmse_o = float(np.sqrt(np.mean(err_oil ** 2)))
    rmse_w = float(np.sqrt(np.mean(err_wat ** 2)))
    norm_inc_o = float(np.sqrt(np.mean(true_inc_oil ** 2)))
    norm_inc_w = float(np.sqrt(np.mean(true_inc_wat ** 2)))
    
    # 6. Terminal errors
    term_cum_err_o = float(err_oil[-1])
    term_cum_err_w = float(err_wat[-1])
    
    term_rate_err_o = (
        float(pred_rate_oil[-1] - true_rate_oil[-1])
        if (pred_rate_oil is not None and true_rate_oil is not None)
        else np.nan
    )
    term_rate_err_w = (
        float(pred_rate_wat[-1] - true_rate_wat[-1])
        if (pred_rate_wat is not None and true_rate_wat is not None)
        else np.nan
    )
    
    return {
        "macro_nrmse": float(macro),
        "oil_nrmse": float(no),
        "water_nrmse": float(nw),
        "oil_bias": float(bo),
        "water_bias": float(bw),
        "rmse_oil_stb": rmse_o,
        "rmse_wat_stb": rmse_w,
        "norm_inc_oil_stb": norm_inc_o,
        "norm_inc_wat_stb": norm_inc_w,
        "terminal_cum_err_oil_stb": term_cum_err_o,
        "terminal_cum_err_wat_stb": term_cum_err_w,
        "terminal_rate_err_oil_stbd": term_rate_err_o,
        "terminal_rate_err_wat_stbd": term_rate_err_w,
        "referee_convention_matched": True,
    }


def integrate_and_score_rates(
    init_state: Dict[str, Any],
    pred_rate_oil: np.ndarray,
    pred_rate_wat: np.ndarray,
    post_obs_df: pd.DataFrame,
    dt_days: float = REFEREE_DT_DAYS_DEFAULT,
) -> Dict[str, Any]:
    """
    Integrates forecast rates into cumulatives following the locked production
    convention and evaluates against observed dataframe.
    """
    qo = np.maximum(np.asarray(pred_rate_oil, dtype=float), 0.0)
    qw = np.maximum(np.asarray(pred_rate_wat, dtype=float), 0.0)
    
    q0_oil = float(init_state["Qo"])
    q0_wat = float(init_state["Qw"])
    
    # Production-faithful integration
    cum_o = q0_oil + np.cumsum(qo * dt_days)
    cum_w = q0_wat + np.cumsum(qw * dt_days)
    cum_o = np.maximum.accumulate(cum_o)
    cum_w = np.maximum.accumulate(cum_w)
    
    # Extract truth columns
    col_o = "oil_cum_stb" if "oil_cum_stb" in post_obs_df.columns else "oil_cum"
    col_w = "water_cum_stb" if "water_cum_stb" in post_obs_df.columns else "water_cum"
    true_co = post_obs_df[col_o].values[:len(cum_o)]
    true_cw = post_obs_df[col_w].values[:len(cum_w)]
    
    true_ro = (
        post_obs_df["oil_rate_stbd"].values[:len(qo)]
        if "oil_rate_stbd" in post_obs_df.columns
        else (post_obs_df["oil_rate"].values[:len(qo)] if "oil_rate" in post_obs_df.columns else None)
    )
    true_rw = (
        post_obs_df["water_rate_stbd"].values[:len(qw)]
        if "water_rate_stbd" in post_obs_df.columns
        else (post_obs_df["water_rate"].values[:len(qw)] if "water_rate" in post_obs_df.columns else None)
    )
    
    res = score_cumulative_forecast(
        pred_cum_oil=cum_o,
        pred_cum_wat=cum_w,
        true_cum_oil=true_co,
        true_cum_wat=true_cw,
        origin_cum_oil=q0_oil,
        origin_cum_wat=q0_wat,
        pred_rate_oil=qo,
        pred_rate_wat=qw,
        true_rate_oil=true_ro,
        true_rate_wat=true_rw,
    )
    res["pred_terminal_oil_rate"] = float(qo[-1])
    res["pred_terminal_water_rate"] = float(qw[-1])
    res["pred_terminal_oil_cum"] = float(cum_o[-1])
    res["pred_terminal_water_cum"] = float(cum_w[-1])
    return res
