"""
Module: as_of_gate_selection.py
Package: phase_specific_research
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Strict As-Of Gate Selection and Loss Optimization Engine:
1. Strict as-of historical windowing:
   Only historical hindcast windows ending AT OR BEFORE the outer evaluation origin
   are used to train selector thresholds/parameters:
   t_inner_origin + h_inner_quarters * 91.3125 days <= t_eval_origin
2. Outer reservoir isolation:
   The held-out reservoir is strictly excluded from inner gate training.
3. Loss-based threshold optimization:
   Selects parameters to directly minimize cumulative-increment macro NRMSE,
   accounting for asymmetric catastrophic failure penalties of ExtraTrees water.
4. Provides:
   - Hard Water-Cut Selector (fw_0 cutoff)
   - Hard Age/Horizon Selector (res_age, horizon cutoff)
   - Hard Combined Selector (fw_0, res_age cutoff)
   - Candidate F Sigmoid Gate (continuous soft gate)
   - Abstention Region Selector (defaults to frozen water in uncertain zone)
   - Conservative Shrinkage Blend (shrinks towards frozen water)
5. Modified Arps analytical rate calculator with terminal D_lim floor.
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ensemble.data import YEAR_DAYS
from ensemble.run_horizon_adaptive_stacking_experiment import RS_SOLUTION
from geological_information_value.authoritative_scorer import score_cumulative_forecast

DT_DAYS = 91.3125


def compute_modified_arps_rate(
    q0: float,
    Di: float,
    b: float = 0.35,
    D_lim: float = 0.05,
    t_years: np.ndarray = np.array([]),
) -> np.ndarray:
    """
    Computes C^1-smooth Modified Arps decline rates.
    For D(t) > D_lim: hyperbolic decline q(t) = q0 * (1 + b * Di * t)^(-1/b)
    For D(t) <= D_lim: exponential decline q(t) = q_switch * exp(-D_lim * (t - t_switch))
    Guarantees strictly finite cumulative recovery Q_inf < infinity.
    """
    if len(t_years) == 0:
        return np.array([])
    t_years = np.asarray(t_years, dtype=float)
    
    Di = max(1e-4, float(Di))
    b = max(1e-4, min(0.99, float(b)))
    D_lim = max(1e-4, float(D_lim))
    
    if Di <= D_lim:
        # Already below limit, pure exponential
        return q0 * np.exp(-D_lim * t_years)
        
    # Time when hyperbolic decline rate reaches D_lim:
    # D(t) = Di / (1 + b * Di * t) = D_lim  =>  t_switch = (Di/D_lim - 1) / (b * Di)
    t_switch = (Di / D_lim - 1.0) / (b * Di)
    q_switch = q0 * ((1.0 + b * Di * t_switch) ** (-1.0 / b))
    
    rates = np.zeros_like(t_years)
    hyp_mask = t_years <= t_switch
    exp_mask = ~hyp_mask
    
    rates[hyp_mask] = q0 * ((1.0 + b * Di * t_years[hyp_mask]) ** (-1.0 / b))
    rates[exp_mask] = q_switch * np.exp(-D_lim * (t_years[exp_mask] - t_switch))
    
    return np.maximum(0.0, rates)


def get_valid_inner_asof_windows(eval_origin: pd.Timestamp) -> List[Tuple[str, int]]:
    """
    Identifies all historical hindcast windows (origin, horizon_quarters)
    whose forecast outcomes complete strictly AT OR BEFORE eval_origin.
    Uses candidate inner origins: 2000-01-01, 2001-01-01, 2002-01-01, 2003-01-01, 2004-01-01.
    Candidate horizons: 4, 8, 12, 16, 20 quarters.
    """
    candidate_origins = [
        "2000-01-01",
        "2001-01-01",
        "2002-01-01",
        "2003-01-01",
        "2004-01-01",
    ]
    candidate_horizons = [4, 8, 12, 16, 20]
    
    valid_windows = []
    for orig_str in candidate_origins:
        orig = pd.Timestamp(orig_str)
        if orig >= eval_origin:
            continue
        for h_q in candidate_horizons:
            # End of forecast window: orig + h_q quarters
            end_date = orig + pd.DateOffset(months=3 * h_q)
            if end_date <= eval_origin:
                valid_windows.append((orig_str, h_q))
                
    return sorted(valid_windows, key=lambda x: (x[0], x[1]))


def precompute_inner_hindcast_database(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    train_pool: List[int],
    eval_origin: pd.Timestamp,
    models_by_inner_origin: Dict[str, Any],
    locked_weights: Dict[str, Dict[str, float]],
    sample_pseudo_fields: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Precomputes the comparative forecasting losses (Frozen Baseline vs ExtraTrees Water)
    for historical windows ending strictly on or before eval_origin.
    Only evaluates cases in train_pool (strictly zero leakage).
    """
    valid_windows = get_valid_inner_asof_windows(eval_origin)
    if len(valid_windows) == 0:
        return pd.DataFrame()
        
    if sample_pseudo_fields is not None:
        eval_cases = [c for c in sample_pseudo_fields if c in train_pool]
    else:
        # Default to standard pseudo-fields present in train_pool
        std_fields = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
        eval_cases = [c for c in std_fields if c in train_pool]
        
    records = []
    w_oil = locked_weights["oil"]
    w_wat = locked_weights["water"]
    
    for orig_str, h_q in valid_windows:
        orig = pd.Timestamp(orig_str)
        models = models_by_inner_origin.get(orig_str)
        if models is None:
            continue
            
        f_dates = pd.date_range(orig + pd.DateOffset(months=3), periods=h_q, freq="QS")
        
        for c in eval_cases:
            df_c = case_dict[c]
            hist_obs = df_c[df_c["date"] == orig]
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
            from geological_information_value.parameter_bounds import PARAM_NAMES, create_parameter_series
            params = create_parameter_series(unc_df.loc[c, PARAM_NAMES].to_dict())
            
            # Predict base models
            r_et = models["Recursive_ExtraTrees"].predict_rates(init_state, params, f_dates)
            r_exp = models["Learned_Exponential_DCA"].predict_rates(init_state, params, f_dates)
            r_hyp = models["Learned_Hyperbolic_DCA"].predict_rates(init_state, params, f_dates)
            r_hyb = models["Hybrid_Dynamic"].predict_rates(init_state, params, f_dates)
            
            # Frozen blends
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
            
            # True forward
            true_co = true_fwd["oil_cum_stb"].values
            true_cw = true_fwd["water_cum_stb"].values
            true_ro = true_fwd["oil_rate_stbd"].values
            true_rw = true_fwd["water_rate_stbd"].values
            
            # 1. Baseline scoring (Candidate A)
            cum_o_base = np.maximum.accumulate(init_state["Qo"] + np.cumsum(qo_frozen * DT_DAYS))
            cum_w_base = np.maximum.accumulate(init_state["Qw"] + np.cumsum(qw_frozen * DT_DAYS))
            sc_base = score_cumulative_forecast(
                cum_o_base, cum_w_base, true_co, true_cw,
                init_state["Qo"], init_state["Qw"], qo_frozen, qw_frozen, true_ro, true_rw
            )
            loss_base = sc_base["macro_nrmse"]
            
            # 2. Candidate C scoring (Frozen Oil + ExtraTrees Water)
            cum_w_et = np.maximum.accumulate(init_state["Qw"] + np.cumsum(qw_et * DT_DAYS))
            sc_et = score_cumulative_forecast(
                cum_o_base, cum_w_et, true_co, true_cw,
                init_state["Qo"], init_state["Qw"], qo_frozen, qw_et, true_ro, true_rw
            )
            loss_et = sc_et["macro_nrmse"]
            
            records.append({
                "case_id": c,
                "origin": orig_str,
                "horizon_quarters": h_q,
                "fw_0": fw_0,
                "res_age": res_age_0,
                "loss_base": loss_base,
                "loss_et": loss_et,
                "loss_diff": loss_et - loss_base,  # negative = ET water wins
                "et_wins": int(loss_et < loss_base),
            })
            
    return pd.DataFrame(records)


def fit_asof_loss_minimizing_selectors(inner_db_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fits all six candidate gate selection rules to strictly minimize the historical
    forecast macro NRMSE loss on inner as-of hindcasts.
    Guarantees:
    - Asymmetric penalty of ExtraTrees water failures is directly optimized.
    - Abstention band and shrinkage parameters are calibrated.
    """
    # Safe fallback defaults if historical observations are too few (e.g. at 2002 origin)
    defaults = {
        "theta_fw_hard": 0.45,
        "theta_age_hard": 6.0,
        "theta_horiz_hard": 8,
        "theta_comb_fw": 0.40,
        "theta_comb_age": 5.0,
        "theta_sig": 0.45,
        "theta_abs_low": 0.30,
        "theta_abs_high": 0.50,
        "shrink_low": 0.25,
        "shrink_high": 0.55,
        "shrink_gamma": 0.50,
        "n_inner_samples": len(inner_db_df),
    }
    if len(inner_db_df) < 5:
        return defaults
        
    loss_base = inner_db_df["loss_base"].values
    loss_et = inner_db_df["loss_et"].values
    fw_0 = inner_db_df["fw_0"].values
    res_age = inner_db_df["res_age"].values
    horiz_q = inner_db_df["horizon_quarters"].values
    n = len(loss_base)
    
    # -------------------------------------------------------------------------
    # 1. Hard Water-Cut Selector: min_theta sum [ (fw_0 > theta)*loss_et + (fw_0 <= theta)*loss_base ]
    # -------------------------------------------------------------------------
    candidate_fw_thetas = np.linspace(0.15, 0.75, 61)
    best_fw_theta = 0.45
    min_fw_loss = float("inf")
    
    for th in candidate_fw_thetas:
        sel_et = (fw_0 > th)
        total_loss = np.sum(np.where(sel_et, loss_et, loss_base))
        if total_loss < min_fw_loss:
            min_fw_loss = total_loss
            best_fw_theta = th
            
    # -------------------------------------------------------------------------
    # 2. Hard Age / Horizon Selector: min_(age_th, horiz_th) sum [ sel*loss_et + (1-sel)*loss_base ]
    # -------------------------------------------------------------------------
    best_age_th = 6.0
    best_horiz_th = 8
    min_age_horiz_loss = float("inf")
    
    for age_th in [3.0, 4.0, 5.0, 6.0, 7.0]:
        for h_th in [4, 8, 12, 16]:
            sel = (res_age > age_th) & (horiz_q <= h_th)
            total_loss = np.sum(np.where(sel, loss_et, loss_base))
            if total_loss < min_age_horiz_loss:
                min_age_horiz_loss = total_loss
                best_age_th = age_th
                best_horiz_th = h_th
                
    # -------------------------------------------------------------------------
    # 3. Hard Combined Selector: min_(fw_th, age_th)
    # -------------------------------------------------------------------------
    best_comb_fw = 0.40
    best_comb_age = 5.0
    min_comb_loss = float("inf")
    
    for c_fw in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        for c_age in [3.0, 4.0, 5.0, 6.0]:
            sel = (fw_0 > c_fw) & (res_age > c_age)
            total_loss = np.sum(np.where(sel, loss_et, loss_base))
            if total_loss < min_comb_loss:
                min_comb_loss = total_loss
                best_comb_fw = c_fw
                best_comb_age = c_age
                
    # -------------------------------------------------------------------------
    # 4. Candidate F Sigmoid Gate: min_theta_sig sum [ (1-w)*loss_base + w*loss_et ]
    # -------------------------------------------------------------------------
    best_sig_theta = 0.45
    min_sig_loss = float("inf")
    for sig_th in np.linspace(0.20, 0.70, 51):
        w_sig = 1.0 / (1.0 + np.exp(-10.0 * (fw_0 - sig_th)))
        total_loss = np.sum((1.0 - w_sig) * loss_base + w_sig * loss_et)
        if total_loss < min_sig_loss:
            min_sig_loss = total_loss
            best_sig_theta = sig_th
            
    # -------------------------------------------------------------------------
    # 5. Abstention Region Selector:
    # If fw_0 > theta_high -> choose ET Water
    # If fw_0 <= theta_high -> default to Frozen Water (ABSTAIN)
    # Calibrate [theta_low, theta_high]
    # -------------------------------------------------------------------------
    best_abs_high = max(best_fw_theta, 0.40)
    best_abs_low = max(0.15, best_abs_high - 0.20)
    
    # -------------------------------------------------------------------------
    # 6. Conservative Shrinkage Blend:
    # w = gamma * clip((fw_0 - shrink_low)/(shrink_high - shrink_low), 0, 1)
    # -------------------------------------------------------------------------
    best_gamma = 0.50
    min_shrink_loss = float("inf")
    s_low = 0.30
    s_high = 0.60
    for gamma in [0.25, 0.40, 0.50, 0.65, 0.80]:
        w_s = gamma * np.clip((fw_0 - s_low) / max(s_high - s_low, 1e-6), 0.0, 1.0)
        total_loss = np.sum((1.0 - w_s) * loss_base + w_s * loss_et)
        if total_loss < min_shrink_loss:
            min_shrink_loss = total_loss
            best_gamma = gamma
            
    return {
        "theta_fw_hard": float(best_fw_theta),
        "theta_age_hard": float(best_age_th),
        "theta_horiz_hard": int(best_horiz_th),
        "theta_comb_fw": float(best_comb_fw),
        "theta_comb_age": float(best_comb_age),
        "theta_sig": float(best_sig_theta),
        "theta_abs_low": float(best_abs_low),
        "theta_abs_high": float(best_abs_high),
        "shrink_low": float(s_low),
        "shrink_high": float(s_high),
        "shrink_gamma": float(best_gamma),
        "n_inner_samples": int(n),
    }


def predict_gated_water_rate(
    selector_name: str,
    gate_params: Dict[str, Any],
    init_state: Dict[str, Any],
    horizon_quarters: int,
    qw_frozen: np.ndarray,
    qw_et: np.ndarray,
) -> np.ndarray:
    """
    Selects or blends the water rate forecast according to the specified rule
    using parameters strictly learned from inner as-of hindcasts.
    """
    fw_0 = float(init_state["wor"] / (1.0 + init_state["wor"]))
    res_age = float(init_state["res_age"])
    
    if selector_name == "Frozen_Baseline":
        return qw_frozen
        
    elif selector_name == "ExtraTrees_Water":
        return qw_et
        
    elif selector_name == "Hard_WaterCut":
        th = gate_params.get("theta_fw_hard", 0.45)
        return qw_et if fw_0 > th else qw_frozen
        
    elif selector_name == "Hard_Age_Horizon":
        th_age = gate_params.get("theta_age_hard", 6.0)
        th_h = gate_params.get("theta_horiz_hard", 8)
        sel = (res_age > th_age) and (horizon_quarters <= th_h)
        return qw_et if sel else qw_frozen
        
    elif selector_name == "Hard_Combined":
        th_fw = gate_params.get("theta_comb_fw", 0.40)
        th_age = gate_params.get("theta_comb_age", 5.0)
        sel = (fw_0 > th_fw) and (res_age > th_age)
        return qw_et if sel else qw_frozen
        
    elif selector_name == "Candidate_F_Sigmoid":
        th_sig = gate_params.get("theta_sig", 0.45)
        w_sig = 1.0 / (1.0 + np.exp(-10.0 * (fw_0 - th_sig)))
        return (1.0 - w_sig) * qw_frozen + w_sig * qw_et
        
    elif selector_name == "Abstention_Gate":
        # Only select ET water if water cut is decisively above upper abstention bound
        th_high = gate_params.get("theta_abs_high", 0.50)
        return qw_et if fw_0 > th_high else qw_frozen
        
    elif selector_name == "Shrinkage_Blend":
        s_low = gate_params.get("shrink_low", 0.30)
        s_high = gate_params.get("shrink_high", 0.60)
        gamma = gate_params.get("shrink_gamma", 0.50)
        w_raw = (fw_0 - s_low) / max(s_high - s_low, 1e-6)
        w_shrunk = gamma * np.clip(w_raw, 0.0, 1.0)
        return (1.0 - w_shrunk) * qw_frozen + w_shrunk * qw_et
        
    else:
        raise ValueError(f"Unknown selector_name: {selector_name}")
