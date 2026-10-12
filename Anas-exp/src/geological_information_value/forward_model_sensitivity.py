"""
Module: forward_model_sensitivity.py
Package: geological_information_value
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Addresses: Section 3 — Quantify Forward-Model Geological Sensitivity
Calculates:
1. Local parameter Jacobians (terminal rate & cumulative) for each model & ensemble.
2. Normalized sensitivities across forecast horizons (4Q, 12Q, 40Q, 80Q).
3. Global sensitivity analysis (Sobol / variance-based main effects) using space-filling sampling.
4. Mathematical inspection of fitted representations (RidgeCV coefficients, ExtraTrees importances).
5. State anchor sensitivity vs. geological sensitivity comparison.
"""
from typing import Dict, List, Tuple, Any, Optional
import time
import numpy as np
import pandas as pd
from scipy.stats import qmc

from ensemble.data import load_uncertainty, resolve_shared_dir, TRAIN_CASES, PARAMS, YEAR_DAYS
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
    STATE_FEATURES,
    RS_SOLUTION,
)
from ensemble.generate_competition_deliverable import (
    load_locked_config,
    generate_scenario_predictions,
)
from forecast_advancement.protocol_framework import fit_base_models_under_protocol
from geological_information_value.parameter_bounds import (
    PARAM_NAMES,
    SHORT_NAMES,
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    RANGES,
    PRIOR_MEANS,
    denormalize_parameters,
    normalize_parameters,
    create_parameter_series,
)


def evaluate_single_model_forecast(
    model: Any,
    model_name: str,
    init_state: Dict[str, Any],
    u_series: pd.Series,
    forecast_dates: pd.DatetimeIndex,
    dt_days: float = 91.3125,
) -> Dict[str, np.ndarray]:
    """Generates rate and cumulative trajectories for a standalone base model."""
    res = model.predict_rates(init_state, u_series, forecast_dates)
    qo = np.maximum(res["oil_rate"], 0.0)
    qw = np.maximum(res["water_rate"], 0.0)
    qg = RS_SOLUTION * qo
    
    Qo = init_state["Qo"] + np.cumsum(qo * dt_days)
    Qw = init_state["Qw"] + np.cumsum(qw * dt_days)
    Qg = init_state["Qg"] + RS_SOLUTION * (Qo - init_state["Qo"])
    
    return {
        "qo_rate": qo,
        "qw_rate": qw,
        "qg_rate": qg,
        "cum_oil": Qo,
        "cum_water": Qw,
        "cum_gas": Qg,
    }


def compute_local_jacobians(
    models: Dict[str, Any],
    weights: Dict[str, Dict[str, float]],
    init_state: Dict[str, Any],
    base_u: np.ndarray,
    horizons: List[int] = [4, 12, 40, 80],
    rel_step: float = 0.02,
) -> pd.DataFrame:
    """
    Computes numerical Jacobians of terminal rates and cumulative production
    with respect to each of the 4 geological parameters.
    """
    dt_days = 91.3125
    records = []
    
    origin = pd.Timestamp(init_state["origin_date"])
    max_h = max(horizons)
    all_dates = pd.date_range(origin + pd.DateOffset(days=1), periods=max_h, freq="QS")
    
    model_keys = list(models.keys()) + ["Blend_Phase_Specific_Simplex"]
    
    for h in horizons:
        f_dates = all_dates[:h]
        
        # 1. Base evaluation at unperturbed u
        u_base_ser = create_parameter_series(base_u)
        base_preds = {}
        for m_name in models.keys():
            base_preds[m_name] = evaluate_single_model_forecast(
                models[m_name], m_name, init_state, u_base_ser, f_dates, dt_days
            )
        _, aud_ens = generate_scenario_predictions(
            models=models,
            weights=weights,
            init_state=init_state,
            params=u_base_ser,
            forecast_dates=f_dates,
            decline_sensitivity=1.0,
        )
        base_preds["Blend_Phase_Specific_Simplex"] = aud_ens
        
        # 2. Perturbation loop over each parameter
        for p_idx, p_name in enumerate(PARAM_NAMES):
            delta = RANGES[p_idx] * rel_step
            u_plus = base_u.copy()
            u_minus = base_u.copy()
            u_plus[p_idx] = min(UPPER_BOUNDS[p_idx], base_u[p_idx] + delta)
            u_minus[p_idx] = max(LOWER_BOUNDS[p_idx], base_u[p_idx] - delta)
            actual_d = u_plus[p_idx] - u_minus[p_idx]
            
            u_p_ser = create_parameter_series(u_plus)
            u_m_ser = create_parameter_series(u_minus)
            
            for m_key in model_keys:
                if m_key == "Blend_Phase_Specific_Simplex":
                    _, aud_p = generate_scenario_predictions(
                        models, weights, init_state, u_p_ser, f_dates, 1.0
                    )
                    _, aud_m = generate_scenario_predictions(
                        models, weights, init_state, u_m_ser, f_dates, 1.0
                    )
                    out_p, out_m = aud_p, aud_m
                else:
                    out_p = evaluate_single_model_forecast(
                        models[m_key], m_key, init_state, u_p_ser, f_dates, dt_days
                    )
                    out_m = evaluate_single_model_forecast(
                        models[m_key], m_key, init_state, u_m_ser, f_dates, dt_days
                    )
                
                # Terminal Rate Jacobian
                d_term_qo = (out_p["qo_rate"][-1] - out_m["qo_rate"][-1]) / actual_d
                d_term_qw = (out_p["qw_rate"][-1] - out_m["qw_rate"][-1]) / actual_d
                
                # Cumulative Production Jacobian
                d_cum_qo = (out_p["cum_oil"][-1] - out_m["cum_oil"][-1]) / actual_d
                d_cum_qw = (out_p["cum_water"][-1] - out_m["cum_water"][-1]) / actual_d
                
                # Normalized Sensitivity S = (RANGES[p] / Base) * dY/du
                base_term_qo = max(base_preds[m_key]["qo_rate"][-1], 1.0)
                base_term_qw = max(base_preds[m_key]["qw_rate"][-1], 1.0)
                base_cum_qo = max(base_preds[m_key]["cum_oil"][-1], 1.0)
                base_cum_qw = max(base_preds[m_key]["cum_water"][-1], 1.0)
                
                norm_sens_term_qo = (RANGES[p_idx] / base_term_qo) * d_term_qo
                norm_sens_term_qw = (RANGES[p_idx] / base_term_qw) * d_term_qw
                norm_sens_cum_qo = (RANGES[p_idx] / base_cum_qo) * d_cum_qo
                norm_sens_cum_qw = (RANGES[p_idx] / base_cum_qw) * d_cum_qw
                
                records.append({
                    "horizon_quarters": h,
                    "model_name": m_key,
                    "parameter": p_name,
                    "short_name": SHORT_NAMES[p_idx],
                    "jac_terminal_oil_rate": d_term_qo,
                    "jac_terminal_water_rate": d_term_qw,
                    "jac_cum_oil": d_cum_qo,
                    "jac_cum_water": d_cum_qw,
                    "norm_sens_terminal_oil": norm_sens_term_qo,
                    "norm_sens_terminal_water": norm_sens_term_qw,
                    "norm_sens_cum_oil": norm_sens_cum_qo,
                    "norm_sens_cum_water": norm_sens_cum_qw,
                })
                
    return pd.DataFrame(records)


def compute_global_sensitivity_analysis(
    models: Dict[str, Any],
    weights: Dict[str, Dict[str, float]],
    init_state: Dict[str, Any],
    n_samples: int = 256,
    seed: int = 42,
    horizon_quarters: int = 80,
) -> pd.DataFrame:
    """
    Computes global variance-based sensitivity indices across a space-filling
    Sobol design on [0, 1]^4.
    """
    dt_days = 91.3125
    origin = pd.Timestamp(init_state["origin_date"])
    f_dates = pd.date_range(origin + pd.DateOffset(days=1), periods=horizon_quarters, freq="QS")
    
    # Sobol sampler (power of 2)
    sampler = qmc.Sobol(d=4, seed=seed)
    # Get 2 * n_samples for first-order estimation
    u_unit = sampler.random(n_samples)
    u_phys = denormalize_parameters(u_unit)
    
    model_keys = list(models.keys()) + ["Blend_Phase_Specific_Simplex"]
    
    # Store outputs: model -> array of shape (n_samples,)
    eur_o_outputs = {m: np.zeros(n_samples) for m in model_keys}
    eur_w_outputs = {m: np.zeros(n_samples) for m in model_keys}
    term_o_outputs = {m: np.zeros(n_samples) for m in model_keys}
    term_w_outputs = {m: np.zeros(n_samples) for m in model_keys}
    
    for i in range(n_samples):
        u_ser = create_parameter_series(u_phys[i])
        for m_name in models.keys():
            res = evaluate_single_model_forecast(
                models[m_name], m_name, init_state, u_ser, f_dates, dt_days
            )
            eur_o_outputs[m_name][i] = res["cum_oil"][-1]
            eur_w_outputs[m_name][i] = res["cum_water"][-1]
            term_o_outputs[m_name][i] = res["qo_rate"][-1]
            term_w_outputs[m_name][i] = res["qw_rate"][-1]
            
        _, aud = generate_scenario_predictions(
            models, weights, init_state, u_ser, f_dates, 1.0
        )
        eur_o_outputs["Blend_Phase_Specific_Simplex"][i] = aud["cum_oil"][-1]
        eur_w_outputs["Blend_Phase_Specific_Simplex"][i] = aud["cum_water"][-1]
        term_o_outputs["Blend_Phase_Specific_Simplex"][i] = aud["qo_rate"][-1]
        term_w_outputs["Blend_Phase_Specific_Simplex"][i] = aud["qw_rate"][-1]
        
    records = []
    
    # Analyze variance decomposition for each model
    for m_key in model_keys:
        y_eur_o = eur_o_outputs[m_key]
        y_eur_w = eur_w_outputs[m_key]
        
        var_eur_o = float(np.var(y_eur_o))
        var_eur_w = float(np.var(y_eur_w))
        std_eur_o = float(np.std(y_eur_o))
        std_eur_w = float(np.std(y_eur_w))
        spread_eur_o = float(np.max(y_eur_o) - np.min(y_eur_o))
        spread_eur_w = float(np.max(y_eur_w) - np.min(y_eur_w))
        
        # Binned main effects (approximation of first-order Sobol index Si)
        n_bins = 4
        for p_idx, p_name in enumerate(PARAM_NAMES):
            # Bin by parameter value
            bin_idx = np.digitize(u_unit[:, p_idx], np.linspace(0.0, 1.0, n_bins + 1)[:-1]) - 1
            
            bin_means_o = [np.mean(y_eur_o[bin_idx == b]) for b in range(n_bins) if np.sum(bin_idx == b) > 0]
            bin_means_w = [np.mean(y_eur_w[bin_idx == b]) for b in range(n_bins) if np.sum(bin_idx == b) > 0]
            
            var_main_o = float(np.var(bin_means_o)) if len(bin_means_o) > 1 else 0.0
            var_main_w = float(np.var(bin_means_w)) if len(bin_means_w) > 1 else 0.0
            
            s1_o = (var_main_o / var_eur_o) if var_eur_o > 1e-12 else 0.0
            s1_w = (var_main_w / var_eur_w) if var_eur_w > 1e-12 else 0.0
            
            records.append({
                "model_name": m_key,
                "parameter": p_name,
                "short_name": SHORT_NAMES[p_idx],
                "total_eur_oil_std_stb": std_eur_o,
                "total_eur_oil_spread_stb": spread_eur_o,
                "total_eur_water_std_stb": std_eur_w,
                "total_eur_water_spread_stb": spread_eur_w,
                "sobol_first_order_eur_oil": min(1.0, s1_o),
                "sobol_first_order_eur_water": min(1.0, s1_w),
            })
            
    return pd.DataFrame(records)


def extract_fitted_representations_audit(models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Direct mathematical demonstration of model representation sensitivities:
    Extracts RidgeCV coefficients and ExtraTrees feature importances.
    """
    out = {}
    
    # 1. Learned Exponential DCA
    m_exp = models.get("Learned_Exponential_DCA")
    if m_exp is not None:
        coef_D = dict(zip(m_exp.feat_cols, [float(c) for c in m_exp.model_D.coef_]))
        coef_w = dict(zip(m_exp.feat_cols, [float(c) for c in m_exp.model_water_growth.coef_]))
        out["exponential_dca"] = {
            "alpha_D": float(m_exp.model_D.alpha_),
            "intercept_D": float(m_exp.model_D.intercept_),
            "coef_D": coef_D,
            "alpha_water": float(m_exp.model_water_growth.alpha_),
            "intercept_water": float(m_exp.model_water_growth.intercept_),
            "coef_water": coef_w,
        }
        
    # 2. Recursive ExtraTrees
    m_et = models.get("Recursive_ExtraTrees")
    if m_et is not None:
        imp_o = dict(zip(STATE_FEATURES, [float(x) for x in m_et.model_qo.feature_importances_]))
        imp_w = dict(zip(STATE_FEATURES, [float(x) for x in m_et.model_qw.feature_importances_]))
        out["extratrees"] = {
            "oil_feature_importances": imp_o,
            "water_feature_importances": imp_w,
        }
        
    return out
